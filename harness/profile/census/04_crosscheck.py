#!/usr/bin/env python3
"""Step 4: cross-check the analytic census (census_lib.py, 03_census.py) against the three Nsight Systems traces
(trace-passes.tsv, trace-mmq-launches.tsv, trace-kernels.tsv written by trace_export.sh + trace_census.py). python3 stdlib only.

Run:  python3 04_crosscheck.py [--out DIR]        (a few seconds, CPU only)

Inputs : trace-passes.tsv, trace-mmq-launches.tsv, trace-kernels.tsv, cases-plan.tsv, tensors.tsv, hparams.tsv, found with
         census_lib.data_file() in the census data set ($DMK_CENSUS_DATA, default data/2026-10-30-shape-census).
Output folder: --out (default: tables/ of the census data set). The program prints the result of each check (passed / total).

Outputs
  crosscheck-requests.tsv  every traced request: observed pass sequence (tokens, outputs) vs the slot arrangements that give it.
  crosscheck-passes.tsv    every traced pass: observed vs predicted launch counts and a launch-by-launch comparison.
  crosscheck-keys.tsv      launch keys (type, tile, grid, fix-up) whose observed and predicted counts differ, summed over all passes.
  crosscheck-kernels.tsv   whole-trace kernel counts per kernel name / grid: observed vs predicted (flash attention, Gated DeltaNet).
  crosscheck-summary.tsv   one row per check: passed / total.
Matmul launches are compared as multisets per pass (key = weight type, tile width, need_check, grid.x, fix-up kernel) because the
graph execution order of ggml is a depth-first order and differs from the order in which qwen35.cpp creates the nodes.
"""
import argparse
import csv
import os
from collections import Counter, OrderedDict, defaultdict

import census_lib as L

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--out", default=os.path.join(L.data_dir(), "tables"), help="output folder (default: tables/ of the census data set)")
OUT = _ap.parse_args().out
os.makedirs(OUT, exist_ok=True)
M = L.Model()


def read_tsv(name):
    with open(L.data_file(name)) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(name, header, rows):
    with open(os.path.join(OUT, name), "w") as o:
        o.write("\t".join(header) + "\n")
        for r in rows:
            o.write("\t".join(f"{r.get(h, ''):.3f}" if isinstance(r.get(h, ""), float) else str(r.get(h, "")) for h in header) + "\n")


def ival(r, k):
    v = r.get(k, "")
    return int(float(v)) if v not in ("", None) else 0


tp = [r for r in read_tsv("trace-passes.tsv") if r["row_kind"] == "pass"]
launches = read_tsv("trace-mmq-launches.tsv")
kern = read_tsv("trace-kernels.tsv")
plans = {(r["variant"], r["case"]): r for r in read_tsv("cases-plan.tsv")}

by_req = OrderedDict()
for r in tp:
    by_req.setdefault((r["trace"], int(r["request_idx"])), []).append(r)
for v in by_req.values():
    v.sort(key=lambda r: int(r["pass_idx_in_request"]))

lj = defaultdict(list)
for r in launches:
    lj[(r["trace"], int(r["pass_idx_in_trace"]))].append(r)

# ------------------------------------------------------------------ requests: which slot arrangement gives the observed passes?
req_rows = []
chosen = {}   # (trace, request_idx) -> simulated passes of the first matching arrangement
for (trace, ri), plist in by_req.items():
    case = plist[0]["case_id_inferred"]
    plan = plans[("ref", case)]
    suff = [int(x) for x in plan["suffix_tokens_per_question"].split()]
    obs = [(int(p["n_tokens"]), int(p["n_outputs"])) for p in plist]
    shared = sum(int(p["n_tokens"]) for p in plist if p["pass_kind"].startswith("prefix"))
    match = []
    first = None
    for slots in (L.arrangements(len(suff)) if len(suff) > 1 else [(0,)]):
        passes = L.simulate_request(shared if len(suff) > 1 else 0, suff, slots)
        if [(p["n_tokens"], p["n_out"]) for p in passes] == obs:
            match.append(",".join(map(str, slots)))
            first = first or passes
    chosen[(trace, ri)] = first
    uni = [(L.simulate_request(shared, suff, s)) for s in (L.arrangements(len(suff)) if len(suff) > 1 else [(0,)])]
    req_rows.append({
        "trace": trace, "request_idx": ri, "case": case, "suffix_tokens_per_question_model": " ".join(map(str, suff)), "shared_prefix_tokens_trace": shared,
        "observed_passes (tokens/outputs)": " ".join(f"{n}/{o}" for n, o in obs), "n_passes_observed": len(obs),
        "slot_arrangements_matching_Q0,Q1,Q2": " | ".join(match) if match else "none", "matches": "yes" if match else "NO",
        "n_passes_model_min": min(len(u) for u in uni), "n_passes_model_max": max(len(u) for u in uni),
        "tokens_observed": sum(n for n, _ in obs), "tokens_model_same": "yes" if sum(n for n, _ in obs) == shared + sum(suff) else "NO"})
write_tsv("crosscheck-requests.tsv", list(req_rows[0].keys()), req_rows)

# ------------------------------------------------------------------ passes
def mmq_key_expected(o):
    return ("MMQ", o["wtype"], int(o["J"]), int(o["fallback"]), int(o["grid_x"]), int(o["fixup"]))


def mmvq_key_expected(o):
    return ("MMVQ", o["wtype"], int(o["N"]), int(bool(o.get("fused"))), int(o["grid_x"]), int(o["small_k"]), int(o["block"].split("x")[1]))


def key_observed(r):
    if r["launch_kind"] == "mmq":
        return ("MMQ", r["quant_type"], int(r["mmq_x"]), int(r["need_check"] == "true"), int(r["grid_x"]), int(r["has_fixup"] in ("1", "true", "True")))
    return ("MMVQ", r["quant_type"], int(r["ncols_dst"]), int(r["has_fusion"] == "true"), int(r["grid_x"]), int(r["small_k"] == "true"), int(r["block_y"]))


pass_rows = []
key_diff = Counter()
key_total_exp = Counter()
key_total_obs = Counter()
checks = OrderedDict()


def chk(name, ok):
    c = checks.setdefault(name, [0, 0])
    c[1] += 1
    c[0] += 1 if ok else 0


for (trace, ri), plist in by_req.items():
    sim = chosen[(trace, ri)]
    for i, p in enumerate(plist):
        n, n_out = int(p["n_tokens"]), int(p["n_outputs"])
        pidx = int(p["pass_idx_in_trace"])
        sp = sim[i] if sim else None
        n_seqs = len(sp["seqs"]) if sp else ""
        mms = L.pass_matmuls(M, n, n_out)
        exp_keys = Counter(mmq_key_expected(o) if o["path"] == "MMQ" else mmvq_key_expected(o) for o in mms)
        has_rows = (trace, pidx) in lj
        obs_keys = Counter(key_observed(r) for r in lj[(trace, pidx)])
        matched = sum((exp_keys & obs_keys).values())
        if has_rows:
            for k, v in (exp_keys - obs_keys).items():
                key_diff[("predicted only",) + k] += v
            for k, v in (obs_keys - exp_keys).items():
                key_diff[("observed only",) + k] += v
            key_total_exp.update(exp_keys)
            key_total_obs.update(obs_keys)
        exp_mmq = sum(1 for o in mms if o["path"] == "MMQ")
        exp_fix = sum(1 for o in mms if o["path"] == "MMQ" and o["fixup"])
        att = L.attn_cost(M, sp["seqs"]) if sp else None
        pc = L.pass_costs(M, sp) if sp else None
        row = {
            "trace": trace, "request_idx": ri, "pass_idx_in_request": int(p["pass_idx_in_request"]), "pass_kind": p["pass_kind"], "N_tokens": n, "n_outputs": n_out,
            "n_seqs_model": n_seqs,
            "n_kernels_observed": ival(p, "n_kernels"),
            "n_mmq_obs": ival(p, "n_mmq"), "n_mmq_pred": exp_mmq,
            "n_mmq_fixup_obs": ival(p, "n_mmq_fixup"), "n_mmq_fixup_pred": exp_fix,
            "n_mmvq_obs": ival(p, "n_mmvq"), "n_mmvq_pred": len(mms) - exp_mmq,
            "n_quantize_obs": ival(p, "n_quantize"), "n_quantize_pred": len(mms),
            "n_flash_obs": ival(p, "n_flash_attn"), "n_flash_pred": M.pattern.count("A"),
            "n_gdn_obs": ival(p, "n_gdn"), "n_gdn_pred": M.pattern.count("R"),
            "launch_keys_pred": sum(exp_keys.values()) if has_rows else "", "launch_keys_obs": sum(obs_keys.values()) if has_rows else "", "launch_keys_matching": matched if has_rows else "",
            "n_kv_padded_obs": ival(p, "n_kv_padded"), "n_kv_window_model": att["n_kv"] if att else "",
            "kv_window_equal": "yes" if att and att["n_kv"] == ival(p, "n_kv_padded") else "no",
            "model_floor_weights_ms": pc["floor_weights_s"] * 1e3 if pc else "", "model_floor_int8_ms": pc["floor_int8_s"] * 1e3 if pc else "",
            "model_floor_pass_max_ms": pc["floor_pass_max_s"] * 1e3 if pc else "", "model_floor_roofline_ms": pc["floor_roofline_s"] * 1e3 if pc else "",
            "observed_gpu_busy_ms": float(p["gpu_busy_ms"]), "observed_wall_ms": float(p["wall_ms"]),
            "observed_gap_total_ms": float(p["gap_total_ms"]),
            "busy_over_pass_max_floor": float(p["gpu_busy_ms"]) / (pc["floor_pass_max_s"] * 1e3) if pc else "",
            "busy_over_roofline_floor": float(p["gpu_busy_ms"]) / (pc["floor_roofline_s"] * 1e3) if pc else "",
            "model_mmq_gflop": sum(o["flops"] for o in mms if o["path"] == "MMQ") / 1e9, "model_mmq_weight_mb": sum(o["weight_bytes"] for o in mms if o["path"] == "MMQ") / 1e6,
            "model_mmvq_gflop": sum(o["flops"] for o in mms if o["path"] == "MMVQ") / 1e9, "model_mmvq_weight_mb": sum(o["weight_bytes"] for o in mms if o["path"] == "MMVQ") / 1e6,
            "observed_mmq_kernel_ms": float(p["t_cat_mmq_ms"]), "observed_mmvq_kernel_ms": float(p["t_cat_mmvq_ms"]), "observed_quantize_kernel_ms": float(p["t_cat_quant_ms"]),
            "observed_flash_kernel_ms": float(p["t_cat_flash_ms"]), "observed_gdn_kernel_ms": float(p["t_cat_gdn_ms"]),
        }
        pass_rows.append(row)
        chk("matmul count (MMQ) per pass", row["n_mmq_obs"] == exp_mmq)
        chk("matmul count (MMVQ) per pass", row["n_mmvq_obs"] == len(mms) - exp_mmq)
        chk("stream-k fix-up launches per pass", row["n_mmq_fixup_obs"] == exp_fix)
        chk("quantize launches per pass", row["n_quantize_obs"] == len(mms))
        chk("flash attention launches per pass", row["n_flash_obs"] == M.pattern.count("A"))
        chk("Gated DeltaNet launches per pass", row["n_gdn_obs"] == M.pattern.count("R"))
        if has_rows:
            chk("launch keys (type, tile, need_check, grid.x, fix-up) identical as multiset, passes with launch records", matched == sum(exp_keys.values()) == sum(obs_keys.values()))
        if att:
            chk("flash-attention K/V window (padded n_kv) per pass", att["n_kv"] == ival(p, "n_kv_padded"))
write_tsv("crosscheck-passes.tsv", list(pass_rows[0].keys()), pass_rows)

for r in req_rows:
    chk("pass sequence of the request reproduced by a slot arrangement", r["matches"] == "yes")
    chk("tokens evaluated per request = shared prefix + sum of suffixes", r["tokens_model_same"] == "yes")

key_rows = []
for k, v in sorted(key_diff.items(), key=lambda kv: -kv[1]):
    key_rows.append({"side": k[0], "path": k[1], "wtype": k[2], "tile_J_or_ncols": k[3], "need_check_or_fused": k[4], "grid_x": k[5], "fixup_or_small_k": k[6], "n_launches_summed_over_passes": v})
if key_rows:
    write_tsv("crosscheck-keys.tsv", list(key_rows[0].keys()), key_rows)
else:
    write_tsv("crosscheck-keys.tsv", ["side", "path", "wtype", "tile_J_or_ncols", "need_check_or_fused", "grid_x", "fixup_or_small_k", "n_launches_summed_over_passes"], [])

# ------------------------------------------------------------------ whole-trace kernel counts
kern_rows = []
for trace in ("short", "medium", "long"):
    # expected from the matched arrangements
    exp_fa = Counter()
    exp_gdn = Counter()
    for (t, ri), plist in by_req.items():
        if t != trace or not chosen[(t, ri)]:
            continue
        for sp in chosen[(t, ri)]:
            att = L.attn_cost(M, sp["seqs"])
            exp_fa[att["ncols1"]] += M.pattern.count("A")
            g = L.gdn_cost(M, sp["n_tokens"], len(sp["seqs"]))
            exp_gdn[f"{g['kernel']} grid {g['grid'][0]}x{g['grid'][1]}x{g['grid'][2]}"] += M.pattern.count("R")
    obs_fa, obs_gdn = Counter(), Counter()
    for r in kern:
        if r["trace"] != trace:
            continue
        if r["kernel_short"].startswith("flash_attn_ext_f16<"):
            obs_fa[int(r["kernel_short"].split(",")[2])] += int(r["n_launches"])
        if r["kernel_short"].startswith("gated_delta_net_cuda"):
            obs_gdn[f"gated_delta_net_cuda<128,false,false,4> grid {r['grid_x']}x{r['grid_y']}x{r['grid_z']}"] += int(r["n_launches"])
        if r["kernel_short"].startswith("gdn_single"):
            obs_gdn[f"{r['kernel_short']} grid {r['grid_x']}x{r['grid_y']}x{r['grid_z']}"] += int(r["n_launches"])
    for k in sorted(set(exp_fa) | set(obs_fa)):
        kern_rows.append({"trace": trace, "kernel": f"flash_attn_ext_f16<256,256,{k},4>", "observed_launches": obs_fa[k], "predicted_launches": exp_fa[k]})
        chk("flash-attention kernel template (ncols1) over the whole trace", obs_fa[k] == exp_fa[k])
    for k in sorted(set(exp_gdn) | set(obs_gdn), key=str):
        kern_rows.append({"trace": trace, "kernel": k, "observed_launches": obs_gdn[k], "predicted_launches": exp_gdn[k]})
        chk("Gated DeltaNet kernel and grid (sequences) over the whole trace", obs_gdn[k] == exp_gdn[k])
write_tsv("crosscheck-kernels.tsv", ["trace", "kernel", "observed_launches", "predicted_launches"], kern_rows)

sum_rows = [{"check": k, "passed": v[0], "total": v[1], "all_passed": "yes" if v[0] == v[1] else "NO"} for k, v in checks.items()]
write_tsv("crosscheck-summary.tsv", ["check", "passed", "total", "all_passed"], sum_rows)
for r in sum_rows:
    print(f"{r['passed']:4d}/{r['total']:<4d} {r['all_passed']:3s}  {r['check']}")
print("launch keys predicted:", sum(key_total_exp.values()), "observed:", sum(key_total_obs.values()), "differing key classes:", len(key_rows))
for r in key_rows[:12]:
    print("   ", r)
