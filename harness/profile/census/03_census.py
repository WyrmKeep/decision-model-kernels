#!/usr/bin/env python3
"""Step 3: analytic census of every model pass of each request type (python3 stdlib only; needs steps 1 and 2).

Run:  python3 03_census.py [--out DIR]        (about 30 s, CPU only)

Inputs : tensors.tsv, hparams.tsv (01), cases-plan.tsv, token-counts.tsv (02), census_lib.py (the model of one pass).
         The tables are found with census_lib.data_file() in the census data set ($DMK_CENSUS_DATA, default data/2026-10-30-shape-census).
Outputs go to --out (default: tables/ of the census data set).
The passes of a request depend on which slots (= seq ids) the parent and the children get (llama-batch.cpp split_equal with
sequential = false, as llama-memory-hybrid.cpp chooses it for --kv-unified; server-context.cpp slot order), which the server decides by least-recently-used order and the benchmark
does not fix. Every table therefore uses all 3*2 (two questions) or 3*2*1 (three questions) slot arrangements.
Outputs:
  census.tsv          one row per (representative request, slot structure, pass, op group): matmul shapes M,N,K, type, CUDA path,
                      launch prediction, FLOPs, weight bytes; plus flash attention, Gated DeltaNet and element-wise rows.
  census-sequence.tsv ordered per-launch list of the mul_mat nodes of every pass of the representative requests.
  census-passes.tsv   one row per pass of the representative requests: tokens, outputs, FLOPs, bytes, roofline floors.
  census-cases.tsv    one row per (variant, case, slot arrangement): passes, FLOPs, bytes, floors.
  census-summary.tsv  per task / request type, per slot structure and uniform over the arrangements.
  census-head.tsv     output head: rows computed vs rows the API reads, per request type.
  request-floors.tsv  one row per request type for the request of the profile runs (check03, triage09, medium01-memory, long00-storage):
                      passes, tokens, weight bytes, floors and launch counts summed over its passes.
Floors (all per request, summed over its passes; nothing measured):
  floor_weights_ms   = weight bytes of the pass / 448 GB/s                      (the first estimate of the project used a 6 ms floor)
  floor_int8_ms      = useful int8 ops (2*M*N*K of the matmuls) / 144.8 TOPS
  floor_pass_max_ms  = per pass max(floor_weights, floor_int8), summed over the passes
  floor_roofline_ms  = sum over kernels of max(bytes / 448 GB/s, ops / peak) with all activation traffic taken from DRAM
                       (matmul: weights + quantised input + output; quantize kernels; Gated DeltaNet; flash attention; element-wise ops)
"""
import argparse
import csv
import os
import statistics
from collections import OrderedDict

import census_lib as L

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--out", default=os.path.join(L.data_dir(), "tables"), help="output folder (default: tables/ of the census data set)")
OUT = _ap.parse_args().out
os.makedirs(OUT, exist_ok=True)
M = L.Model()
BW = L.BW_GBPS * 1e9
INT8 = L.INT8_TOPS * 1e12
FP16 = L.FP16_ACC32_TFLOPS * 1e12
FP32 = L.FP32_TFLOPS * 1e12


def read_tsv(name):
    with open(L.data_file(name)) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fmt(x, nd=3):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_tsv(name, header, rows):
    with open(os.path.join(OUT, name), "w") as o:
        o.write("\t".join(header) + "\n")
        for r in rows:
            o.write("\t".join(fmt(r.get(h, "")) for h in header) + "\n")


def pass_costs(p):
    return L.pass_costs(M, p)


def describe(passes):
    parts = []
    for p in passes:
        if p["kind"] == "prefix":
            parts.append(f"P{p['n_tokens']}")
        elif p["kind"].startswith("single"):
            parts.append(f"F{p['n_tokens']}")
        else:
            parts.append(f"S{len(p['seqs'])}x{p['seqs'][0][1]}")
    return "+".join(parts)


def structure(passes):
    """Slot structure of the request = number of sequences of every pass, e.g. P1 S3 S2 S1."""
    return " ".join(("P" if p["kind"] == "prefix" else "F" if p["kind"].startswith("single") else "S") + str(len(p["seqs"])) for p in passes)


# ------------------------------------------------------------------ cases x slot arrangements
plans = read_tsv("cases-plan.tsv")
tcounts = read_tsv("token-counts.tsv")
qinfo = {}
for r in tcounts:
    qinfo.setdefault((r["variant"], r["case"]), []).append(r)

GROUPS = OrderedDict([  # task -> request type
    ("check", "short-1q"), ("commit", "short-1q"), ("triage", "short-2q"), ("style", "short-2q"),
    ("syslog-medium", "medium-3q"), ("syslog-long", "long-3q"),
])

case_rows = []
per_case = {}      # (variant, case) -> {slots: (passes, costs)}
for plan in plans:
    shared = int(plan["shared_prefix_tokens"])
    suff = [int(x) for x in plan["suffix_tokens_per_question"].split()]
    qs = qinfo[(plan["variant"], plan["case"])]
    labels_needed = sum(int(q["n_options"]) for q in qs)
    arrs = L.arrangements(len(suff)) if len(suff) > 1 else [(0,)]
    d = {}
    for slots in arrs:
        passes = L.simulate_request(shared, suff, slots)
        costs = [pass_costs(p) for p in passes]
        d[slots] = (passes, costs)
        row = {
            "variant": plan["variant"], "case": plan["case"], "task": plan["task"], "request_type": GROUPS[plan["task"]],
            "n_questions": plan["n_questions"], "sum_prompt_tokens": plan["sum_prompt_tokens"],
            "shared_prefix_tokens": plan["shared_prefix_tokens"], "suffix_tokens_per_question": plan["suffix_tokens_per_question"],
            "slots_Q0_Q1_Q2": ",".join(map(str, slots)) if len(suff) > 1 else "-", "structure": structure(passes),
            "n_passes": len(passes), "passes": describe(passes),
            "tokens_evaluated": sum(p["n_tokens"] for p in passes),
            "n_matmul_launches": sum(c["n_matmul"] for c in costs), "n_mmq": sum(c["n_mmq"] for c in costs), "n_mmvq": sum(c["n_mmvq"] for c in costs),
            "matmul_gflop": sum(c["flops"] for c in costs) / 1e9,
            "matmul_padded_gops": sum(c["padded"] for c in costs) / 1e9,
            "weight_gb_read": sum(c["wbytes"] for c in costs) / 1e9,
            "head_gb_read": sum(c["head_bytes"] for c in costs) / 1e9,
            "fa_gflop": sum(c["fa_flops"] for c in costs) / 1e9,
            "gdn_gflop": sum(c["gdn_flops"] for c in costs) / 1e9,
            "state_traffic_gb": sum(c["state_bytes"] for c in costs) / 1e9,
            "other_activation_traffic_gb": sum(c["side_bytes"] + c["quant_bytes"] + c["mmout_bytes"] for c in costs) / 1e9,
            "floor_weights_ms": sum(c["floor_weights_s"] for c in costs) * 1e3,
            "floor_int8_ms": sum(c["floor_int8_s"] for c in costs) * 1e3,
            "floor_pass_max_ms": sum(c["floor_pass_max_s"] for c in costs) * 1e3,
            "floor_roofline_ms": sum(c["floor_roofline_s"] for c in costs) * 1e3,
            "head_rows_computed": sum(p["n_out"] for p in passes),
            "head_rows_needed": labels_needed,
        }
        case_rows.append(row)
    per_case[(plan["variant"], plan["case"])] = d

CASE_HDR = list(case_rows[0].keys())
write_tsv("census-cases.tsv", CASE_HDR, case_rows)

# ------------------------------------------------------------------ summary per task / request type / slot structure
SUM_COLS = ["n_passes", "tokens_evaluated", "sum_prompt_tokens", "n_matmul_launches", "n_mmq", "n_mmvq", "matmul_gflop", "weight_gb_read",
            "head_gb_read", "fa_gflop", "gdn_gflop", "state_traffic_gb", "other_activation_traffic_gb", "floor_weights_ms", "floor_int8_ms", "floor_pass_max_ms", "floor_roofline_ms"]
sum_rows = []


def summarize(label, kind, struct, rows):
    d = {"group": label, "kind": kind, "variant": rows[0]["variant"], "slot_structure": struct, "n_cases": len({r["case"] for r in rows}),
         "n_case_arrangements": len(rows), "passes_example": rows[len(rows) // 2]["passes"]}
    for c in SUM_COLS:
        vals = [float(r[c]) for r in rows]
        d[c + "_mean"] = statistics.fmean(vals)
        d[c + "_min"] = min(vals)
        d[c + "_max"] = max(vals)
    return d


for variant in ("ref", "raw"):
    vrows = [r for r in case_rows if r["variant"] == variant]
    for kind, keyf, labels in (("task", lambda r: r["task"], list(GROUPS)),
                               ("request_type", lambda r: r["request_type"], ["short-1q", "short-2q", "medium-3q", "long-3q"])):
        for lab in labels:
            sel = [r for r in vrows if keyf(r) == lab]
            sum_rows.append(summarize(lab, kind, "all (uniform over slot arrangements)", sel))
            structs = []
            for r in sel:
                if r["structure"] not in structs:
                    structs.append(r["structure"])
            if len(structs) > 1:
                for s in sorted(structs):
                    sum_rows.append(summarize(lab, kind, s, [r for r in sel if r["structure"] == s]))
    sum_rows.append(summarize("all 48 cases", "benchmark", "all (uniform over slot arrangements)", vrows))
SUM_HDR = list(sum_rows[0].keys())
write_tsv("census-summary.tsv", SUM_HDR, sum_rows)

# ------------------------------------------------------------------ representative requests: census.tsv etc.
reps = OrderedDict()
for task in GROUPS:
    sel = sorted([r for r in case_rows if r["variant"] == "ref" and r["task"] == task], key=lambda r: int(r["sum_prompt_tokens"]))
    reps[task] = sel[len(sel) // 2]["case"]

census_rows, seq_rows, pass_rows = [], [], []
CENSUS_HDR = ["request_type", "task", "rep_case", "slot_structure", "slots_Q0_Q1_Q2", "pass_idx", "pass_kind", "n_seqs", "tokens_per_seq", "N_tokens", "n_outputs",
              "op", "n_layers", "M_out_rows", "N_tokens_op", "K", "wtype", "path", "kernel_predicted", "mmq_J", "mmq_tiles_m", "mmq_tiles_n",
              "grid_x_predicted", "fixup_predicted", "tile_eff_pct", "flops_total_mflop", "padded_ops_total_mop", "weight_bytes_total_mb", "other_bytes_total_mb"]
n_r, n_a = M.pattern.count("R"), M.pattern.count("A")

for task, case in reps.items():
    d = per_case[("ref", case)]
    rtype = GROUPS[task]
    seen = set()
    for slots, (passes, costs) in d.items():
        st = structure(passes)
        if st in seen:
            continue
        seen.add(st)
        slots_s = ",".join(map(str, slots)) if len(slots) > 1 else "-"
        for pi, (p, c) in enumerate(zip(passes, costs)):
            n_seqs = c["n_seqs"]
            base = {"request_type": rtype, "task": task, "rep_case": case, "slot_structure": st, "slots_Q0_Q1_Q2": slots_s, "pass_idx": pi, "pass_kind": p["kind"],
                    "n_seqs": n_seqs, "tokens_per_seq": p["seqs"][0][1], "N_tokens": p["n_tokens"], "n_outputs": p["n_out"]}
            groups = OrderedDict()
            for o in c["mms"]:
                last = o["layer"] == M.n_layer - 1 and o["op"].startswith("ffn")
                name = o["op"] + (" @layer31 (n_outputs rows only)" if last else "")
                groups.setdefault((name, o["M"], o["N"], o["K"], o["wtype"], o["path"]), []).append(o)
            for (name, m, n, k, wt, path), os_ in groups.items():
                o = os_[0]
                kernel = o["kernel"].replace("ncols=1>", "1,fusion>") if o.get("fused") else o["kernel"]
                r = dict(base)
                r.update({"op": name, "n_layers": len(os_), "M_out_rows": m, "N_tokens_op": n, "K": k, "wtype": wt, "path": path,
                          "kernel_predicted": kernel, "mmq_J": o["J"], "mmq_tiles_m": o["tiles_m"], "mmq_tiles_n": o["tiles_n"],
                          "grid_x_predicted": o["grid_x"], "fixup_predicted": o["fixup"], "tile_eff_pct": o["tile_eff_pct"],
                          "flops_total_mflop": len(os_) * o["flops"] / 1e6, "padded_ops_total_mop": len(os_) * o["padded_ops"] / 1e6,
                          "weight_bytes_total_mb": len(os_) * o["weight_bytes"] / 1e6,
                          "other_bytes_total_mb": len(os_) * (o["act_in_f32_bytes"] + o["act_q8_bytes"] * 2 + o["act_out_f32_bytes"]) / 1e6})
                census_rows.append(r)
            att, gdn = c["att"], c["gdn"]
            r = dict(base); r.update({"op": "gated_delta_net (fused op)", "n_layers": n_r, "M_out_rows": M.gdn_heads_v, "N_tokens_op": p["n_tokens"], "K": M.gdn_dim,
                                       "wtype": "F32 state", "path": "gated_delta_net", "kernel_predicted": f"{gdn['kernel']} grid {gdn['grid'][0]}x{gdn['grid'][1]}x{gdn['grid'][2]}",
                                       "grid_x_predicted": gdn["blocks"], "flops_total_mflop": n_r * gdn["flops"] / 1e6,
                                       "other_bytes_total_mb": n_r * (gdn["bytes"] + gdn["state_copy_bytes"]) / 1e6})
            census_rows.append(r)
            r = dict(base); r.update({"op": "flash_attn_ext (attention layers)", "n_layers": n_a, "M_out_rows": M.n_head, "N_tokens_op": p["n_tokens"], "K": M.head_dim,
                                       "wtype": "F16 KV", "path": "flash-attn mma", "kernel_predicted": att["kernel"],
                                       "grid_x_predicted": att["q_tiles"], "flops_total_mflop": n_a * att["flops"] / 1e6,
                                       "other_bytes_total_mb": n_a * (att["kv_bytes"] + att["kv_write_bytes"]) / 1e6})
            census_rows.append(r)
            r = dict(base); r.update({"op": "other activation traffic (norms, conv, silu, swiglu, adds, rope, cpy; estimate)", "n_layers": M.n_layer,
                                       "N_tokens_op": p["n_tokens"], "path": "element-wise", "other_bytes_total_mb": c["side_bytes"] / 1e6})
            census_rows.append(r)

            pass_rows.append({
                "request_type": rtype, "task": task, "rep_case": case, "slot_structure": st, "slots_Q0_Q1_Q2": slots_s, "pass_idx": pi, "pass_kind": p["kind"], "n_seqs": n_seqs,
                "tokens_per_seq": p["seqs"][0][1], "N_tokens": p["n_tokens"], "n_outputs": p["n_out"], "start_pos": p["seqs"][0][0], "n_kv_window": att["n_kv"], "fa_ncols1": att["ncols1"],
                "n_matmul": c["n_matmul"], "n_mmq": c["n_mmq"], "n_mmvq": c["n_mmvq"], "n_mmq_fixup_pred": c["n_mmq_fixup"], "gdn_kernel": c["gdn"]["kernel"],
                "matmul_gflop": c["flops"] / 1e9, "padded_gops": c["padded"] / 1e9, "weight_mb": c["wbytes"] / 1e6, "head_mb": c["head_bytes"] / 1e6,
                "fa_gflop": c["fa_flops"] / 1e9, "gdn_gflop": c["gdn_flops"] / 1e9, "state_mb": c["state_bytes"] / 1e6, "kv_mb": c["kv_bytes"] / 1e6,
                "side_activation_mb": c["side_bytes"] / 1e6, "quant_mb": c["quant_bytes"] / 1e6, "mmout_mb": c["mmout_bytes"] / 1e6,
                "floor_weights_ms": c["floor_weights_s"] * 1e3, "floor_int8_ms": c["floor_int8_s"] * 1e3, "floor_int8_padded_ms": c["floor_int8_padded_s"] * 1e3,
                "floor_pass_max_ms": c["floor_pass_max_s"] * 1e3, "floor_roofline_ms": c["floor_roofline_s"] * 1e3})
            for i, o in enumerate(c["mms"]):
                seq_rows.append({"rep_case": case, "slots_Q0_Q1_Q2": slots_s, "pass_idx": pi, "N_tokens": p["n_tokens"], "n_outputs": p["n_out"], "order": i,
                                 "layer": o["layer"], "op": o["op"], "wtype": o["wtype"], "M": o["M"], "N": o["N"], "K": o["K"], "path": o["path"],
                                 "kernel_predicted": o["kernel"], "mmq_J": o["J"], "grid_x_predicted": o["grid_x"], "block": o["block"], "fixup_predicted": o["fixup"]})

write_tsv("census.tsv", CENSUS_HDR, census_rows)
SEQ_HDR = ["rep_case", "slots_Q0_Q1_Q2", "pass_idx", "N_tokens", "n_outputs", "order", "layer", "op", "wtype", "M", "N", "K", "path", "kernel_predicted", "mmq_J", "grid_x_predicted", "block", "fixup_predicted"]
write_tsv("census-sequence.tsv", SEQ_HDR, seq_rows)
write_tsv("census-passes.tsv", list(pass_rows[0].keys()), pass_rows)

# ------------------------------------------------------------------ one row per request type: the request that the profile runs use
# (request type, case, slot structure). Each is a row group of census-passes.tsv; the sums run over the passes of the request.
# pass_structure is the column `passes` of census-cases.tsv (ref variant, the first slot arrangement with this structure).
PROFILE_REQUESTS = [("short-1q", "check03", "F1"), ("short-2q", "triage09", "P1 S2 S1"),
                    ("medium-3q", "medium01-memory", "P1 S3 S2 S1"), ("long-3q", "long00-storage", "P1 P1 S3 S2 S1")]
floor_rows = []
for rtype, case, struct in PROFILE_REQUESTS:
    sel = [p for p in pass_rows if p["request_type"] == rtype and p["rep_case"] == case and p["slot_structure"] == struct]
    assert sel, f"no pass rows for {rtype} {case} {struct}"
    arr_passes = next(ps for ps, _ in per_case[("ref", case)].values() if structure(ps) == struct)
    floor_rows.append({
        "request_type": rtype, "rep_case": case, "slot_structure": struct, "pass_structure": describe(arr_passes), "n_passes": len(sel),
        "tokens_evaluated": sum(p["N_tokens"] for p in sel),
        "weight_gb_read": sum(p["weight_mb"] for p in sel) / 1e3,
        "floor_weights_ms": sum(p["floor_weights_ms"] for p in sel),
        "floor_pass_max_ms": sum(p["floor_pass_max_ms"] for p in sel),
        "floor_roofline_ms": sum(p["floor_roofline_ms"] for p in sel),
        "n_mmq": sum(p["n_mmq"] for p in sel), "n_mmvq": sum(p["n_mmvq"] for p in sel)})
write_tsv("request-floors.tsv", list(floor_rows[0].keys()), floor_rows)

# ------------------------------------------------------------------ output head: rows computed vs needed
head_rows = []
for task, case in reps.items():
    qs = qinfo[("ref", case)]
    n_need = sum(int(q["n_options"]) for q in qs)
    passes = next(iter(per_case[("ref", case)].values()))[0]
    n_comp = sum(p["n_out"] for p in passes)
    n_head = sum(1 for p in passes if p["n_out"] > 0)
    head_rows.append({
        "request_type": GROUPS[task], "task": task, "rep_case": case, "n_questions": len(qs),
        "passes_with_head": n_head, "head_rows_computed_per_request": n_comp * M.n_vocab, "head_rows_needed_per_request": n_need,
        "head_weight_mb_read_per_request": n_head * M.tok_embd_bytes / 1e6,
        "head_weight_mb_needed_per_request": n_need * L.weight_bytes("Q8_0", 1, M.n_embd) / 1e6,
        "head_flops_computed_gflop": 2 * n_comp * M.n_vocab * M.n_embd / 1e9,
        "head_flops_needed_gflop": 2 * n_need * M.n_embd / 1e9,
        "head_time_at_448GBps_ms": n_head * M.tok_embd_bytes / BW * 1e3})
write_tsv("census-head.tsv", list(head_rows[0].keys()), head_rows)

print("representative cases:", dict(reps))
print("wrote census.tsv, census-sequence.tsv, census-passes.tsv, request-floors.tsv, census-cases.tsv, census-summary.tsv, census-head.tsv")
