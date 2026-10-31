#!/usr/bin/env python3
"""Synthetic data with planted answers for analyze_childpass.py and analyze_kvwin.py (python 3 stdlib only; no GPU).

Everything is generated from the print formats of the real server, so the parsers see what the GPU run will give them:
  * server logs: `DMK_TIMING ubatch|decode|sync|request` lines (src/dmk-timing.h, server-context.cpp:6706) and slot lines in the
    format of a server log of the data set data/2026-10-30-window1 (`0.08.748.281 I slot get_availabl: id  0 | task -1 | selected slot by LRU, ...`);
  * client JSON of kvwin_client.py (schema of the p3 clients: requests[idx, kind, case, phase, pair, ref, t_send_mono_ns, ...]);
  * nsys passes TSV with the header of the census table trace-passes.tsv plus run, segment, client_idx, kind, case.

Scenarios (written to --out, default <temporary folder>/dmk-leads-synthetic; every file has a `*.truth.json` with the planted answer):
  childpass-census   16 + 3 requests (1q, 2q, 3q medium and long) over ALL slot placements, pass structure = the census model (default split =
                     unified, sequential = false: --kv-unified, the production setting); the log says kv_unified = 'true'.
                     truth: ubatch / decode counts per request, requests with a child below the parent, fixed-order counts.
  childpass-seq      the same requests with the sequential split (sequential = true: what --no-kv-unified does; the log says
                     kv_unified = 'false'): differs from the census only for the two-question placement (0,2) (ubatch sizes, not counts).
  childpass-extra    census structure, plus one planted extra ubatch AND one extra decode call in every medium/long request with a
                     child below the parent (a "third decode pass"): truth = the answer `yes`.
  kvwin-growth       nsys passes + timing log of the K/V test with a planted window law: T2..T4 sit behind stale cells (window +3328).
  kvwin-strong       the same with the FA law x6 (a window effect large enough to move the pass time clearly).
  kvwin-null         the same sequence without any window growth.
  kvwin-nokvu        timing log of the --no-kv-unified run (no growth; one extra synchronize + 0.9 ms host copy time per child before the suffix).
FA time law planted in the kvwin data (per pass, 8 attention layers):  fa_ms = A + B * N * w   with A = 0.48 ms, B = 3.03e-6 ms
per (token x cell) (the fit of the three old traces); w = window seen for N < 1024 (the kernel reads the whole window) and
w = eff_front + stale offset for N >= 1024 (tail cells are skipped, interior stale cells are not).
"""
import argparse
import json
import os
import random
import sys
import tempfile

import leadlib as LL

L = LL.L

FA_A, FA_B = 0.48, 3.03e-6
FA_SCALE = 1.0       # multiplies FA_B (set per scenario by generate())
KINDS = ["short1q", "short2q", "medium3q", "long3q"]
REP = {"short1q": "check03", "short2q": "triage09", "medium3q": "medium01-memory", "long3q": "long00-storage"}


def plan_of(case):
    p = LL.case_plan()[case]
    return p["shared"], p["suffix"], p["n_questions"]


def kind_of_case(case):
    for k, c in REP.items():
        if c == case:
            return k
    return "long3q" if case.startswith("long") else "medium3q" if case.startswith("medium") else "short2q"


# ------------------------------------------------------------------------------------------ log writer
class Log:
    def __init__(self, t_mono0=5000.0, t_log0=8.0, seed=1):
        self.lines = []
        self.t = t_mono0              # steady clock, seconds
        self.t_log0 = t_log0 - t_mono0
        self.rng = random.Random(seed)
        self.ub_id = 0
        self.task = 0
        self.gpu_free = self.t

    def tlog(self, t=None):
        x = (self.t if t is None else t) + self.t_log0
        m, s = divmod(x, 60.0)
        us = min(int(round((s - int(s)) * 1e6)), 999999)
        return f"{int(m)}.{int(s):02d}.{us // 1000:03d}.{us % 1000:03d}"

    def slot(self, func, slot, task, text):
        self.lines.append(f"{self.tlog()} I slot {func:>12}: id {slot:2d} | task {task} | {text}")

    def dmk(self, kind, **kv):
        s = " ".join(f"{k}={v}" for k, v in kv.items())
        self.lines.append(f"DMK_TIMING {kind} {s}")


def gpu_ms_of(p, window, off, base_scale=4.0, noise=0.0, rng=None):
    """GPU time of one pass (ms) and its FA part (ms) under the planted law."""
    c = L.pass_costs(LL.model(), p)
    base = base_scale * c["floor_pass_max_s"] * 1e3 + 6.0
    n = p["n_tokens"]
    w = window["seen"] if n < 1024 else window["eff_front"] + off
    fa = FA_A + FA_B * FA_SCALE * n * w
    t = base + fa
    if noise and rng:
        f = 1 + rng.gauss(0, noise)
        return t * f, fa * f
    return t, fa


def planted_passes(spec):
    """Pass list of one request spec: mode unified / seq / fixed / extra."""
    shared, suffix, n_q = spec["shared"], spec["suffix"], spec["n_questions"]
    slots = spec["slots"]
    mode = spec.get("mode", "unified")
    if mode == "extra":
        ps = [dict(p) for p in LL.simulate(shared, suffix, slots, "unified")]
        below = n_q > 1 and any(s < slots[0] for s in slots[1:])
        singles = [i for i, p in enumerate(ps) if p["kind"] == "suffix" and len(p["seqs"]) == 1 and p["n_tokens"] > 1]
        if below and n_q == 3 and singles:
            # one extra ubatch in its own llama_decode: a one-token ubatch is split off the last single-sequence suffix pass
            i = singles[-1]
            seq0, n0 = ps[i]["seqs"][0]
            extra = {"kind": "suffix", "seqs": [(seq0 + n0 - 1, 1)], "slots": ps[i]["slots"], "questions": ps[i]["questions"], "n_tokens": 1,
                     "n_out": ps[i]["n_out"], "iteration": ps[-1]["iteration"] + 1}
            ps[i] = dict(ps[i], seqs=[(seq0, n0 - 1)], n_tokens=n0 - 1, n_out=0)
            ps.append(extra)
        return ps
    return LL.simulate(shared, suffix, slots, mode)


def emit_request(log, spec, t_send, law=None, copy_sync=False):
    """Append one request to the log; returns the client record and the pass records (with the planted timing)."""
    rng = log.rng
    ps = planted_passes(spec)
    shared, suffix, n_q, slots = spec["shared"], spec["suffix"], spec["n_questions"], spec["slots"]
    wins = LL.pass_windows(ps)
    off = spec.get("offset", 0)
    log.t = t_send
    t_start = t_send + 0.00035
    log.t = t_start
    parse_us, build_us = 240, 1500 + rng.randint(0, 300)
    log.t += (parse_us + build_us) / 1e6
    t_built = log.t
    # slot lines
    tid = log.task
    log.task += n_q + 3
    sel = "LRU, n_tokens = 0, t_last = %d" % rng.randint(10**11, 10**12)
    log.slot("get_availabl", slots[0], -1, "selected slot by " + sel)
    for q in range(1, n_q):
        log.slot("launch_slot_", slots[q], tid + q, "processing task, is_child = 1")
    log.slot("launch_slot_", slots[0], tid, "processing task, is_child = 0")
    recs = []
    pending = 0
    pend_tok = 0
    t_first = None
    t_last_sub_end = None
    iters = sorted({p["iteration"] for p in ps})
    for it in iters:
        group = [p for p in ps if p["iteration"] == it]
        if copy_sync and it > 0 and group[0]["kind"] == "suffix" and pending and not any(p["n_out"] for p in ps if p["iteration"] < it):
            # --no-kv-unified: seq_cp of the shared prefix to a child copies the whole K/V stream buffer after llama_synchronize()
            t_begin = log.t
            t_end = max(t_begin, log.gpu_free) + 0.00008
            log.dmk("sync", t=f"{t_end:.6f}", passes=pending, tokens=pend_tok, wait_us=round((t_end - t_begin) * 1e6),
                    since_first_submit_us=round((t_end - t_first) * 1e6), since_last_submit_us=round((t_end - t_last_sub_end) * 1e6))
            log.t = t_end + 0.0009 * (n_q - 1)
            pending, pend_tok, t_first = 0, 0, None
        t_dec0 = log.t
        n_tok_dec = 0
        for p in group:
            w = wins[ps.index(p)]
            win = {"seen": w["needed_seq"] + off, "eff_front": w["eff_front"]}
            if win["seen"] > 16384:
                win["seen"] = 16384
            host_us = 900 + rng.randint(0, 900) + (2500 if log.ub_id == 0 else 0)
            t_sub0 = log.t
            log.t += host_us / 1e6
            log.ub_id += 1
            nseq = len(p["seqs"])
            log.dmk("ubatch", t=f"{log.t:.6f}", id=log.ub_id, n_tokens=p["n_tokens"], n_seqs=nseq, n_seq_tokens=p["n_tokens"] // nseq, n_outputs=p["n_out"],
                    n_nodes=1370 + 40 * rng.randint(0, 1), reuse=0, apply_us=40, build_us=rng.randint(300, 700), alloc_us=rng.randint(100, 250),
                    inputs_us=rng.randint(100, 300), submit_us=host_us - 800, total_us=host_us)
            g_ms, fa_ms = gpu_ms_of(p, win, off, noise=law.get("noise", 0.01) if law else 0.0, rng=rng)
            start = max(log.gpu_free, log.t)
            log.gpu_free = start + g_ms / 1e3
            if t_first is None:
                t_first = t_sub0
            t_last_sub_end = log.t
            pending += 1
            pend_tok += p["n_tokens"]
            n_tok_dec += p["n_tokens"]
            recs.append({"pass": p, "win": win, "n_kv_seen": pad_seen(win["seen"]), "needed_seq": w["needed_seq"], "gpu_ms": g_ms, "fa_ms": fa_ms,
                         "iteration": it, "t_start_gpu": start, "wall_ms": g_ms + 0.9})
        log.dmk("decode", t=f"{log.t + 0.0001:.6f}", n_tokens=n_tok_dec, n_ubatches=len(group), init_us=rng.randint(60, 120), total_us=round((log.t - t_dec0) * 1e6 + 100))
        log.t += 0.0001
        if any(p["n_out"] > 0 for p in group):
            t_begin = log.t
            t_end = max(t_begin, log.gpu_free) + 0.00008
            log.dmk("sync", t=f"{t_end:.6f}", passes=pending, tokens=pend_tok, wait_us=round((t_end - t_begin) * 1e6),
                    since_first_submit_us=round((t_end - t_first) * 1e6), since_last_submit_us=round((t_end - t_last_sub_end) * 1e6))
            log.t = t_end + 0.0003
            pending, pend_tok, t_first = 0, 0, None
        log.t += 0.0002
    t_done = log.t
    for q in range(n_q):
        rel = shared + suffix[q]
        log.slot("release", slots[q], tid + q, f"stop processing: n_tokens = {rel}, truncated = 0")
    log.t += 0.00015
    log.dmk("request", t=f"{log.t:.6f}", n_questions=n_q, n_groups=1, parse_us=parse_us, build_us=build_us, wait_us=round((t_done - t_built) * 1e6),
            format_us=150, total_us=round((log.t - t_start) * 1e6))
    t_recv = log.t + 0.00025
    log.gpu_free = max(log.gpu_free, log.t)
    n_in = n_q * shared + sum(suffix) if shared else suffix[0]
    return {"t_send_mono_ns": int(t_send * 1e9), "t_header_mono_ns": int((t_recv - 0.00005) * 1e9), "t_recv_mono_ns": int(t_recv * 1e9),
            "wall_ms": (t_recv - t_send) * 1e3, "input_tokens": n_in}, recs


def pad_seen(x):
    return LL.pad256(x)


# ------------------------------------------------------------------------------------------ childpass scenarios
def childpass_specs(mode):
    specs = []
    for kind in KINDS:
        case = REP[kind]
        shared, suffix, n_q = plan_of(case)
        if n_q == 1:
            specs.append(dict(kind=kind, case=case, shared=shared, suffix=suffix, n_questions=1, slots=(0,), mode=mode))
            specs.append(dict(kind=kind, case=case, shared=shared, suffix=suffix, n_questions=1, slots=(2,), mode=mode))
            continue
        for slots in L.arrangements(n_q):
            specs.append(dict(kind=kind, case=case, shared=shared, suffix=suffix, n_questions=n_q, slots=slots, mode=mode))
    # second case of each multi-question type, parent in slot 0 and 2 only (different token counts)
    for case in ("triage02", "style03", "medium04-network", "long05-memory"):
        if case not in LL.case_plan():
            continue
        shared, suffix, n_q = plan_of(case)
        for slots in L.arrangements(n_q)[::2]:
            specs.append(dict(kind=kind_of_case(case), case=case, shared=shared, suffix=suffix, n_questions=n_q, slots=slots, mode=mode))
    return specs


def truth_childpass(specs):
    out = []
    for s in specs:
        slots = s["slots"]
        below = s["n_questions"] > 1 and any(x < slots[0] for x in slots[1:])
        census = LL.simulate(s["shared"], s["suffix"], slots, "unified")
        planted = planted_passes(s)
        fixed = LL.simulate(s["shared"], s["suffix"], slots, "fixed")
        out.append({"kind": s["kind"], "case": s["case"], "slots": list(slots), "child_below_parent": below,
                    "planted_ubatches": len(planted), "planted_decodes": LL.n_decodes(planted),
                    "census_ubatches": len(census), "census_decodes": LL.n_decodes(census),
                    "fixed_ubatches": len(fixed), "fixed_decodes": LL.n_decodes(fixed),
                    "planted_equals_census": LL.ub_key(planted) == LL.ub_key(census),
                    "planted_structure": LL.structure(planted)})
    return out


def gen_childpass(mode, seed):
    specs = childpass_specs(mode)
    log = Log(seed=seed)
    log.t += 1.0
    reqs = []
    t = log.t
    for i, s in enumerate(specs):
        client, _ = emit_request(log, s, t, law={"noise": 0.01})
        reqs.append({"idx": i, "kind": s["kind"], "case": s["case"], "phase": "bench", "pair": None, "ref": f"{i:010d}",
                     "t_send_epoch_ns": 1790000000_000000000 + client["t_send_mono_ns"], **client})
        t = log.t + 0.004
    return specs, log, reqs


# ------------------------------------------------------------------------------------------ kvwin scenarios
def kvwin_sequence(offsets, nokvu=False, others=6, repeat=1):
    """(phase, case, slots) of the kvwin_client.py sequence with planted placements and window offsets."""
    seq = []
    seq.append(("warmup", REP["short1q"], (0,), 0))
    seq.append(("warmup", REP["short2q"], (1, 2), 0))
    place = {"T1": (0, 1, 2), "T2": (0, 1, 2), "T3": (2, 0, 1), "T4": (0, 1, 2)}
    seq.append(("T1", REP["long3q"], place["T1"], offsets["T1"]))
    perms = L.arrangements(3)
    for i in range(others):
        seq.append(("other", f"long0{i + 1}-" + ["memory", "network", "none", "storage", "memory", "network"][i], perms[(i * 5 + 1) % 6], offsets["other"]))
    seq.append(("T2", REP["long3q"], place["T2"], offsets["T2"]))
    seq.append(("T3", REP["long3q"], place["T3"], offsets["T3"]))
    for i in range(repeat):
        seq.append((f"T{4 + i}", REP["long3q"], place["T4"], offsets["T4"]))
    return seq


def gen_kvwin(growth, nokvu=False, seed=5):
    if growth:
        offsets = {"T1": 256, "T2": 3328, "T3": 3328, "T4": 3328, "other": 3584}
    else:
        offsets = {"T1": 256, "T2": 256, "T3": 256, "T4": 256, "other": 256}
    seq = kvwin_sequence(offsets)
    log = Log(seed=seed)
    log.t += 1.0
    t = log.t
    reqs, passes_all = [], []
    cp = LL.case_plan()
    for i, (phase, case, slots, off) in enumerate(seq):
        shared, suffix, n_q = plan_of(case) if case in cp else plan_of(REP["long3q"])
        spec = dict(kind=kind_of_case(case), case=case, shared=shared, suffix=suffix, n_questions=n_q, slots=slots, mode="seq" if nokvu else "unified", offset=off)
        if nokvu:
            spec["offset"] = 0
        client, recs = emit_request(log, spec, t, law={"noise": 0.01}, copy_sync=nokvu)
        reqs.append({"idx": i, "kind": spec["kind"], "case": case, "phase": phase, "pair": None, "ref": f"{i:010d}",
                     "t_send_epoch_ns": 1790000000_000000000 + client["t_send_mono_ns"], **client})
        passes_all.append((i, spec, recs))
        t = log.t + 0.004
    return seq, log, reqs, passes_all


def passes_tsv(passes_all, reqs, run, path):
    """Passes TSV with the header of the census table trace-passes.tsv plus run, segment, client_idx, kind, case."""
    head = open(L.data_file("trace-passes.tsv")).readline().rstrip("\n").split("\t")
    cols = head + ["run", "segment", "client_idx", "kind", "case"]
    t_ns = 1_000_000_000
    n_trace = 0
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for i, spec, recs in passes_all:
            for j, r in enumerate(recs):
                p = r["pass"]
                kind = "prefix-large" if (p["kind"] == "prefix" and p["n_tokens"] >= 2048) else ("prefix-small" if p["kind"] == "prefix" else "suffix" if p["kind"] == "suffix" else "single")
                fa = r["fa_ms"]
                row = {c: "" for c in cols}
                row.update({"row_kind": "pass", "trace": run, "request_idx": i, "pass_idx_in_request": j, "pass_idx_in_trace": n_trace,
                            "case_id_inferred": spec["case"], "pass_kind": kind, "n_tokens": p["n_tokens"], "n_kv_padded": r["n_kv_seen"],
                            "n_outputs": p["n_out"], "n_passes": len(recs), "start_ns": t_ns, "wall_ms": f"{r['wall_ms']:.6f}",
                            "gpu_busy_ms": f"{r['gpu_ms']:.6f}", "kernel_time_sum_ms": f"{r['gpu_ms']:.6f}", "n_flash_attn": 8, "n_flash_aux": 8,
                            "t_cat_flash_ms": f"{fa:.6f}", "n_module_load_events": 0, "module_load_ms": 0.0,
                            "run": run, "segment": 0, "client_idx": i, "kind": spec["kind"], "case": spec["case"]})
                f.write("\t".join(str(row[c]) for c in cols) + "\n")
                n_trace += 1
                t_ns += int(r["wall_ms"] * 1e6) + 500_000


# ------------------------------------------------------------------------------------------ writing
def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def generate(out, seed=20261030):
    os.makedirs(out, exist_ok=True)
    made = []
    for name, mode in (("childpass-census", "unified"), ("childpass-seq", "seq"), ("childpass-extra", "extra")):
        specs, log, reqs = gen_childpass(mode, seed)
        base = os.path.join(out, "syn-" + name)
        kvu = mode != "seq"
        write(base + ".server.log", "\n".join(["0.00.387.436 I srv  llama_server: initializing ...", "0.07.554.165 I srv    load_model: initializing, n_slots = 3, n_ctx_slot = %d, kv_unified = '%s'"
                                                % (16384 if kvu else 5461, "true" if kvu else "false")] + log.lines) + "\n")
        write_json(base + ".json", {"meta": {"scenario": name, "synthetic": True}, "requests": reqs})
        write_json(base + ".truth.json", {"scenario": name, "requests": truth_childpass(specs)})
        made.append(base)
    global FA_SCALE
    for name, growth, nokvu, scale in (("kvwin-growth", True, False, 1.0), ("kvwin-strong", True, False, 6.0), ("kvwin-null", False, False, 1.0), ("kvwin-nokvu", False, True, 1.0)):
        FA_SCALE = scale
        seq, log, reqs, passes_all = gen_kvwin(growth, nokvu, seed)
        FA_SCALE = 1.0
        base = os.path.join(out, "syn-" + name)
        write(base + ".server.log", "\n".join(["0.00.387.436 I srv  llama_server: initializing ...", "0.07.554.165 I srv    load_model: initializing, n_slots = 3, n_ctx_slot = %d, kv_unified = '%s'"
                                                % (5461 if nokvu else 16384, "false" if nokvu else "true")] + log.lines) + "\n")
        write_json(base + ".json", {"meta": {"scenario": name, "synthetic": True, "target": REP["long3q"]}, "requests": reqs})
        passes_tsv(passes_all, reqs, "syn-" + name, base + ".passes.tsv")
        rq = [{"idx": r["idx"], "phase": r["phase"], "fa_ms_sum": round(sum(x["fa_ms"] for x in rec), 6),
               "gpu_ms_sum": round(sum(x["gpu_ms"] for x in rec), 6)} for r, (_, _, rec) in zip(reqs, passes_all)]
        first = {}
        for r in rq:
            first.setdefault(r["phase"], r)
        same = [p for p in ("T2", "T4") if p in first]      # T2 and T4 have the placement of T1: sums are comparable
        planted = {"fa": sum(first[p]["fa_ms_sum"] for p in same) / len(same) / first["T1"]["fa_ms_sum"] - 1,
                   "pass_time": sum(first[p]["gpu_ms_sum"] for p in same) / len(same) / first["T1"]["gpu_ms_sum"] - 1}
        truth = {"scenario": name, "fa_law": {"A_ms": FA_A, "B_ms_per_token_cell": FA_B * scale}, "growth": growth, "fa_scale": scale,
                 "offsets_cells": {"T1": 256, "T2": 3328 if growth else 256, "T3": 3328 if growth else 256, "T4": 3328 if growth else 256},
                 "planted_effect_T2_T4_vs_T1": planted,
                 "expect": {"fa_effect": "yes" if growth else "no", "pass_time_effect": "yes" if scale > 1 else ("no" if not growth else "any")},
                 "requests": rq}
        write_json(base + ".truth.json", truth)
        made.append(base)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "dmk-leads-synthetic"))
    ap.add_argument("--seed", type=int, default=20261030)
    a = ap.parse_args()
    for b in generate(a.out, a.seed):
        print("wrote", b + ".*")


if __name__ == "__main__":
    main()
