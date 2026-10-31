#!/usr/bin/env python3
"""nsys_client.py - request sequences for the whole-server nsys captures (phase 3, package 2). python3.12 stdlib only.

Runs inside python:3.12-slim on the GPU host (harness `client` line, /extra/prof/nsys_client.py, `import common`).

  --kind short1q|short2q|medium3q|long3q   --warmup W requests (phase "warmup", recorded) + --n N requests (phase "timed")
         cycling over the cases of the kind in cases.json order, starting at the representative case (common.REP_CASE).
         The warm-up requests use the W cases that come just before the representative case in that cycle.
  --kind all48    2 warm-up requests (short1q and short2q representative cases, --warmup of them), then the 48 cases of
         cases.json once, in file order (--n is ignored).
  --kind replay   the unprofiled control run: the five segments short1q, short2q, medium3q, long3q, all48 back to back in ONE
         server run, each exactly as its own nsys run (same cases, same order, same Refs: the Ref of a request is
         common.tag("p3-nsys", segment, phase, position, case id), it does not contain the run name). Use the same --n and
         --warmup as the nsys runs.

Every request is sequential on one keep-alive connection and has a unique "Ref: <10 hex>" first line (common.body), so no
cached prompt can answer it and the tokenisation of a request is identical in a nsys run and in its segment of the replay run.

Output (--out, e.g. /runs/{name}.client.json):
  {"meta": {...}, "requests": [{"idx","segment","kind","case","phase","ref","t_send_mono_ns","t_header_mono_ns",
            "t_recv_mono_ns","t_send_epoch_ns","wall_ms","input_tokens"}, ...], "errors": [{"idx","segment","case","error"}]}
  idx = running index over the whole file; kind = request type of the case (short1q ... long3q); phase = warmup | timed.
  t_*_mono_ns: CLOCK_MONOTONIC (same clock as the nsys `systemClockNs`), t_send_epoch_ns: wall clock (nsys `utcEpochNs`).
A summary per segment is printed. Exit status 2 when a request failed (the JSON is written anyway).
"""
import argparse
import hashlib
import json
import platform
import sys
import time

import common

SEGMENTS = ["short1q", "short2q", "medium3q", "long3q", "all48"]   # = order of the five nsys runs = order of the replay
REF_PREFIX = "p3-nsys"


def plan_kind(cases, kind, n, warmup):
    """[(segment, phase, position, case)] of one request type"""
    cyc = common.cases_of(cases, kind)
    rep = cyc.index(common.rep_case(cases, kind))
    out = []
    for i in range(warmup):
        out.append((kind, "warmup", i, cyc[(rep - warmup + i) % len(cyc)]))
    for i in range(n):
        out.append((kind, "timed", i, cyc[(rep + i) % len(cyc)]))
    return out


def plan_all48(cases, warmup):
    warm = [common.rep_case(cases, "short1q"), common.rep_case(cases, "short2q")]
    out = [("all48", "warmup", i, warm[i % 2]) for i in range(warmup)]
    out += [("all48", "timed", i, c) for i, c in enumerate(cases)]
    return out


def build_plan(cases, kind, n, warmup):
    if kind == "all48":
        return plan_all48(cases, warmup)
    if kind == "replay":
        plan = []
        for seg in SEGMENTS:
            plan += plan_all48(cases, warmup) if seg == "all48" else plan_kind(cases, seg, n, warmup)
        return plan
    return plan_kind(cases, kind, n, warmup)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", required=True, choices=list(common.KINDS) + ["all48", "replay"])
    ap.add_argument("--n", type=int, default=10, help="timed requests per request type (default 10)")
    ap.add_argument("--warmup", type=int, default=2, help="warm-up requests per segment (default 2)")
    ap.add_argument("--out", required=True, help="client JSON, e.g. /runs/{name}.client.json")
    ap.add_argument("--cases", default=common.CASES_PATH)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--segment-pause-s", type=float, default=0.0, help="replay only: sleep between segments (default 0)")
    args = ap.parse_args()

    cases = common.load_cases(args.cases)
    plan = build_plan(cases, args.kind, args.n, args.warmup)
    started_epoch, started_mono = common.epoch_ns(), common.mono_ns()
    cli = common.Client(args.host, args.port)
    reqs, errors, segs = [], [], []
    prev_seg = None
    for idx, (seg, phase, pos, case) in enumerate(plan):
        if seg != prev_seg:
            if prev_seg is not None and args.segment_pause_s > 0:
                time.sleep(args.segment_pause_s)
            segs.append({"name": seg, "first_idx": idx, "last_idx": idx, "n_warmup": 0, "n_timed": 0})
            prev_seg = seg
        segs[-1]["last_idx"] = idx
        segs[-1]["n_warmup" if phase == "warmup" else "n_timed"] += 1
        ref = common.tag(REF_PREFIX, seg, phase, pos, case["id"])
        try:
            r = cli.post(common.body(case, ref))
        except Exception as e:   # recorded, the run goes on
            errors.append({"idx": idx, "segment": seg, "case": case["id"], "error": repr(e)[:300]})
            print(f"ERROR idx={idx} segment={seg} case={case['id']}: {e!r}", flush=True)
            continue
        reqs.append({"idx": idx, "segment": seg, "kind": case["kind"], "case": case["id"], "phase": phase, "ref": ref,
                     "t_send_mono_ns": r["t_send_mono_ns"], "t_header_mono_ns": r["t_header_mono_ns"],
                     "t_recv_mono_ns": r["t_recv_mono_ns"], "t_send_epoch_ns": r["t_send_epoch_ns"],
                     "wall_ms": round(r["wall_ms"], 6), "input_tokens": r["input_tokens"]})
    raw = open(args.cases, "rb").read()
    meta = {"script": "nsys_client.py", "argv": sys.argv[1:], "kind": args.kind, "n": args.n, "warmup": args.warmup,
            "cases_path": args.cases, "cases_sha256": hashlib.sha256(raw).hexdigest(), "n_cases": len(cases),
            "segments": segs, "n_requests_planned": len(plan), "n_requests_ok": len(reqs), "n_errors": len(errors),
            "ref_scheme": f"common.tag('{REF_PREFIX}', segment, phase, position, case id)",
            "started_epoch_ns": started_epoch, "started_mono_ns": started_mono,
            "finished_epoch_ns": common.epoch_ns(), "finished_mono_ns": common.mono_ns(),
            "reconnects": cli.reconnects, "python": platform.python_version(), "host": args.host, "port": args.port}
    common.write_json(args.out, {"meta": meta, "requests": reqs, "errors": errors})

    print(f"nsys_client: kind={args.kind} requests ok={len(reqs)} errors={len(errors)} -> {args.out}")
    print(f"{'segment':9s} {'phase':7s} {'kind':9s} {'n':>3s} {'mean ms':>9s} {'p50 ms':>9s} {'p95 ms':>9s} {'min ms':>9s} {'max ms':>9s}")
    keys = []
    for r in reqs:
        k = (r["segment"], r["phase"], r["kind"])
        if k not in keys:
            keys.append(k)
    for seg, phase, kind in keys:
        w = [r["wall_ms"] for r in reqs if (r["segment"], r["phase"], r["kind"]) == (seg, phase, kind)]
        print(f"{seg:9s} {phase:7s} {kind:9s} {len(w):3d} {sum(w) / len(w):9.2f} {common.pct(w, 50):9.2f} {common.pct(w, 95):9.2f} "
              f"{min(w):9.2f} {max(w):9.2f}")
    sys.exit(2 if errors else 0)


if __name__ == "__main__":
    main()
