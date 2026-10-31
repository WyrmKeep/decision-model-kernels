#!/usr/bin/env python3
"""kvwin_client.py - request sequence of the K/V-window test (upstream lead #28495, phase 3, package 3). python3.12 stdlib only.

Runs inside python:3.12-slim on the GPU host (harness `client` line, /extra/prof/kvwin_client.py, `import common`).

Question: is the flash-attention window (n_kv) of a request wider than its own cells because cells of other sequences sit in the shared
(--kv-unified) pool, and what does that cost? The same request (the TARGET, a long 3-question case) is sent four times:

   warmup  2 requests: short1q and short2q representative cases (module loads, first-use costs), outside the comparison
   T1      TARGET on an (almost) fresh pool: only the ~400 cells the two warm-up requests left in their slots
   other   N long cases (default 6: long01..long06): every one leaves its cells in a slot, so the pool fills up
   T2      TARGET again: behind the stale cells of the other requests
   T3      TARGET again
   T4..    TARGET again (--repeat-target R: R further repeats, default 1: T4)

Sequential, one keep-alive connection. Every request has a unique "Ref: <10 digits>" first line. DIGITS only (not hex): a digit-only value
is always 14 tokens for "Ref: <value>\\n" (llama-tokenize, 40 of 40 samples), so T1..T4 have identical token counts and the same
prompt structure; hex values vary between 9 and 14 tokens. The Ref is derived from the sequence position, so a second run (for
example the --no-kv-unified run) sends exactly the same Refs.

Output (--out, e.g. /runs/{name}.json):
  {"meta": {...}, "requests": [{"idx","segment","kind","case","phase","ref","t_send_mono_ns","t_header_mono_ns","t_recv_mono_ns",
            "t_send_epoch_ns","wall_ms","input_tokens","n_questions"}, ...], "errors": [{"idx","case","error"}]}
  segment = "kvwin"; phase = warmup | T1 | other | T2 | T3 | T4 ...; t_*_mono_ns = CLOCK_MONOTONIC (same clock as the nsys `systemClockNs`).
The same fields as nsys_client.py, so harness/profile/nsys/nsys_census.py reads it. Exit status 2 when a request failed (the JSON is written anyway).
--plan prints the sequence and exits (no server needed).
"""
import argparse
import hashlib
import platform
import sys
import time

import common

SEGMENT = "kvwin"
REF_PREFIX = "p3-kvwin"


def digit_ref(*parts):
    """10 digits (zero padded) from the parts: a deterministic unique Ref value that always tokenizes to 14 tokens in "Ref: <value>\\n"."""
    h = hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()
    return f"{int(h, 16) % 10 ** 10:010d}"


def build_plan(cases, target, others, repeat, warmup_cases):
    """[(phase, case)] in sending order."""
    tgt = common.case_by_id(cases, target)
    pool = [c for c in common.cases_of(cases, tgt["kind"]) if c["id"] != target]
    if len(pool) < others:
        sys.exit(f"only {len(pool)} other cases of kind {tgt['kind']}, --others {others} asked")
    plan = [("warmup", common.case_by_id(cases, w)) for w in warmup_cases]
    plan.append(("T1", tgt))
    plan += [("other", c) for c in pool[:others]]
    plan += [("T2", tgt), ("T3", tgt)]
    plan += [(f"T{4 + i}", tgt) for i in range(repeat)]
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="client JSON, e.g. /runs/{name}.json (required unless --plan)")
    ap.add_argument("--target", default=common.REP_CASE["long3q"], help="case id of the target request (default long00-storage)")
    ap.add_argument("--others", type=int, default=6, help="number of other cases of the same kind between T1 and T2 (default 6)")
    ap.add_argument("--repeat-target", type=int, default=1, help="repeats of the target after T3: T4.. (default 1)")
    ap.add_argument("--warmup-cases", default="%s,%s" % (common.REP_CASE["short1q"], common.REP_CASE["short2q"]),
                    help="comma separated case ids of the warm-up requests (default check03,triage09)")
    ap.add_argument("--pause-s", type=float, default=0.0, help="sleep between requests (default 0)")
    ap.add_argument("--linger-s", type=float, default=0.0, help="sleep after the last response before exiting (default 0)")
    ap.add_argument("--cases", default=common.CASES_PATH)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    args = ap.parse_args()
    if not args.plan and not args.out:
        ap.error("--out is required")

    cases = common.load_cases(args.cases)
    warm = [w for w in args.warmup_cases.split(",") if w]
    plan = build_plan(cases, args.target, args.others, args.repeat_target, warm)
    refs = [digit_ref(REF_PREFIX, i, ph, c["id"]) for i, (ph, c) in enumerate(plan)]
    if len(set(refs)) != len(refs):
        sys.exit("duplicate Ref values")
    if args.plan:
        for i, ((ph, c), r) in enumerate(zip(plan, refs)):
            print(f"{i:2d} {ph:7s} {c['id']:18s} {c['kind']:9s} ref={r}")
        return

    started_epoch, started_mono = common.epoch_ns(), common.mono_ns()
    cli = common.Client(args.host, args.port)
    reqs, errors = [], []
    for idx, ((phase, case), ref) in enumerate(zip(plan, refs)):
        if idx and args.pause_s > 0:
            time.sleep(args.pause_s)
        try:
            r = cli.post(common.body(case, ref))
        except Exception as e:   # recorded, the run goes on
            errors.append({"idx": idx, "case": case["id"], "error": repr(e)[:300]})
            print(f"ERROR idx={idx} phase={phase} case={case['id']}: {e!r}", flush=True)
            continue
        reqs.append({"idx": idx, "segment": SEGMENT, "kind": case["kind"], "case": case["id"], "phase": phase, "ref": ref,
                     "t_send_mono_ns": r["t_send_mono_ns"], "t_header_mono_ns": r["t_header_mono_ns"],
                     "t_recv_mono_ns": r["t_recv_mono_ns"], "t_send_epoch_ns": r["t_send_epoch_ns"],
                     "wall_ms": round(r["wall_ms"], 6), "input_tokens": r["input_tokens"], "n_questions": len(case["questions"])})
        print(f"{idx:2d} {phase:7s} {case['id']:18s} input_tokens={r['input_tokens']} wall_ms={r['wall_ms']:.2f}", flush=True)
    if args.linger_s > 0:
        time.sleep(args.linger_s)
    raw = open(args.cases, "rb").read()
    meta = {"script": "kvwin_client.py", "argv": sys.argv[1:], "target": args.target, "others": args.others, "repeat_target": args.repeat_target,
            "warmup_cases": warm, "pause_s": args.pause_s, "linger_s": args.linger_s, "cases_path": args.cases,
            "cases_sha256": hashlib.sha256(raw).hexdigest(), "n_cases": len(cases), "n_requests_planned": len(plan),
            "n_requests_ok": len(reqs), "n_errors": len(errors),
            "ref_scheme": f"10 digits from sha256('{REF_PREFIX}/<position>/<phase>/<case id>') (fixed 14 tokens for the Ref line)",
            "started_epoch_ns": started_epoch, "started_mono_ns": started_mono,
            "finished_epoch_ns": common.epoch_ns(), "finished_mono_ns": common.mono_ns(),
            "reconnects": cli.reconnects, "python": platform.python_version(), "host": args.host, "port": args.port}
    common.write_json(args.out, {"meta": meta, "requests": reqs, "errors": errors})

    tt = [(r["phase"], r["input_tokens"]) for r in reqs if r["phase"].startswith("T")]
    if len({t for _, t in tt}) > 1:
        print(f"WARNING: the target requests have different input_tokens: {tt}", flush=True)
    print(f"kvwin_client: requests ok={len(reqs)} errors={len(errors)} -> {args.out}")
    print(f"{'phase':7s} {'case':18s} {'input_tokens':>12s} {'wall ms':>9s}")
    for r in reqs:
        if r["phase"] != "other":
            print(f"{r['phase']:7s} {r['case']:18s} {str(r['input_tokens']):>12s} {r['wall_ms']:9.2f}")
    sys.exit(2 if errors else 0)


if __name__ == "__main__":
    main()
