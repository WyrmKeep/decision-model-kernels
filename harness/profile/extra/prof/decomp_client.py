#!/usr/bin/env python3
"""Decomposition client (phase 3, package 1): client-side wall time of sequential requests, with CLOCK_MONOTONIC stamps.

Runs in python:3.12-slim with host networking (harness `client` line), python stdlib only, `import common`.
It does not profile anything. It records, for every request, the client stamps that the analysis
(harness/profile/decomp/analyze_decomp.py) matches against the server's DMK_TIMING lines (same clock).

Per request type (kind) in the order of --kinds, strictly sequential on ONE keep-alive connection, no sleep between requests:
  1. warmup  --warmup requests (phase "warmup", recorded, not for statistics): the cases that FOLLOW the representative case
             in the kind's case list (cyclic), one each, unique Ref.
  2. unique  --reps requests of the representative case (common.REP_CASE), each with its own Ref.
  3. cycle   --cycle-rounds rounds over ALL cases of the kind; each round in an order shuffled by random.Random(seed)
             (a new Random(seed) per kind, so one kind's order does not depend on the kinds before it); unique Refs.
  4. pair1/pair2  --pairs pairs: the SAME body (same Ref, same case) sent twice back to back; every pair has its own Ref.
             Pair i (1-based) uses the representative case when i is odd and the case that follows it in the kind's case list
             when i is even (alternating, so slow drift hits both cases alike).
`--sleep-between-kinds` seconds of idle between two kinds.

Usage: python -u /extra/prof/decomp_client.py --out /runs/{name}.client.json [--kinds short1q,short2q,medium3q,long3q]
         [--reps 30] [--cycle-rounds 2] [--pairs 15] [--warmup 3] [--seed 20261031] [--sleep-between-kinds 1.0] [--port 18080]
Requests per kind with the defaults: warmup + reps + cases x cycle-rounds + 2 x pairs = 3 + 30 + 2n + 30
  (n = 12 / 20 / 8 / 8 cases: 87 / 103 / 79 / 79).
Output JSON: {"meta": {...}, "requests": [{idx, kind, case, phase, pair, ref, t_send_mono_ns, t_header_mono_ns, t_recv_mono_ns,
  t_send_epoch_ns, wall_ms, input_tokens}], "errors": [...]}. Answers are not kept. Exit status 1 if there was any error.
wall_ms = from just before the request is written to the socket until the response body is read completely
(common.Client.post; the client's own json.loads of the answer comes after the stamp).
"""
import argparse
import gc
import json
import random
import sys
import time

import common

DEFAULT_KINDS = "short1q,short2q,medium3q,long3q"


def plan_kind(cases, kind, reps, cycle_rounds, pairs, warmup, seed):
    """Requests of one kind in send order: list of {kind, case (dict), phase, pair, ref}. No network, no clock."""
    kc = common.cases_of(cases, kind)
    if not kc:
        raise SystemExit(f"no cases of kind {kind}")
    rep = common.rep_case(cases, kind)
    i_rep = [c["id"] for c in kc].index(rep["id"])
    following = [kc[(i_rep + 1 + j) % len(kc)] for j in range(len(kc))]   # cases after the representative one, cyclic
    plan = []

    def add(case, phase, ref, pair=None):
        plan.append({"kind": kind, "case": case, "phase": phase, "pair": pair, "ref": ref})

    for j in range(warmup):
        c = following[j % len(following)]
        add(c, "warmup", common.tag("decomp", seed, kind, "warmup", j, c["id"]))
    for j in range(reps):
        add(rep, "unique", common.tag("decomp", seed, kind, "unique", j))
    rng = random.Random(seed)
    for r in range(cycle_rounds):
        order = list(kc)
        rng.shuffle(order)
        for c in order:
            add(c, "cycle", common.tag("decomp", seed, kind, "cycle", r, c["id"]))
    for i in range(1, pairs + 1):
        c = rep if i % 2 == 1 else following[0]
        ref = common.tag("decomp", seed, kind, "pair", i, c["id"])
        add(c, "pair1", ref, i)
        add(c, "pair2", ref, i)
    return plan


def plan_all(cases, kinds, reps, cycle_rounds, pairs, warmup, seed):
    return [plan_kind(cases, k, reps, cycle_rounds, pairs, warmup, seed) for k in kinds]


def summary_table(requests, kinds):
    rows = []
    for kind in kinds:
        for phase in ("warmup", "unique", "cycle", "pair1", "pair2"):
            w = [r["wall_ms"] for r in requests if r["kind"] == kind and r["phase"] == phase]
            if not w:
                continue
            rows.append((kind, phase, len(w), sum(w) / len(w), common.pct(w, 50), common.pct(w, 95), min(w), max(w)))
    lines = [f"{'kind':9s} {'phase':7s} {'n':>4s} {'mean_ms':>9s} {'p50_ms':>9s} {'p95_ms':>9s} {'min_ms':>9s} {'max_ms':>9s}"]
    for k, p, n, m, p50, p95, lo, hi in rows:
        lines.append(f"{k:9s} {p:7s} {n:4d} {m:9.2f} {p50:9.2f} {p95:9.2f} {lo:9.2f} {hi:9.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kinds", default=DEFAULT_KINDS, help="comma separated request types, in this order")
    ap.add_argument("--reps", type=int, default=30, help="requests of the representative case per kind (phase unique)")
    ap.add_argument("--cycle-rounds", type=int, default=2, help="rounds over all cases of the kind (phase cycle)")
    ap.add_argument("--pairs", type=int, default=15, help="pairs of identical requests (phases pair1/pair2)")
    ap.add_argument("--warmup", type=int, default=3, help="untimed requests at the start of each kind")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--sleep-between-kinds", type=float, default=1.0)
    ap.add_argument("--out", required=True, help="JSON report, /runs/{name}.client.json")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cases", default=common.CASES_PATH)
    args = ap.parse_args()

    kinds = [k for k in args.kinds.split(",") if k]
    for k in kinds:
        if k not in common.KINDS:
            sys.exit(f"unknown kind {k!r}, choose from {','.join(common.KINDS)}")
    cases = common.load_cases(args.cases)
    plans = plan_all(cases, kinds, args.reps, args.cycle_rounds, args.pairs, args.warmup, args.seed)
    n_total = sum(len(p) for p in plans)

    client = common.Client(host=args.host, port=args.port)
    started_epoch_ns = common.epoch_ns()
    started_mono_ns = common.mono_ns()
    requests, errors = [], []
    meta = {
        "argv": sys.argv, "started_epoch_ns": started_epoch_ns, "started_mono_ns": started_mono_ns,
        "kinds": kinds, "reps": args.reps, "cycle_rounds": args.cycle_rounds, "pairs": args.pairs, "warmup": args.warmup,
        "seed": args.seed, "sleep_between_kinds": args.sleep_between_kinds, "host": args.host, "port": args.port,
        "python": sys.version.split()[0], "n_planned": n_total,
        "n_cases": {k: len(common.cases_of(cases, k)) for k in kinds},
        "rep_case": {k: common.REP_CASE[k] for k in kinds},
        "pair_cases": {p[0]["kind"]: [r["case"]["id"] for r in p if r["phase"] == "pair1"] for p in plans if p},
        "ref_format": "Ref: <10 hex>\\n as first line of the state (common.body)",
        "clock": "t_*_mono_ns = CLOCK_MONOTONIC (time.clock_gettime_ns), same clock as the t= of the server DMK_TIMING lines",
    }

    def write(final):
        meta["reconnects"] = client.reconnects
        meta["n_requests"] = len(requests)
        meta["n_errors"] = len(errors)
        meta["finished_epoch_ns"] = common.epoch_ns() if final else None
        meta["finished"] = final
        common.write_json(args.out, {"meta": meta, "requests": requests, "errors": errors})

    idx = 0
    consecutive_errors = 0
    gc.collect()
    for i_kind, plan in enumerate(plans):
        if i_kind and args.sleep_between_kinds > 0:
            time.sleep(args.sleep_between_kinds)
        for step in plan:
            case = step["case"]
            data = common.body(case, step["ref"])
            try:
                r = client.post(data)
                consecutive_errors = 0
            except Exception as e:  # keep going: one failed request must not end the run
                errors.append({"idx": idx, "kind": step["kind"], "case": case["id"], "phase": step["phase"],
                               "pair": step["pair"], "ref": step["ref"], "error": f"{type(e).__name__}: {e}"[:300]})
                consecutive_errors += 1
                idx += 1
                if consecutive_errors >= 5:
                    print("5 consecutive errors, giving up", file=sys.stderr)
                    break
                continue
            requests.append({
                "idx": idx, "kind": step["kind"], "case": case["id"], "phase": step["phase"], "pair": step["pair"],
                "ref": step["ref"], "t_send_mono_ns": r["t_send_mono_ns"], "t_header_mono_ns": r["t_header_mono_ns"],
                "t_recv_mono_ns": r["t_recv_mono_ns"], "t_send_epoch_ns": r["t_send_epoch_ns"],
                "wall_ms": r["wall_ms"], "input_tokens": r["input_tokens"],
            })
            idx += 1
        write(False)
        if consecutive_errors >= 5:
            break
    write(True)

    print(f"decomp_client: {len(requests)} requests ok, {len(errors)} errors, {client.reconnects} reconnects, "
          f"{(common.mono_ns() - started_mono_ns) / 1e9:.1f} s, planned {n_total}")
    print(summary_table(requests, kinds))
    for e in errors[:10]:
        print("ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
