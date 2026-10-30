#!/usr/bin/env python3
"""Blocked A/B statistics for runs made from a make_schedule.py schedule.

For each block and metric: ratio = arm B / arm A. Estimate = mean of block ratios.
95 % interval = percentile bootstrap over blocks (resample blocks with replacement).

Usage:
  ab_stats.py <results dir> --prefix p1-bench --a S --b D [--margin 2] [--tsv out.tsv]
Metrics:
  bench runs (<name>.json):  lat_ms, short_ms, medium_ms, long_ms (lower is better), rps8
  llama bench runs (<name>.out, JSON): pp<N>_ts (tokens/s, higher is better)
"""
import argparse
import glob
import json
import os
import random
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--prefix", required=True)
ap.add_argument("--a", required=True, help="tag of the reference arm")
ap.add_argument("--b", required=True, help="tag of the compared arm")
ap.add_argument("--margin", type=float, default=None, help="equivalence margin in percent")
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--tsv", default=None)
args = ap.parse_args()


def metrics(path_base):
    m = {}
    if os.path.exists(path_base + ".json"):
        r = json.load(open(path_base + ".json"))
        s = r.get("seq") or {}
        if s.get("lat_ms") is not None:
            m["lat_ms"] = s["lat_ms"]
            for size in ("short", "medium", "long"):
                if size in s.get("by_size", {}):
                    m[f"{size}_ms"] = s["by_size"][size]["mean"]
        if r.get("tput"):
            m["rps8"] = r["tput"]["rps"]
    elif os.path.exists(path_base + ".out"):
        txt = open(path_base + ".out").read()
        try:
            rows = json.loads(txt)
        except ValueError:
            rows = []
        for row in rows:
            if row.get("n_gen", 0) == 0 and row.get("n_prompt"):
                m[f"pp{row['n_prompt']}_ts"] = row["avg_ts"]
    return m


pat = re.compile(re.escape(args.prefix) + r"-b(\d+)-(\w+)$")
blocks = {}
for f in glob.glob(os.path.join(args.results, args.prefix + "-b*")):
    base = re.sub(r"\.(json|out)$", "", f)
    if base == f:
        continue
    mt = pat.search(os.path.basename(base))
    if not mt:
        continue
    blk, tag = int(mt.group(1)), mt.group(2)
    m = metrics(base)
    if m:
        blocks.setdefault(blk, {})[tag] = m

complete = {b: v for b, v in sorted(blocks.items()) if args.a in v and args.b in v}
if not complete:
    raise SystemExit("no complete blocks")
names = sorted(set.intersection(*(set(v[args.a]) & set(v[args.b]) for v in complete.values())))
rng = random.Random(args.seed)
rows = []
for name in names:
    ratios = [v[args.b][name] / v[args.a][name] for v in complete.values()]
    est = statistics.fmean(ratios)
    boots = []
    for _ in range(args.boot):
        sample = [ratios[rng.randrange(len(ratios))] for _ in ratios]
        boots.append(statistics.fmean(sample))
    boots.sort()
    lo, hi = boots[int(0.025 * args.boot)], boots[int(0.975 * args.boot) - 1]
    a_mean = statistics.fmean(v[args.a][name] for v in complete.values())
    b_mean = statistics.fmean(v[args.b][name] for v in complete.values())
    row = {"metric": name, "blocks": len(ratios), "a_mean": a_mean, "b_mean": b_mean,
           "diff_pct": 100 * (est - 1), "ci_lo_pct": 100 * (lo - 1), "ci_hi_pct": 100 * (hi - 1)}
    if args.margin is not None:
        row["within_margin"] = -args.margin <= row["ci_lo_pct"] and row["ci_hi_pct"] <= args.margin
    rows.append(row)

cols = list(rows[0].keys())
print("\t".join(cols))
for r in rows:
    print("\t".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
if args.tsv:
    with open(args.tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
