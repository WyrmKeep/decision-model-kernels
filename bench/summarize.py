#!/usr/bin/env python3
"""Summarize one benchmark run (<base>.json plus its sampler files).

Usage: summarize.py RUN_BASE [--vram-cap-mib N]
RUN_BASE is the path of a run without extension, for example runs/base-1.
Prints a readable summary and a TSV fragment for a results table."""
import argparse
import json
import os
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("base", help="run path without extension, for example runs/base-1")
ap.add_argument("--vram-cap-mib", type=float, default=float(os.environ.get("BENCH_VRAM_CAP_MIB", "6144")),
                help="guard: peak GPU memory in MiB (default 6144, or BENCH_VRAM_CAP_MIB)")
args = ap.parse_args()

VRAM_CAP_MIB = args.vram_cap_mib

base = args.base
name = base.rsplit("/", 1)[-1]
r = json.load(open(f"{base}.json"))

samples = []
try:
    for line in open(f"{base}.smi"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            samples.append((float(parts[0]), float(parts[1]), float(parts[2]), parts[3], float(parts[4])))
        except ValueError:
            continue
except FileNotFoundError:
    pass


def window(phase):
    lo, hi = r["phases"].get(phase, [None, None])
    if lo is None or hi is None:
        return []
    return [s for s in samples if lo <= s[0] <= hi]


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


vram_peak = max((s[1] for s in samples), default=float("nan"))
seq_w, tput_w = window("seq"), window("tput")
power_seq = mean([s[2] for s in seq_w])
power_tput = mean([s[2] for s in tput_w])
sm_seq = mean([s[4] for s in seq_w])
pstates = sorted({s[3] for s in seq_w})
load = []
for k in ("load0", "load1"):
    try:
        load.append(open(f"{base}.{k}").read().split()[0])
    except FileNotFoundError:
        load.append("?")

q, seq, tput = r["quality"], r["seq"], r["tput"]
guard = dict(r.get("guard", {}))
guard["vram"] = vram_peak <= VRAM_CAP_MIB
if not all(guard.values()):
    verdict = "FAIL"
elif r.get("guard_pass") is None:
    verdict = "n/a"  # no measured baseline: the quality guard did not run
else:
    verdict = "PASS"
bs = seq["by_size"]

print(f"run {name}")
print(f"  lat_ms {seq['lat_ms']:.1f}  (pass means {', '.join(f'{x:.1f}' for x in seq['pass_means'])}; spread {seq['spread_pct']:.1f}%)")
print(f"  by size mean/p95: " + "  ".join(f"{s} {v['mean']:.1f}/{v['p95']:.1f}" for s, v in bs.items()))
print(f"  tokens mean: {q['input_tokens_mean']}")
print(f"  tput conc {tput['conc']}: {tput['rps']:.2f} req/s, {tput['decisions_per_s']:.2f} decisions/s, p95 {tput['lat']['p95']:.0f} ms")
print(f"  vram peak {vram_peak:.0f} MiB  power seq {power_seq:.0f} W / tput {power_tput:.0f} W  sm clock seq {sm_seq:.0f} MHz  pstates {pstates}")
print(f"  startup {r['meta'].get('startup_s')} s  load avg {load[0]} -> {load[1]}  reconnects {seq['reconnects']}")
if "cold_ms" in r:
    print(f"  cold ms {[round(x, 1) for x in r['cold_ms']]}")
print(f"  quality: correct {q['correct']}/{q['total']} {q['correct_by_size']}  agree {q.get('agree_n')}  "
      f"mean_dp {q.get('mean_dp')}  flips {q.get('flips')}")
print(f"  errors {len(r['errors'])} {r['errors'][:3]}")
print(f"  guard {verdict} {guard}")
fmt = lambda x, d=1: "nan" if x is None else f"{x:.{d}f}"  # noqa: E731
print("TSV\t" + "\t".join([
    name, fmt(seq["lat_ms"]), fmt(bs["short"]["mean"]), fmt(bs["medium"]["mean"]),
    fmt(bs["long"]["mean"]), fmt(seq["all"]["p95"]), fmt(tput["rps"], 2), f"{vram_peak:.0f}",
    f"{q['correct']}/{q['total']}", str(q.get("agree_n")), fmt(q.get("mean_dp"), 4),
    verdict]))
