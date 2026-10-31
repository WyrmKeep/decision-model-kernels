#!/usr/bin/env python3
"""Summarise the runs of the ablation schedules. Python standard library only.

Usage: summarise_ablate.py [--tsv-dir DIR] RESULTS_FOLDER [RESULTS_FOLDER ...]

Reads each run in the folders. The files are the files that harness/run_schedule.sh writes:
<run>.meta, <run>.json (ablate_bench.py), <run>.server.log, <run>.out (llama bench).
The script ignores the answers. It uses the latencies and the DMK_* lines of the logs.

The name of a run gives its arm: the last part of abl-s1-b1-<arm>, abl-ops-<arm>, abl-llb-b1-<arm>.
The three runs abl-s1-b1-ref-start, -ref-mid and -ref-end are one arm, ref. The runs abl-smoke-* are
only in the table ops-lines.

Without --tsv-dir the script prints the tables as text. With --tsv-dir it writes one TSV file for each
table to the folder DIR and prints the file names.

Tables:
  drift-ref          the arm ref (stock b12004 build) at the start, in the middle and at the end of stage 1
  caps-request       latency of a request, for each arm and request size. cap = base minus arm, in ms and
                     in % of base. A cap is an upper bound of the time of the skipped part.
  pass-spread        the means of the timed passes of each run: lowest, highest and spread
  caps-llama-bench   llama bench: time of one prompt of N tokens, for each arm. The same cap.
  timing-ubatch      DMK_TIMING lines of the arms timing and all_timing: medians for each ubatch class
  timing-summary     the other DMK_TIMING lines of these arms, and the number of passes for each CUDA mode
  ops-lines          for each run with DMK_ABLATE_SYNC: the number of DMK_OPS lines in the server log
  ops                the DMK_OPS table of the arm sync, minus the cost of the arm sync_all
                     (written only if the logs have DMK_OPS lines)
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys

SIZES = ("short", "medium", "long")
UBATCH_CLASSES = (("1-8", 1, 8), ("9-32", 9, 32), ("33-128", 33, 128), ("129-512", 129, 512), ("513-2048", 513, 2048))


def arm_of(run):
    m = re.match(r"abl-(?:s\d|llb|ops)(?:-b\d+)?-(.+)$", run)
    arm = m.group(1) if m else run
    return re.sub(r"^ref-(start|mid|end)$", "ref", arm)


def kind_of(run):
    if run.startswith("abl-smoke-"):
        return "smoke"
    return "ops" if run.startswith("abl-ops-") else "llb" if run.startswith("abl-llb-") else "srv"


def fmt(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"


def cap_cols(val, base):
    """cap in ms and in % of base, for one value and the base value."""
    if val is None or base is None or base == 0:
        return "", ""
    return fmt(base - val), fmt(100.0 * (base - val) / base, 2)


def parse_timing(path):
    """DMK_TIMING lines of a server log: {kind: [{key: value}]}."""
    out = collections.defaultdict(list)
    for line in open(path, errors="replace"):
        i = line.find("DMK_TIMING ")
        if i < 0:
            continue
        parts = line[i + 11:].split()
        kind, kv = parts[0], {}
        for p in parts[1:]:
            k, _, v = p.partition("=")
            try:
                kv[k] = float(v)
            except ValueError:
                kv[k] = v
        out[kind].append(kv)
    return out


def parse_ops(path):
    """DMK_OPS<TAB>bucket<TAB>label<TAB>n<TAB>total_us<TAB>avg_us, summed over all dumps of the log."""
    d = collections.defaultdict(lambda: [0, 0.0])
    for line in open(path, errors="replace"):
        i = line.find("DMK_OPS\t")
        if i < 0:
            continue
        f = line[i:].rstrip("\n").split("\t")
        if len(f) != 6:
            continue
        d[(f[1], f[2])][0] += int(f[3])
        d[(f[1], f[2])][1] += float(f[4])
    return d


def table_requests(runs):
    by = collections.defaultdict(lambda: collections.defaultdict(list))        # arm -> size -> [run means]
    passes = collections.defaultdict(lambda: collections.defaultdict(list))    # arm -> size -> [pass means]
    drift = {}                                                                 # position -> {size: mean}
    for r, path in runs:
        if kind_of(r) != "srv":
            continue
        try:
            d = json.load(open(path))
        except (OSError, ValueError):
            continue
        m = re.match(r"abl-s1-b1-ref-(start|mid|end)$", r)
        if m:
            drift[m.group(1)] = {s: v["mean"] for s, v in d.get("by_size", {}).items()}
        for s, v in d.get("by_size", {}).items():
            by[arm_of(r)][s].append(v["mean"])
            passes[arm_of(r)][s] += v.get("pass_means_ms", [])
    tables = []
    if len(drift) == 3:
        rows = []
        for s in SIZES:
            v = [drift[p][s] for p in ("start", "mid", "end")]
            rows.append([s, *(fmt(x) for x in v), fmt(100.0 * (max(v) - min(v)) / statistics.fmean(v), 2)])
        tables.append(("drift-ref", ["size", "start_ms", "mid_ms", "end_ms", "spread_pct"], rows))
    if not by:
        return tables
    base = {s: statistics.fmean(by["base"][s]) if by.get("base", {}).get(s) else None for s in SIZES}
    header = ["arm", "runs"]
    for s in SIZES:
        header += [f"{s}_mean_ms", f"{s}_cap_ms", f"{s}_cap_pct"]
    rows = []
    for arm in sorted(by, key=lambda a: (a != "base", a)):
        cells = []
        for s in SIZES:
            v = statistics.fmean(by[arm][s]) if by[arm].get(s) else None
            cells += [fmt(v), *cap_cols(v, base[s])]
        rows.append([arm, max(len(x) for x in by[arm].values()), *cells])
    tables.append(("caps-request", header, rows))
    rows = []
    for arm in sorted(passes, key=lambda a: (a != "base", a)):
        for s in SIZES:
            p = passes[arm].get(s)
            if p:
                rows.append([arm, s, len(p), fmt(min(p)), fmt(max(p)), fmt(100.0 * (max(p) - min(p)) / statistics.fmean(p), 2)])
    tables.append(("pass-spread", ["arm", "size", "passes", "pass_min_ms", "pass_max_ms", "spread_pct"], rows))
    return tables


def table_llb(runs):
    by = collections.defaultdict(lambda: collections.defaultdict(list))   # arm -> n_prompt -> [ms per prompt]
    for r, path in runs:
        if kind_of(r) != "llb":
            continue
        try:
            data = json.load(open(path[:-5] + ".out"))
        except (OSError, ValueError):
            continue
        for x in data:
            if x.get("n_gen", 0) == 0 and x.get("avg_ts"):
                by[arm_of(r)][int(x["n_prompt"])].append(1000.0 * x["n_prompt"] / x["avg_ts"])
    if not by:
        return []
    sizes = sorted({n for v in by.values() for n in v})
    header = ["arm"]
    for n in sizes:
        header += [f"pp{n}_ms", f"pp{n}_cap_ms", f"pp{n}_cap_pct"]
    base = {n: statistics.fmean(by["base"][n]) if by.get("base", {}).get(n) else None for n in sizes}
    rows = []
    for arm in sorted(by, key=lambda a: (a != "base", a)):
        cells = []
        for n in sizes:
            v = statistics.fmean(by[arm][n]) if by[arm].get(n) else None
            cells += [fmt(v), *cap_cols(v, base[n])]
        rows.append([arm, *cells])
    return [("caps-llama-bench", header, rows)]


def table_timing(runs):
    ub_rows, sum_rows = [], []
    for r, path in runs:
        if kind_of(r) != "srv" or arm_of(r) not in ("timing", "all_timing"):
            continue
        log = path[:-5] + ".server.log"
        if not os.path.exists(log):
            continue
        t = parse_timing(log)
        for name, lo, hi in UBATCH_CLASSES:
            u = [x for x in t["ubatch"] if lo <= x["n_tokens"] <= hi]
            if u:
                med = lambda k: statistics.median(x[k] for x in u)
                ub_rows.append([r, name, len(u), *(fmt(med(k), 0) for k in
                                ("apply_us", "build_us", "alloc_us", "inputs_us", "submit_us", "total_us")),
                                fmt(100.0 * sum(x["reuse"] for x in u) / len(u), 1)])
        modes = collections.Counter(x["mode"] for x in t["cuda_compute"])
        waits = sorted(x["wait_us"] for x in t["sync"])
        req = t["request"]
        sum_rows.append([
            r, len(t["ubatch"]), len(t["decode"]), len(t["sync"]), len(req),
            fmt(statistics.median(waits), 0) if waits else "",
            fmt(waits[int(0.9 * (len(waits) - 1))], 0) if waits else "",
            fmt(statistics.median(x["passes"] for x in t["sync"]), 0) if waits else "",
            *(fmt(statistics.median(x[k] for x in req), 0) if req else "" for k in ("total_us", "wait_us", "build_us", "format_us")),
            modes.get("direct", 0), modes.get("graph_capture", 0), modes.get("graph_replay", 0),
            fmt(statistics.median(x["host_us"] for x in t["cuda_compute"]), 0) if t["cuda_compute"] else ""])
    if not ub_rows:
        return []
    return [("timing-ubatch", ["run", "n_tokens_class", "ubatches", "apply_us", "build_us", "alloc_us", "inputs_us",
                               "submit_us", "total_us", "reuse_pct"], ub_rows),
            ("timing-summary", ["run", "ubatches", "decodes", "syncs", "requests", "sync_wait_median_us", "sync_wait_p90_us",
                                "passes_per_sync_median", "request_total_median_us", "request_wait_median_us",
                                "request_build_median_us", "request_format_median_us", "cuda_direct", "cuda_graph_capture",
                                "cuda_graph_replay", "cuda_host_median_us"], sum_rows)]


def table_ops(runs):
    """One row for each run that has DMK_ABLATE_SYNC set, and the DMK_OPS table if the logs have one."""
    tables, rows, logs = [], [], {}
    for r, path in runs:
        log = path[:-5] + ".server.log"
        if kind_of(r) not in ("ops", "smoke") or not os.path.exists(log):
            continue
        active, ops_lines, begin_lines = None, 0, 0
        for line in open(log, errors="replace"):
            m = re.search(r"DMK_ABLATE active: .* sync=(\d+)", line)
            if m:
                active = int(m.group(1))
            if "DMK_OPS\t" in line:
                ops_lines += 1
            if "DMK_OPS begin" in line:
                begin_lines += 1
        if active:
            rows.append([r, active, begin_lines, ops_lines])
            logs[arm_of(r)] = log
    if rows:
        tables.append(("ops-lines", ["run", "sync_flag", "ops_dumps", "ops_lines"], rows))
    if "sync" in logs and sum(v[0] for v in parse_ops(logs["sync"]).values()):
        ops = parse_ops(logs["sync"])
        floor = parse_ops(logs["sync_all"]) if "sync_all" in logs else {}
        f_n = sum(v[0] for v in floor.values())
        f_us = sum(v[1] for v in floor.values()) / f_n if f_n else 0.0     # cost of one node that launches nothing
        tot = collections.defaultdict(float)
        for (b, _), (n, us) in ops.items():
            tot[b] += us
        out = [[b, label, n, fmt(us, 1), fmt(us / n, 2), fmt(us / n - f_us, 2), fmt(100 * us / tot[b], 2)]
               for (b, label), (n, us) in sorted(ops.items(), key=lambda kv: (kv[0][0], -kv[1][1]))]
        tables.append(("ops", ["token_bucket", "label", "nodes", "total_us", "avg_us", "avg_minus_floor_us", "share_of_bucket_pct"], out))
    return tables


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv-dir", help="write one TSV file for each table to this folder")
    ap.add_argument("folders", nargs="+", help="results folders of run_schedule.sh")
    args = ap.parse_args()
    runs = []
    for d in args.folders:
        for p in sorted(glob.glob(os.path.join(d, "abl-*.meta"))):
            runs.append((os.path.basename(p)[:-5], p[:-5] + ".json"))
    if not runs:
        sys.exit("no abl-* runs found")
    tables = table_requests(runs) + table_llb(runs) + table_timing(runs) + table_ops(runs)
    if args.tsv_dir:
        os.makedirs(args.tsv_dir, exist_ok=True)
    for name, header, rows in tables:
        if args.tsv_dir:
            with open(os.path.join(args.tsv_dir, name + ".tsv"), "w") as f:
                f.write("\t".join(header) + "\n")
                for row in rows:
                    assert len(row) == len(header), (name, row)
                    f.write("\t".join(str(v) for v in row) + "\n")
            print(f"{name}.tsv: {len(rows)} rows")
        else:
            print(f"\n## {name}\n")
            print("\t".join(header))
            for row in rows:
                print("\t".join(str(v) for v in row))


if __name__ == "__main__":
    main()
