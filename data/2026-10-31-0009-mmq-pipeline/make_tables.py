#!/usr/bin/env python3
"""Makes all tables of the data set 2026-10-31-0009-mmq-pipeline from the files raw/*.out.

Each run has the baseline of a cell (the kernel without the pipeline, column ref 1) and the variant with the pipeline (name ends in _abl1, ref 0).
The ratio of a cell is the time of the baseline divided by the time of the pipeline in the same run.
The table vs-ceiling.tsv also reads the ablation runs of the data set 2026-10-31-0006-mmq-tiles.
Usage: python3 make_tables.py
"""
import csv
import os
import statistics
from collections import defaultdict
import math

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "tables")
ABL = os.path.join(HERE, "..", "2026-10-31-0006-mmq-tiles", "raw", "0006-abl-r3-q4.out")

GATE_N = 1025   # the limit of the change looks at the cells with this number of tokens and more
GATE = 1.08     # ratio of the limit
SHAPES = [("9216", "2560", "ffn_up, ffn_gate"), ("2560", "9216", "ffn_down")]
N_CLASSES = [("9-64", 9, 64), ("65-256", 65, 256), ("257-1024", 257, 1024), ("1025-2048", 1025, 2048)]

# version, short name, runs
VERSIONS = [
    ("1", "weights only", ["0009-pipe-r1", "0009-pipe-r2"]),
    ("2", "weights and activations, 4 barriers", ["0009-pipe2-r1", "0009-pipe2-r2"]),
    ("2a", "version 2, loops not expanded", ["0009-pipe3-r1", "0009-pipe3-r2"]),
    ("2b", "version 2a, loops that make the int8 values expanded 2 times", ["0009-pipe4-r1", "0009-pipe4-r2"]),
    ("3", "3 barriers, work copy", ["0009-pipe5-r1", "0009-pipe5-r2"]),
    ("3c", "version 3 (commit)", ["0009-pipef-r1", "0009-pipef-r2", "0009-pipef-r3"]),
]
FINAL = "3c"


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(name, header, rows):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def fmt(x, d=3):
    return f"{x:.{d}f}"


def load_run(run):
    """cells[(M, K, N)] = {"base": row, "pipe": row}"""
    cells = defaultdict(dict)
    for r in read_tsv(os.path.join(RAW, run + "-q4.out")):
        if r["ref"] not in ("0", "1"):
            continue
        key = (r["M"], r["K"], int(r["N"]))
        if r["ref"] == "1":
            cells[key]["base"] = r
        elif r["variant"].endswith("_abl1"):
            cells[key]["pipe"] = r
    return {k: v for k, v in cells.items() if "base" in v and "pipe" in v}


RUNS = {run: load_run(run) for _, _, runs in VERSIONS for run in runs}


def ratios(design_runs):
    """ratio[(M, K, N)] = list of ratios, one for each run"""
    out = defaultdict(list)
    for run in design_runs:
        for k, c in RUNS[run].items():
            out[k].append(float(c["base"]["time_us"]) / float(c["pipe"]["time_us"]))
    return out


def mean(xs):
    return sum(xs) / len(xs)


def class_of(n):
    for name, lo, hi in N_CLASSES:
        if lo <= n <= hi:
            return name
    return None


# ---------------------------------------------------------------- versions.tsv
rows = []
for d, name, runs in VERSIONS:
    rt = ratios(runs)
    cell_mean = {k: mean(v) for k, v in rt.items()}
    gate_cells = {k: v for k, v in cell_mean.items() if k[2] >= GATE_N}
    up = [v for k, v in gate_cells.items() if k[0] == "9216"]
    down = [v for k, v in gate_cells.items() if k[0] == "2560"]
    ndiff = max(int(c["pipe"]["ndiff"]) for run in runs for c in RUNS[run].values())
    rows.append([d, name, " ".join(runs), len(gate_cells), fmt(mean(list(gate_cells.values()))), fmt(min(gate_cells.values())),
                 fmt(max(gate_cells.values())), fmt(mean(up)), fmt(mean(down)),
                 sum(1 for v in cell_mean.values() if v < 1.0), len(cell_mean), ndiff,
                 "yes" if mean(list(gate_cells.values())) >= GATE else "no"])
write_tsv("versions.tsv", ["version", "name", "runs", "cells_ge_1025", "mean_ratio_ge_1025", "min_ratio_ge_1025", "max_ratio_ge_1025",
                           "mean_ratio_ffn_up_gate", "mean_ratio_ffn_down", "cells_slower_than_baseline", "cells", "max_ndiff", "limit_met"], rows)

# ---------------------------------------------------------------- final-speedup.tsv
final_runs = dict((d, runs) for d, _, runs in VERSIONS)[FINAL]
rows = []
for M, K, op in SHAPES:
    for n in sorted({k[2] for k in RUNS[final_runs[0]] if k[0] == M and k[1] == K}):
        key = (M, K, n)
        base = [float(RUNS[r][key]["base"]["time_us"]) for r in final_runs]
        pipe = [float(RUNS[r][key]["pipe"]["time_us"]) for r in final_runs]
        rr = [b / p for b, p in zip(base, pipe)]
        c = RUNS[final_runs[0]][key]
        rows.append([M, K, op, n, c["base"]["J"], fmt(mean(base), 1), fmt(mean(pipe), 1), fmt(mean(rr)), fmt(min(rr)), fmt(max(rr)),
                     c["base"]["regs"], c["pipe"]["regs"], c["pipe"]["local_bytes"], c["base"]["smem_dynamic"], c["pipe"]["smem_dynamic"],
                     c["pipe"]["blocks_per_sm"], max(int(RUNS[r][key]["pipe"]["ndiff"]) for r in final_runs)])
write_tsv("final-speedup.tsv", ["M", "K", "operations", "tokens", "J", "baseline_us", "pipeline_us", "ratio", "ratio_min", "ratio_max",
                                "regs_baseline", "regs_pipeline", "local_bytes_pipeline", "smem_baseline", "smem_pipeline",
                                "blocks_per_sm_pipeline", "ndiff"], rows)

# ---------------------------------------------------------------- by-version.tsv
per_design = {d: {k: mean(v) for k, v in ratios(runs).items()} for d, _, runs in VERSIONS}
keys = sorted(per_design[FINAL], key=lambda k: (k[0] != "9216", k[2]))
rows = []
for k in keys:
    rows.append([k[0], k[1], k[2]] + [fmt(per_design[d][k]) if k in per_design[d] else "" for d, _, _ in VERSIONS])
write_tsv("by-version.tsv", ["M", "K", "tokens"] + [f"ratio_version_{d}" for d, _, _ in VERSIONS], rows)

# ---------------------------------------------------------------- registers.tsv
rows = []
for d, _, runs in VERSIONS:
    for J in (16, 64, 120, 128):
        for k, c in sorted(RUNS[runs[0]].items()):
            if k[0] == "9216" and int(c["base"]["J"]) == J:
                rows.append([d, J, k[2], c["base"]["regs"], c["base"]["local_bytes"], c["base"]["smem_dynamic"], c["base"]["blocks_per_sm"],
                             c["pipe"]["regs"], c["pipe"]["local_bytes"], c["pipe"]["smem_dynamic"], c["pipe"]["blocks_per_sm"]])
                break
write_tsv("registers.tsv", ["version", "J", "tokens_of_cell", "regs_baseline", "local_bytes_baseline", "smem_baseline", "blocks_per_sm_baseline",
                            "regs_pipeline", "local_bytes_pipeline", "smem_pipeline", "blocks_per_sm_pipeline"], rows)

# ---------------------------------------------------------------- limit.tsv
rows = []
for M, K, op in SHAPES:
    rr = [(k[2], per_design[FINAL][k]) for k in per_design[FINAL] if k[0] == M and k[2] >= GATE_N]
    rr.sort()
    rows.append([M, K, op, " ".join(f"{n}:{fmt(v)}" for n, v in rr), fmt(mean([v for _, v in rr]))])
allv = [v for k, v in per_design[FINAL].items() if k[2] >= GATE_N]
rows.append(["all", "all", "the two shapes", "", fmt(mean(allv))])
rows.append(["limit", "", "ratio of the limit", "", fmt(GATE, 2)])
write_tsv("limit.tsv", ["M", "K", "operations", "ratio_of_each_cell_ge_1025", "mean_ratio"], rows)

# ---------------------------------------------------------------- vs-ceiling.tsv
abl = defaultdict(dict)
for r in read_tsv(ABL):
    if r["ref"] not in ("0", "1") or int(r["streamk"]) != 1:
        continue
    key = (r["M"], r["K"], int(r["N"]))
    if r["ref"] == "1":
        abl[key]["base"] = float(r["time_us"])
    elif r["variant"].endswith("_abl12"):
        abl[key]["same"] = float(r["time_us"])
rows = []
by_class = defaultdict(list)
for k in keys:
    if k not in abl or "same" not in abl[k] or "base" not in abl[k]:
        continue
    c = RUNS[final_runs[0]][k]
    base = mean([float(RUNS[r][k]["base"]["time_us"]) for r in final_runs])
    pipe = mean([float(RUNS[r][k]["pipe"]["time_us"]) for r in final_runs])
    same = abl[k]["same"] * base / abl[k]["base"]   # the time of same_loads, scaled to the baseline of this data set
    frac = (base - pipe) / (base - same) if base > same else float("nan")
    rows.append([k[0], k[1], k[2], fmt(base, 1), fmt(pipe, 1), fmt(same, 1), fmt(base / pipe), fmt(base / same), fmt(100 * frac, 1)])
    by_class[(k[0], class_of(k[2]))].append(frac)
    by_class[("all", class_of(k[2]))].append(frac)
write_tsv("vs-ceiling.tsv", ["M", "K", "tokens", "baseline_us", "pipeline_us", "same_loads_us", "ratio_pipeline", "ratio_same_loads", "pct_of_ceiling_removed"], rows)

rows = []
for shape in ("9216", "2560", "all"):
    for name, lo, hi in N_CLASSES:
        v = by_class.get((shape, name), [])
        if v:
            rows.append([shape, name, len(v), fmt(100 * mean(v), 1)])
write_tsv("vs-ceiling-by-class.tsv", ["M", "tokens_of_pass", "cells", "mean_pct_of_ceiling_removed"], rows)

# ---------------------------------------------------------------- projection.tsv
# The time that the pipeline removes from the Q4_0 operations of each request type, with the census of the data set 2026-10-31-0006-mmq-tiles.
# The number of tokens of a pass has the cell with the nearest number of tokens of the ablation runs (logarithmic distance).
T0006 = os.path.join(HERE, "..", "2026-10-31-0006-mmq-tiles")
TYPES = ["short1q", "short2q", "medium", "long"]
Q4_OPS = [("ffn_up", "9216", "2560", 31), ("ffn_gate", "9216", "2560", 31), ("ffn_down", "2560", "9216", 31)]
requests = {r["request_type"]: r for r in read_tsv(os.path.join(T0006, "raw", "inputs", "requests.tsv"))}
weights = defaultdict(float)
for r in read_tsv(os.path.join(T0006, "tables", "weights.tsv")):
    weights[(r["request_type"], int(r["N_tokens"]))] += float(r["passes_in_a_run_of_the_benchmark"])
grid = sorted({k[2] for k in abl if k[0] == "9216" and "same" in abl[k]})


def nearest(n):
    return min(grid, key=lambda g: abs(math.log(g / n)))


res = {}
for t in TYPES:
    cases = int(requests[t]["cases"])
    wall = float(requests[t]["wall_ms"])
    base_ms = pipe_ms = pos_ms = ceil_ms = 0.0
    for (rt, n), w in weights.items():
        if rt != t:
            continue
        g = nearest(n)
        for _, M, K, layers in Q4_OPS:
            k = (M, K, g)
            b = mean([float(RUNS[r][k]["base"]["time_us"]) for r in final_runs])
            p = mean([float(RUNS[r][k]["pipe"]["time_us"]) for r in final_runs])
            f = w / cases * layers / 1000.0
            base_ms += f * b
            pipe_ms += f * (b - p)
            pos_ms += f * max(b - p, 0.0)   # a rule that uses the pipeline only in the cells where it is faster
            ceil_ms += f * (abl[k]["base"] - abl[k]["same"])
    res[t] = (cases, wall, base_ms, pipe_ms, pos_ms, ceil_ms)
tot_cases = sum(v[0] for v in res.values())
rows = []
for t in TYPES:
    cases, wall, base_ms, pipe_ms, pos_ms, ceil_ms = res[t]
    rows.append([t, fmt(wall, 1), fmt(base_ms, 2), fmt(pipe_ms, 2), fmt(100 * pipe_ms / wall, 2), fmt(pos_ms, 2), fmt(100 * pos_ms / wall, 2),
                 fmt(ceil_ms, 2), fmt(100 * ceil_ms / wall, 2), fmt(100 * pipe_ms / ceil_ms, 1)])
lat = [sum(res[t][i] * res[t][0] for t in TYPES) / tot_cases for i in range(1, 6)]
rows.append(["lat_ms", fmt(lat[0], 1), fmt(lat[1], 2), fmt(lat[2], 2), fmt(100 * lat[2] / lat[0], 2), fmt(lat[3], 2), fmt(100 * lat[3] / lat[0], 2),
             fmt(lat[4], 2), fmt(100 * lat[4] / lat[0], 2), fmt(100 * lat[2] / lat[4], 1)])
write_tsv("projection.tsv", ["request_type", "request_ms", "q4_0_mmq_ms", "removed_ms", "removed_pct_of_request", "removed_where_faster_ms",
                             "removed_where_faster_pct_of_request", "ceiling_removed_ms", "ceiling_removed_pct_of_request",
                             "pct_of_ceiling_removed"], rows)
print("tables written to", OUT)
