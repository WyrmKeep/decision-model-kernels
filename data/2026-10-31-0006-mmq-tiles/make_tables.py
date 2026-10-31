#!/usr/bin/env python3
"""Make all derived tables of the data set 2026-10-31-0006-mmq-tiles from the files in raw/.

Inputs (raw/):
  0006-sweep-r1-q4.out, 0006-sweep-r1-q8.out   the sweep of window 0006-sweep-r1
  0006-sweep-r1b-q8.out                        the repeat of two Q8_0 shapes (window 0006-sweep-r1b)
  inputs/census-cases.tsv                      copy of the table of the data set 2026-10-30-shape-census
  inputs/requests.tsv, pass_kernels.tsv, targets.tsv   copies of the files of the data set 2026-10-31-phase4-model
Output: tables/*.tsv

Python 3 standard library only. Run from any folder:

    python3 make_tables.py
"""
import csv
import math
import os
import re
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "tables")

FILES_R1 = ["0006-sweep-r1-q4.out", "0006-sweep-r1-q8.out"]
FILE_R1B = "0006-sweep-r1b-q8.out"
# The two shapes of the repeat. The tables use the rows of the repeat for these shapes.
# The rows of window 0006-sweep-r1 stay in tables/disturbed.tsv only.
REPEATED = {("Q8_0", 8192, 2560), ("Q8_0", 4096, 2560)}
TYPES = ["short1q", "short2q", "medium", "long"]
CENSUS_NAME = {"short-1q": "short1q", "short-2q": "short2q", "medium-3q": "medium", "long-3q": "long"}

# The MMQ launches of one pass: (name, type, M rows, K, launches). From the op table of the census.
# The two shapes with 32 rows (ssm_alpha, ssm_beta) use the fallback table and are not in the sweep.
OPS = [
    ("attn_qkv", "Q8_0", 8192, 2560, 24),
    ("attn_gate", "Q8_0", 4096, 2560, 24),
    ("ssm_out", "Q8_0", 2560, 4096, 24),
    ("ffn_up", "Q4_0", 9216, 2560, 31),
    ("ffn_gate", "Q4_0", 9216, 2560, 31),
    ("ffn_down", "Q4_0", 2560, 9216, 31),
    ("attn_q", "Q8_0", 8192, 2560, 8),
    ("attn_k", "Q8_0", 1024, 2560, 8),
    ("attn_v", "Q8_0", 1024, 2560, 8),
    ("attn_output", "Q8_0", 2560, 4096, 8),
]


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def pct(x):
    return f"{x:.2f}"


# ------------------------------------------------------------------------------------------ sweep rows
def load_sweep(fname):
    rows = []
    for r in read_tsv(os.path.join(RAW, fname)):
        if r["time_us"] in ("", "ERR") or r.get("regs") == "ERR":
            continue
        rows.append({
            "type": r["type"], "M": int(r["M"]), "K": int(r["K"]), "N": int(r["N"]), "variant": r["variant"],
            "nt": int(r["nt"]), "occ": int(r["occ_req"]), "I": int(r["I"]), "J": int(r["J"]), "policy": int(r["policy"]),
            "regs": int(r["regs"]), "local": int(r["local_bytes"]), "smem": int(r["smem_dynamic"]), "bps": int(r["blocks_per_sm"]),
            "ntiles": int(r["ntiles"]), "grid": int(r["grid"]), "fixup": int(r["fixup"]), "t": float(r["time_us"]),
            "ndiff": int(r["ndiff"]), "maxabs": float(r["maxabs"]), "ref": r["ref"] == "1", "src": fname,
        })
    return rows


r1 = []
for fn in FILES_R1:
    r1 += load_sweep(fn)
r1b = load_sweep(FILE_R1B)
rows = [r for r in r1 if (r["type"], r["M"], r["K"]) not in REPEATED] + r1b
for r in rows:
    r["family"] = f"nt{r['nt']}_o{r['occ']}_I{r['I']}"

by = defaultdict(list)
for r in rows:
    by[(r["type"], r["M"], r["K"], r["N"])].append(r)
base = {k: next(r for r in v if r["ref"] and r["policy"] == 0) for k, v in by.items()}
shapes = sorted({k[:3] for k in by})
grid_ns = sorted({k[3] for k in by})


def tops(M, K, N, t_us):
    return 2.0 * M * K * N / (t_us * 1e-6) / 1e12


# ------------------------------------------------------------------------------------------ table: baseline
out = []
for k in sorted(base):
    b = base[k]
    out.append([k[0], k[1], k[2], k[3], b["J"], b["grid"], b["fixup"], b["ntiles"], b["bps"], b["regs"], f"{b['t']:.2f}", f"{tops(k[1], k[2], k[3], b['t']):.1f}"])
write_tsv("baseline.tsv", ["type", "M_rows", "K", "N_tokens", "J_tile_tokens", "grid_blocks", "fixup", "tiles", "blocks_per_sm", "registers", "time_us", "tops_int8"], out)

# ------------------------------------------------------------------------------------------ table: best per shape
out = []
for k in sorted(by):
    b = base[k]
    best = min(by[k], key=lambda r: r["t"])
    ex = min((r for r in by[k] if r["ndiff"] == 0), key=lambda r: r["t"])
    out.append([k[0], k[1], k[2], k[3], f"{b['t']:.2f}", b["J"], b["fixup"],
                best["variant"], best["policy"], f"{best['t']:.2f}", f"{b['t'] / best['t']:.3f}", best["ndiff"],
                ex["variant"], ex["policy"], f"{ex['t']:.2f}", f"{b['t'] / ex['t']:.3f}"])
write_tsv("best-per-shape.tsv",
          ["type", "M_rows", "K", "N_tokens", "base_time_us", "base_J", "base_fixup",
           "best_variant", "best_policy", "best_time_us", "best_speedup", "best_ndiff",
           "exact_variant", "exact_policy", "exact_time_us", "exact_speedup"], out)

# ------------------------------------------------------------------------------------------ table: families
out = []
for k in sorted(by):
    b = base[k]
    fams = defaultdict(list)
    for r in by[k]:
        fams[r["family"]].append(r)
    for fam in sorted(fams):
        r = min(fams[fam], key=lambda x: x["t"])
        out.append([k[0], k[1], k[2], k[3], fam, r["J"], r["policy"], r["bps"], r["regs"], r["local"], f"{r['t']:.2f}", f"{r['t'] / b['t']:.3f}", r["ndiff"]])
write_tsv("families.tsv", ["type", "M_rows", "K", "N_tokens", "family", "J_of_best", "policy_of_best", "blocks_per_sm", "registers", "local_bytes", "time_us", "time_over_base", "ndiff"], out)

# ------------------------------------------------------------------------------------------ table: g-test
# Does more than one block for each SM recover time at large N? The base has one block of 8 warps for each SM.
tests = [
    ("two blocks of 4 warps, same J", lambda r, b: r["family"] == "nt128_o1_I64" and r["J"] == b["J"]),
    ("two blocks of 8 warps, best J", lambda r, b: r["family"] == "nt256_o2_I128"),
    ("three blocks of 4 warps, best J", lambda r, b: r["family"] == "nt128_o3_I64"),
]
out = []
for k in sorted(by):
    if k[3] not in (512, 1024, 1173, 1288, 2048):
        continue
    b = base[k]
    out.append([k[0], k[1], k[2], k[3], "base: one block of 8 warps", b["variant"], b["J"], b["bps"], 8 * b["bps"], b["regs"], b["local"], f"{b['t']:.2f}", "1.000", b["fixup"], 0])
    for name, sel in tests:
        c = [r for r in by[k] if sel(r, b)]
        if not c:
            continue
        r = min(c, key=lambda x: x["t"])
        warps = (r["nt"] // 32) * r["bps"]
        out.append([k[0], k[1], k[2], k[3], name, r["variant"], r["J"], r["bps"], warps, r["regs"], r["local"], f"{r['t']:.2f}", f"{r['t'] / b['t']:.3f}", r["fixup"], r["ndiff"]])
write_tsv("g-test.tsv", ["type", "M_rows", "K", "N_tokens", "test", "variant", "J_tile_tokens", "blocks_per_sm", "warps_per_sm", "registers", "local_bytes",
                         "time_us", "time_over_base", "fixup", "ndiff"], out)

# Summary of the test: the cells with 1,173 tokens or more in which the base and the other build run one block for each tile (no fixup).
out = []
for name, sel in tests:
    xs = []
    for k in sorted(by):
        b = base[k]
        if k[3] < 1173 or b["fixup"]:
            continue
        c = [r for r in by[k] if sel(r, b) and not r["fixup"]]
        if c:
            xs.append(min(r["t"] for r in c) / b["t"])
    out.append([name, len(xs), f"{min(xs):.3f}", f"{statistics.median(xs):.3f}", f"{max(xs):.3f}"])
write_tsv("g-summary.tsv", ["test", "cells", "min_time_over_base", "median_time_over_base", "max_time_over_base"], out)

# The ladder of J for one block of 8 warps and 128 rows with the grid rule of the base. A larger J needs more registers and more shared memory.
out = []
for k in sorted(by):
    if k[3] not in (1173, 2048) or (k[0], k[1], k[2]) not in (("Q4_0", 9216, 2560), ("Q8_0", 8192, 2560)):
        continue
    b = base[k]
    for r in sorted((r for r in by[k] if r["family"] == "nt256_o1_I128" and r["policy"] == 0 and r["J"] >= 32), key=lambda x: x["J"]):
        out.append([k[0], k[1], k[2], k[3], r["J"], r["regs"], r["local"], r["smem"], r["bps"], r["grid"], r["fixup"], f"{r['t']:.2f}", f"{r['t'] / b['t']:.3f}"])
write_tsv("j-ladder.tsv", ["type", "M_rows", "K", "N_tokens", "J_tile_tokens", "registers", "local_bytes", "shared_memory_bytes", "blocks_per_sm", "grid_blocks", "fixup",
                           "time_us", "time_over_base"], out)

# Fit T(J) = A + B / J to J of 64 and J of 128 at 2,048 tokens (2,048 is a multiple of both J, so no tile has padding).
# B / 128 as a part of T(128) is the part of the time that is inversely proportional to J: the work that each column tile repeats.
out = []
for shape in (("Q4_0", 9216, 2560), ("Q8_0", 8192, 2560)):
    k = shape + (2048,)
    t = {r["J"]: r["t"] for r in by[k] if r["family"] == "nt256_o1_I128" and r["policy"] == 0 and r["J"] in (64, 128)}
    share = (t[64] - t[128]) / t[128]
    out.append([shape[0], shape[1], shape[2], 2048, f"{t[64]:.2f}", f"{t[128]:.2f}", f"{100 * share:.1f}", f"{1 / (1 - share):.2f}"])
write_tsv("j-fit.tsv", ["type", "M_rows", "K", "N_tokens", "time_J64_us", "time_J128_us", "part_inverse_to_J_pct_of_time_J128", "ceiling_if_part_removed"], out)

# ------------------------------------------------------------------------------------------ weights
census = [r for r in read_tsv(os.path.join(RAW, "inputs", "census-cases.tsv")) if r["variant"] == "raw"]
requests = {r["request_type"]: r for r in read_tsv(os.path.join(RAW, "inputs", "requests.tsv"))}
cases = {t: int(requests[t]["cases"]) for t in TYPES}


def pass_tokens(passes):
    """'P68+S2x51+S1x69' -> [68, 102, 69]; 'F132' -> [132]."""
    res = []
    for item in passes.split("+"):
        m = re.fullmatch(r"([PSF])(?:(\d+)x)?(\d+)", item)
        if not m:
            raise ValueError("bad pass item " + item)
        res.append((int(m.group(2)) if m.group(2) else 1) * int(m.group(3)))
    return res


rows_by_type = defaultdict(list)
for r in census:
    rows_by_type[CENSUS_NAME[r["request_type"]]].append(r)
weights = defaultdict(float)  # (type, N) -> passes of one benchmark run
for t, rs in rows_by_type.items():
    w = cases[t] / len(rs)
    for r in rs:
        for n in pass_tokens(r["passes"]):
            if n >= 9:  # MMVQ below 9 tokens
                weights[(t, n)] += w
out = [[t, n, f"{w:.4f}"] for (t, n), w in sorted(weights.items(), key=lambda x: (TYPES.index(x[0][0]), x[0][1]))]
write_tsv("weights.tsv", ["request_type", "N_tokens", "passes_in_a_run_of_the_benchmark"], out)


def nearest(n):
    return min(grid_ns, key=lambda g: abs(math.log(g / n)))


# ------------------------------------------------------------------------------------------ rules
def pick_oracle(k, b):
    return min(by[k], key=lambda r: r["t"])


def pick_oracle_guard(k, b):
    r = min(by[k], key=lambda r: r["t"])
    return r if r["t"] * 1.03 <= b["t"] else b


def pick_exact(k, b):
    return min((r for r in by[k] if r["ndiff"] == 0), key=lambda r: r["t"])


def pick_exact_guard(k, b):
    r = pick_exact(k, b)
    return r if r["t"] * 1.03 <= b["t"] else b


def pick_i64(k, b):
    if not b["fixup"]:
        return b
    c = [r for r in by[k] if r["family"] == "nt128_o1_I64" and r["J"] == b["J"] and r["policy"] == 1]
    return c[0] if c else b


def pick_i64_bestj(k, b):
    if not b["fixup"]:
        return b
    c = [r for r in by[k] if r["nt"] == 128 and r["policy"] == 1]
    return min(c, key=lambda r: r["t"]) if c else b

def pick_q8_i64(k, b):
    return pick_i64(k, b) if k[0] == "Q8_0" else b


def pick_q8_i64_bestj(k, b):
    return pick_i64_bestj(k, b) if k[0] == "Q8_0" else b



RULES = [
    ("oracle-all", "Best row for each shape and N (picks the noise too)", pick_oracle),
    ("oracle-all-guard3", "Best row if it is at least 3 % faster, else the base", pick_oracle_guard),
    ("oracle-exact", "Best row with equal bits", pick_exact),
    ("oracle-exact-guard3", "Best row with equal bits if it is at least 3 % faster, else the base", pick_exact_guard),
    ("rule-i64-same-J", "Where the base runs the fixup path: 128 threads, 64 rows, the J of the base, one block for each tile", pick_i64),
    ("rule-i64-best-J", "Where the base runs the fixup path: the best J of the 128 thread, 64-row tile with one block for each tile", pick_i64_bestj),
    ("rule-q8-i64-same-J", "The rule rule-i64-same-J for Q8_0 only", pick_q8_i64),
    ("rule-q8-i64-best-J", "The rule rule-i64-best-J for Q8_0 only", pick_q8_i64_bestj),
]

wall = {t: float(requests[t]["wall_ms"]) for t in TYPES}


def model(pick):
    """MMQ ms of one request of each type, and the part that comes from rows with equal bits."""
    ms = {t: 0.0 for t in TYPES}
    ex = {t: 0.0 for t in TYPES}
    for (t, n), w in weights.items():
        g = nearest(n)
        for _, typ, M, K, layers in OPS:
            k = (typ, M, K, g)
            b = base[k]
            r = pick(k, b)
            ms[t] += w * layers * r["t"] / 1000.0 / cases[t]
            if r["ndiff"] == 0:
                ex[t] += w * layers * (b["t"] - r["t"]) / 1000.0 / cases[t]
    return ms, ex


base_ms, _ = model(lambda k, b: b)
out = []
for rule, text, pick in [("base", "The base", lambda k, b: b)] + RULES:
    ms, ex = model(pick)
    lat_saved = sum(cases[t] * (base_ms[t] - ms[t]) for t in TYPES) / sum(cases.values())
    lat = sum(cases[t] * wall[t] for t in TYPES) / sum(cases.values())
    for t in TYPES:
        saved = base_ms[t] - ms[t]
        out.append([rule, t, f"{base_ms[t]:.2f}", f"{ms[t]:.2f}", f"{saved:.2f}", pct(100 * saved / base_ms[t]),
                    pct(100 * saved / wall[t]), f"{ex[t]:.2f}"])
    out.append([rule, "lat_ms", f"{sum(cases[t] * base_ms[t] for t in TYPES) / sum(cases.values()):.2f}",
                f"{sum(cases[t] * ms[t] for t in TYPES) / sum(cases.values()):.2f}", f"{lat_saved:.2f}",
                pct(100 * lat_saved / (sum(cases[t] * base_ms[t] for t in TYPES) / sum(cases.values()))), pct(100 * lat_saved / lat),
                f"{sum(cases[t] * ex[t] for t in TYPES) / sum(cases.values()):.2f}"])
write_tsv("rules.tsv", ["rule", "request_type", "base_mmq_ms", "rule_mmq_ms", "saved_ms", "saved_pct_of_mmq", "saved_pct_of_request", "saved_ms_with_equal_bits"], out)
write_tsv("rules-text.tsv", ["rule", "meaning"], [["base", "The launch rule of the base build (J with the fewest tiles, 256 threads, 128 rows, stream-k)"]] + [[r, t] for r, t, _ in RULES])


# The same model for each operation: ms of one request, mean over the cases of the benchmark.
def model_ops(pick):
    ms = defaultdict(float)
    for (t, n), w in weights.items():
        g = nearest(n)
        for name, typ, M, K, layers in OPS:
            k = (typ, M, K, g)
            ms[name] += w * layers * pick(k, base[k])["t"] / 1000.0 / sum(cases.values())
    return ms


base_ops = model_ops(lambda k, b: b)
out = []
for rule, text, pick in RULES:
    ops_ms = model_ops(pick)
    for name, typ, M, K, layers in OPS:
        out.append([rule, name, typ, M, K, layers, f"{base_ops[name]:.2f}", f"{ops_ms[name]:.2f}", pct(100 * (base_ops[name] - ops_ms[name]) / base_ops[name])])
write_tsv("rules-by-op.tsv", ["rule", "operation", "type", "M_rows", "K", "launches_per_pass", "base_ms_per_case", "rule_ms_per_case", "saved_pct_of_operation"], out)

# The same model for each class of the number of tokens of a pass: ms of one request, mean over the cases of the benchmark.
N_CLASSES = [("9-64", 9, 64), ("65-256", 65, 256), ("257-1024", 257, 1024), ("1025-2048", 1025, 2048)]


def model_n(pick):
    ms = defaultdict(float)
    for (t, n), w in weights.items():
        cls = next(c for c, lo, hi in N_CLASSES if lo <= n <= hi)
        g = nearest(n)
        for name, typ, M, K, layers in OPS:
            k = (typ, M, K, g)
            ms[cls] += w * layers * pick(k, base[k])["t"] / 1000.0 / sum(cases.values())
    return ms


base_n = model_n(lambda k, b: b)
out = []
for rule, text, pick in RULES:
    n_ms = model_n(pick)
    for cls, lo, hi in N_CLASSES:
        out.append([rule, cls, f"{base_n[cls]:.2f}", f"{n_ms[cls]:.2f}", pct(100 * (base_n[cls] - n_ms[cls]) / base_n[cls])])
write_tsv("rules-by-n.tsv", ["rule", "tokens_of_pass", "base_ms_per_case", "rule_ms_per_case", "saved_pct_of_class"], out)

# ------------------------------------------------------------------------------------------ phase-4 prediction of H and G
passes = {r["pass_type"]: r for r in read_tsv(os.path.join(RAW, "inputs", "pass_kernels.tsv"))}


def req_passes(t):
    return [passes[p] for p in requests[t]["passes"].split(",")]


def mmq_by_size(plist, large, mid, small):
    total = 0.0
    for p in plist:
        tok = p["tokens"]
        f = large if tok in ("2048", "1025-2047") else mid if tok in ("65-128", "129-256") else small
        total += float(p["mmq_ms"]) * f
    return total


lat_wall = sum(cases[t] * wall[t] for t in TYPES) / sum(cases.values())
# (name, confidence, saved fraction of the MMQ time at >= 1025 tokens, at 65 to 256 tokens, at 9 to 64 tokens)
# The confidence of H and G is the value of the file inputs/targets.tsv. The revised G has the confidence 0.25.
scen = [
    ("H phase-4 prediction", 0.30, 0.08, 0.05, 0.05),
    ("G phase-4 prediction (speedup 1.35, 1.30, 1.15)", 0.40, 1 - 1 / 1.35, 1 - 1 / 1.30, 1 - 1 / 1.15),
    ("G revised (speedup 1.08, 1.10, 1.08)", 0.25, 1 - 1 / 1.08, 1 - 1 / 1.10, 1 - 1 / 1.08),
    ("G revised, low (speedup 1.00, 1.05, 1.03)", 0.25, 0.0, 1 - 1 / 1.05, 1 - 1 / 1.03),
    ("G revised, high (speedup 1.20, 1.15, 1.10)", 0.25, 1 - 1 / 1.20, 1 - 1 / 1.15, 1 - 1 / 1.10),
]
out = []
for name, conf, sl, sm, ss in scen:
    ms = {t: mmq_by_size(req_passes(t), sl, sm, ss) for t in TYPES}
    lat_ms = sum(cases[t] * ms[t] for t in TYPES) / sum(cases.values())
    row = [name, f"{conf:.2f}"]
    for t in TYPES:
        row += [f"{ms[t]:.1f}", pct(100 * ms[t] / wall[t])]
    row += [f"{lat_ms:.1f}", pct(100 * lat_ms / lat_wall), pct(conf * max(100 * ms[t] / wall[t] for t in TYPES))]
    out.append(row)
hdr = ["scenario", "confidence"]
for t in TYPES:
    hdr += [f"{t}_saved_ms", f"{t}_saved_pct"]
hdr += ["lat_saved_ms", "lat_saved_pct", "max_expected_pct"]
write_tsv("g-h-scenarios.tsv", hdr, out)

# ------------------------------------------------------------------------------------------ noise
# Two rows of one launch with the same variant, the same grid and the same fixup differ only by the noise of the run.
def spreads(rs):
    g = defaultdict(list)
    for r in rs:
        g[(r["type"], r["M"], r["K"], r["N"], r["variant"], r["grid"], r["fixup"])].append(r["t"])
    return [max(v) / min(v) for v in g.values() if len(v) >= 2]


def quant(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


out = []
for label, rs in (("window r1", r1), ("window r1b", r1b)):
    for sh in sorted({(r["type"], r["M"], r["K"]) for r in rs}):
        sp = spreads([r for r in rs if (r["type"], r["M"], r["K"]) == sh])
        out.append([label, sh[0], sh[1], sh[2], len(sp), pct(100 * (statistics.median(sp) - 1)), pct(100 * (quant(sp, 0.9) - 1)), pct(100 * (max(sp) - 1))])
write_tsv("noise.tsv", ["window", "type", "M_rows", "K", "pairs", "median_spread_pct", "p90_spread_pct", "max_spread_pct"], out)

# ------------------------------------------------------------------------------------------ disturbed shapes: window r1 against window r1b
key = lambda r: (r["type"], r["M"], r["K"], r["N"], r["variant"], r["policy"])
old = {key(r): r for r in r1 if (r["type"], r["M"], r["K"]) in REPEATED}
out = []
for sh in sorted(REPEATED):
    for n in grid_ns:
        rat = [old[key(r)]["t"] / r["t"] for r in r1b if (r["type"], r["M"], r["K"], r["N"]) == sh + (n,) and key(r) in old]
        brat = [old[key(r)]["t"] / r["t"] for r in r1b if (r["type"], r["M"], r["K"], r["N"]) == sh + (n,) and r["ref"] and r["policy"] == 0 and key(r) in old]
        if rat:
            out.append([sh[0], sh[1], sh[2], n, len(rat), f"{statistics.median(rat):.3f}", f"{quant(rat, 0.9):.3f}", f"{max(rat):.3f}", f"{brat[0]:.3f}" if brat else ""])
write_tsv("disturbed.tsv", ["type", "M_rows", "K", "N_tokens", "rows_compared", "median_r1_over_r1b", "p90_r1_over_r1b", "max_r1_over_r1b", "base_row_r1_over_r1b"], out)

# ------------------------------------------------------------------------------------------ class of the best rows
out = []
for sh in shapes:
    n_total = n_eq = n_gain10 = n_gain10_eq = 0
    for n in grid_ns:
        k = sh + (n,)
        b = base[k]
        best = min(by[k], key=lambda r: r["t"])
        n_total += 1
        n_eq += best["ndiff"] == 0
        if b["t"] / best["t"] >= 1.10:
            n_gain10 += 1
            n_gain10_eq += best["ndiff"] == 0
    nfix = sum(base[sh + (n,)]["fixup"] for n in grid_ns)
    out.append([sh[0], sh[1], sh[2], n_total, nfix, n_eq, n_gain10, n_gain10_eq])
write_tsv("class.tsv", ["type", "M_rows", "K", "N_values", "N_values_where_base_runs_fixup", "N_values_best_has_equal_bits", "N_values_best_is_10pct_faster", "of_these_with_equal_bits"], out)

print("rows used:", len(rows), "r1:", len(r1), "r1b:", len(r1b), "shapes:", len(shapes), "N values:", len(grid_ns))
