#!/usr/bin/env python3
"""Make all tables of the data set 2026-10-31-0002-gdn-value-split from the files in raw/.

Run it in any folder: python3 make_tables.py
It uses only the standard library of Python 3. It writes the folder tables/.
"""
import collections
import glob
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)

WHOLE = "128"
SPLITS = ["64_1", "64_2", "32_1", "32_2", "16_1", "16_2", "16_3"]
PATHS = [WHOLE] + SPLITS
# Slice width V, thread blocks of one head, and the value of BLOCKS (the launch bound) of each forced path.
PATH_INFO = {"128": (128, 1, 1), "64_1": (64, 2, 1), "64_2": (64, 2, 2), "32_1": (32, 4, 1), "32_2": (32, 4, 2),
             "16_1": (16, 8, 1), "16_2": (16, 8, 2), "16_3": (16, 8, 3)}
CONTENDED = 1.10   # a time is contended when it is more than 1.10 times the smallest time of the 3 repetitions
SM_COUNT = 38
HEADS = 32


def write_tsv(name, header, rows):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")


def fmt(x, digits=1):
    return f"{x:.{digits}f}"


# ---------------------------------------------------------------- the perf runs
ANSI = re.compile(r"\x1b\[[0-9;]*m")   # the program colors the rate in the line
PERF_LINE = re.compile(
    r"n_seq_tokens=(\d+),n_seqs=(\d+),.*?\):\s+(\d+) runs -\s+([\d.]+) us/run -\s+(\d+) kB/run -\s+([\d.]+) GB/s")
perf = {}   # (path, rep, n_seqs, n_tokens) -> (runs, us, kb, gbs)
runs_rows = []
for meta in sorted(glob.glob(os.path.join(RAW, "perf", "perf-r*-*.meta"))):
    base = meta[:-5]
    m = re.search(r"perf-r(\d+)-(.+)$", os.path.basename(base))
    rep, path = int(m.group(1)), m.group(2)
    lines = open(meta).read().splitlines()
    head = dict(kv.split("=", 1) for kv in lines[0].split())
    load_before = lines[1].split()[0]
    drv, pl, clk0, temp0, thr0 = [x.strip() for x in lines[2].split(",")]
    rc = lines[3].split("=")[1]
    load_after = lines[4].split()[0]
    clk1, temp1, pw1 = [x.strip() for x in lines[5].split(",")]
    n = 0
    for line in open(base + ".out"):
        mm = PERF_LINE.search(ANSI.sub("", line))
        if mm:
            tok, seqs, runs, us, kb, gbs = mm.groups()
            perf[(path, rep, int(seqs), int(tok))] = (int(runs), float(us), int(kb), float(gbs))
            n += 1
    runs_rows.append([path, rep, head["date_utc"], load_before, load_after, drv, pl.replace(" W", ""),
                      clk0.replace(" MHz", ""), temp0, thr0, clk1.replace(" MHz", ""), temp1,
                      pw1.replace(" W", ""), rc, n])
runs_rows.sort(key=lambda r: r[2])
shapes = sorted({(s, t) for (_, _, s, t) in perf})
reps = sorted({r for (_, r, _, _) in perf})
assert len(runs_rows) == len(PATHS) * len(reps), len(runs_rows)
assert all(r[-1] == len(shapes) for r in runs_rows)

# contended: one time is more than CONTENDED times the smallest time of the same path and shape
contended = {}
for (p, s, t) in {(p, s, t) for (p, _, s, t) in perf}:
    us = {r: perf[(p, r, s, t)][1] for r in reps}
    low = min(us.values())
    for r in reps:
        contended[(p, r, s, t)] = us[r] > CONTENDED * low
    assert sum(contended[(p, r, s, t)] for r in reps) <= 1, (p, s, t)   # the median of 3 stays a clean time

count_by_run = collections.Counter(k[:2] for k, v in contended.items() if v)
for row in runs_rows:
    row.append(count_by_run[(row[0], row[1])])
write_tsv("runs.tsv", ["path", "rep", "start_utc", "load1_before", "load1_after", "driver", "power_limit_w",
                       "sm_clock_mhz_start", "temperature_c_start", "throttle_reasons_at_start", "sm_clock_mhz_end",
                       "temperature_c_end", "power_draw_w_end", "exit_code", "shapes", "contended_shapes"], runs_rows)

write_tsv("perf-long.tsv", ["path", "rep", "n_seqs", "n_tokens", "runs", "us_per_run", "kb_per_run", "gb_per_s",
                            "contended"],
          [[p, r, s, t, *perf[(p, r, s, t)][:1], fmt(perf[(p, r, s, t)][1], 2), perf[(p, r, s, t)][2],
            fmt(perf[(p, r, s, t)][3], 2), "true" if contended[(p, r, s, t)] else "false"]
           for p in PATHS for r in reps for (s, t) in shapes])

med = {}
summary_rows = []
for (s, t) in shapes:
    for p in PATHS:
        us = [perf[(p, r, s, t)][1] for r in reps]
        med[(p, s, t)] = statistics.median(us)
for (s, t) in shapes:
    for p in PATHS:
        us = [perf[(p, r, s, t)][1] for r in reps]
        clean = sum(not contended[(p, r, s, t)] for r in reps)
        summary_rows.append([s, t, p, len(us), fmt(med[(p, s, t)], 2), fmt(min(us), 2), fmt(max(us), 2), clean,
                             f"{med[(p, s, t)] / med[(WHOLE, s, t)]:.4f}"])
write_tsv("perf-summary.tsv", ["n_seqs", "n_tokens", "path", "reps", "median_us", "min_us", "max_us", "clean_reps",
                               "ratio_to_whole"], summary_rows)

best_rows = []
for (s, t) in shapes:
    best = min(SPLITS, key=lambda p: med[(p, s, t)])
    best_rows.append([s, t, fmt(med[(WHOLE, s, t)], 1), best, fmt(med[(best, s, t)], 1),
                      f"{med[(best, s, t)] / med[(WHOLE, s, t)]:.3f}",
                      "true" if med[(WHOLE, s, t)] < min(med[(p, s, t)] for p in SPLITS) else "false"])
write_tsv("perf-best-split.tsv", ["n_seqs", "n_tokens", "whole_us", "best_split", "best_split_us", "ratio_to_whole",
                                  "whole_is_fastest"], best_rows)
assert all(r[-1] == "true" for r in best_rows)


spread_rows = []
for p in PATHS:
    worst = 0.0
    n_cont = 0
    min_clean = 3
    for (s, t) in shapes:
        clean = [perf[(p, r, s, t)][1] for r in reps if not contended[(p, r, s, t)]]
        n_cont += len(reps) - len(clean)
        min_clean = min(min_clean, len(clean))
        worst = max(worst, 100.0 * (max(clean) / min(clean) - 1.0))
    spread_rows.append([p, len(shapes) * len(reps), n_cont, min_clean, f"{worst:.2f}"])
write_tsv("perf-rep-spread.tsv", ["path", "times", "contended_times", "fewest_clean_reps", "largest_spread_clean_pct"],
          spread_rows)


# difference of the best value-split path to the whole-head path, with the range of the clean times
diff_rows = []
for (s, t) in shapes:
    best = min(SPLITS, key=lambda p: med[(p, s, t)])
    cw = [perf[(WHOLE, r, s, t)][1] for r in reps if not contended[(WHOLE, r, s, t)]]
    cb = [perf[(best, r, s, t)][1] for r in reps if not contended[(best, r, s, t)]]
    d = 100.0 * (med[(best, s, t)] / med[(WHOLE, s, t)] - 1.0)
    lo = 100.0 * (min(cb) / max(cw) - 1.0)
    hi = 100.0 * (max(cb) / min(cw) - 1.0)
    diff_rows.append([s, t, fmt(med[(WHOLE, s, t)], 1), best, fmt(med[(best, s, t)], 1), f"{d:.1f}", f"{lo:.1f}",
                      f"{hi:.1f}", len(cw), len(cb)])
write_tsv("perf-diff-interval.tsv", ["n_seqs", "n_tokens", "whole_us", "best_split", "best_split_us", "difference_pct",
                                     "low_pct", "high_pct", "clean_reps_whole", "clean_reps_split"], diff_rows)

range_rows = []
for p in SPLITS:
    ratios = [med[(p, s, t)] / med[(WHOLE, s, t)] for (s, t) in shapes]
    v, slices, blocks = PATH_INFO[p]
    ctas1 = HEADS * slices
    range_rows.append([p, v, blocks, ctas1, f"{ctas1 / SM_COUNT:.2f}", f"{min(ratios):.3f}", f"{max(ratios):.3f}",
                       f"{med[(p, 1, 2048)] / med[(WHOLE, 1, 2048)]:.3f}"])
write_tsv("perf-ratio-range.tsv", ["path", "slice_width", "blocks_bound", "thread_blocks_1_seq", "blocks_per_sm_1_seq",
                                   "ratio_min", "ratio_max", "ratio_1_seq_2048"], range_rows)

scale_rows = []
for t in sorted({t for (_, t) in shapes}):
    scale_rows.append([t] + [f"{med[(WHOLE, s, t)] / med[(WHOLE, 1, t)]:.3f}" for s in (1, 2, 3)])
write_tsv("perf-whole-scaling.tsv", ["n_tokens", "ratio_1_seq", "ratio_2_seq", "ratio_3_seq"], scale_rows)

# ---------------------------------------------------------------- MMA count (from the loop bounds of gdn_single)
# Counted for one chunk of C = 16 tokens and D = 128. Each product() call of K has K/16 steps of 3 MMA instructions.
D, C, N = 128, 16, 8
mma_rows = []
for p in PATHS:
    v, slices, _ = PATH_INFO[p]
    gram = 2 * (C // 16) * (C // N) * (D // 16) * 3          # K K^T and Q K^T
    proj = (C // 16) * (v // N) * 2 * (D // 16) * 3          # Q S and K S
    solve = (C // 16) * (v // N) * (C // 16) * 3             # inverse * delta
    out = (C // 16) * (v // N) * (C // 16) * 3               # P * solved
    state = (D // 16) * (v // N) * (C // 16) * 3             # K^T * delta, for each state tile
    block = gram + proj + solve + out + state
    mma_rows.append([p, v, slices, gram, proj, solve, out, state, block, block * slices, ""])
whole_total = next(r[9] for r in mma_rows if r[0] == WHOLE)
for r in mma_rows:
    r[10] = f"{r[9] / whole_total:.3f}"
write_tsv("mma-count.tsv", ["path", "slice_width", "slices_per_head", "gram", "projection", "solve", "out", "state",
                            "per_thread_block", "per_head", "ratio_to_whole"], mma_rows)

# ---------------------------------------------------------------- test-backend-ops in test mode, one run for each path
TEST_LINE = re.compile(r"^\s+GATED_DELTA_NET\((.*)\): (OK|FAIL.*)$")
test_rows = []
for meta in sorted(glob.glob(os.path.join(RAW, "test", "test-*.meta"))):
    base = meta[:-5]
    path = os.path.basename(base)[len("test-"):]
    rc = [l for l in open(meta).read().splitlines() if l.startswith("rc=")][0][3:]
    cases = ok = 0
    backend_ok = False
    for line in open(base + ".out"):
        line = ANSI.sub("", line)
        m = TEST_LINE.match(line)
        if m:
            cases += 1
            ok += m.group(2) == "OK"
        backend_ok = backend_ok or line.strip() == "Backend CUDA0: OK"
    test_rows.append([path, rc, cases, ok, cases - ok, "true" if backend_ok else "false"])
order = ["default"] + PATHS
test_rows.sort(key=lambda r: order.index(r[0]))
write_tsv("test-paths.tsv", ["path", "exit_code", "cases", "cases_ok", "cases_failed", "backend_ok"], test_rows)
assert all(r[4] == 0 and r[5] == "true" and r[1] == "0" for r in test_rows)

# ---------------------------------------------------------------- bit compare of all paths on the same input
CHECK = re.compile(r"GDN_CHECK H=(\d+) seqs=(\d+) tokens=(\d+) path=(\S+) bytes_differ=(\d+) of (\d+)")
check_rows = []
for line in open(os.path.join(RAW, "bitcompare", "bitcompare.err")):
    m = CHECK.search(line)
    if m:
        h, s, t, p, bad, n = m.groups()
        check_rows.append([h, s, t, p, bad, n])
write_tsv("bitcompare.tsv", ["heads", "n_seqs", "n_tokens", "path", "floats_differ", "floats_compared"], check_rows)
by_shape = collections.defaultdict(list)
for r in check_rows:
    by_shape[tuple(r[:3]) + (r[5],)].append(r)
sum_rows = [[*k[:3], k[3], len(v), sum(int(x[4]) for x in v)] for k, v in sorted(by_shape.items(), key=lambda kv: tuple(int(x) for x in kv[0]))]
write_tsv("bitcompare-shapes.tsv", ["heads", "n_seqs", "n_tokens", "floats_compared", "paths_compared",
                                    "floats_differ_in_all_paths"], sum_rows)
assert all(int(r[4]) == 0 for r in check_rows) and all(r[4] == len(PATHS) for r in sum_rows)

# ---------------------------------------------------------------- estimate of the candidate for the benchmark requests
rows_in = [l.rstrip("\n").split("\t") for l in open(os.path.join(RAW, "inputs", "gdn-ms-per-pass.tsv"))]
gdn = {r[0]: (r[1], float(r[2]), r[3]) for r in rows_in[1:]}
reqs = [l.rstrip("\n").split("\t") for l in open(os.path.join(RAW, "inputs", "request-passes.tsv"))][1:]
# The shape of the micro-benchmark that stands for the pass type (one sequence): the best split path of that shape.
STAND_FOR = {"prefix_large": 2048, "prefix_small_1025_2047": 1173, "prefix_small_65_128": 128, "single_129_256": 256}
est_rows = []
tot_w = tot_n = 0
for rt, cases, wall, passes in reqs:
    gdn_ms = extra = 0.0
    for pt in passes.split(","):
        if gdn[pt][2] != "gdn_single":
            continue
        t = STAND_FOR[pt]
        best = min(SPLITS, key=lambda p: med[(p, 1, t)])
        ratio = med[(best, 1, t)] / med[(WHOLE, 1, t)]
        gdn_ms += gdn[pt][1]
        extra += gdn[pt][1] * (ratio - 1.0)
    est_rows.append([rt, cases, wall, fmt(gdn_ms, 1), fmt(extra, 1), fmt(100 * extra / float(wall), 1)])
    tot_w += float(cases) * extra
    tot_n += int(cases)
lat_wall = sum(float(r[1]) * float(r[2]) for r in reqs) / tot_n
est_rows.append(["lat_ms", tot_n, fmt(lat_wall, 1), "", fmt(tot_w / tot_n, 1), fmt(100 * tot_w / tot_n / lat_wall, 1)])
write_tsv("candidate-estimate.tsv", ["request_type", "cases", "wall_ms", "gdn_single_ms", "extra_ms_estimate",
                                     "extra_pct_estimate"], est_rows)


# ---------------------------------------------------------------- tests of the builds of the branches
BT = os.path.join(RAW, "build-tests")
ERRSUM = re.compile(r"(?:ERROR SUMMARY: |RACECHECK SUMMARY: \d+ hazards displayed \()(\d+) error")
bt_rows = []
for build in ("0002-pr", "0002-candidate"):
    for tag in ("test", "memcheck", "racecheck", "initcheck", "synccheck"):
        base = os.path.join(BT, build, tag)
        if not os.path.exists(base + ".meta"):
            continue
        meta = open(base + ".meta").read().splitlines()
        rc = [l for l in meta if l.startswith("rc=")][0][3:]
        text = ANSI.sub("", open(base + ".out").read())
        cases = ok = 0
        for line in text.splitlines():
            m = TEST_LINE.match(line)
            if m:
                cases += 1
                ok += m.group(2) == "OK"
        errs = ERRSUM.findall(text + ANSI.sub("", open(base + ".err").read()))
        bt_rows.append([build, "none" if tag == "test" else tag, rc, cases, ok, cases - ok,
                        "" if tag == "test" else (errs[-1] if errs else "")])
write_tsv("build-tests.tsv", ["build", "sanitizer_tool", "exit_code", "cases", "cases_ok", "cases_failed",
                              "sanitizer_errors"], bt_rows)
assert all(r[2] == "0" and r[5] == 0 for r in bt_rows)
assert all(r[6] == "0" for r in bt_rows if r[1] != "none")

# ---------------------------------------------------------------- SASS compare of each build with the parent build
sass_rows = []
for label, folder in (("0002-pr", "sass-compare-pr"), ("0002-candidate", "sass-compare-candidate")):
    d = json.load(open(os.path.join(RAW, folder, "sass-compare.json")))
    sm = d["summary"]
    sass_rows.append([label, sm["cubins_per_side"], sm["names_in_both"], sm["names_only_A"], sm["names_only_B"],
                      sm["names_sass_identical"], sm["names_sass_different"],
                      sm["names_encoding_identical_incl_control_bits"], sm["names_encoding_different"],
                      sm["names_resusage_different"], sm["instr_total_A"], sm["instr_total_B"],
                      ";".join(d["onlyB"])])
write_tsv("sass-summary.tsv", ["build", "cubins_per_side", "names_in_both", "names_only_parent", "names_only_build",
                               "names_sass_identical", "names_sass_different", "names_encoding_identical",
                               "names_encoding_different", "names_resusage_different", "instructions_parent",
                               "instructions_build", "names_only_in_build"], sass_rows)

print(f"wrote {len(glob.glob(os.path.join(OUT, '*.tsv')))} tables to {OUT}")
