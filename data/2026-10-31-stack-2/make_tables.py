#!/usr/bin/env python3
"""Make the derived tables of the data set 2026-10-31-stack-2 from the files in raw/.

Usage: python3 data/2026-10-31-stack-2/make_tables.py
The script writes into the folder tables/ next to it. It calls
harness/analysis/ab_stats.py and harness/analysis/answers_compare.py from this
repository. It also reads the table speed-bench.tsv and the table speed-llama-bench.tsv
of the data sets of the three earlier tests that the prediction multiplied (see
EARLIER_TESTS). It uses only the Python standard library. It changes no file in raw/.
Run it again after each change of a file in raw/.
"""
import json
import math
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "tables")
ANALYSIS = os.path.join(ROOT, "harness", "analysis")
SIZES = ("short", "medium", "long")
PROMPTS = (64, 256, 3000)
BLOCKS_BENCH = 6
BLOCKS_LLB = 4
SCHEDULES = ("stack2-bench.txt", "stack2-llb.txt")

# The prediction entry docs/notebook/2026-10-31-stack-2-prediction.md, table "Predicted gain":
# metric, predicted change (%), the two ends of the range (%).
PREDICTION = [
    ("lat_ms", -9.8, -11.8, -7.8),
    ("short_ms", -9.8, -11.8, -7.8),
    ("medium_ms", -9.8, -11.8, -7.8),
    ("long_ms", -9.4, -11.4, -7.4),
    ("rps8", 10.7, 8.7, 12.7),
    ("pp64_ts", 6.0, 4.0, 8.0),
    ("pp256_ts", 7.7, 5.7, 9.7),
    ("pp3000_ts", 9.5, 7.5, 11.5),
]
LOWER_IS_BETTER = ("lat_ms", "short_ms", "medium_ms", "long_ms")
LLB_METRICS = ("pp64_ts", "pp256_ts", "pp3000_ts")

# The three earlier tests that the prediction multiplied: key, data set folder, the two tables
# (private benchmark, llama bench), and the two arms of the test (arm A is the reference arm).
# Phase 1 has the stock image as arm A, and build D as arm B. The first test of the stack has
# build D as arm A and the stack at fbfd9cb8c as arm B. Change 0003 has the stack at fbfd9cb8c
# as arm A and the stack with the change (A3R) as arm B. The prediction used the ratio 1 for
# change 0003 in llama bench, thus the test of change 0003 has no table of llama bench here.
EARLIER_TESTS = [
    ("phase1", "2026-10-30-window1", ("speed-bench.tsv", "speed-llama-bench.tsv"), "stock image", "build D"),
    ("stack1", "2026-10-31-stack-1", ("speed-bench.tsv", "speed-llama-bench.tsv"), "build D", "stack at fbfd9cb8c"),
    ("c0003", "2026-10-31-0003-answer-head", ("speed-rebased.tsv", None), "stack at fbfd9cb8c", "stack with change 0003"),
]


def raw(name):
    return os.path.join(RAW, name)


def write_tsv(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            assert len(r) == len(header), (name, r)
            f.write("\t".join("" if v is None else str(v) for v in r) + "\n")


def num(v, places):
    return None if v is None else f"{v:.{places}f}"


# ---------------------------------------------------------------- the schedules and the run files
def schedule():
    """Return {run name: line type} from the schedule files that the harness ran."""
    lines = {}
    for name in SCHEDULES:
        if not os.path.exists(raw(name)):
            continue
        for line in open(raw(name)):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words = line.split()
            while re.match(r"DMK_[A-Z0-9_]+=", words[0]):
                words = words[1:]
            lines[words[1]] = words[0]
    return lines


def test_name(run):
    for prefix, label in (("stack2-bench-", "private benchmark"), ("stack2-llb-", "llama bench")):
        if run.startswith(prefix):
            return label
    raise SystemExit(f"make_tables.py: unknown run name {run}")


def arm_tag(run):
    return run.rsplit("-", 1)[1]


def block_of(run):
    return int(re.search(r"-b(\d+)-", run).group(1))


def load1(path):
    """The one-minute load average from a line of /proc/loadavg."""
    return float(open(path).read().split()[0])


def read_run(run):
    """Facts of one run from its .meta file (and .load0, .load1 for bench lines)."""
    meta = open(raw(run + ".meta")).read().splitlines()
    d = {"run": run, "arm": meta[0].split()[0].split("=", 1)[1], "exit_code": None}
    d["start_utc"] = [m for m in meta if m.startswith("date_utc=")][0].split("=", 1)[1]
    if os.path.exists(raw(run + ".load0")):  # a bench line
        d["load_before"] = load1(raw(run + ".load0"))
        d["load_after"] = load1(raw(run + ".load1"))
        info = meta[2].split(", ")
        d["throttle"] = info[2] if len(info) > 2 else None
    else:  # a tool line: the load averages are lines 3 and 6 of the meta file
        d["load_before"] = float(meta[2].split()[0])
        d["load_after"] = float(meta[5].split()[0])
        info = meta[3].split(", ")
        d["throttle"] = None
        d["exit_code"] = int([m for m in meta if m.startswith("rc=")][0].split("=", 1)[1])
    d["driver"] = info[0]
    d["power_limit_w"] = info[1].replace(" W", "")
    return d


def smi_stats(run):
    """Samples of the GPU sampler: five values on each line, separated by a comma and a space."""
    rows = [l.strip().split(", ") for l in open(raw(run + ".smi")) if l.strip()]
    rows = [r for r in rows if len(r) == 5 and r[4].isdigit()]
    return {
        "mem": max(float(r[1]) for r in rows),
        "power": statistics.median(float(r[2]) for r in rows),
        "clock": statistics.median(int(r[4]) for r in rows),
    }


def bench_json(run):
    p = raw(run + ".json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    s, q = j["seq"], j["quality"]
    return {
        "lat_ms": s["lat_ms"],
        **{f"{k}_ms": s["by_size"][k]["mean"] for k in SIZES},
        "rps8": j["tput"]["rps"],
        "correct": q["correct"],
        "total": q["total"],
        "agree_n": q["agree_n"],
        "mean_dp": q["mean_dp"],
        "max_dp": q["max_dp"],
        "flips": len(q["flips"]),
        "errors": len(j["errors"]),
        "guard_pass": j["guard_pass"],
    }


def guards(run):
    """The four guards of the benchmark in one run, as {name: bool}."""
    return json.load(open(raw(run + ".json")))["guard"]


def llb_values(run):
    """The prompt speed in tokens/s for each prompt length from the JSON in the .out file of llama bench."""
    p = raw(run + ".out")
    if not os.path.exists(p):
        return None
    return {f"pp{r['n_prompt']}_ts": r["avg_ts"] for r in json.load(open(p)) if r.get("n_gen", 0) == 0 and r.get("n_prompt")}


def runs_table():
    lines = schedule()
    runs = [read_run(r) for r in lines]
    runs.sort(key=lambda d: (d["start_utc"], d["run"]))
    rows = []
    for d in runs:
        run = d["run"]
        b, p = bench_json(run), llb_values(run)
        smi = smi_stats(run)
        rows.append([
            run, test_name(run), lines[run], block_of(run), arm_tag(run), d["arm"], d["start_utc"],
            num(d["load_before"], 2), num(d["load_after"], 2),
            d["driver"], d["power_limit_w"], d["throttle"], d["exit_code"],
            num(b["lat_ms"], 3) if b else None, num(b["short_ms"], 3) if b else None,
            num(b["medium_ms"], 3) if b else None, num(b["long_ms"], 3) if b else None,
            num(b["rps8"], 3) if b else None,
            *[num(p[f"pp{n}_ts"], 3) if p else None for n in PROMPTS],
            num(smi["clock"], 1), num(smi["power"], 2), num(smi["mem"], 0),
        ])
    write_tsv("runs.tsv", [
        "run", "test", "line_type", "block", "arm_tag", "arm", "start_utc", "load1_before", "load1_after",
        "driver", "power_limit_w", "throttle_reasons_at_start", "exit_code",
        "lat_ms", "short_ms", "medium_ms", "long_ms", "rps8", "pp64_ts", "pp256_ts", "pp3000_ts",
        "sm_clock_mhz_median", "power_draw_w_median", "memory_used_mib_max"], rows)
    return {d["run"]: d for d in runs}


# ---------------------------------------------------------------- ab_stats and answers_compare
def run_tool(args):
    return subprocess.run([sys.executable] + args, cwd=HERE, check=True, capture_output=True, text=True).stdout


def read_tsv(path):
    lines = open(path).read().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, l.split("\t"))) for l in lines[1:]]


def ab_tables():
    os.makedirs(OUT, exist_ok=True)
    for prefix, name in (("stack2-bench", "speed-bench.tsv"), ("stack2-llb", "speed-llama-bench.tsv")):
        table = run_tool([os.path.relpath(os.path.join(ANALYSIS, "ab_stats.py"), HERE), "raw", "--prefix", prefix,
                          "--a", "P", "--b", "S2", "--margin", "2"])
        with open(os.path.join(OUT, name), "w") as f:
            f.write(table)
    out = run_tool([os.path.relpath(os.path.join(ANALYSIS, "answers_compare.py"), HERE),
                    "--a", "raw/stack2-bench-b*-P.reference.json", "--b", "raw/stack2-bench-b*-S2.reference.json"])
    with open(os.path.join(OUT, "answers-compare-bench.json"), "w") as f:
        f.write(out)
    return json.loads(out)


def block_tables(runs):
    metrics = ["lat_ms", "short_ms", "medium_ms", "long_ms", "rps8"]
    rows = []
    for block in range(1, BLOCKS_BENCH + 1):
        p, s = f"stack2-bench-b{block}-P", f"stack2-bench-b{block}-S2"
        bp, bs = bench_json(p), bench_json(s)
        first = "P" if runs[p]["start_utc"] < runs[s]["start_utc"] else "S2"
        rows.append([block, first,
                     num(runs[p]["load_before"], 2), num(runs[p]["load_after"], 2),
                     num(runs[s]["load_before"], 2), num(runs[s]["load_after"], 2)]
                    + [num(100 * (bs[m] / bp[m] - 1), 3) for m in metrics])
    write_tsv("speed-bench-blocks.tsv", [
        "block", "first_arm", "load1_p_before", "load1_p_after", "load1_s2_before", "load1_s2_after",
        "lat_ms_diff_pct", "short_ms_diff_pct", "medium_ms_diff_pct", "long_ms_diff_pct", "rps8_diff_pct"], rows)

    rows = []
    for block in range(1, BLOCKS_LLB + 1):
        p, s = f"stack2-llb-b{block}-P", f"stack2-llb-b{block}-S2"
        vp, vs = llb_values(p), llb_values(s)
        first = "P" if runs[p]["start_utc"] < runs[s]["start_utc"] else "S2"
        rows.append([block, first,
                     num(runs[p]["load_before"], 2), num(runs[p]["load_after"], 2),
                     num(runs[s]["load_before"], 2), num(runs[s]["load_after"], 2)]
                    + [num(100 * (vs[f"pp{n}_ts"] / vp[f"pp{n}_ts"] - 1), 3) for n in PROMPTS])
    write_tsv("speed-llama-bench-blocks.tsv", [
        "block", "first_arm", "load1_p_before", "load1_p_after", "load1_s2_before", "load1_s2_after",
        "pp64_ts_diff_pct", "pp256_ts_diff_pct", "pp3000_ts_diff_pct"], rows)


def quality_table(runs):
    rows = []
    guard_rows = []
    for run in sorted((r for r in runs if bench_json(r)), key=lambda r: (block_of(r), arm_tag(r))):
        b = bench_json(run)
        rows.append([run, block_of(run), arm_tag(run), b["correct"], b["total"], b["agree_n"].split("/")[0],
                     b["mean_dp"], b["max_dp"], b["flips"], b["errors"], b["guard_pass"]])
        g = guards(run)
        guard_rows.append([run, block_of(run), arm_tag(run), g["errors"], g["correct"], g["agree"], g["mean_dp"], b["guard_pass"],
                           num(smi_stats(run)["mem"], 0)])
    write_tsv("quality-bench.tsv", [
        "run", "block", "arm_tag", "correct", "decisions", "agree", "mean_dp", "max_dp",
        "top_answer_differs_from_reference", "errors", "guards_true"], rows)
    write_tsv("guards-bench.tsv", [
        "run", "block", "arm_tag", "guard_errors", "guard_correct", "guard_agree", "guard_mean_dp", "guards_true",
        "memory_used_mib_max"], guard_rows)


# ---------------------------------------------------------------- prediction against result
def speed_rows():
    rows = {}
    for name in ("speed-bench.tsv", "speed-llama-bench.tsv"):
        for r in read_tsv(os.path.join(OUT, name)):
            rows[r["metric"]] = r
    return rows


def prediction_table():
    result = speed_rows()
    rows = []
    for metric, predicted, lo, hi in PREDICTION:
        r = result[metric]
        value = float(r["diff_pct"])
        inside = lo <= value <= hi
        better = (value < lo) if metric in LOWER_IS_BETTER else (value > hi)
        rows.append([metric, num(predicted, 1), num(lo, 1), num(hi, 1), num(value, 3), r["ci_lo_pct"], r["ci_hi_pct"],
                     num(value - predicted, 3), inside, "inside" if inside else ("better" if better else "worse")])
    write_tsv("prediction-vs-result.tsv", [
        "metric", "predicted_pct", "range_min_pct", "range_max_pct", "result_pct", "ci_lo_pct", "ci_hi_pct",
        "result_minus_predicted_points", "in_range", "position"], rows)


def earlier_rows():
    """The rows of the three earlier tests: {key: {metric: row}}. A test with no table has no row for llama bench."""
    out = {}
    for key, folder, tables, arm_a, arm_b in EARLIER_TESTS:
        rows = {}
        for name in tables:
            if name is None:
                continue
            for r in read_tsv(os.path.join(ROOT, "data", folder, "tables", name)):
                rows[r["metric"]] = r
        out[key] = rows
    return out


def estimate_tables():
    """The product of the ratios of the three earlier tests, its interval, and the result of the test of the stack."""
    result = speed_rows()
    earlier = earlier_rows()
    rows = []
    for metric, predicted, lo, hi in PREDICTION:
        ratio = 1.0
        squares = 0.0
        cells = []
        for key, folder, tables, arm_a, arm_b in EARLIER_TESTS:
            r = earlier[key].get(metric)
            if r is None:
                assert key == "c0003" and metric in LLB_METRICS, (key, metric)  # the prediction used the ratio 1
                cells += [None, None]
                continue
            half = (float(r["ci_hi_pct"]) - float(r["ci_lo_pct"])) / 2
            ratio *= 1 + float(r["diff_pct"]) / 100
            squares += half ** 2
            cells += [r["diff_pct"], num(half, 3)]
        product = 100 * (ratio - 1)
        assert abs(product - predicted) < 0.06, (metric, product, predicted)  # the prediction used this product
        half = math.sqrt(squares)
        value = float(result[metric]["diff_pct"])
        rows.append([metric] + cells + [num(ratio, 4), num(product, 3), num(half, 3), num(product - half, 3),
                     num(product + half, 3), num(value, 3), num(value - product, 3),
                     min(product - half, product + half) <= value <= max(product - half, product + half)])
    write_tsv("stack-estimate.tsv", [
        "metric", "phase1_diff_pct", "phase1_half_width_points", "stack1_diff_pct", "stack1_half_width_points",
        "c0003_diff_pct", "c0003_half_width_points", "product_of_ratios", "product_diff_pct",
        "combined_half_width_points", "combined_end_a_pct", "combined_end_b_pct", "result_pct",
        "result_minus_product_points", "result_in_combined_interval"], rows)


def millisecond_table():
    """The gain in ms of each earlier test and of the test of the stack, for the metrics of the private benchmark in ms."""
    result = {r["metric"]: r for r in read_tsv(os.path.join(OUT, "speed-bench.tsv"))}
    earlier = earlier_rows()
    rows = []
    for metric in ("lat_ms", "short_ms", "medium_ms", "long_ms"):
        gains = [float(earlier[key][metric]["a_mean"]) - float(earlier[key][metric]["b_mean"])
                 for key, folder, tables, arm_a, arm_b in EARLIER_TESTS]
        stack = float(result[metric]["a_mean"]) - float(result[metric]["b_mean"])
        rows.append([metric] + [num(g, 3) for g in gains] + [num(sum(gains), 3), num(stack, 3), num(stack - sum(gains), 3)])
    write_tsv("gain-ms.tsv", [
        "metric", "gain_phase1_ms", "gain_stack1_ms", "gain_0003_ms", "sum_of_gains_ms",
        "gain_of_stack2_ms", "stack2_minus_sum_ms"], rows)


def comparison_table():
    """The test of the stack against the first test of the stack, and against the test of change 0003."""
    result = speed_rows()
    earlier = earlier_rows()
    rows = []
    for metric, predicted, lo, hi in PREDICTION:
        e, s = earlier["stack1"][metric], result[metric]
        first, second = float(e["diff_pct"]), float(s["diff_pct"])
        # The ratio of this test (stack at 419b8f612 against the image) divided by the ratio of the first test
        # (stack at fbfd9cb8c against build D). It has the effect of change 0003 and the difference of build D and the image.
        ratio = 100 * ((1 + second / 100) / (1 + first / 100) - 1)
        cells = [metric, e["a_mean"], e["b_mean"], num(first, 3), s["a_mean"], s["b_mean"], num(second, 3),
                 num(ratio, 3),
                 num(100 * (float(s["a_mean"]) / float(e["a_mean"]) - 1), 3),
                 num(100 * (float(s["b_mean"]) / float(e["b_mean"]) - 1), 3)]
        r = earlier["c0003"].get(metric)
        if r is None:
            rows.append(cells + [None, None, None, None])
            continue
        c_lo, c_hi = float(r["ci_lo_pct"]), float(r["ci_hi_pct"])
        rows.append(cells + [r["diff_pct"], r["ci_lo_pct"], r["ci_hi_pct"], c_lo <= ratio <= c_hi])
    write_tsv("comparison-stack-1.tsv", [
        "metric", "stack1_a_mean", "stack1_b_mean", "stack1_diff_pct", "stack2_a_mean", "stack2_b_mean",
        "stack2_diff_pct", "stack2_over_stack1_pct", "a_mean_stack2_over_stack1_pct", "b_mean_stack2_over_stack1_pct",
        "c0003_diff_pct", "c0003_ci_lo_pct", "c0003_ci_hi_pct", "stack2_over_stack1_in_c0003_interval"], rows)


if __name__ == "__main__":
    runs = runs_table()
    answers = ab_tables()
    block_tables(runs)
    quality_table(runs)
    prediction_table()
    estimate_tables()
    millisecond_table()
    comparison_table()
    print("answers: self repeat P", answers["self_repeat_a"], "- self repeat S2", answers["self_repeat_b"],
          "- cross identical", answers["cross_identical_all"], "- cross max |dp|", answers["cross_max_abs_dp"],
          "- top flips", answers["cross_top_flips"], "- decisions", answers["decisions"])
    print("tables written to", os.path.relpath(OUT, ROOT))
