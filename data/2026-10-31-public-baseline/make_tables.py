#!/usr/bin/env python3
"""Make the derived tables of the data set 2026-10-31-public-baseline from the files in raw/.

Usage: python3 data/2026-10-31-public-baseline/make_tables.py
The script writes into the folder tables/ next to it. It calls
harness/analysis/answers_compare.py and bench/summarize.py from this repository.
It uses only the Python standard library. It changes no file in raw/.
It also checks that raw/p3-pub-mkbase.baseline.json has the values that
bench/tools/make_baseline.py gives for the three baseline runs, and that
bench/baseline.json has the same bytes. It writes the result of each check
on the standard output. Run it again after each change of a file in raw/.
"""
import glob
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "tables")
SCHEDULE = "window2-pub.txt"
BASELINE_RUNS = ["p3-pub-base-1", "p3-pub-base-2", "p3-pub-base-3"]
CHECK_RUN = "p3-pub-base-4"
MKBASE_RUN = "p3-pub-mkbase"
SIZES = ("short", "medium", "long")


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


def load1(path):
    """The one-minute load average from a line of /proc/loadavg."""
    return float(open(path).read().split()[0])


# ---------------------------------------------------------------- the schedule and the run files
def schedule():
    """Return {run name: (line type, script, line)} from the schedule file that the harness ran."""
    lines = {}
    for line in open(raw(SCHEDULE)):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words = line.split()
        lines[words[1]] = (words[0], os.path.basename(words[3]), line)
    return lines


def step_name(script, line):
    if script == "make_baseline.py":
        return "make baseline"
    if "--baseline /runs/" in line:
        return "check run"
    return "baseline run"


def read_run(run):
    """Facts of one run from its .meta, .load0 and .load1 files."""
    meta = open(raw(run + ".meta")).read().splitlines()
    info = meta[2].split(", ")
    return {
        "arm": meta[0].split("=", 1)[1],
        "start_utc": [m for m in meta if m.startswith("date_utc=")][0].split("=", 1)[1],
        "load_before": load1(raw(run + ".load0")),
        "load_after": load1(raw(run + ".load1")),
        "driver": info[0],
        "power_limit_w": info[1].replace(" W", ""),
        "throttle": info[2],
    }


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
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------- runs.tsv and public-bench.tsv
def runs_table(lines):
    runs = {r: read_run(r) for r in lines}
    rows = []
    for run in sorted(runs, key=lambda r: (runs[r]["start_utc"], r)):
        d, b, smi = runs[run], bench_json(run), smi_stats(run)
        line_type, script, line = lines[run]
        s = b["seq"] if b else None
        rows.append([
            run, step_name(script, line), line_type, d["arm"], d["start_utc"],
            num(d["load_before"], 2), num(d["load_after"], 2),
            d["driver"], d["power_limit_w"], d["throttle"],
            num(s["lat_ms"], 3) if b else None,
            *[num(s["by_size"][k]["mean"], 3) if b else None for k in SIZES],
            num(b["tput"]["rps"], 3) if b else None,
            num(smi["clock"], 1), num(smi["power"], 2), num(smi["mem"], 0),
        ])
    write_tsv("runs.tsv", [
        "run", "step", "line_type", "arm", "start_utc", "load1_before", "load1_after",
        "driver", "power_limit_w", "throttle_reasons_at_start",
        "lat_ms", "short_ms", "medium_ms", "long_ms", "rps8",
        "sm_clock_mhz_median", "power_draw_w_median", "memory_used_mib_max"], rows)


def public_table(lines):
    rows = []
    for run in BASELINE_RUNS + [CHECK_RUN]:
        b = bench_json(run)
        s, q = b["seq"], b["quality"]
        rows.append([
            run, step_name(lines[run][1], lines[run][2]), os.path.basename(q["model"]), b["meta"]["passes"],
            num(s["lat_ms"], 3), num(s["spread_pct"], 2),
            *[num(s["by_size"][k]["mean"], 3) for k in SIZES], num(s["all"]["p95"], 3),
            num(b["tput"]["rps"], 3), num(b["tput"]["decisions_per_s"], 3),
            q["correct"], q["total"], q["agree_n"].split("/")[0], num(q["agree"], 2),
            num(q["mean_dp"], 6), num(q["max_dp"], 6), " ".join(q["flips"]),
            len(b["errors"]), b["guard_pass"],
        ])
    write_tsv("public-bench.tsv", [
        "run", "step", "model", "timed_passes", "lat_ms", "spread_pct",
        "short_ms", "medium_ms", "long_ms", "p95_ms", "rps8", "decisions_per_s",
        "correct", "decisions", "agree_decisions", "agree", "mean_dp", "max_dp", "flips",
        "errors", "guard_pass"], rows)


# ---------------------------------------------------------------- baseline-check.tsv and guard-check.tsv
def vram_peak(run):
    """The same rule as bench/tools/make_baseline.py: the largest memory.used value of the sampler file."""
    return max(float(l.split(",")[1]) for l in open(raw(run + ".smi")) if len(l.split(",")) == 5)


def baseline_table():
    runs = [bench_json(r) for r in BASELINE_RUNS]
    baseline = json.load(open(raw(MKBASE_RUN + ".baseline.json")))
    expected = {
        "correct": ("lowest of the runs", min(r["quality"]["correct"] for r in runs)),
        "agree": ("lowest of the runs", min(r["quality"]["agree"] for r in runs)),
        "mean_dp": ("highest of the runs", max(r["quality"]["mean_dp"] for r in runs)),
        "lat_ms": ("`seq.lat_ms` of each run, 1 decimal place", [round(r["seq"]["lat_ms"], 1) for r in runs]),
        "short_ms": ("`seq.by_size.short.mean` of each run, 1 decimal place",
                     [round(r["seq"]["by_size"]["short"]["mean"], 1) for r in runs]),
        "rps": ("median of `tput.rps`, 2 decimal places", round(statistics.median(r["tput"]["rps"] for r in runs), 2)),
        "conc": ("`tput.conc` of the first run", runs[0]["tput"]["conc"]),
        "vram_mib": ("largest `memory.used` of the sampler files, rounded", round(max(vram_peak(r) for r in BASELINE_RUNS))),
    }
    rows, ok = [], baseline["status"] == "measured" and baseline["runs"] == BASELINE_RUNS
    for field, (rule, value) in expected.items():
        same = baseline[field] == value
        ok = ok and same
        rows.append([field, rule, json.dumps(value), json.dumps(baseline[field]), same])
    write_tsv("baseline-check.tsv", ["field", "rule", "value_from_runs", "value_in_baseline_json", "equal"], rows)
    print("baseline.json values equal the values that make_baseline.py gives for the three runs:", ok)
    same_bytes = open(raw(MKBASE_RUN + ".baseline.json"), "rb").read() == open(
        os.path.join(ROOT, "bench", "baseline.json"), "rb").read()
    print("bench/baseline.json has the same bytes as raw/" + MKBASE_RUN + ".baseline.json:", same_bytes)
    return baseline


def guard_table(baseline):
    b = bench_json(CHECK_RUN)
    q = b["quality"]
    vram = vram_peak(CHECK_RUN)
    rows = [
        ["errors", len(b["errors"]), 0, "None", "equal to the limit", len(b["errors"]) == 0, b["guard"]["errors"]],
        ["correct", q["correct"], baseline["correct"] - 1, "decisions", "not less than the limit",
         q["correct"] >= baseline["correct"] - 1, b["guard"]["correct"]],
        ["agree", q["agree"], round(baseline["agree"] - 0.02, 2), "0 to 1", "not less than the limit",
         q["agree"] >= baseline["agree"] - 0.02, b["guard"]["agree"]],
        ["mean_dp", num(q["mean_dp"], 6), num(baseline["mean_dp"] + 0.01, 6), "probability", "not more than the limit",
         q["mean_dp"] <= baseline["mean_dp"] + 0.01, b["guard"]["mean_dp"]],
        ["vram", num(vram, 0), 6144, "MiB", "not more than the limit", vram <= 6144, None],
    ]
    write_tsv("guard-check.tsv", ["guard", "run_value", "limit", "unit", "rule", "passes", "guard_in_run_json"], rows)
    same = all(r[6] is None or r[6] == r[5] for r in rows)
    print("guard values of " + CHECK_RUN + " equal the values that bench.py wrote:", same)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "bench", "summarize.py"), raw(CHECK_RUN)],
                         check=True, capture_output=True, text=True).stdout
    verdict = re.search(r"^\s*guard (\w+/?\w*)", out, re.M).group(1)
    print("bench/summarize.py gives for " + CHECK_RUN + " the result:", verdict)


# ---------------------------------------------------------------- answers_compare
def answers_compare():
    cmd = [sys.executable, os.path.join(ROOT, "harness", "analysis", "answers_compare.py"),
           "--a", "raw/p3-pub-base-[123].reference.json", "--b", "raw/p3-pub-base-4.reference.json"]
    out = subprocess.run(cmd, cwd=HERE, check=True, capture_output=True, text=True).stdout
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "answers-compare-baseline.json"), "w") as f:
        f.write(out)


if __name__ == "__main__":
    lines = schedule()
    runs_table(lines)
    public_table(lines)
    baseline = baseline_table()
    guard_table(baseline)
    answers_compare()
    print("tables written to", os.path.relpath(OUT, ROOT))
