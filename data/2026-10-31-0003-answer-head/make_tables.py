#!/usr/bin/env python3
"""Make the derived tables of the data set 2026-10-31-0003-answer-head from the files in raw/.

Usage: python3 data/2026-10-31-0003-answer-head/make_tables.py
The script writes into the folder tables/ next to it. It calls
harness/analysis/ab_stats.py and harness/analysis/answers_compare.py from this
repository. It uses only the Python standard library. It changes no file in raw/.
Run it again after each change of a file in raw/.
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
UNIT = os.path.join(RAW, "unit-tests")
OUT = os.path.join(HERE, "tables")
ANALYSIS = os.path.join(ROOT, "harness", "analysis")
SIZES = ("short", "medium", "long")
PROMPTS = (64, 256, 3000)
SCHEDULE_AB = "0003-answer-head.txt"
SCHEDULE_CORR = "0003-correctness.txt"
SCHEDULE_CORR2 = "0003-correctness2.txt"
ARMS = {"L": "build:lead-30489", "A3": "build:0003-answer-head"}


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
def schedule(name):
    """Return {run name: (line type, arm, {DMK_ overrides})} from a schedule file that the harness ran."""
    lines = {}
    for line in open(raw(name)):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        envs = {}
        while True:
            m = re.match(r'(DMK_[A-Z0-9_]+)=("([^"\']*)"|[^\s"\']+)\s+(.*)$', line)
            if not m:
                break
            envs[m.group(1)] = m.group(3) if m.group(2).startswith('"') else m.group(2)
            line = m.group(4)
        words = line.split()
        lines[words[1]] = (words[0], words[2], envs)
    return lines


def arm_tag(run):
    return run.rsplit("-", 1)[1] if re.search(r"-b\d+-", run) else run.split("-")[2]


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


def llb_values(run):
    """The prompt speed in tokens/s for each prompt length from the JSON in the .out file of llama bench."""
    p = raw(run + ".out")
    if not os.path.exists(p):
        return None
    return {f"pp{r['n_prompt']}_ts": r["avg_ts"] for r in json.load(open(p)) if r.get("n_gen", 0) == 0 and r.get("n_prompt")}


def test_name(run):
    if run.startswith("0003-bench-"):
        return "private benchmark"
    if run.startswith("0003-llb-"):
        return "llama bench"
    raise SystemExit(f"make_tables.py: unknown run name {run}")


def runs_table():
    lines = schedule(SCHEDULE_AB)
    runs = [read_run(r) for r in lines]
    runs.sort(key=lambda d: (d["start_utc"], d["run"]))
    rows = []
    for d in runs:
        run = d["run"]
        b, p = bench_json(run), llb_values(run)
        smi = smi_stats(run)
        rows.append([
            run, test_name(run), lines[run][0], block_of(run), arm_tag(run), d["arm"], d["start_utc"],
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


def tool_path(name):
    return os.path.relpath(os.path.join(ANALYSIS, name), HERE)


def ab_tables():
    os.makedirs(OUT, exist_ok=True)
    for prefix, name in (("0003-bench", "speed-bench.tsv"), ("0003-llb", "speed-llama-bench.tsv")):
        table = run_tool([tool_path("ab_stats.py"), "raw", "--prefix", prefix, "--a", "L", "--b", "A3"])
        with open(os.path.join(OUT, name), "w") as f:
            f.write(table)
    out = run_tool([tool_path("answers_compare.py"), "--a", "raw/0003-bench-b*-L.reference.json",
                    "--b", "raw/0003-bench-b*-A3.reference.json"])
    with open(os.path.join(OUT, "answers-compare-bench.json"), "w") as f:
        f.write(out)


def block_tables(runs):
    metrics = ["lat_ms", "short_ms", "medium_ms", "long_ms", "rps8"]
    rows = []
    for block in range(1, 5):
        a, b = f"0003-bench-b{block}-L", f"0003-bench-b{block}-A3"
        ja, jb = bench_json(a), bench_json(b)
        first = "L" if runs[a]["start_utc"] < runs[b]["start_utc"] else "A3"
        rows.append([block, first,
                     num(runs[a]["load_before"], 2), num(runs[a]["load_after"], 2),
                     num(runs[b]["load_before"], 2), num(runs[b]["load_after"], 2)]
                    + [num(100 * (jb[m] / ja[m] - 1), 3) for m in metrics])
    write_tsv("speed-bench-blocks.tsv", [
        "block", "first_arm", "load1_l_before", "load1_l_after", "load1_a3_before", "load1_a3_after",
        "lat_ms_diff_pct", "short_ms_diff_pct", "medium_ms_diff_pct", "long_ms_diff_pct", "rps8_diff_pct"], rows)

    rows = []
    for block in range(1, 5):
        a, b = f"0003-llb-b{block}-L", f"0003-llb-b{block}-A3"
        va, vb = llb_values(a), llb_values(b)
        first = "L" if runs[a]["start_utc"] < runs[b]["start_utc"] else "A3"
        rows.append([block, first,
                     num(runs[a]["load_before"], 2), num(runs[a]["load_after"], 2),
                     num(runs[b]["load_before"], 2), num(runs[b]["load_after"], 2)]
                    + [num(100 * (vb[f"pp{n}_ts"] / va[f"pp{n}_ts"] - 1), 3) for n in PROMPTS])
    write_tsv("speed-llama-bench-blocks.tsv", [
        "block", "first_arm", "load1_l_before", "load1_l_after", "load1_a3_before", "load1_a3_after",
        "pp64_ts_diff_pct", "pp256_ts_diff_pct", "pp3000_ts_diff_pct"], rows)


def quality_table(runs):
    rows = []
    for run in sorted((r for r in runs if bench_json(r)), key=lambda r: (block_of(r), arm_tag(r))):
        b = bench_json(run)
        rows.append([run, block_of(run), arm_tag(run), b["correct"], b["total"], b["agree_n"].split("/")[0],
                     b["mean_dp"], b["max_dp"], b["flips"], b["errors"], b["guard_pass"]])
    write_tsv("quality-bench.tsv", [
        "run", "block", "arm_tag", "correct", "decisions", "agree", "mean_dp", "max_dp",
        "top_answer_differs_from_reference", "errors", "guards_true"], rows)


# ---------------------------------------------------------------- the server logs
REALLOC = re.compile(r"graph reallocation: nodes = (\d+) \(changed = (\d)\), splits = (\d+), n_same = (\d+), n_diff = (\d+)")
ROWS_PATH = re.compile(r"the head reads the rows of (\d+) tokens for (\d+) output rows")
FULL_PATH = re.compile(r"the full head, then the values of (\d+) tokens for (\d+) output rows")
RESERVING = re.compile(r"sched_reserve: reserving \.\.\.")
SUMMARY = re.compile(r"ERROR SUMMARY: (\d+) errors?")


def log_facts(run):
    """Counts from the server log of one run."""
    text = open(raw(run + ".server.log"), errors="replace").read()
    summary = SUMMARY.findall(text)
    return {
        "reallocs": len(REALLOC.findall(text)),
        "reserves": len(RESERVING.findall(text)),
        "rows_lines": len(ROWS_PATH.findall(text)),
        "full_lines": len(FULL_PATH.findall(text)),
        "memcheck_errors": summary[-1] if summary else None,
        "log_level_4": "verbosity = 4" in text,
    }


# ---------------------------------------------------------------- the correctness runs
# For each schedule file of a correctness window: the commit of the arm of the change in this window.
WINDOWS = [
    ("0003-correctness.txt", 1, {"A3": "c1d3d481b"}),
    ("0003-correctness2.txt", 2, {"A3": "4bd0c75b5"}),
    ("0003-correctness3.txt", 3, {"A3": "4bd0c75b5"}),
    ("0003-correctness4.txt", 4, {"A3": "f0a2033e7"}),
    ("0003-rebased.txt", 5, {"A3R": "ca69612da"}),
    ("0003-correctness5.txt", 6, {"A3": "f0a2033e7", "A3R": "ca69612da"}),
    ("0003-final.txt", 7, {"F": "419b8f612", "A3R": "ca69612da"}),
]
COMMITS = {"L": "6b8b65e66", "S": "fbfd9cb8c"}
ALL_ARMS = {"build:lead-30489": "L", "build:0003-answer-head": "A3", "build:stack-fbfd9cb": "S",
            "build:0003-answer-head-rebased": "A3R", "build:0003-answer-head-final": "F"}


def number_diff(x, y):
    """The largest absolute difference of the numbers in two JSON values of the same shape, or None for a shape that is not the same."""
    if isinstance(x, bool) or isinstance(y, bool):
        return 0.0 if x == y else None
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(x - y)
    if isinstance(x, dict) and isinstance(y, dict):
        if x.keys() != y.keys():
            return None
        ds = [number_diff(x[k], y[k]) for k in x]
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return None
        ds = [number_diff(a, b) for a, b in zip(x, y)]
    else:
        return 0.0 if x == y else None
    return None if any(d is None for d in ds) else max(ds, default=0.0)


def answers(run):
    return json.load(open(raw(run + ".answers.json")))


def correctness_runs():
    """One row for each run of the correctness windows."""
    rows = []
    for sched, window, own in WINDOWS:
        if not os.path.exists(raw(sched)):
            continue
        for run, (kind, arm, envs) in schedule(sched).items():
            d = read_run(run)
            sargs = envs.get("DMK_SERVER_ARGS", "")
            np_ = re.search(r"-np (\d+)", sargs)
            tag = ALL_ARMS[arm]
            commit = own.get(tag, COMMITS.get(tag))
            ot = re.search(r"-ot (\S+)", sargs)
            f = log_facts(run)
            saved = os.path.exists(raw(run + ".reference.json")) or os.path.exists(raw(run + ".answers.json"))
            rows.append([
                window, run, kind, tag, commit, np_.group(1) if np_ else "3",
                "4" if f["log_level_4"] else "3",
                "memcheck" if "san-wrap" in envs.get("DMK_SERVER_WRAP", "") or "compute-sanitizer" in envs.get("DMK_SERVER_WRAP", "") else "none",
                ot.group(1) if ot else "none", d["start_utc"],
                "answers saved" if saved else "no answers: the client stopped with an error",
                f["reserves"], f["reallocs"] if f["log_level_4"] else None, f["rows_lines"], f["full_lines"], f["memcheck_errors"]])
    rows.sort(key=lambda r: r[9])
    write_tsv("correctness-runs.tsv", [
        "window", "run", "line_type", "arm_tag", "arm_commit", "slots", "log_level", "wrapper", "tensor_override", "start_utc",
        "result", "sched_reserve_lines", "graph_reallocation_lines", "head_rows_info_lines", "head_full_info_lines",
        "memcheck_error_summary"], rows)


def compare_tables():
    """The answers of the benchmark runs of the correctness windows, and the answers of the client runs."""
    for name, a, b in (
            ("answers-compare-window1.json", "raw/0003-corr-L-*.reference.json", "raw/0003-corr-A3-*.reference.json"),
            ("answers-compare-window4.json", "raw/0003-r-bench-L-*.reference.json", "raw/0003-r-bench-A3-*.reference.json"),
            ("answers-compare-rebased-bench.json", "raw/0003-rb-bench-S-*.reference.json", "raw/0003-rb-bench-A3R-*.reference.json"),
            ("answers-compare-rebased-ab.json", "raw/0003-rb-ab-b*-S.reference.json", "raw/0003-rb-ab-b*-A3R.reference.json"),
            ("answers-compare-final.json", "raw/0003-final-bench-F-*.reference.json", "raw/0003-rb-ab-b*-S.reference.json")):
        if not glob.glob(os.path.join(HERE, a)):
            continue
        with open(os.path.join(OUT, name), "w") as f:
            f.write(run_tool([tool_path("answers_compare.py"), "--a", a, "--b", b]))

    pairs = []
    for prefix, (a, b) in (("0003-fin-np6", ("A3", "L")), ("0003-r-np6", ("A3", "L")),
                           ("0003-r-cpuhead", ("A3", "L")), ("0003-rb-cpuhead", ("A3R", "S")),
                           ("0003-g-np6", ("A3", "L")), ("0003-g-np6", ("A3R", "S")), ("0003-g-ngl32", ("A3", "L"))):
        pairs += [(f"{prefix}-{a}-1", f"{prefix}-{b}-1"), (f"{prefix}-{a}-1", f"{prefix}-{b}-2"),
                  (f"{prefix}-{a}-2", f"{prefix}-{b}-1"), (f"{prefix}-{a}-2", f"{prefix}-{b}-2"),
                  (f"{prefix}-{b}-1", f"{prefix}-{b}-2"), (f"{prefix}-{a}-1", f"{prefix}-{a}-2")]
    pairs += [("0003-fin-types-A3", "0003-fin-types-L"), ("0003-fin-memcheck-A3", "0003-fin-types-A3")]
    pairs += [("0003-final-mixed-F", "0003-rb-cpuhead-S-1"), ("0003-final-mixed-F", "0003-rb-cpuhead-S-2"),
              ("0003-final-mixed-A3R", "0003-rb-cpuhead-S-1"), ("0003-final-mixed-A3R", "0003-rb-cpuhead-S-2"),
              ("0003-final-mixed-F", "0003-final-mixed-A3R")]
    rows = []
    for a, b in pairs:
        if not (os.path.exists(raw(a + ".answers.json")) and os.path.exists(raw(b + ".answers.json"))):
            continue
        xa, xb = answers(a), answers(b)
        ids = sorted(xa)
        same = sum(1 for i in ids if i in xb and xa[i] == xb[i])
        diffs = [number_diff(xa[i], xb[i]) for i in ids if i in xb]
        finite = [d for d in diffs if d is not None]
        rows.append([a, b, len(ids), same, len(ids) - same, num(max(finite), 9) if finite else None,
                     sum(1 for v in xa.values() if v["status"] != 200)])
    write_tsv("answers-client-runs.tsv", [
        "run_a", "run_b", "cases", "cases_identical", "cases_different", "max_abs_difference", "requests_with_error_in_a"], rows)

    rows = []
    for run in ("0003-memcheck-A3", "0003-fin-memcheck-A3", "0003-lv-memcheck-A3", "0003-r-memcheck-A3", "0003-rb-memcheck-A3R"):
        if os.path.exists(raw(run + ".answers.json")):
            for k, v in sorted(answers(run).items()):
                rows.append([run, k, v["id"], v["status"], log_facts(run)["memcheck_errors"]])
    write_tsv("memcheck-requests.tsv", ["run", "request_type", "case", "http_status", "memcheck_error_summary"], rows)


# ---------------------------------------------------------------- the unit tests
def unit_table():
    commands = {
        "mmr-before-repair": "test-mul-mat-rows, first build, before the repair of mmvq.cu",
        "mmr": "test-mul-mat-rows", "mmr-full": "test-mul-mat-rows --full", "mmr-all": "test-mul-mat-rows --all",
        "bo-id": "test-backend-ops test -b CUDA0 -o MUL_MAT_ID", "bo-mm": "test-backend-ops test -b CUDA0 -o MUL_MAT",
        "rows": "test-backend-ops test -b CUDA0 -p same_rows",
        "rows-1": "test-backend-ops test -b CUDA0 -p same_rows (run 1)", "rows-2": "test-backend-ops test -b CUDA0 -p same_rows (run 2)",
        "rows-3": "test-backend-ops test -b CUDA0 -p same_rows (run 3)",
        "rows-o-1": "test-backend-ops test -b CUDA0 -o MUL_MAT_ROWS (run 1)", "rows-o-2": "test-backend-ops test -b CUDA0 -o MUL_MAT_ROWS (run 2)",
        "rows-o-3": "test-backend-ops test -b CUDA0 -o MUL_MAT_ROWS (run 3)",
    }
    rows = []
    for build, sub, names in (
            ("first build, dirty tree of 6b8b65e66", "c1d3d481b", ["mmr-before-repair"]),
            ("c1d3d481b", "c1d3d481b", ["mmr", "mmr-full", "mmr-all", "bo-id", "bo-mm"]),
            ("f0a2033e7", "f0a2033e7", ["mmr", "mmr-full", "mmr-all", "bo-id", "bo-mm", "rows"]),
            ("ca69612da", "ca69612da", ["mmr", "mmr-full", "bo-id", "bo-mm", "rows"]),
            ("419b8f612 (the final commit)", "419b8f612", ["rows-1", "rows-2", "rows-3", "rows-o-1", "rows-o-2", "rows-o-3", "bo-id", "bo-mm", "mmr", "mmr-full"])):
        folder = os.path.join(UNIT, sub)
        summary = {}
        if os.path.exists(os.path.join(folder, "summary.txt")):
            summary = dict(l.split()[0:2] for l in open(os.path.join(folder, "summary.txt")) if len(l.split()) > 1 and l.split()[1].startswith("rc="))
        rows_summary = os.path.join(folder, "rows-summary.txt")
        if os.path.exists(rows_summary):
            # the file has one line for each build directory: <directory> rc=<code>
            for l in open(rows_summary):
                if len(l.split()) > 1 and l.split()[1].startswith("rc=") and l.split()[0].endswith("rebased") == (sub == "ca69612da"):
                    summary["rows"] = l.split()[1]
        for name in names:
            if not os.path.exists(os.path.join(folder, name + ".log")):
                continue
            text = open(os.path.join(folder, name + ".log"), errors="replace").read()
            m1 = re.search(r"(\d+) cases, (\d+) with different values, (\d+) failed", text)
            m2 = re.search(r"(\d+)/(\d+) tests passed", text)
            rc = summary[name].replace("rc=", "") if name in summary else None
            rows.append([build, name, commands[name], rc,
                         m1.group(1) if m1 else None, m1.group(2) if m1 else None, m1.group(3) if m1 else None,
                         f"{m2.group(1)}/{m2.group(2)}" if m2 else None,
                         "yes" if "GGML_ASSERT" in text else "no"])
    write_tsv("unit-tests.tsv", [
        "build", "log", "command", "exit_code", "cases", "cases_with_different_values", "cases_failed", "tests_passed",
        "assert_in_log"], rows)


CASE_RE = re.compile(r"(CONT\(type_a=(\w+),n_rows=(\d+),k=(\d+),n_cols=(\d+),n_ids=(\d+),same_rows=(\d)\)): (OK|FAIL)")
NMSE_RE = re.compile(r"ERR = ([0-9.]+) > [0-9.]+\s+\[CONT\] ERR = [0-9.]+ > [0-9.]+\s+(CONT\([^)]*\)): FAIL")


def rows_cases(path):
    """The cases of one run of test-backend-ops -p same_rows: (case, type, n_rows, k, n_cols, n_ids, same_rows, result, nmse)."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(path, errors="replace").read())
    nmse = {m.group(2): m.group(1) for m in NMSE_RE.finditer(text)}
    out = []
    for m in CASE_RE.finditer(text):
        out.append([m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)), m.group(7),
                    m.group(8), nmse.get(m.group(1))])
    return out


def rows_repeat_tables():
    """Every case of every run of the cases of test_mul_mat_rows, and the number of failed cases for each number of ids."""
    groups = (
        ("first run, f0a2033e7 and ca69612da", [("ca69612da", "rows"), ("f0a2033e7", "rows")]),
        ("repeat, ca69612da, n_ids=1", [("ca69612da/repeat", "ids1-%d" % i) for i in range(1, 11)]),
        ("repeat, ca69612da, n_ids=10", [("ca69612da/repeat", "ids10-%d" % i) for i in range(1, 11)]),
        ("419b8f612, the cases have 10 or 33 ids", [("419b8f612", "rows-%d" % i) for i in range(1, 4)]),
    )
    all_rows, counts = [], {}
    for group, logs in groups:
        for folder, name in logs:
            path = os.path.join(UNIT, folder, name + ".log")
            if not os.path.exists(path):
                continue
            for c in rows_cases(path):
                all_rows.append([group, folder, name] + c[0:1] + [c[1], c[2], c[3], c[4], c[5], c[4] * c[5], c[6], c[7], c[8]])
                key = (group, c[5])
                n, f = counts.get(key, (0, 0))
                counts[key] = (n + 1, f + (c[7] == "FAIL"))
    write_tsv("rows-cases.tsv", ["group", "folder", "log", "case", "type_a", "n_rows", "k", "n_cols", "n_ids", "n_values",
                                 "same_rows", "result", "nmse"], all_rows)
    write_tsv("rows-by-ids.tsv", ["group", "n_ids", "case_results", "failed", "failed_percent"],
              [[g, i, n, f, num(100.0 * f / n, 1)] for (g, i), (n, f) in sorted(counts.items())])


def rebased_tables():
    """The short A/B of the branch on the stack (arms S and A3R), if its runs are in raw/."""
    if not glob.glob(os.path.join(RAW, "0003-rb-ab-b*-S.json")):
        return
    table = run_tool([tool_path("ab_stats.py"), "raw", "--prefix", "0003-rb-ab", "--a", "S", "--b", "A3R"])
    with open(os.path.join(OUT, "speed-rebased.tsv"), "w") as f:
        f.write(table)
    metrics = ["lat_ms", "short_ms", "medium_ms", "long_ms", "rps8"]
    rows = []
    for block in range(1, 5):
        a, b = f"0003-rb-ab-b{block}-S", f"0003-rb-ab-b{block}-A3R"
        ja, jb = bench_json(a), bench_json(b)
        da, db = read_run(a), read_run(b)
        rows.append([block, "S" if da["start_utc"] < db["start_utc"] else "A3R",
                     num(da["load_before"], 2), num(da["load_after"], 2), num(db["load_before"], 2), num(db["load_after"], 2)]
                    + [num(100 * (jb[m] / ja[m] - 1), 3) for m in metrics])
    write_tsv("speed-rebased-blocks.tsv", [
        "block", "first_arm", "load1_s_before", "load1_s_after", "load1_a3r_before", "load1_a3r_after",
        "lat_ms_diff_pct", "short_ms_diff_pct", "medium_ms_diff_pct", "long_ms_diff_pct", "rps8_diff_pct"], rows)

    rows = []
    for run in sorted(r for r in schedule("0003-rebased.txt") if r.startswith("0003-rb-ab-")):
        b, d = bench_json(run), read_run(run)
        smi = smi_stats(run)
        rows.append([run, block_of(run), arm_tag(run), d["start_utc"], num(d["load_before"], 2), num(d["load_after"], 2),
                     num(b["lat_ms"], 3), num(b["short_ms"], 3), num(b["medium_ms"], 3), num(b["long_ms"], 3), num(b["rps8"], 3),
                     num(smi["clock"], 1), num(smi["power"], 2)])
    write_tsv("runs-rebased.tsv", [
        "run", "block", "arm_tag", "start_utc", "load1_before", "load1_after", "lat_ms", "short_ms", "medium_ms", "long_ms",
        "rps8", "sm_clock_mhz_median", "power_draw_w_median"], rows)


MIXED_RUNS = (("0003-final-mixed-F", "F", "419b8f612"), ("0003-final-mixed-A3R", "A3R", "ca69612da"))


def final_tables():
    """The runs of the final commit: the two passes of the private benchmark and the two mixed-load runs, if they are in raw/."""
    if not os.path.exists(raw("0003-final.txt")):
        return
    rows = []
    for run in sorted(r for r in schedule("0003-final.txt") if r.startswith("0003-final-bench-")):
        b, d = bench_json(run), read_run(run)
        smi = smi_stats(run)
        rows.append([run, "F", d["start_utc"], num(d["load_before"], 2), num(d["load_after"], 2),
                     num(b["lat_ms"], 3), num(b["short_ms"], 3), num(b["medium_ms"], 3), num(b["long_ms"], 3), num(b["rps8"], 3),
                     b["correct"], b["total"], b["errors"], b["guard_pass"], num(smi["clock"], 1), num(smi["power"], 2)])
    write_tsv("runs-final.tsv", [
        "run", "arm_tag", "start_utc", "load1_before", "load1_after", "lat_ms", "short_ms", "medium_ms", "long_ms", "rps8",
        "decisions_correct", "decisions_total", "errors", "guards_true", "sm_clock_mhz_median", "power_draw_w_median"], rows)

    reference = answers("0003-rb-cpuhead-S-1")
    rows = []
    for run, tag, commit in MIXED_RUNS:
        others = json.load(open(raw(run + ".others.json")))
        decisions = answers(run)
        spans = [(o["start"], o["end"]) for o in others["requests"]]
        overlapped = sum(1 for t in others["decision_requests"].values() if any(s < t["end"] and e > t["start"] for s, e in spans))
        starts = len(re.findall(r"server_run_i: model loaded", open(raw(run + ".server.log"), errors="replace").read()))
        same = sum(1 for k, v in decisions.items() if reference.get(k) == v)
        rows.append([run, tag, commit, "decision", len(decisions), sum(1 for v in decisions.values() if v["status"] == 200),
                     "200:%d" % len(decisions), "not used", overlapped, starts, "%d of %d" % (same, len(decisions)), None])
        for kind in ("completion", "chat"):
            rs = [o for o in others["requests"] if o["kind"] == kind]
            counts = {}
            for o in rs:
                counts[o["status"]] = counts.get(o["status"], 0) + 1
            error = next((o["error"] for o in rs if o["error"]), None)
            message = json.loads(error)["error"]["message"] if error else None
            rows.append([run, tag, commit, kind, len(rs), counts.get(200, 0),
                         " ".join("%s:%d" % kv for kv in sorted(counts.items())), sum(1 for o in rs if o["non_empty"]),
                         None, None, None, message])
    write_tsv("mixed-load.tsv", [
        "run", "arm_tag", "arm_commit", "request_type", "requests", "http_200", "status_counts", "requests_with_text",
        "decision_requests_with_another_request_in_flight", "server_starts_in_log", "decision_cases_same_as_s_one_at_a_time",
        "error_message"], rows)


if __name__ == "__main__":
    runs = runs_table()
    ab_tables()
    block_tables(runs)
    quality_table(runs)
    correctness_runs()
    compare_tables()
    unit_table()
    rows_repeat_tables()
    rebased_tables()
    final_tables()
    print("tables written to", os.path.relpath(OUT, ROOT))
