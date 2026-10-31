#!/usr/bin/env python3
"""Make the derived tables of the data set 2026-10-31-0004-gpu-mask from the files in raw/.

Usage: python3 data/2026-10-31-0004-gpu-mask/make_tables.py
The script writes into the folder tables/ next to it. It calls
harness/analysis/ab_stats.py and harness/analysis/answers_compare.py from this
repository. It uses only the Python standard library. It changes no file in raw/.
Run it again after each change of a file in raw/.
"""
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(HERE, "raw")
CHECKS = ("0004-check-unified", "0004-check-streams")
OUT = os.path.join(HERE, "tables")
ANALYSIS = os.path.join(ROOT, "harness", "analysis")
SIZES = ("short", "medium", "long")
PROMPTS = (64, 256, 3000)
SCHEDULES = ("0004-ab-bench.txt", "0004-ab-llb.txt")


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


# ---------------------------------------------------------------- the schedule and the run files
def schedule():
    """Return {run name: line type} from the schedule file that the harness ran."""
    lines = {}
    for name in SCHEDULES:
        for line in open(raw(name)):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words = line.split()
            while re.match(r"DMK_[A-Z0-9_]+=", words[0]):
                words = words[1:]
            lines[words[1]] = words[0]
    return lines


TESTS = [
    (r"gpumask-bench-", "private benchmark"),
    (r"gpumask-llb-", "llama bench"),
]


def test_name(run):
    for rx, label in TESTS:
        if re.match(rx, run):
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


def ab_tables():
    os.makedirs(OUT, exist_ok=True)
    for prefix, name in (("gpumask-bench", "speed-bench.tsv"), ("gpumask-llb", "speed-llama-bench.tsv")):
        table = run_tool([os.path.relpath(os.path.join(ANALYSIS, "ab_stats.py"), HERE), "raw", "--prefix", prefix,
                          "--a", "D", "--b", "L"])
        with open(os.path.join(OUT, name), "w") as f:
            f.write(table)
    out = run_tool([os.path.relpath(os.path.join(ANALYSIS, "answers_compare.py"), HERE),
                    "--a", "raw/gpumask-bench-b*-D.reference.json", "--b", "raw/gpumask-bench-b*-L.reference.json"])
    with open(os.path.join(OUT, "answers-compare-bench.json"), "w") as f:
        f.write(out)
    return json.loads(out)


def block_tables(runs):
    metrics = ["lat_ms", "short_ms", "medium_ms", "long_ms", "rps8"]
    rows = []
    for block in range(1, 5):
        d, l = f"gpumask-bench-b{block}-D", f"gpumask-bench-b{block}-L"
        bd, bl = bench_json(d), bench_json(l)
        first = "D" if runs[d]["start_utc"] < runs[l]["start_utc"] else "L"
        rows.append([block, first,
                     num(runs[d]["load_before"], 2), num(runs[d]["load_after"], 2),
                     num(runs[l]["load_before"], 2), num(runs[l]["load_after"], 2)]
                    + [num(100 * (bl[m] / bd[m] - 1), 3) for m in metrics])
    write_tsv("speed-bench-blocks.tsv", [
        "block", "first_arm", "load1_d_before", "load1_d_after", "load1_l_before", "load1_l_after",
        "lat_ms_diff_pct", "short_ms_diff_pct", "medium_ms_diff_pct", "long_ms_diff_pct", "rps8_diff_pct"], rows)

    rows = []
    for block in range(1, 5):
        d, l = f"gpumask-llb-b{block}-D", f"gpumask-llb-b{block}-L"
        vd, vl = llb_values(d), llb_values(l)
        first = "D" if runs[d]["start_utc"] < runs[l]["start_utc"] else "L"
        rows.append([block, first,
                     num(runs[d]["load_before"], 2), num(runs[d]["load_after"], 2),
                     num(runs[l]["load_before"], 2), num(runs[l]["load_after"], 2)]
                    + [num(100 * (vl[f"pp{n}_ts"] / vd[f"pp{n}_ts"] - 1), 3) for n in PROMPTS])
    write_tsv("speed-llama-bench-blocks.tsv", [
        "block", "first_arm", "load1_d_before", "load1_d_after", "load1_l_before", "load1_l_after",
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


# ---------------------------------------------------------------- correctness runs
CHECK_RX = re.compile(r"KQ mask check: n_kv = (\d+), n_tokens = (\d+), type = (\w+), n_different = (\d+)")
SAN_RUNS = ("0004-san-memcheck-ppl256-L", "0004-san-memcheck-ppl2048-L")


def correctness_table():
    """One row for each correctness run: the compare of the mask from the GPU with the mask from the host, and compute-sanitizer."""
    rows = []
    for run in CHECKS:
        log = open(raw(run + ".server.log")).read()
        found = CHECK_RX.findall(log)
        rows.append([run, "mask compare", len(found), sum(int(f[3]) for f in found), max(int(f[0]) for f in found),
                     max(int(f[1]) for f in found), "KQ mask on the device enabled" in log, "ABORT" in log])
    mem = open(raw("0004-san-memcheck.server.log")).read()
    answers = json.load(open(raw("0004-san-memcheck.answers.json")))
    errors = [l for l in mem.splitlines() if l.startswith("=========") and "COMPUTE-SANITIZER" not in l]
    rows.append(["0004-san-memcheck", "memcheck, server, 4 requests", len(answers), len(errors), None, None,
                 "KQ mask on the device enabled" in mem, None])
    for run in SAN_RUNS:
        out = open(raw(run + ".out")).read()
        summary = [l for l in out.splitlines() if "ERROR SUMMARY" in l]
        rows.append([run, "memcheck, llama perplexity", None, summary[0].split("ERROR SUMMARY:")[1].split()[0] if summary else None,
                     None, None, None, None])
    write_tsv("correctness.tsv", [
        "run", "test", "count", "different_or_errors", "max_n_kv", "max_n_tokens", "device_mask_enabled_in_log", "abort_in_log"], rows)


if __name__ == "__main__":
    runs = runs_table()
    answers = ab_tables()
    block_tables(runs)
    quality_table(runs)
    correctness_table()
    print("answers: self repeat D", answers["self_repeat_a"], "- self repeat L", answers["self_repeat_b"],
          "- cross identical", answers["cross_identical_all"], "- cross max |dp|", answers["cross_max_abs_dp"],
          "- top flips", answers["cross_top_flips"])
    print("tables written to", os.path.relpath(OUT, ROOT))
