#!/usr/bin/env python3
"""Make the derived tables of the data set 2026-10-30-window1 from the files in raw/.

Usage: python3 data/2026-10-30-window1/make_tables.py
The script writes into the folder tables/ next to it. It calls
harness/analysis/ab_stats.py and harness/analysis/answers_compare.py from this
repository. It uses only the Python standard library. It changes no file in raw/.
Run it again after each change of a file in raw/.
"""
import glob
import hashlib
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
ANALYSIS = os.path.join(ROOT, "harness", "analysis")
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


# ---------------------------------------------------------------- the schedule and the run files
def schedule():
    """Return {run name: line type} from the schedule file that the harness ran."""
    lines = {}
    for line in open(raw("window1.txt")):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words = line.split()
        while re.match(r"DMK_[A-Z0-9_]+=", words[0]):
            words = words[1:]
        lines[words[1]] = words[0]
    return lines


TESTS = [
    (r"p1-bench-", "private benchmark"),
    (r"p1-llb-", "llama bench"),
    (r"p[12]-kld-", "KL divergence"),
    (r"p2-dectest-", "52 decisions"),
    (r"p2-pub-", "public benchmark"),
    (r"p2-san-", "sanitizer"),
]


def test_name(run):
    for rx, label in TESTS:
        if re.match(rx, run):
            return label
    raise SystemExit(f"make_tables.py: unknown run name {run}")


def load1(path):
    """The one-minute load average from a line of /proc/loadavg."""
    return float(open(path).read().split()[0])


def read_run(run):
    """Facts of one run from its .meta file (and .load0, .load1 for bench and client lines)."""
    meta = open(raw(run + ".meta")).read().splitlines()
    d = {"run": run, "arm": meta[0].split()[0].split("=", 1)[1]}
    d["start_utc"] = [m for m in meta if m.startswith("date_utc=")][0].split("=", 1)[1]
    if os.path.exists(raw(run + ".load0")):  # a bench or client line
        d["load_before"] = load1(raw(run + ".load0"))
        d["load_after"] = load1(raw(run + ".load1"))
        info = meta[2].split(", ")
        d["throttle"] = info[2] if len(info) > 2 else None
    else:  # a tool line: the load averages are lines 3 and 6 of the meta file
        d["load_before"] = float(meta[2].split()[0])
        d["load_after"] = float(meta[5].split()[0])
        info = meta[3].split(", ")
        d["throttle"] = None
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
    s = j["seq"]
    return {
        "lat_ms": s["lat_ms"],
        **{f"{k}_ms": s["by_size"][k]["mean"] for k in SIZES},
        "rps8": j["tput"]["rps"],
        "correct": j["quality"]["correct"],
        "total": j["quality"]["total"],
        "passes": j["meta"]["passes"],
        "model": os.path.basename(j["quality"]["model"]),
        "errors": len(j["errors"]),
    }


def runs_table():
    lines = schedule()
    runs = [read_run(r) for r in lines]
    runs.sort(key=lambda d: (d["start_utc"], d["run"]))
    rows = []
    for d in runs:
        b = bench_json(d["run"])
        smi = smi_stats(d["run"])
        rows.append([
            d["run"], test_name(d["run"]), lines[d["run"]], d["arm"], d["start_utc"],
            num(d["load_before"], 2), num(d["load_after"], 2),
            d["driver"], d["power_limit_w"], d["throttle"],
            num(b["lat_ms"], 3) if b else None, num(b["short_ms"], 3) if b else None,
            num(b["medium_ms"], 3) if b else None, num(b["long_ms"], 3) if b else None,
            num(b["rps8"], 3) if b else None,
            num(smi["clock"], 1), num(smi["power"], 2), num(smi["mem"], 0),
        ])
    write_tsv("runs.tsv", [
        "run", "test", "line_type", "arm", "start_utc", "load1_before", "load1_after",
        "driver", "power_limit_w", "throttle_reasons_at_start",
        "lat_ms", "short_ms", "medium_ms", "long_ms", "rps8",
        "sm_clock_mhz_median", "power_draw_w_median", "memory_used_mib_max"], rows)
    return {d["run"]: d for d in runs}


# ---------------------------------------------------------------- ab_stats and answers_compare
def run_tool(args):
    return subprocess.run([sys.executable] + args, cwd=HERE, check=True, capture_output=True, text=True).stdout


def ab_tables():
    os.makedirs(OUT, exist_ok=True)
    for prefix, name in (("p1-bench", "speed-bench.tsv"), ("p1-llb", "speed-llama-bench.tsv")):
        table = run_tool([os.path.relpath(os.path.join(ANALYSIS, "ab_stats.py"), HERE), "raw", "--prefix", prefix,
                          "--a", "S", "--b", "D", "--margin", "2"])
        with open(os.path.join(OUT, name), "w") as f:
            f.write(table)
    out = run_tool([os.path.relpath(os.path.join(ANALYSIS, "answers_compare.py"), HERE),
                    "--a", "raw/p1-bench-b*-S.reference.json", "--b", "raw/p1-bench-b*-D.reference.json"])
    with open(os.path.join(OUT, "answers-compare-bench.json"), "w") as f:
        f.write(out)


def block_table(runs):
    metrics = ["lat_ms", "short_ms", "medium_ms", "long_ms", "rps8"]
    rows = []
    for block in range(1, 7):
        s, d = f"p1-bench-b{block}-S", f"p1-bench-b{block}-D"
        bs, bd = bench_json(s), bench_json(d)
        first = "S" if runs[s]["start_utc"] < runs[d]["start_utc"] else "D"
        rows.append([block, first,
                     num(runs[s]["load_before"], 2), num(runs[s]["load_after"], 2),
                     num(runs[d]["load_before"], 2), num(runs[d]["load_after"], 2)]
                    + [num(100 * (bd[m] / bs[m] - 1), 3) for m in metrics])
    write_tsv("speed-bench-blocks.tsv", [
        "block", "first_arm", "load1_s_before", "load1_s_after", "load1_d_before", "load1_d_after",
        "lat_ms_diff_pct", "short_ms_diff_pct", "medium_ms_diff_pct", "long_ms_diff_pct", "rps8_diff_pct"], rows)


# ---------------------------------------------------------------- KL divergence
KLD_RUNS = [
    # run, text, compared arm, base
    ("p1-kld-wiki-S-self", "WikiText-2 test", "S", "S base"),
    ("p1-kld-wiki-D-vsS", "WikiText-2 test", "D", "S base"),
    ("p1-kld-domain-S-self", "benchmark states", "S", "S base"),
    ("p1-kld-domain-D-vsS", "benchmark states", "D", "S base"),
    ("p2-kld-wiki-D-vsQ8", "WikiText-2 test", "D", "Q8_0"),
    ("p2-kld-domain-D-vsQ8", "benchmark states", "D", "Q8_0"),
]


def value_and_error(text, label):
    """A line 'label: value ± error' of the llama perplexity summary. The error can be -nan."""
    m = re.search(r"^" + label + r"\s*:\s*(-?[\d.]+)(?: ± +(-?[\w.]+))?", text, re.M)
    if not m:
        raise SystemExit(f"make_tables.py: no line {label}")
    err = m.group(2)
    return m.group(1), (None if err is None or "nan" in err else err)


def kld_table():
    rows = []
    for run, text_set, arm, base in KLD_RUNS:
        t = open(raw(run + ".out")).read()
        mean, mean_err = value_and_error(t, r"Mean\s+KLD")
        mx, _ = value_and_error(t, r"Maximum KLD")
        p999, _ = value_and_error(t, r"99\.9%\s+KLD")
        top, top_err = value_and_error(t, r"Same top p")
        ppl_q, _ = value_and_error(t, r"Mean PPL\(Q\)")
        ppl_b, _ = value_and_error(t, r"Mean PPL\(base\)")
        ratio, ratio_err = value_and_error(t, r"Mean PPL\(Q\)/PPL\(base\)")
        rows.append([run, text_set, arm, base, mean, mean_err, mx, p999, top, top_err, ppl_q, ppl_b, ratio, ratio_err])
    write_tsv("kld.tsv", [
        "run", "text", "compared_arm", "base", "mean_kld", "mean_kld_error", "max_kld", "kld_p99_9",
        "same_top_token_pct", "same_top_token_error_pct", "ppl_compared", "ppl_base", "ppl_ratio", "ppl_ratio_error"], rows)


# ---------------------------------------------------------------- 52 decisions, sanitizers, public benchmark
def decisions_table():
    rows = []
    for arm in ("S", "D"):
        t = open(raw(f"p2-dectest-{arm}.client.out")).read()
        total = re.search(r"^correct (\d+)/(\d+)", t, re.M)
        misses = re.search(r"^misses: (.*)$", t, re.M).group(1).split(", ")
        rows.append([arm, "all", total.group(1), total.group(2), len(misses), " ".join(misses)])
        for g in re.finditer(r"^\s+([a-z]+)\s+(\d+)/(\d+)$", t, re.M):
            m = [x for x in misses if re.match(g.group(1) + r"\d", x)]
            rows.append([arm, g.group(1), g.group(2), g.group(3), len(m), " ".join(m)])
    write_tsv("decisions52.tsv", ["arm", "group", "correct", "decisions", "misses", "missed_decisions"], rows)
    same = (hashlib.sha256(open(raw("p2-dectest-S.answers.json"), "rb").read()).hexdigest()
            == hashlib.sha256(open(raw("p2-dectest-D.answers.json"), "rb").read()).hexdigest())
    print("52 decisions: the two answer files are byte-identical:", same)


def sanitizer_table():
    rows = []
    for tool in ("memcheck", "initcheck", "synccheck", "racecheck"):
        run = f"p2-san-{tool}-D"
        out = open(raw(run + ".out")).read()
        err = open(raw(run + ".err")).read()
        m = re.search(r"ERROR SUMMARY: (\d+) errors", out)
        if m:
            errors, hazards, warnings = m.group(1), None, None
        else:
            m = re.search(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)", out)
            hazards, errors, warnings = m.group(1), m.group(2), m.group(3)
        seconds = re.search(r"([\d.]+) seconds per pass", err).group(1)
        ppl = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", err + out) or re.search(r"\[1\]([\d.]+),", out)
        rows.append([tool, errors, hazards, warnings, seconds, ppl.group(1)])
    write_tsv("sanitizers.tsv", ["tool", "errors", "hazards_displayed", "warnings", "seconds_per_pass", "ppl"], rows)


def public_table():
    rows = []
    for run in ("p2-pub-ref-q8", "p2-pub-base-1", "p2-pub-base-2", "p2-pub-base-3"):
        b = bench_json(run)
        rows.append([run, "reference" if "ref" in run else "baseline", b["model"], b["passes"],
                     num(b["lat_ms"], 3), num(b["short_ms"], 3), num(b["medium_ms"], 3), num(b["long_ms"], 3),
                     num(b["rps8"], 3), b["correct"], b["total"], b["errors"]])
    write_tsv("public-bench.tsv", [
        "run", "role", "model", "timed_passes", "lat_ms", "short_ms", "medium_ms", "long_ms", "rps8",
        "correct", "decisions", "errors"], rows)


if __name__ == "__main__":
    runs = runs_table()
    ab_tables()
    block_table(runs)
    kld_table()
    decisions_table()
    sanitizer_table()
    public_table()
    print("tables written to", os.path.relpath(OUT, ROOT))
