#!/usr/bin/env python3
"""Make the derived tables of the data set 2026-10-31-lead-30489 from the files in raw/.

Usage: python3 data/2026-10-31-lead-30489/make_tables.py
The script writes into the folder tables/ next to it. It calls
harness/analysis/ab_stats.py and harness/analysis/answers_compare.py from this
repository. It uses only the Python standard library. It changes no file in raw/.
Run it again after each change of a file in raw/.
"""
import collections
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(HERE, "raw")
SASS = os.path.join(RAW, "sass-compare")
OUT = os.path.join(HERE, "tables")
ANALYSIS = os.path.join(ROOT, "harness", "analysis")
SIZES = ("short", "medium", "long")
PROMPTS = (64, 256, 3000)
SCHEDULE = "lead-30489.txt"


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
    for line in open(raw(SCHEDULE)):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words = line.split()
        while re.match(r"DMK_[A-Z0-9_]+=", words[0]):
            words = words[1:]
        lines[words[1]] = words[0]
    return lines


TESTS = [
    (r"lead30489-bench-", "private benchmark"),
    (r"lead30489-llb-", "llama bench"),
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
    for prefix, name in (("lead30489-bench", "speed-bench.tsv"), ("lead30489-llb", "speed-llama-bench.tsv")):
        table = run_tool([os.path.relpath(os.path.join(ANALYSIS, "ab_stats.py"), HERE), "raw", "--prefix", prefix,
                          "--a", "D", "--b", "L", "--margin", "2"])
        with open(os.path.join(OUT, name), "w") as f:
            f.write(table)
    out = run_tool([os.path.relpath(os.path.join(ANALYSIS, "answers_compare.py"), HERE),
                    "--a", "raw/lead30489-bench-b*-D.reference.json", "--b", "raw/lead30489-bench-b*-L.reference.json"])
    with open(os.path.join(OUT, "answers-compare-bench.json"), "w") as f:
        f.write(out)
    return json.loads(out)


def block_tables(runs):
    metrics = ["lat_ms", "short_ms", "medium_ms", "long_ms", "rps8"]
    rows = []
    for block in range(1, 5):
        d, l = f"lead30489-bench-b{block}-D", f"lead30489-bench-b{block}-L"
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
        d, l = f"lead30489-llb-b{block}-D", f"lead30489-llb-b{block}-L"
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


# ---------------------------------------------------------------- SASS
GLU_OPS = {"1": "GEGLU", "2": "SWIGLU", "3": "SWIGLU_OAI", "6": "SWIGLU_CLAMP", "7": "none (COUNT)"}
LAYOUTS = {"0": "D4", "1": "DS4", "2": "D2S6"}


def family(mangled):
    """The name of the function in an Itanium mangled name, without the template arguments."""
    s = mangled[2:]  # after _Z
    nested = s.startswith("N")
    s = s[1:] if nested else s
    s = s[1:] if s.startswith("L") else s
    parts = []
    while s and s[0].isdigit():
        m = re.match(r"\d+", s)
        k = int(m.group(0))
        s = s[m.end():]
        parts.append(s[:k])
        s = s[k:]
        if not nested:
            break
    return "::".join(parts) if parts else mangled


def sass_tables():
    j = json.load(open(os.path.join(SASS, "sass-compare.json")))
    summary = j["summary"]
    write_tsv("sass-summary.tsv", ["item", "value"], [[k, v] for k, v in summary.items()])

    rows = []
    for change, names in (("removed", j["onlyA"]), ("added", j["onlyB"])):
        for n in names:
            lay = re.search(r"mmq_q8_1_ds_layout(\d)", n)
            op = re.search(r"ggml_glu_op(\d)", n)
            rows.append([n, change, family(n), LAYOUTS[lay.group(1)] if lay else None,
                         GLU_OPS[op.group(1)] if op else "none (COUNT)" if change == "added" else None, None, None])
    for d in j["different"]:
        rows.append([d["name"], "different", family(d["name"]), None, None, d["instrA"][0], d["instrB"][0]])
    write_tsv("sass-names-changed.tsv", [
        "name", "change", "function", "layout", "glu_op", "instructions_a", "instructions_b"], rows)

    fam = collections.OrderedDict()
    with open(os.path.join(SASS, "key-kernels.tsv")) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(header, line.rstrip("\n").split("\t")))
            fam.setdefault(family(r["name"]), []).append(r)

    def top(rs, key):
        vals = [int(r[key]) for r in rs if r[key].isdigit()]
        return max(vals) if vals else None

    rows = []
    for name in sorted(fam):
        rs = fam[name]
        rows.append([name, len(rs), sum(int(r["sass_identical"]) for r in rs), sum(1 - int(r["sass_identical"]) for r in rs),
                     sum(int(r["encoding_identical"]) for r in rs),
                     sum(int(r["instrA"]) for r in rs), sum(int(r["instrB"]) for r in rs),
                     top(rs, "regA"), top(rs, "regB"), top(rs, "sharedA"), top(rs, "sharedB")])
    write_tsv("sass-functions.tsv", [
        "function", "names", "sass_identical", "sass_different", "encoding_identical", "instructions_a", "instructions_b",
        "max_registers_a", "max_registers_b", "max_static_shared_a_bytes", "max_static_shared_b_bytes"], rows)


if __name__ == "__main__":
    runs = runs_table()
    answers = ab_tables()
    block_tables(runs)
    quality_table(runs)
    sass_tables()
    print("answers: self repeat D", answers["self_repeat_a"], "- self repeat L", answers["self_repeat_b"],
          "- cross identical", answers["cross_identical_all"], "- cross max |dp|", answers["cross_max_abs_dp"],
          "- top flips", answers["cross_top_flips"])
    print("tables written to", os.path.relpath(OUT, ROOT))
