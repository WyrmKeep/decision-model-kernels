#!/usr/bin/env python3
"""Make the derived tables of the data set 2026-10-31-0007-input-copies from the files in raw/.

Usage: python3 data/2026-10-31-0007-input-copies/make_tables.py
The script writes into the folder tables/ next to it. It calls
harness/analysis/ab_stats.py and harness/analysis/answers_compare.py from this
repository. It uses only the Python standard library. It changes no file in raw/.
Run it again after each change of a file in raw/.

The table mechanism.tsv comes from the two Nsight Systems reports c0007-nsys-L.nsys-rep
and c0007-nsys-I.nsys-rep. The reports are not in this repository. To make the table
again, put them in a folder, set DMK_PROFILE_DIR to that folder and set DMK_NSYS to the
command line tool of Nsight Systems. Without these variables the script leaves the
committed file mechanism.tsv as it is.
"""
import collections
import json
import os
import re
import sqlite3
import statistics
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "tables")
ANALYSIS = os.path.join(ROOT, "harness", "analysis")
SIZES = ("short", "medium", "long")
PROMPTS = (64, 256, 3000)
SCHEDULES = ("c0007.txt",)
BENCH = "c0007-bench"
LLB = "c0007-llb"
NSYS_RUNS = ("c0007-nsys-L", "c0007-nsys-I")
ARM_A, ARM_B = "L", "I"  # the ratio is arm B divided by arm A


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
    """Return {run name: line type} from the schedule files that the harness ran."""
    lines = {}
    for name in SCHEDULES:
        for line in open(raw(name)):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words = shlex.split(line)
            while re.match(r"DMK_[A-Z0-9_]+=", words[0]):
                words = words[1:]
            lines[words[1]] = words[0]
    return lines


TESTS = [
    (BENCH + "-", "private benchmark"),
    (LLB + "-", "llama bench"),
    ("c0007-nsys-", "nsys"),
]


def test_name(run):
    for rx, label in TESTS:
        if run.startswith(rx):
            return label
    raise SystemExit(f"make_tables.py: unknown run name {run}")


def arm_tag(run):
    return run.rsplit("-", 1)[1]


def block_of(run):
    m = re.search(r"-b(\d+)-", run)
    return int(m.group(1)) if m else None


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


def ab_stats(prefix, results="raw"):
    return run_tool([os.path.relpath(os.path.join(ANALYSIS, "ab_stats.py"), HERE), results, "--prefix", prefix,
                     "--a", ARM_A, "--b", ARM_B])


def ab_tables():
    os.makedirs(OUT, exist_ok=True)
    for prefix, name in ((BENCH, "speed-bench.tsv"), (LLB, "speed-llama-bench.tsv")):
        with open(os.path.join(OUT, name), "w") as f:
            f.write(ab_stats(prefix))
    out = run_tool([os.path.relpath(os.path.join(ANALYSIS, "answers_compare.py"), HERE),
                    "--a", f"raw/{BENCH}-b*-{ARM_A}.reference.json", "--b", f"raw/{BENCH}-b*-{ARM_B}.reference.json"])
    with open(os.path.join(OUT, "answers-compare-bench.json"), "w") as f:
        f.write(out)
    return {"private benchmark": json.loads(out)}


def block_tables(runs):
    metrics = ["lat_ms", "short_ms", "medium_ms", "long_ms", "rps8"]
    rows = []
    for block in range(1, 5):
        a, b = f"{BENCH}-b{block}-{ARM_A}", f"{BENCH}-b{block}-{ARM_B}"
        ba, bb = bench_json(a), bench_json(b)
        first = ARM_A if runs[a]["start_utc"] < runs[b]["start_utc"] else ARM_B
        rows.append([block, first,
                     num(runs[a]["load_before"], 2), num(runs[a]["load_after"], 2),
                     num(runs[b]["load_before"], 2), num(runs[b]["load_after"], 2)]
                    + [num(100 * (bb[m] / ba[m] - 1), 3) for m in metrics])
    write_tsv("speed-bench-blocks.tsv", [
        "block", "first_arm", "load1_l_before", "load1_l_after", "load1_i_before", "load1_i_after",
        "lat_ms_diff_pct", "short_ms_diff_pct", "medium_ms_diff_pct", "long_ms_diff_pct", "rps8_diff_pct"], rows)

    rows = []
    for block in range(1, 5):
        a, b = f"{LLB}-b{block}-{ARM_A}", f"{LLB}-b{block}-{ARM_B}"
        va, vb = llb_values(a), llb_values(b)
        first = ARM_A if runs[a]["start_utc"] < runs[b]["start_utc"] else ARM_B
        rows.append([block, first,
                     num(runs[a]["load_before"], 2), num(runs[a]["load_after"], 2),
                     num(runs[b]["load_before"], 2), num(runs[b]["load_after"], 2)]
                    + [num(100 * (vb[f"pp{n}_ts"] / va[f"pp{n}_ts"] - 1), 3) for n in PROMPTS])
    write_tsv("speed-llama-bench-blocks.tsv", [
        "block", "first_arm", "load1_l_before", "load1_l_after", "load1_i_before", "load1_i_after",
        "pp64_ts_diff_pct", "pp256_ts_diff_pct", "pp3000_ts_diff_pct"], rows)


def quality_table(runs):
    rows = []
    for run in sorted((r for r in runs if bench_json(r)), key=lambda r: runs[r]["start_utc"]):
        b = bench_json(run)
        rows.append([run, block_of(run), arm_tag(run), b["correct"], b["total"], b["agree_n"].split("/")[0],
                     b["mean_dp"], b["max_dp"], b["flips"], b["errors"], b["guard_pass"]])
    write_tsv("quality-bench.tsv", [
        "run", "block", "arm_tag", "correct", "decisions", "agree", "mean_dp", "max_dp",
        "top_answer_differs_from_reference", "errors", "guards_true"], rows)


# ---------------------------------------------------------------- the mechanism check: uploads and sync calls in a pass
def passes_of(db):
    """One record for each pass of the Nsight Systems export db.

    A pass is a run of 900 or more kernel launches. Its upload phase is the host to device copies that
    come before it in the sequence of CUDA runtime calls, with the sync call before the first copy and the sync
    call after the last copy. The records have the number of copies, the number of sync calls, the bytes, and
    the time of the phase in ms."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.cursor()
    rows = cur.execute(
        "select r.start, r.end, s.value, r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r "
        "join StringIds s on s.id = r.nameId order by r.start").fetchall()
    copy = {c: (b, k) for c, b, k in cur.execute("select correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY")}
    seq, run = [], 0
    for r in rows:
        if r[2].startswith("cudaLaunchKernel"):
            run += 1
        else:
            if run:
                seq.append(("launches", run, None))
                run = 0
            seq.append(("call", 0, r))
    if run:
        seq.append(("launches", run, None))
    out = []
    for i, item in enumerate(seq):
        if item[0] != "launches" or item[1] < 900:
            continue
        j, segment = i - 1, []
        while j >= 0 and seq[j][0] == "call":
            segment.append(seq[j][2])
            j -= 1
        segment.reverse()
        up = [k for k, r in enumerate(segment) if r[3] in copy and copy[r[3]][1] == 1 and r[2].startswith("cudaMemcpyAsync")]
        if not up or up[0] == 0:
            continue  # the segment does not start with a sync call before the first upload
        first, last = up[0] - 1, up[-1] + 1
        while last < len(segment) and not segment[last][2].startswith("cudaStreamSynchronize"):
            last += 1
        if last >= len(segment) or not segment[first][2].startswith("cudaStreamSynchronize"):
            continue
        phase = segment[first:last + 1]
        out.append({
            "uploads": len(up),
            "bytes": sum(copy[segment[k][3]][0] for k in up),
            "sync_calls": sum(1 for r in phase if r[2].startswith("cudaStreamSynchronize")),
            "phase_ms": (phase[-1][1] - phase[0][1]) / 1e6,
            "tokens": max(copy[segment[k][3]][0] for k in up) // 10240,
        })
    return out


def mechanism_table():
    """The tables mechanism.tsv and mechanism-pass-types.tsv. They need the Nsight Systems captures, which are not in this repository."""
    folder, nsys = os.environ.get("DMK_PROFILE_DIR"), os.environ.get("DMK_NSYS")
    if not folder:
        print("mechanism.tsv: DMK_PROFILE_DIR not set, the committed files stay")
        return
    rows, kinds = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for run in NSYS_RUNS:
            db = os.path.join(folder, run + ".sqlite")
            if not os.path.exists(db):  # export the capture
                db = os.path.join(tmp, run + ".sqlite")
                subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db,
                                os.path.join(folder, run + ".nsys-rep")], check=True, capture_output=True)
            ps = [p for p in passes_of(db) if p["tokens"] > 3]
            for (uploads, syncs), n in sorted(collections.Counter((p["uploads"], p["sync_calls"]) for p in ps).items()):
                kinds.append([run, uploads, syncs, n])
            buckets = collections.OrderedDict([("1 to 64 tokens", (1, 64)), ("65 to 256 tokens", (65, 256))])
            for label, (lo, hi) in buckets.items():
                sel = [p for p in ps if lo <= p["tokens"] <= hi]
                if not sel:
                    continue
                rows.append([run, label, len(sel),
                             num(statistics.mean(p["uploads"] for p in sel), 2),
                             num(statistics.mean(p["sync_calls"] for p in sel), 2),
                             num(statistics.mean(p["bytes"] for p in sel) / 1e6, 3),
                             num(statistics.mean(p["phase_ms"] for p in sel), 3)])
    write_tsv("mechanism.tsv", [
        "run", "tokens_in_pass", "passes", "uploads_per_pass", "sync_calls_in_upload_phase",
        "uploaded_mb_per_pass", "upload_phase_ms"], rows)
    write_tsv("mechanism-pass-types.tsv", ["run", "uploads", "sync_calls_in_upload_phase", "passes"], kinds)


def sass_summary():
    """The summary of the SASS compare of libggml-cuda.so of the two builds for sm_86."""
    j = json.load(open(raw("sass-compare/sass-compare.json")))
    write_tsv("sass-summary.tsv", ["item", "value"], [[k, v] for k, v in j["summary"].items()])


if __name__ == "__main__":
    runs = runs_table()
    answers = ab_tables()
    block_tables(runs)
    quality_table(runs)
    mechanism_table()
    sass_summary()
    for key, a in answers.items():
        print("answers,", key, "blocks: self repeat", ARM_A, a["self_repeat_a"], "- self repeat", ARM_B, a["self_repeat_b"],
              "- cross identical", a["cross_identical_all"], "- cross max |dp|", a["cross_max_abs_dp"],
              "- top flips", len(a["cross_top_flips"]))
    print("tables written to", os.path.relpath(OUT, ROOT))
