#!/usr/bin/env python3
"""Make the derived tables of the data set 2026-10-31-0008-ragged-suffix from the files in raw/.

Usage: python3 data/2026-10-31-0008-ragged-suffix/make_tables.py
The script writes into the folder tables/ next to it. It calls
harness/analysis/ab_stats.py and harness/analysis/answers_compare.py from this
repository. It uses only the Python standard library. It changes no file in raw/.
Run it again after each change of a file in raw/.
"""
import collections
import glob
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(HERE, "raw")
TESTS_DIR = os.path.join(RAW, "tests")
OUT = os.path.join(HERE, "tables")
ANALYSIS = os.path.join(ROOT, "harness", "analysis")
SIZES = ("short", "medium", "long")
PROMPTS = (64, 256, 3000)
SCHEDULE = "0008-ab.txt"
BENCH = "ragged0008-bench"
LLB = "ragged0008-llb"


def raw(name):
    return os.path.join(RAW, name)


def tests(name):
    return os.path.join(TESTS_DIR, name)


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


TEST_LABELS = [(BENCH + "-", "private benchmark"), (LLB + "-", "llama bench")]


def test_name(run):
    for prefix, label in TEST_LABELS:
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


def rel(path):
    return os.path.relpath(path, HERE)


def ab_tables():
    os.makedirs(OUT, exist_ok=True)
    for prefix, name in ((BENCH, "speed-bench.tsv"), (LLB, "speed-llama-bench.tsv")):
        table = run_tool([rel(os.path.join(ANALYSIS, "ab_stats.py")), "raw", "--prefix", prefix,
                          "--a", "A", "--b", "B", "--margin", "1"])
        with open(os.path.join(OUT, name), "w") as f:
            f.write(table)
    out = run_tool([rel(os.path.join(ANALYSIS, "answers_compare.py")),
                    "--a", f"raw/{BENCH}-b*-A.reference.json", "--b", f"raw/{BENCH}-b*-B.reference.json"])
    with open(os.path.join(OUT, "answers-compare-bench.json"), "w") as f:
        f.write(out)
    return json.loads(out)


def block_tables(runs):
    metrics = ["lat_ms", "short_ms", "medium_ms", "long_ms", "rps8"]
    rows = []
    for block in range(1, 5):
        a, b = f"{BENCH}-b{block}-A", f"{BENCH}-b{block}-B"
        ba, bb = bench_json(a), bench_json(b)
        first = "A" if runs[a]["start_utc"] < runs[b]["start_utc"] else "B"
        rows.append([block, first,
                     num(runs[a]["load_before"], 2), num(runs[a]["load_after"], 2),
                     num(runs[b]["load_before"], 2), num(runs[b]["load_after"], 2)]
                    + [num(100 * (bb[m] / ba[m] - 1), 3) for m in metrics])
    write_tsv("speed-bench-blocks.tsv", [
        "block", "first_arm", "load1_a_before", "load1_a_after", "load1_b_before", "load1_b_after",
        "lat_ms_diff_pct", "short_ms_diff_pct", "medium_ms_diff_pct", "long_ms_diff_pct", "rps8_diff_pct"], rows)

    rows = []
    for block in range(1, 5):
        a, b = f"{LLB}-b{block}-A", f"{LLB}-b{block}-B"
        va, vb = llb_values(a), llb_values(b)
        first = "A" if runs[a]["start_utc"] < runs[b]["start_utc"] else "B"
        rows.append([block, first,
                     num(runs[a]["load_before"], 2), num(runs[a]["load_after"], 2),
                     num(runs[b]["load_before"], 2), num(runs[b]["load_after"], 2)]
                    + [num(100 * (vb[f"pp{n}_ts"] / va[f"pp{n}_ts"] - 1), 3) for n in PROMPTS])
    write_tsv("speed-llama-bench-blocks.tsv", [
        "block", "first_arm", "load1_a_before", "load1_a_after", "load1_b_before", "load1_b_after",
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


# ---------------------------------------------------------------- the correctness runs
def check_runs_table():
    """The three single passes of the private benchmark of the correctness windows."""
    rows = []
    for run, arm, log_level in (("0008-check-A", "A", "4"), ("0008-check-B", "B", "4"), ("0008-check2-B", "B", "5")):
        b = bench_json(run)
        rows.append([run, arm, log_level, b["correct"], b["total"], b["agree_n"].split("/")[0], b["mean_dp"], b["max_dp"],
                     b["flips"], b["errors"], b["guard_pass"], num(b["lat_ms"], 3), num(b["rps8"], 3)])
    write_tsv("check-runs.tsv", [
        "run", "arm_tag", "server_log_verbosity", "correct", "decisions", "agree", "mean_dp", "max_dp",
        "top_answer_differs_from_reference", "errors", "guards_true", "lat_ms", "rps8"], rows)
    out = run_tool([rel(os.path.join(ANALYSIS, "answers_compare.py")),
                    "--a", "raw/0008-check-A.reference.json", "--b", "raw/0008-check-B.reference.json"])
    with open(os.path.join(OUT, "answers-compare-check.json"), "w") as f:
        f.write(out)
    return json.loads(out)


def decisions_table():
    rows = []
    for arm in ("A", "B"):
        text = open(raw(f"0008-dectest-{arm}.client.out")).read()
        correct = re.search(r"^correct (\d+)/(\d+)", text, re.M)
        misses = re.search(r"^misses: (.*)$", text, re.M)
        rows.append([f"0008-dectest-{arm}", arm, correct.group(1), correct.group(2), misses.group(1) if misses else ""])
    write_tsv("decisions-52.tsv", ["run", "arm_tag", "correct", "decisions", "wrong_decisions"], rows)
    # answers_compare.py reads the file shape of the benchmark: put the answers of the 52 decisions into that shape
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for arm in ("A", "B"):
            p = os.path.join(tmp, f"{arm}.json")
            json.dump({"answers": json.load(open(raw(f"0008-dectest-{arm}.answers.json")))}, open(p, "w"))
            paths.append(p)
        out = run_tool([rel(os.path.join(ANALYSIS, "answers_compare.py")), "--a", paths[0], "--b", paths[1]])
    out = re.sub(r"/tmp/[^\"]+/", "", out)
    with open(os.path.join(OUT, "answers-compare-decisions-52.json"), "w") as f:
        f.write(out)
    return json.loads(out)


def ops_table():
    """The result of each run of test-backend-ops."""
    rows = []
    items = [
        ("SSM_CONV_RAGGED", "-o SSM_CONV_RAGGED", "conv.log"),
        ("GATED_DELTA_NET_RAGGED, without the fusion", "-o GATED_DELTA_NET_RAGGED", "gdn.log"),
        ("GATED_DELTA_NET_RAGGED, with the fusion into CPY", "-p cache=1,lens=", "tbo-cache.log"),
        ("Built-in cases of the five operations of the layer", "-o GATED_DELTA_NET,CPY,SSM_CONV,L2_NORM,SOFTPLUS", "builtin.log"),
        ("187 operations of the graph of mix8", "--test-file <file of the 187 operations>", "graph.log"),
    ]
    for label, flt, log in items:
        text = open(tests(log)).read()
        m = re.findall(r"(\d+)/(\d+) tests passed", text)[-1]
        failing = collections.Counter(re.findall(r"^  (\w+)\(name=", text.split("Failing tests:")[1], re.M)) if "Failing tests:" in text else {}
        rows.append([label, flt, "tests/" + log, m[0], m[1], ", ".join(f"{k} x{v}" for k, v in sorted(failing.items()))])
    write_tsv("ops-tests.tsv", ["test", "filter", "log", "passed", "cases", "failing_operations"], rows)

    text = open(tests("test-batch-alloc.log")).read()
    vals = {k: re.search(rf"^{k}\s*: (\d+)", text, re.M).group(1) for k in ("tests", "assertions", "failures", "exceptions", "skipped")}
    write_tsv("unit-tests.tsv", ["test", "tests", "assertions", "failures", "exceptions", "skipped"],
              [["test-batch-alloc (CPU, includes split_equal_ragged and split_equal_ragged_falls_back)",
                vals["tests"], vals["assertions"], vals["failures"], vals["exceptions"], vals["skipped"]]])


CASES = [("medium", "inproc-medium.log"), ("short2q", "inproc-short2q.log"), ("five", "inproc-five.log"), ("alllogits", "inproc-alllogits.log")]
COMPARE_RX = re.compile(
    r"^(floor \(equal again\)|ragged vs equal|ragged again vs ragged|equal vs one at a time|ragged vs one at a time)\s+"
    r"kl_mean (\S+)\s+kl_max (\S+)\s+max_abs_logit (\S+)\s+top1 (\d+)/(\d+) \| "
    r"conv: max_abs (\S+)\s+nmse (\S+)\s+identical (\d+)/(\d+) \| ssm: max_abs (\S+)\s+nmse (\S+)\s+identical (\d+)/(\d+)$")


def suffix_tables():
    cmp_rows, pass_rows, check_rows, layer_rows = [], [], [], []
    for case, log in CASES:
        text = open(tests(log)).read().splitlines()
        head = [re.match(r"^lens ([\d ]+)  prefix (\d+)  rows (\d+)$", l) for l in text]
        head = [m for m in head if m][0]
        lens = head.group(1).replace(" ", ",")
        for l in text:
            m = COMPARE_RX.match(l)
            if m:
                cmp_rows.append([case, lens, head.group(2), head.group(3), m.group(1)] + list(m.groups()[1:]))
            m = re.match(r"^(equal split|equal split again|ragged split|ragged split again|one sequence a time)\s+passes of the suffixes: (\d+)$", l)
            if m:
                pass_rows.append([case, lens, m.group(1), m.group(2)])
            m = re.match(r"^(ok  |FAIL): (.*)$", l)
            if m:
                check_rows.append([case, lens, m.group(2), m.group(1).strip()])
            m = re.match(r"^floor: kl_mean (\S+) kl_max (\S+) -> limits: kl_mean (\S+) kl_max (\S+)$", l)
            if m:
                floor = m.groups()
        in_table = False
        for l in text:
            if l.startswith("layer  ragged-vs-equal"):
                in_table = True
                continue
            m = re.match(r"^\s+(\d+)  (\S+) (\S+)\s+(\S+) (\S+)\s+(\S+) (\S+)$", l) if in_table else None
            if m:
                layer_rows.append([case, lens] + list(m.groups()))
            elif in_table:
                in_table = False
        check_rows.append([case, lens, f"floor of the logits: mean KL {floor[0]}, largest KL {floor[1]}; limits: mean KL {floor[2]}, largest KL {floor[3]}", "info"])
    write_tsv("suffix-compare.tsv", [
        "case", "suffix_lengths", "prefix_tokens", "rows", "comparison", "kl_mean", "kl_max", "max_abs_logit", "same_top_token", "rows_compared",
        "conv_max_abs", "conv_nmse", "conv_identical", "conv_vectors", "ssm_max_abs", "ssm_nmse", "ssm_identical", "ssm_vectors"], cmp_rows)
    write_tsv("suffix-passes.tsv", ["case", "suffix_lengths", "decode", "passes_of_the_suffixes"], pass_rows)
    write_tsv("suffix-checks.tsv", ["case", "suffix_lengths", "check", "result"], check_rows)
    write_tsv("suffix-layers.tsv", [
        "case", "suffix_lengths", "layer", "ragged_vs_equal_conv_nmse", "ragged_vs_equal_ssm_nmse",
        "equal_vs_one_at_a_time_conv_nmse", "equal_vs_one_at_a_time_ssm_nmse",
        "ragged_vs_one_at_a_time_conv_nmse", "ragged_vs_one_at_a_time_ssm_nmse"], layer_rows)


def kl_table():
    rows = []
    for text_set, what in (("wiki", "WikiText-2, 20 chunks of 512 tokens"), ("domain", "text of the states of the benchmark, 20 chunks of 512 tokens")):
        for run, label in ((f"0008-kld-{text_set}-A-self", "A against its own base file (the KL floor)"), (f"0008-kld-{text_set}-B-vsA", "B against the base file of A")):
            t = open(raw(run + ".out")).read()

            def one(rx):
                return re.search(rx, t).groups()

            rows.append([run, text_set, what, label,
                         one(r"Mean    KLD:\s+(\S+) ±")[0], one(r"Maximum KLD:\s+(\S+)")[0], one(r"99\.9%   KLD:\s+(\S+)")[0],
                         one(r"Same top p:\s+(\S+) ±")[0], one(r"RMS Δp\s+:\s+(\S+) ±")[0],
                         one(r"Mean PPL\(Q\)\s+:\s+(\S+) ±")[0], one(r"Mean PPL\(base\)\s+:\s+(\S+) ±")[0]])
    write_tsv("kl-perplexity.tsv", [
        "run", "text_set", "text", "comparison", "mean_kld", "max_kld", "kld_percentile_99_9", "same_top_p_pct", "rms_dp_pct", "mean_ppl_q", "mean_ppl_base"], rows)


def ubatch_rows(lines_file, phases):
    """Ubatches with suffixes of different length in a run with the debug log, by phase. phases is a list of (name, requests)."""
    first, total = {}, 0
    for name, n in phases:
        first[name] = total
        total += n

    def phase_of(request):
        name = phases[0][0]
        for p, _ in phases:
            if request >= first[p]:
                name = p
        return name

    stats = {p: collections.Counter() for p, _ in phases}
    sizes = {p: collections.Counter() for p, _ in phases}
    idle = 0
    for line in open(raw(lines_file)):
        if "all slots are idle" in line:
            idle += 1
            continue
        m = re.search(r"n_seqs = (\d+), n_tokens = (\d+), max_step = (\d+), single = (\d), ragged = (\d)", line)
        n_seqs, n_tok, max_step, single, ragged = map(int, m.groups())
        phase = phase_of(idle - 1)  # the first idle line comes before the first request
        key = "taken" if ragged else "not_taken"
        stats[phase][key] += 1
        stats[phase]["tokens_" + key] += n_tok
        stats[phase]["max_step_" + key] = max(stats[phase]["max_step_" + key], max_step)
        sizes[phase][f"{n_seqs}"] += 1
    rows = []
    for p, n in phases:
        s = stats[p]
        rows.append([p, n, s["taken"] + s["not_taken"], s["taken"], s["not_taken"], s["tokens_taken"], s["tokens_not_taken"],
                     s["max_step_taken"], s["max_step_not_taken"],
                     "; ".join(f"{k} sequences: {v}" for k, v in sorted(sizes[p].items()))])
    return rows


UBATCH_HEADER = [
    "phase", "requests", "ubatches_with_different_lengths", "ragged_taken", "ragged_not_taken", "tokens_in_taken", "tokens_in_not_taken",
    "largest_step_taken", "largest_step_not_taken", "sequences_per_ubatch"]


def ubatch_table():
    """The pass of the private benchmark with the debug log, and the pass of the client with the debug log."""
    j = json.load(open(raw("0008-check2-B.json")))
    n_quality = j["meta"]["cases"]
    write_tsv("ragged-ubatches.tsv", UBATCH_HEADER, ubatch_rows(
        "0008-check2-B.server-ubatch-lines.txt",
        [("warmup", 3), ("quality", n_quality), ("seq", n_quality * j["meta"]["passes"]), ("tput", j["tput"]["requests"])]))
    if os.path.exists(raw("0008-type-census.server-ubatch-lines.txt")):
        write_tsv("census-type.tsv", UBATCH_HEADER, ubatch_rows(
            "0008-type-census.server-ubatch-lines.txt", [("warmup", 3), ("seq", n_quality)]))


def memcheck_table():
    t = open(tests("memcheck2.out")).read()
    rows = [["error_summary", re.search(r"ERROR SUMMARY: (.*)", t).group(1)],
            ["server_exit_code", re.search(r"server exit code: (\d+)", t).group(1)],
            ["client_exit_code", re.search(r"client rc=(\d+)", t).group(1)]]
    for m in re.finditer(r"^(short1q|short2q|medium|long) (\S+) (\d+) (\d+) ms$", t, re.M):
        rows.append([f"request {m.group(1)} {m.group(2)}", f"HTTP {m.group(3)}, {m.group(4)} ms"])
    write_tsv("memcheck.tsv", ["item", "value"], rows)


TYPE_PREFIX = "ragged0008-type"
TYPES = ("short1q", "short2q", "medium", "long")


def block_stat(ratios, boot=10000, seed=12345):
    """Mean of the block ratios and its 95 % percentile bootstrap interval, as in harness/analysis/ab_stats.py."""
    rng = random.Random(seed)
    boots = []
    for _ in range(boot):
        boots.append(statistics.fmean([ratios[rng.randrange(len(ratios))] for _ in ratios]))
    boots.sort()
    return statistics.fmean(ratios), boots[int(0.025 * boot)], boots[int(0.975 * boot) - 1]


def type_tables():
    """The latency of each request type, from the runs of the client latency_by_type.py."""
    res = {}
    for path in sorted(glob.glob(raw(TYPE_PREFIX + "-b*-*.client.out"))):
        run = os.path.basename(path)[:-len(".client.out")]
        res[run] = json.loads(re.search(r"^RESULT (.*)$", open(path).read(), re.M).group(1))
    rows = []
    for run in sorted(res, key=lambda r: read_run(r)["start_utc"]):
        d = read_run(run)
        rows.append([run, block_of(run), arm_tag(run), d["arm"], d["start_utc"], num(d["load_before"], 2), num(d["load_after"], 2)]
                    + [num(res[run][t]["mean_ms"], 3) for t in TYPES])
    write_tsv("type-runs.tsv", ["run", "block", "arm_tag", "arm", "start_utc", "load1_before", "load1_after",
                                "short1q_ms", "short2q_ms", "medium_ms", "long_ms"], rows)
    blocks = sorted({block_of(r) for r in res})
    rows = []
    for t in TYPES:
        a = [res[f"{TYPE_PREFIX}-b{b}-A"][t]["mean_ms"] for b in blocks]
        b_ = [res[f"{TYPE_PREFIX}-b{b}-B"][t]["mean_ms"] for b in blocks]
        est, lo, hi = block_stat([y / x for x, y in zip(a, b_)])
        rows.append([t + "_ms", len(blocks), num(statistics.fmean(a), 3), num(statistics.fmean(b_), 3),
                     num(100 * (est - 1), 3), num(100 * (lo - 1), 3), num(100 * (hi - 1), 3)]
                    + [num(100 * (y / x - 1), 3) for x, y in zip(a, b_)])
    write_tsv("speed-type.tsv", ["metric", "blocks", "a_mean", "b_mean", "diff_pct", "ci_lo_pct", "ci_hi_pct"]
              + [f"block{b}_diff_pct" for b in blocks], rows)


def check_block_stat():
    """The function block_stat gives the numbers of ab_stats.py for the metric lat_ms of the private benchmark."""
    ratios = [bench_json(f"{BENCH}-b{b}-B")["lat_ms"] / bench_json(f"{BENCH}-b{b}-A")["lat_ms"] for b in range(1, 5)]
    est, lo, hi = block_stat(ratios)
    row = [l.split("\t") for l in open(os.path.join(OUT, "speed-bench.tsv")).read().splitlines() if l.startswith("lat_ms")][0]
    assert abs(100 * (est - 1) - float(row[4])) < 1e-3 and abs(100 * (lo - 1) - float(row[5])) < 1e-3 \
        and abs(100 * (hi - 1) - float(row[6])) < 1e-3, "block_stat differs from ab_stats.py"

if __name__ == "__main__":
    runs = runs_table()
    answers = ab_tables()
    block_tables(runs)
    quality_table(runs)
    check = check_runs_table()
    dec = decisions_table()
    ops_table()
    suffix_tables()
    kl_table()
    ubatch_table()
    memcheck_table()
    check_block_stat()
    if glob.glob(raw(TYPE_PREFIX + "-b*-*.client.out")):
        type_tables()
    print("answers: self repeat A", answers["self_repeat_a"], "- self repeat B", answers["self_repeat_b"],
          "- cross identical", answers["cross_identical_all"], "- cross max |dp|", answers["cross_max_abs_dp"],
          "- top flips", answers["cross_top_flips"])
    print("check runs: cross identical", check["cross_identical_all"], "- cross max |dp|", check["cross_max_abs_dp"],
          "- top flips", check["cross_top_flips"])
    print("52 decisions: cross identical", dec["cross_identical_all"], "- cross max |dp|", dec["cross_max_abs_dp"],
          "- top flips", dec["cross_top_flips"])
    print("tables written to", os.path.relpath(OUT, ROOT))
