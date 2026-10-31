#!/usr/bin/env python3
"""Make the derived tables of the data set 2026-10-31-0008-ragged-suffix-retest from the files in raw/.

Usage: python3 data/2026-10-31-0008-ragged-suffix-retest/make_tables.py
The script writes into the folder tables/ next to it. It calls
harness/analysis/ab_stats.py and harness/analysis/answers_compare.py from this
repository. It uses only the Python standard library. It changes no file in raw/.
Run it again after each change of a file in raw/.
The speed tables and the tables of the graph file with the new operation
numbers need raw files that the script finds only when they exist. It skips each
table group whose files are not all in raw/.
"""
import collections
import glob
import json
import os
import random
import re
import shlex
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
SCHEDULE = "0008r-ab.txt"
CORRECTNESS_SCHEDULE = "0008r-correctness.txt"
BENCH = "ragged0008r-bench"
LLB = "ragged0008r-llb"
TYPE_PREFIX = "ragged0008r-type"
TYPES = ("short1q", "short2q", "medium", "long")
PRODUCTION_ARGS = "-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots"


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


def sci(v):
    return None if v is None else f"{v:.6e}"


def run_tool(args):
    return subprocess.run([sys.executable] + args, cwd=HERE, check=True, capture_output=True, text=True).stdout


def rel(path):
    return os.path.relpath(path, HERE)


# ---------------------------------------------------------------- the schedules and the run files
def parse_schedule(name):
    """Lines of a schedule file of the harness: type, run name, arm, arguments and environment variables."""
    lines = []
    for line in open(raw(name)):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words = shlex.split(line)
        env = {}
        while re.match(r"DMK_[A-Z0-9_]+=", words[0]):
            key, value = words.pop(0).split("=", 1)
            env[key] = value
        lines.append({"type": words[0], "run": words[1], "arm": words[2], "args": words[3:], "env": env})
    return lines


def schedule():
    """Return {run name: line type} from the schedule file of the speed test."""
    return {l["run"]: l["type"] for l in parse_schedule(SCHEDULE)}


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
    if os.path.exists(raw(run + ".load0")):  # a bench line or a client line
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


# ---------------------------------------------------------------- the answers of the benchmark
def answers_of(run):
    return json.load(open(raw(run + ".reference.json")))["answers"]


def probs(ans):
    """The probability of each answer key of a decision, as in harness/analysis/answers_compare.py."""
    if ans["type"] == "noul":
        return {"yes": ans["noul"]}
    if "probabilities" in ans:
        return dict(ans["probabilities"])
    return {"score": ans.get("score")}


def top(ans):
    """The top answer of a decision, as in harness/analysis/answers_compare.py."""
    if ans["type"] == "noul":
        return ans["noul"] >= 0.5
    if ans["type"] == "choice":
        return ans["choice"]
    return round(ans["score"])


def top_label(ans):
    t = top(ans)
    if ans["type"] == "noul":
        return "yes" if t else "no"
    return str(t)


def dp_of(a, b):
    """The largest difference of the probability over the answer keys of one decision."""
    pa, pb = probs(a), probs(b)
    return max([0.0] + [abs((pa.get(k) or 0.0) - (pb.get(k) or 0.0)) for k in set(pa) | set(pb)])


def pair_stats(x, y):
    """The difference of the probability of each decision between two answer sets, and the flips of the top answer."""
    dps, flips = {}, []
    for cid, qs in x.items():
        for qid, a in qs.items():
            b = y[cid][qid]
            dps[f"{cid}.{qid}"] = dp_of(a, b)
            if top(a) != top(b):
                flips.append((f"{cid}.{qid}", top_label(a), top_label(b)))
    return {"decisions": len(dps), "identical": x == y, "max_dp": max(dps.values()), "mean_dp": statistics.fmean(dps.values()),
            "differing": sum(1 for v in dps.values() if v > 0), "flips": flips, "dps": dps}


def flip_text(flips):
    return "; ".join(f"{d} (a: {ta}, b: {tb})" for d, ta, tb in flips)


# The pairs of runs that the answers table compares: name, run a, run b, meaning.
PAIRS = [
    ("A3a-vs-A3b", "0008r-A3a", "0008r-A3b", "Arm A with -np 3, two runs (self repeat of arm A)"),
    ("B3a-vs-B3b", "0008r-B3a", "0008r-B3b", "Arm B with -np 3, two runs (self repeat of arm B)"),
    ("A3a-vs-A1", "0008r-A3a", "0008r-A1", "Arm A with -np 3 against arm A with -np 1 (the reference decode), run a"),
    ("A3b-vs-A1", "0008r-A3b", "0008r-A1", "Arm A with -np 3 against arm A with -np 1, run b"),
    ("B3a-vs-A1", "0008r-B3a", "0008r-A1", "Arm B with -np 3 against arm A with -np 1, run a"),
    ("B3b-vs-A1", "0008r-B3b", "0008r-A1", "Arm B with -np 3 against arm A with -np 1, run b"),
    ("B1-vs-A1", "0008r-B1", "0008r-A1", "Arm B with -np 1 against arm A with -np 1"),
    ("B3a-vs-A3a", "0008r-B3a", "0008r-A3a", "Arm B against arm A, both with -np 3, run a"),
    ("B3b-vs-A3b", "0008r-B3b", "0008r-A3b", "Arm B against arm A, both with -np 3, run b"),
    ("B3lv5-vs-B3a", "0008r-B3-lv5", "0008r-B3a", "Arm B with -np 3 and the debug log, against run a without it"),
]

# The groups that answers_compare.py gets as file patterns: name, pattern a, pattern b.
GROUPS = [
    ("np3_A_vs_B", "0008r-A3[ab]", "0008r-B3[ab]"),
    ("A_np3_vs_A_np1", "0008r-A3[ab]", "0008r-A1"),
    ("B_np3_vs_A_np1", "0008r-B3[ab]", "0008r-A1"),
    ("B_np1_vs_A_np1", "0008r-B1", "0008r-A1"),
]


def answers_tables(quality, dec):
    """answers-compare.tsv, answers-compare.json and answers-verdict.tsv."""
    answers = {run: answers_of(run) for run in {p[1] for p in PAIRS} | {p[2] for p in PAIRS}}
    stats = {name: pair_stats(answers[a], answers[b]) for name, a, b, _ in PAIRS}
    rows = []
    for name, a, b, meaning in PAIRS:
        s = stats[name]
        rows.append([name, a, b, meaning, s["identical"], s["decisions"], num(s["max_dp"], 6), num(s["mean_dp"], 6), s["differing"],
                     len(s["flips"]), flip_text(s["flips"])])
    write_tsv("answers-compare.tsv", [
        "pair", "run_a", "run_b", "meaning", "identical_bit_for_bit", "decisions", "max_abs_dp", "mean_abs_dp",
        "decisions_with_dp_above_0", "top_answer_flips", "top_answer_flip_list"], rows)

    groups = {}
    for name, pa, pb in GROUPS:
        out = json.loads(run_tool([rel(os.path.join(ANALYSIS, "answers_compare.py")),
                                   "--a", f"raw/{pa}.reference.json", "--b", f"raw/{pb}.reference.json"]))
        groups[name] = out
    with open(os.path.join(OUT, "answers-compare.json"), "w") as f:
        f.write(json.dumps(groups, indent=1) + "\n")

    def runs_of(pattern):
        return sorted(os.path.basename(p)[:-len(".reference.json")] for p in glob.glob(raw(pattern + ".reference.json")))

    # The own numbers must give the numbers of answers_compare.py for each group of files.
    for name, pa, pb in GROUPS:
        runs_a, runs_b = runs_of(pa), runs_of(pb)
        for run in runs_a + runs_b:
            answers.setdefault(run, answers_of(run))
        combos = [pair_stats(answers[x], answers[y]) for x in runs_a for y in runs_b]
        g = groups[name]
        assert runs_a and runs_b and g["files_a"] == [f"raw/{r}.reference.json" for r in runs_a], name
        assert abs(max(c["max_dp"] for c in combos) - g["cross_max_abs_dp"]) < 1e-12, name
        assert sorted({f for c in combos for f, _, _ in c["flips"]}) == g["cross_top_flips"], name
        assert all(c["identical"] for c in combos) == g["cross_identical_all"], name
        assert g["self_repeat_a"] == all(answers[r] == answers[runs_a[0]] for r in runs_a), name
        assert g["self_repeat_b"] == all(answers[r] == answers[runs_b[0]] for r in runs_b), name

    # The limits for the decision on the answers.
    b_runs, a_runs = ("B3a-vs-A1", "B3b-vs-A1"), ("A3a-vs-A1", "A3b-vs-A1")
    verdict = []
    for key, label in (("max_dp", "max_abs_dp"), ("mean_dp", "mean_abs_dp")):
        b, a = max(stats[n][key] for n in b_runs), max(stats[n][key] for n in a_runs)
        ok = b <= 1.5 * a
        note = (f"Arm B runs a and b: {stats[b_runs[0]][key]:.6f}, {stats[b_runs[1]][key]:.6f}. "
                f"Arm A runs a and b: {stats[a_runs[0]][key]:.6f}, {stats[a_runs[1]][key]:.6f}. Each side uses the larger value of its two runs.")
        verdict.append([f"{label} of -np 3 against arm A with -np 1", label, num(b, 6), num(a, 6), num(b / a, 3) if a else None, "1.5",
                        "pass" if ok else "FAIL", note])
    b_flips = {f: (ta, tb) for n in b_runs for f, ta, tb in stats[n]["flips"]}
    a_flips = {f: (ta, tb) for n in a_runs for f, ta, tb in stats[n]["flips"]}
    only_b = sorted(set(b_flips) - set(a_flips))
    note = ("Arm B -np 3 against arm A -np 1: " + (flip_text([(f, *b_flips[f]) for f in sorted(b_flips)]) or "no decision") + ". "
            "Arm A -np 3 against arm A -np 1: " + (flip_text([(f, *a_flips[f]) for f in sorted(a_flips)]) or "no decision") + ".")
    verdict.append(["Top answer of the decisions", "decisions with a different top answer in arm B that arm A does not have",
                    len(only_b), len(a_flips), None, "0", "pass" if not only_b else "FAIL", note])
    verdict.append(["self_repeat of arm B", "the two runs of arm B with -np 3 are the same bit for bit", stats["B3a-vs-B3b"]["identical"], None, None,
                    "True", "pass" if stats["B3a-vs-B3b"]["identical"] else "FAIL", ""])
    verdict.append(["self_repeat of arm A", "the two runs of arm A with -np 3 are the same bit for bit", None, stats["A3a-vs-A3b"]["identical"], None,
                    "True", "pass" if stats["A3a-vs-A3b"]["identical"] else "FAIL", ""])
    verdict.append(["Arm B with -np 1 against arm A with -np 1", "the answers are the same bit for bit", stats["B1-vs-A1"]["identical"], None, None,
                    "True", "pass" if stats["B1-vs-A1"]["identical"] else "FAIL",
                    f"max_abs_dp {stats['B1-vs-A1']['max_dp']:.6f}, decisions with a difference {stats['B1-vs-A1']['differing']}"])
    ca, cb = int(dec["A"]["correct"]), int(dec["B"]["correct"])
    verdict.append(["52-decision set", "correct decisions", cb, ca, None, "equal", "pass" if ca == cb else "FAIL",
                    f"Items with a different top answer: {len(dec['flips'])}. The table dectest-items.tsv has each item."])
    guards = quality
    bad = [r["run"] for r in guards if not r["guard_pass"] or r["errors"]]
    verdict.append(["Guards of the benchmark and errors", "runs of the private benchmark with all guards true and 0 errors", len(guards) - len(bad), len(guards),
                    None, "all", "pass" if not bad else "FAIL", ("Runs with a guard that is not true, or with an error: " + ", ".join(bad)) if bad else ""])
    write_tsv("answers-verdict.tsv", ["rule", "measure", "build_b", "build_a", "ratio_b_to_a", "limit", "result", "note"], verdict)
    return stats


def decision_tables():
    """dectest.tsv, dectest-items.tsv and answers-compare-decisions-52.json."""
    rows, answers, wrong = [], {}, {}
    for arm in ("A", "B"):
        run = f"0008r-dectest-{arm}"
        text = open(raw(run + ".client.out")).read()
        correct = re.search(r"^correct (\d+)/(\d+)", text, re.M)
        misses = re.search(r"^misses: (.*)$", text, re.M)
        wrong[arm] = [m.strip() for m in misses.group(1).split(",")] if misses else []
        rows.append([run, arm, correct.group(1), correct.group(2), misses.group(1) if misses else ""])
        answers[arm] = json.load(open(raw(run + ".answers.json")))
    write_tsv("dectest.tsv", ["run", "arm_tag", "correct", "decisions", "wrong_decisions"], rows)
    items, flips = [], []
    for cid, qs in answers["A"].items():
        for qid, a in qs.items():
            b = answers["B"][cid][qid]
            item = f"{cid}.{qid}"
            items.append([item, top_label(a), top_label(b), top(a) == top(b), item not in wrong["A"], item not in wrong["B"],
                          num(dp_of(a, b), 6), a == b])
            if top(a) != top(b):
                flips.append(item)
    write_tsv("dectest-items.tsv", ["item", "top_answer_a", "top_answer_b", "same_top_answer", "correct_a", "correct_b", "max_abs_dp", "identical"], items)
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for arm in ("A", "B"):
            p = os.path.join(tmp, f"{arm}.json")
            json.dump({"answers": answers[arm]}, open(p, "w"))
            paths.append(p)
        out = run_tool([rel(os.path.join(ANALYSIS, "answers_compare.py")), "--a", paths[0], "--b", paths[1]])
    out = out.replace(tmp + "/", "")
    with open(os.path.join(OUT, "answers-compare-decisions-52.json"), "w") as f:
        f.write(out)
    g = json.loads(out)
    assert g["cross_top_flips"] == sorted(flips)
    return {"A": {"correct": rows[0][2]}, "B": {"correct": rows[1][2]}, "flips": flips, "compare": g}


# ---------------------------------------------------------------- the correctness runs of the private benchmark
def correctness_runs_table():
    """One row for each run of the private benchmark of the correctness measurement window."""
    rows, quality = [], []
    for l in parse_schedule(CORRECTNESS_SCHEDULE):
        if l["type"] != "bench":
            continue
        run = l["run"]
        args = l["env"].get("DMK_SERVER_ARGS", PRODUCTION_ARGS)
        slots = re.search(r"-np (\d+)", args).group(1)
        log_level = (re.search(r"-lv (\d+)", args) or [None, "default"])[1]
        d, b = read_run(run), bench_json(run)
        tag = re.match(r"0008r-([AB])", run).group(1)
        rows.append([run, tag, d["arm"], slots, log_level, d["start_utc"], b["correct"], b["total"], b["agree_n"].split("/")[0],
                     b["mean_dp"], b["max_dp"], b["flips"], b["errors"], b["guard_pass"], num(b["lat_ms"], 3), num(b["rps8"], 3)])
        quality.append({"run": run, "guard_pass": b["guard_pass"], "errors": b["errors"]})
    write_tsv("correctness-runs.tsv", [
        "run", "arm_tag", "arm", "slots", "server_log_verbosity", "start_utc", "correct", "decisions", "agree", "mean_dp", "max_dp",
        "top_answer_differs_from_reference", "errors", "guards_true", "lat_ms", "rps8"], rows)
    return quality


# ---------------------------------------------------------------- test-backend-ops, test-batch-alloc and compute-sanitizer
def read_return_codes():
    codes = {}
    for name in ("summary.txt", "summary-graph.txt"):
        if os.path.exists(tests(name)):
            for m in re.finditer(r"^(\S+) rc=(\d+)$", open(tests(name)).read(), re.M):
                codes[m.group(1)] = int(m.group(2))
    return codes


CASE_RX = re.compile(r"(?:^|\s)([A-Z][A-Z0-9_]*\(.*\)): (OK|FAIL)\s*$")
ANSI_RX = re.compile(r"\x1b\[[0-9;]*m")  # test-backend-ops writes OK and FAIL in colors


def op_cases(log):
    """The cases of a log of test-backend-ops: a list of (case text, OK or FAIL). Some case texts occur more than once."""
    cases = []
    for line in ANSI_RX.sub("", open(tests(log)).read()).splitlines():
        m = CASE_RX.search(line)
        if m:
            cases.append((m.group(1), m.group(2)))
    return cases


OPS_ITEMS = [
    # label, arm, command after "test -b CUDA0", log, key of the return code
    ("SSM_CONV_RAGGED", "B", "-o SSM_CONV_RAGGED", "conv-B.log", "conv-B"),
    ("GATED_DELTA_NET_RAGGED, without the fusion", "B", "-o GATED_DELTA_NET_RAGGED", "gdn-B.log", "gdn-B"),
    ("GATED_DELTA_NET_RAGGED, with the fusion into CPY", "B", "-p cache=1,lens=", "fusion-B.log", "fusion-B"),
    ("Built-in cases of five operations", "A", "-o GATED_DELTA_NET,CPY,SSM_CONV,KQ_MASK,CONCAT", "builtin-A.log", "builtin-A"),
    ("Built-in cases of five operations", "B", "-o GATED_DELTA_NET,CPY,SSM_CONV,KQ_MASK,CONCAT", "builtin-B.log", "builtin-B"),
    ("187 operations of the graph, file of the first test", "A", "--test-file <file of the first test>", "graph-A.log", "graph-A"),
    ("187 operations of the graph, file of the first test", "B", "--test-file <file of the first test>", "graph-B.log", "graph-B"),
    ("187 operations of the graph, file with the new operation numbers", "A", "--test-file <file with the new operation numbers>", "graph2-A.log", "graph2-A"),
    ("187 operations of the graph, file with the new operation numbers", "B", "--test-file <file with the new operation numbers>", "graph2-B.log", "graph2-B"),
]


def ops_tables():
    codes = read_return_codes()
    rows = []
    for label, arm, flt, log, key in OPS_ITEMS:
        if not os.path.exists(tests(log)):
            continue
        text = open(tests(log)).read()
        found = re.findall(r"(\d+)/(\d+) tests passed", text)
        failing = collections.Counter(re.findall(r"^  (\w+)\(name=", text.split("Failing tests:")[1], re.M)) if "Failing tests:" in text else {}
        rc = codes.get(key)
        if found:
            passed, cases = found[-1]
            result = "all cases OK" if passed == cases and rc == 0 else f"{int(cases) - int(passed)} cases FAIL"
        else:
            passed, cases = None, None
            result = f"no result: the program stopped with exit code {rc}" + (" (signal 11)" if rc == 139 else "")
        rows.append([label, arm, flt, "tests/" + log, rc, passed, cases, ", ".join(f"{k} x{v}" for k, v in sorted(failing.items())), result])
    write_tsv("ops.tsv", ["test", "arm_tag", "filter", "log", "exit_code", "passed", "cases", "failing_operations", "result"], rows)

    rows = []
    for label, a_log, b_log in (("Built-in cases of five operations", "builtin-A.log", "builtin-B.log"),
                                ("187 operations of the graph, file with the new operation numbers", "graph2-A.log", "graph2-B.log")):
        if not (os.path.exists(tests(a_log)) and os.path.exists(tests(b_log))):
            continue
        a, b = collections.Counter(op_cases(a_log)), collections.Counter(op_cases(b_log))
        only_a, only_b = a - b, b - a
        rows.append([label, sum(a.values()), sum(b.values()), sum((a & b).values()),
                     sum(n for (_, r), n in a.items() if r == "FAIL"), sum(n for (_, r), n in b.items() if r == "FAIL"),
                     sum(only_a.values()), sum(only_b.values()),
                     "; ".join(f"{c}: {r}" for (c, r) in sorted(only_a)), "; ".join(f"{c}: {r}" for (c, r) in sorted(only_b))])
    write_tsv("ops-compare.tsv", [
        "test", "cases_a", "cases_b", "cases_with_the_same_result_in_a_and_b", "cases_with_FAIL_a", "cases_with_FAIL_b",
        "cases_of_a_without_the_same_result_in_b", "cases_of_b_without_the_same_result_in_a",
        "cases_of_a_without_the_same_result_in_b_list", "cases_of_b_without_the_same_result_in_a_list"], rows)

    text = open(tests("batchalloc.log")).read()
    vals = {k: re.search(rf"^{k}\s*: (\d+)", text, re.M).group(1) for k in ("tests", "assertions", "failures", "exceptions", "skipped")}
    write_tsv("unit-tests.tsv", ["test", "tests", "assertions", "failures", "exceptions", "skipped"],
              [["test-batch-alloc (CPU, includes split_equal_ragged and split_equal_ragged_falls_back)",
                vals["tests"], vals["assertions"], vals["failures"], vals["exceptions"], vals["skipped"]]])


def memcheck_table():
    t = open(tests("memcheck.out")).read()
    rows = [["error_summary", re.search(r"ERROR SUMMARY: (.*)", t).group(1)],
            ["server_exit_code", re.search(r"server exit code: (\d+)", t).group(1)],
            ["client_exit_code", re.search(r"client rc=(\d+)", t).group(1)]]
    for m in re.finditer(r"^(short1q|short2q|medium|long) (\S+) (\d+) (\d+) ms$", t, re.M):
        rows.append([f"request {m.group(1)} {m.group(2)}", f"HTTP {m.group(3)}, {m.group(4)} ms"])
    write_tsv("memcheck.tsv", ["item", "value"], rows)


# ---------------------------------------------------------------- the logits and the states of the 8 inputs
NOISE_RX = re.compile(r"~llama_context:[^\n]*\n")  # the destructor of the context writes to stderr inside the lines of the table
QUANTITIES = ("kl_mean", "kl_max", "top_miss", "conv_nmse", "ssm_nmse")
QUANTITY_LABELS = {
    "kl_mean": "mean KL divergence", "kl_max": "largest KL divergence", "top_miss": "rows with another top token",
    "conv_nmse": "NMSE of the conv state", "ssm_nmse": "NMSE of the SSM state"}
BUDGET_TEXT = {"0.027830": "WikiText-2", "0.057587": "text of the states"}
COMPARE_RX = re.compile(
    r"^(floor \(equal again\)|ragged vs equal|ragged again vs ragged|equal vs one at a time|ragged vs one at a time)\s+"
    r"kl_mean (\S+)\s+kl_max (\S+)\s+max_abs_logit (\S+)\s+top1 (\d+)/(\d+) \| "
    r"conv: max_abs (\S+)\s+nmse (\S+)\s+identical (\d+)/(\d+) \| ssm: max_abs (\S+)\s+nmse (\S+)\s+identical (\d+)/(\d+)\s*$")
QUANTITY_RX = re.compile(r"^(kl_mean|kl_max|top_miss|conv_nmse|ssm_nmse)\s+(\S+)\s+(\S+)\s+(\S+)\s+(ok|FAIL)\s+(\d)\s*$", re.M)


def within(ragged, equal, k):
    """The rule of the prediction: d(ragged) is k times d(equal) or less. When d(equal) is 0, the limit is 0."""
    return ragged <= k * equal


def info_class(ragged_vs_equal, equal):
    return 1 if within(ragged_vs_equal, equal, 1) else 2 if within(ragged_vs_equal, equal, 2) else 3


def parse_input(n):
    text = NOISE_RX.sub("", open(tests(f"input{n}.log")).read())
    lines = text.splitlines()
    head = [m for m in (re.match(r"^lens ([\d ]+)  prefix (\d+)  rows (\d+)$", l) for l in lines) if m][0]
    d = {"n": n, "lens": head.group(1).replace(" ", ","), "prefix": head.group(2), "rows": int(head.group(3)),
         "compare": [], "passes": [], "checks": [], "layers": [], "quantities": {}}
    for l in lines:
        m = COMPARE_RX.match(l)
        if m:
            d["compare"].append(list(m.groups()))
        m = re.match(r"^(equal split|equal split again|ragged split|ragged split again|one sequence a time)\s+passes of the suffixes: (\d+)$", l)
        if m:
            d["passes"].append([m.group(1), int(m.group(2))])
        m = re.match(r"^(ok  |FAIL): (.*)$", l)
        if m:
            d["checks"].append([m.group(2), m.group(1).strip()])
        m = re.match(r"^budget: mean KL of ragged against one at a time (\S+), limit (\S+) \(10 % of (\S+)\)$", l)
        if m:
            d["budget"] = m.groups()
    in_table = False
    for l in lines:
        if l.startswith("layer  ragged-vs-equal"):
            in_table = True
            continue
        m = re.match(r"^\s+(\d+)  (\S+) (\S+)\s+(\S+) (\S+)\s+(\S+) (\S+)\s*$", l) if in_table else None
        if m:
            d["layers"].append(list(m.groups()))
        elif in_table:
            in_table = False
    block = text[text.index("QUANTITY"):]
    for m in QUANTITY_RX.finditer(block):
        d["quantities"][m.group(1)] = {"ragged": m.group(2), "equal": m.group(3), "ragged_vs_equal": m.group(4), "x3": m.group(5), "info": int(m.group(6))}
    assert sorted(d["quantities"]) == sorted(QUANTITIES), n
    d["text"] = BUDGET_TEXT[d["budget"][2]]
    d["logits"] = "last token of each suffix" if d["rows"] == len(d["lens"].split(",")) else f"all {d['rows']} tokens"
    # The numbers of the table must be the numbers of the lines above it (the lines have fewer digits).
    cmp_ = {r[0]: r for r in d["compare"]}
    seq, eq = cmp_["ragged vs one at a time"], cmp_["equal vs one at a time"]
    for key, row, col in (("kl_mean", seq, 1), ("kl_max", seq, 2), ("conv_nmse", seq, 7), ("ssm_nmse", seq, 11)):
        assert abs(float(d["quantities"][key]["ragged"]) / float(row[col]) - 1) < 2e-3, (n, key)
    for key, row, col in (("kl_mean", eq, 1), ("kl_max", eq, 2), ("conv_nmse", eq, 7), ("ssm_nmse", eq, 11)):
        assert abs(float(d["quantities"][key]["equal"]) / float(row[col]) - 1) < 2e-3, (n, key)
    assert float(d["quantities"]["top_miss"]["ragged"]) == int(seq[5]) - int(seq[4]), n
    assert float(d["quantities"]["top_miss"]["equal"]) == int(eq[5]) - int(eq[4]), n
    return d


def logits_tables():
    inputs = [parse_input(n) for n in range(1, 9)]
    rows, cmp_rows, pass_rows, check_rows, layer_rows = [], [], [], [], []
    for d in inputs:
        n = d["n"]
        for q in QUANTITIES:
            v = d["quantities"][q]
            r, e, re_ = float(v["ragged"]), float(v["equal"]), float(v["ragged_vs_equal"])
            x3 = "ok" if within(r, e, 3) else "FAIL"
            cls = info_class(re_, e)
            assert x3 == v["x3"] and cls == v["info"], (n, q, "the rule of the test program differs from the rule of the prediction")
            rows.append([n, d["text"], d["lens"], d["prefix"], d["rows"], d["logits"], q, v["ragged"], v["equal"], v["ragged_vs_equal"],
                         num(r / e, 3) if e else None, v["x3"], x3, v["info"], cls])
        for c in d["compare"]:
            cmp_rows.append([n, d["lens"], d["prefix"], d["rows"], c[0]] + c[1:])
        for p in d["passes"]:
            pass_rows.append([n, d["lens"], p[0], p[1]])
        for text, result in d["checks"]:
            check_rows.append([n, d["lens"], text, result])
        check_rows.append([n, d["lens"], f"budget: mean KL of ragged batching against one at a time {d['budget'][0]}, limit {d['budget'][1]} (10 % of {d['budget'][2]})", "info"])
        for layer in d["layers"]:
            layer_rows.append([n] + layer)
        # the largest NMSE of the layers must be the NMSE of the table
        seq_conv = max(float(l[5]) for l in d["layers"])
        assert abs(seq_conv / float(d["quantities"]["conv_nmse"]["ragged"]) - 1) < 2e-3, n
        seq_ssm = max(float(l[6]) for l in d["layers"])
        assert abs(seq_ssm / float(d["quantities"]["ssm_nmse"]["ragged"]) - 1) < 2e-3, n
    write_tsv("logits-inputs.tsv", [
        "input", "text", "suffix_lengths", "prefix_tokens", "rows", "logits", "quantity", "ragged_vs_seq", "equal_vs_seq", "ragged_vs_eq",
        "ratio_ragged_to_equal", "x3_input_log", "x3_input", "info_class_log", "info_class"], rows)
    write_tsv("logits-compare.tsv", [
        "input", "suffix_lengths", "prefix_tokens", "rows", "comparison", "kl_mean", "kl_max", "max_abs_logit", "same_top_token", "rows_compared",
        "conv_max_abs", "conv_nmse", "conv_identical", "conv_vectors", "ssm_max_abs", "ssm_nmse", "ssm_identical", "ssm_vectors"], cmp_rows)
    write_tsv("logits-passes.tsv", ["input", "suffix_lengths", "decode", "passes_of_the_suffixes"], pass_rows)
    write_tsv("logits-checks.tsv", ["input", "suffix_lengths", "check", "result"], check_rows)
    write_tsv("logits-layers.tsv", [
        "input", "layer", "ragged_vs_equal_conv_nmse", "ragged_vs_equal_ssm_nmse",
        "equal_vs_one_at_a_time_conv_nmse", "equal_vs_one_at_a_time_ssm_nmse",
        "ragged_vs_one_at_a_time_conv_nmse", "ragged_vs_one_at_a_time_ssm_nmse"], layer_rows)

    verdict = []
    # 1. The sum of the 8 inputs: d(ragged) is 1.5 times the sum of d(equal) or less.
    for q in QUANTITIES:
        r = sum(float(d["quantities"][q]["ragged"]) for d in inputs)
        e = sum(float(d["quantities"][q]["equal"]) for d in inputs)
        verdict.append(["Sum of the 8 inputs", QUANTITY_LABELS[q], sci(r), sci(e), num(r / e, 3) if e else None, "1.5",
                        "pass" if within(r, e, 1.5) else "FAIL", ""])
    # 2. Each input: d(ragged) is 3 times d(equal) or less.
    total = []
    for q in QUANTITIES:
        bad = [(d["n"], float(d["quantities"][q]["ragged"]), float(d["quantities"][q]["equal"])) for d in inputs
               if not within(float(d["quantities"][q]["ragged"]), float(d["quantities"][q]["equal"]), 3)]
        total += [(n, q, r, e) for n, r, e in bad]
        note = "; ".join(f"input {n}: ragged {r:.6e}, equal {e:.6e}" + (f", ratio {r / e:.2f}" if e else ", d(equal) is 0") for n, r, e in bad)
        verdict.append(["Each input, 3 times", QUANTITY_LABELS[q], len(bad), None, None, "0 inputs", "pass" if not bad else "FAIL", note])
    note = "; ".join(f"input {n} {QUANTITY_LABELS[q]}: ragged {r:.6e}, equal {e:.6e}" + (f", ratio {r / e:.2f}" if e else ", d(equal) is 0") for n, q, r, e in total)
    verdict.append(["Each input, 3 times", "all 5 quantities: pairs of input and quantity over the limit", len(total), None, None, "0 pairs",
                    "pass" if not total else "FAIL", note])
    # 3. The mean KL divergence of ragged batching is 10 % of the KL budget or less.
    over = []
    for d in inputs:
        kl, limit, budget = float(d["quantities"]["kl_mean"]["ragged"]), float(d["budget"][1]), float(d["budget"][2])
        assert abs(limit / (0.1 * budget) - 1) < 2e-3, d["n"]
        ok = kl <= 0.1 * budget
        assert ok == (["ok"] == [r for t, r in d["checks"] if t.startswith("mean KL of the logits")]), d["n"]
        if not ok:
            over.append(d["n"])
        verdict.append(["10 % of the KL budget", f"mean KL divergence, input {d['n']} ({d['text']})", f"{kl:.6e}", None, None, f"{0.1 * budget:.6f}",
                        "pass" if ok else "FAIL", f"KL budget {budget:.6f}"])
    verdict.append(["10 % of the KL budget", "inputs over the limit", len(over), None, None, "0 inputs", "pass" if not over else "FAIL",
                    ", ".join(map(str, over))])
    # 4. The three checks of the test program that must be true for each input.
    wanted = (("the ragged split decodes all the suffixes in one pass", "Ragged batching decodes the suffixes in one pass"),
              ("the equal split needs more than one pass", "The equal split has more than one pass"),
              ("a second ragged run gives the same logits and the same states bit for bit", "A second run of ragged batching is the same bit for bit"))
    for text, label in wanted:
        ok = [d["n"] for d in inputs if [r for t, r in d["checks"] if t == text] == ["ok"]]
        verdict.append([label, "inputs with the check ok", len(ok), None, None, "8 inputs", "pass" if len(ok) == 8 else "FAIL", ""])
    # the same facts from the numbers of the logs
    one_pass = all(dict((p[0], p[1]) for p in d["passes"])["ragged split"] == 1 for d in inputs)
    more_passes = all(dict((p[0], p[1]) for p in d["passes"])["equal split"] > 1 for d in inputs)
    same = all(c[0] != "ragged again vs ragged" or (c[8] == c[9] and c[12] == c[13] and float(c[1]) == 0 and float(c[2]) == 0 and float(c[7]) == 0 and float(c[11]) == 0)
               for d in inputs for c in d["compare"])
    assert one_pass and more_passes and same
    # 5. Information only: ragged batching against the equal split, 1 time and 2 times d(equal).
    for q in QUANTITIES:
        groups = {1: [], 2: [], 3: []}
        for d in inputs:
            groups[d["quantities"][q]["info"]].append(d["n"])
        verdict.append(["Information only: ragged against equal split", QUANTITY_LABELS[q],
                        f"{len(groups[1])} inputs with a ratio of 1 or less, {len(groups[2])} with a ratio of 2 or less, {len(groups[3])} with a larger ratio",
                        None, None, "none", "info",
                        f"Ratio of 1 or less: inputs {','.join(map(str, groups[1])) or 'none'}. Ratio of 2 or less: inputs {','.join(map(str, groups[2])) or 'none'}. "
                        f"Larger ratio: inputs {','.join(map(str, groups[3])) or 'none'}."])
    write_tsv("logits-verdict.tsv", ["rule", "quantity", "value_ragged", "value_equal", "ratio", "limit", "result", "note"], verdict)
    return inputs


# ---------------------------------------------------------------- the ubatches with suffixes of different lengths
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
    """The pass of the private benchmark with the debug log (0008r-B3-lv5)."""
    j = json.load(open(raw("0008r-B3-lv5.json")))
    n_quality = j["meta"]["cases"]
    write_tsv("ubatches.tsv", UBATCH_HEADER, ubatch_rows(
        "0008r-B3-lv5.server-ubatch-lines.txt",
        [("warmup", 3), ("quality", n_quality), ("seq", n_quality * j["meta"]["passes"]), ("tput", j["tput"]["requests"])]))


# ---------------------------------------------------------------- the speed tests
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


def ab_tables():
    """ab_stats.py without the flag --margin: the prediction declares no equivalence margin."""
    os.makedirs(OUT, exist_ok=True)
    for prefix, name in ((BENCH, "speed-bench.tsv"), (LLB, "speed-llama-bench.tsv")):
        table = run_tool([rel(os.path.join(ANALYSIS, "ab_stats.py")), "raw", "--prefix", prefix, "--a", "A", "--b", "B"])
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
    return [{"run": r[0], "guard_pass": r[10], "errors": r[9]} for r in rows]


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


def speed_data_complete():
    """True when each run of the speed schedule and of the type schedule has its files in raw/."""
    names = [l["run"] for l in parse_schedule(SCHEDULE)] + [l["run"] for l in parse_schedule("0008r-type.txt")]
    return all(os.path.exists(raw(n + ".meta")) for n in names)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    quality = correctness_runs_table()
    dec = decision_tables()
    speed = speed_data_complete()
    if speed:
        runs = runs_table()
        ab = ab_tables()
        block_tables(runs)
        quality += quality_table(runs)
        check_block_stat()
        type_tables()
    stats = answers_tables(quality, {"A": dec["A"], "B": dec["B"], "flips": dec["flips"]})
    ops_tables()
    memcheck_table()
    inputs = logits_tables()
    ubatch_table()
    print("answers: max |dp| B -np 3 vs A -np 1", max(stats[n]["max_dp"] for n in ("B3a-vs-A1", "B3b-vs-A1")),
          "- A -np 3 vs A -np 1", max(stats[n]["max_dp"] for n in ("A3a-vs-A1", "A3b-vs-A1")))
    print("52 decisions: correct A", dec["A"]["correct"], "B", dec["B"]["correct"], "- top flips", dec["compare"]["cross_top_flips"])
    if speed:
        print("speed: answers of the speed runs: self repeat A", ab["self_repeat_a"], "B", ab["self_repeat_b"], "- cross max |dp|", ab["cross_max_abs_dp"])
    else:
        print("speed tables skipped: the files of the speed runs are not all in raw/")
    print("tables written to", os.path.relpath(OUT, ROOT))
