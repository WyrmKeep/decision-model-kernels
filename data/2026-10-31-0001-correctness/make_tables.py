#!/usr/bin/env python3
"""Make the derived tables of the data set 2026-10-31-0001-correctness from the files in raw/.

Usage: python3 data/2026-10-31-0001-correctness/make_tables.py
The script writes into the folder tables/ next to it. It reads the logs of
test-backend-ops and compute-sanitizer in raw/. It also reads the log of build D from
the data set 2026-10-30-test-backend-ops, to compare the results of the two builds.
It uses only the Python standard library. It changes no file in raw/. It stops with
an error when a count of tests in a log differs from the summary line of the log.
"""
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "tables")
LOG_D = os.path.join(os.path.dirname(HERE), "2026-10-30-test-backend-ops", "raw", "tbo-D.txt")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# A test line: an optional diagnostic "[OP] text" and three spaces, two spaces of
# indent for a line without a diagnostic, then OP(arguments): STATUS.
TEST = re.compile(
    r"^(?:\[(?P<tag>[A-Z0-9_]+)\] (?P<msg>.*?)   )?(?:  )?"
    r"(?P<op>[A-Z][A-Z0-9_]*)\((?P<args>.*)\): (?P<status>OK|FAIL|not supported)(?: \[\S+\])?\s*$"
)
SUMMARY = re.compile(r"^\s*(\d+)/(\d+) tests passed")

# log, build, tool, selection of tests
LOGS = [
    ("tbo-L-glu.txt", "L", "none", "-o GLU_MUL_MAT,SWIGLU_MUL_MAT_ID,SWIGLU_MUL_MAT_ID_REUSE"),
    ("tbo-L.txt", "L", "none", "--test-file ops-all.txt"),
    ("san-memcheck-L.txt", "L", "memcheck", "-o GLU_MUL_MAT,SWIGLU_MUL_MAT_ID,SWIGLU_MUL_MAT_ID_REUSE"),
    ("san-initcheck-L.txt", "L", "initcheck", "-o GLU_MUL_MAT,SWIGLU_MUL_MAT_ID,SWIGLU_MUL_MAT_ID_REUSE"),
    ("san-synccheck-L.txt", "L", "synccheck", "-o GLU_MUL_MAT,SWIGLU_MUL_MAT_ID,SWIGLU_MUL_MAT_ID_REUSE"),
    ("san-racecheck-L.txt", "L", "racecheck", "-o GLU_MUL_MAT,SWIGLU_MUL_MAT_ID,SWIGLU_MUL_MAT_ID_REUSE"),
    ("san-initcheck-quantize-L.txt", "L", "initcheck, only kernels with quantize_mmq_q8_1 in the name",
     "-o GLU_MUL_MAT,SWIGLU_MUL_MAT_ID,SWIGLU_MUL_MAT_ID_REUSE"),
    ("san-initcheck-nofixup-L.txt", "L", "initcheck, no kernel with mul_mat_q in the name",
     "-o GLU_MUL_MAT,SWIGLU_MUL_MAT_ID,SWIGLU_MUL_MAT_ID_REUSE"),
    ("initmm-b12004-sm86.txt", "D", "initcheck", "-o MUL_MAT -p q4_0"),
    ("initmm-lead-30489.txt", "L", "initcheck", "-o MUL_MAT -p q4_0"),
]
SAN_RUNS = [
    ("memcheck", "lead30489-san-memcheck-L"),
    ("initcheck", "lead30489-san-initcheck-L"),
    ("synccheck", "lead30489-san-synccheck-L"),
    ("racecheck", "lead30489-san-racecheck-L"),
]


def write_tsv(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            assert len(r) == len(header), (name, r)
            f.write("\t".join("" if v is None else str(v) for v in r) + "\n")


def clean_lines(path):
    for line in open(path, errors="replace"):
        yield ANSI.sub("", line.rstrip("\n"))


def parse(path):
    """Return the test lines of one log as dicts, the numbers of its summary line, and the text."""
    tests, summary, text = [], None, []
    for line in clean_lines(path):
        text.append(line)
        m = SUMMARY.match(line)
        if m:
            summary = (int(m.group(1)), int(m.group(2)))
            continue
        m = TEST.match(line)
        if m and not line.startswith("Failing tests"):
            tests.append(m.groupdict())
    return tests, summary, text


def sanitizer_errors(text):
    """The number of errors in the summary line of compute-sanitizer, or None for no sanitizer."""
    body = "\n".join(text)
    m = re.search(r"ERROR SUMMARY: (\d+) errors", body)
    if m:
        return int(m.group(1))
    m = re.search(r"RACECHECK SUMMARY: \d+ hazards displayed \((\d+) errors, \d+ warnings\)", body)
    if m:
        return int(m.group(1))
    return None


def main():
    rows, ops_rows = [], []
    for log, build, tool, selection in LOGS:
        tests, summary, text = parse(os.path.join(RAW, log))
        count = collections.Counter(t["status"] for t in tests)
        ok, tested = count["OK"], count["OK"] + count["FAIL"]
        if summary != (ok, tested):
            sys.exit(f"{log}: the summary line {summary} differs from the count {(ok, tested)}")
        rows.append((log, build, tool, selection, ok, tested, sanitizer_errors(text)))
        if log == "tbo-L-glu.txt":
            per_op = collections.defaultdict(collections.Counter)
            for t in tests:
                per_op[t["op"]][t["status"]] += 1
            for op in sorted(per_op):
                c = per_op[op]
                ops_rows.append((op, c["OK"], c["FAIL"], c["not supported"]))
    write_tsv("tests.tsv", ("log", "build", "sanitizer", "selection", "tests_ok", "tests_run", "sanitizer_errors"), rows)
    write_tsv("glu-ops.tsv", ("op", "ok", "failed", "not_supported"), ops_rows)

    # The results of the 187 operations of the model graph on build D and on build L.
    def results(path):
        tests = parse(path)[0]
        return {(t["op"], t["args"]): t["status"] for t in tests}

    d, l = results(LOG_D), results(os.path.join(RAW, "tbo-L.txt"))
    different = sorted(k for k in set(d) | set(l) if d.get(k) != l.get(k))
    write_tsv(
        "compare-model-ops.tsv",
        ("tests_d", "tests_l", "same_set_of_tests", "tests_with_different_result", "failed_d", "failed_l"),
        [(len(d), len(l), set(d) == set(l), len(different),
          sum(v == "FAIL" for v in d.values()), sum(v == "FAIL" for v in l.values()))],
    )

    # compute-sanitizer on the model: one row for each tool, as in the data set of window 1.
    san = []
    for tool, run in SAN_RUNS:
        out = open(os.path.join(RAW, run + ".out"), errors="replace").read()
        err = open(os.path.join(RAW, run + ".err"), errors="replace").read()
        m = re.search(r"ERROR SUMMARY: (\d+) errors", out)
        if m:
            errors, hazards, warnings = m.group(1), None, None
        else:
            m = re.search(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)", out)
            hazards, errors, warnings = m.group(1), m.group(2), m.group(3)
        seconds = re.search(r"([\d.]+) seconds per pass", err).group(1)
        ppl = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", err + out) or re.search(r"\[1\]([\d.]+),", out)
        san.append((tool, errors, hazards, warnings, seconds, ppl.group(1)))
    write_tsv("sanitizers-model.tsv", ("tool", "errors", "hazards_displayed", "warnings", "seconds_per_pass", "ppl"), san)

    for r in rows:
        print(*r, sep=" | ")
    print("model ops: different results between D and L:", len(different))
    print("model sanitizers:", *san, sep="\n  ")


if __name__ == "__main__":
    main()
