#!/usr/bin/env python3
"""Make the derived tables of the data set 2026-10-30-test-backend-ops from the files in raw/.

Usage: python3 data/2026-10-30-test-backend-ops/make_tables.py
The script writes into the folder tables/ next to it. It reads the two logs of
test-backend-ops (tbo-D.txt and tbo-D-builtin.txt) and the four files of exported
operations (ops-*.txt). It uses only the Python standard library. It changes no
file in raw/. It stops with an error when a count in a log differs from its
summary line, or when the merged file is not the union of the three exports.
"""
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "tables")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# A test line: an optional diagnostic "[OP] text" and three spaces, two spaces of
# indent for a line without a diagnostic, then OP(arguments): STATUS.
TEST = re.compile(
    r"^(?:\[(?P<tag>[A-Z0-9_]+)\] (?P<msg>.*?)   )?(?:  )?"
    r"(?P<op>[A-Z][A-Z0-9_]*)\((?P<args>.*)\): (?P<status>OK|FAIL|not supported)(?: \[\S+\])?\s*$"
)
SUMMARY = re.compile(r"^\s*(\d+)/(\d+) tests passed")

LOGS = [
    ("model-ops", "tbo-D.txt"),
    ("built-in", "tbo-D-builtin.txt"),
]
EXPORTS = [
    ("ops-ub2048.txt", "2048"),
    ("ops-ub160.txt", "160"),
    ("ops-ub48.txt", "48"),
    ("ops-all.txt", "all"),
]


def write_tsv(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            assert len(r) == len(header), (name, r)
            f.write("\t".join(str(v) for v in r) + "\n")


def parse(log):
    """Return the test lines of one log as dicts, and the numbers of its summary line."""
    tests, summary = [], None
    for line in open(os.path.join(RAW, log), errors="replace"):
        line = ANSI.sub("", line.rstrip("\n"))
        m = SUMMARY.match(line)
        if m:
            summary = (int(m.group(1)), int(m.group(2)))
            continue
        m = TEST.match(line)
        if m and not line.startswith("Failing tests"):
            tests.append(m.groupdict())
    return tests, summary


def main():
    results, failures, ok_by_log = [], [], {}
    for label, log in LOGS:
        tests, summary = parse(log)
        count = collections.defaultdict(collections.Counter)
        for t in tests:
            count[t["op"]][t["status"]] += 1
            if t["status"] == "FAIL":
                name = re.search(r"name=([^,]*)", t["args"])
                ne = re.search(r"ne=\[([^\]]*)\]", t["args"])
                failures.append((label, t["op"], name.group(1) if name else "", ne.group(1) if ne else "",
                                 t["msg"] or ""))
        for op in sorted(count):
            c = count[op]
            results.append((label, op, c["OK"], c["FAIL"], c["not supported"]))
        ok = sum(count[op]["OK"] for op in count)
        tested = ok + sum(count[op]["FAIL"] for op in count)
        if summary != (ok, tested):
            sys.exit(f"{log}: the summary line {summary} differs from the count {(ok, tested)}")
        ok_by_log[label] = (ok, tested)

    exports, union = [], set()
    for name, ubatch in EXPORTS:
        lines = {l for l in open(os.path.join(RAW, name)).read().split("\n") if l}
        exports.append((name, ubatch, len(lines)))
        if ubatch != "all":
            union |= lines
        elif lines != union:
            sys.exit("ops-all.txt is not the union of the three exports")

    write_tsv("export.tsv", ("file", "ubatch_tokens", "unique_ops"), exports)
    write_tsv("results.tsv", ("log", "op", "ok", "failed", "not_supported"), results)
    write_tsv("failures.tsv", ("log", "op", "name", "ne", "diagnostic"), failures)
    for label, (ok, tested) in ok_by_log.items():
        print(f"{label}: {ok} of {tested} tests pass")
    print(f"failing tests: {len(failures)}")
    print(f"unique ops in the merged file: {exports[-1][2]}")


if __name__ == "__main__":
    main()
