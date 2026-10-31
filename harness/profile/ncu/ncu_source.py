#!/usr/bin/env python3
"""ncu_source.py - the per-instruction table of one profiled kernel (stall samples and executed instructions of each SASS line).

Usage:
  ncu_source.py --report FILE.ncu-repz --kernel-regex REGEX --launch-skip N --out DIR [--ncu PATH] [--name NAME]

The report must come from a run with `--set full` (the source counters). The script calls the local `ncu` command line
(`--ncu`, else $DMK_NCU, else `ncu` on PATH) as `ncu --import FILE --csv --page source --print-source sass -k regex:REGEX -s N -c 1`.
`-s N` skips the first N kernels of the report that match the regex, so the kernel is the (N+1)-th match.
It writes two tables (TSV, one tab between the fields) to DIR:

  <name>-instructions.tsv  one row per SASS line that has an opcode: address, sass text, opcode, stall samples, executed instructions and the
                           stall samples of each stall reason
  <name>-opcodes.tsv       one row per opcode, sorted by stall samples (all samples), with the shares and the stall reasons per opcode;
                           the last row `TOTAL` sums all rows

The opcode is the first word of the SASS text without a predicate (`@P0 IMMA.16832.S8.S8 ...` -> `IMMA`; the part after the first `.` is dropped).
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile

STALLS = ["barrier", "branch_resolving", "dispatch", "drain", "imc", "lg", "long_sb", "math", "membar", "mio", "misc",
          "no_inst", "not_selected", "selected", "short_sb", "sleep", "tex", "wait"]
PREDICATE = re.compile(r"^@!?U?P[T0-9]+\s+")


def find_ncu(arg):
    for cand in (arg, os.environ.get("DMK_NCU")):
        if cand:
            return cand
    path = shutil.which("ncu")
    if path:
        return path
    sys.exit("ncu_source: no ncu command line tool: use --ncu, set DMK_NCU, or put ncu on PATH")


def num(text):
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return 0.0


def opcode_of(sass):
    text = PREDICATE.sub("", sass.strip())
    if not text or text.startswith(".") or text.endswith(":"):
        return ""
    return text.split()[0].split(".")[0]


def read_source_csv(ncu, report, regex, skip):
    home = tempfile.mkdtemp(prefix="ncu-home-")
    try:
        env = dict(os.environ, HOME=home)
        cmd = [ncu, "--import", report, "--csv", "--page", "source", "--print-source", "sass", "-k", "regex:" + regex, "-s", str(skip), "-c", "1"]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    finally:
        shutil.rmtree(home, ignore_errors=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        sys.exit("ncu_source: ncu failed (exit %d): %s" % (proc.returncode, (proc.stderr or proc.stdout)[:500]))
    rows = list(csv.reader(proc.stdout.splitlines()))
    if len(rows) < 3 or rows[0][0] != "Kernel Name":
        sys.exit("ncu_source: unexpected output of ncu (first line: %r)" % (rows[0] if rows else None))
    return rows[0][1], rows[1], rows[2:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True)
    ap.add_argument("--kernel-regex", required=True)
    ap.add_argument("--launch-skip", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ncu")
    ap.add_argument("--name", default="kernel")
    args = ap.parse_args()

    kernel, header, rows = read_source_csv(find_ncu(args.ncu), args.report, args.kernel_regex, args.launch_skip)
    col = {name: i for i, name in enumerate(header)}
    need = ["Address", "Source", "Warp Stall Sampling (All Samples)", "Warp Stall Sampling (Not-issued Samples)", "Instructions Executed"]
    for name in need + ["stall_" + s for s in STALLS]:
        if name not in col:
            sys.exit("ncu_source: column %r is missing in the source page (the report needs --set full)" % name)

    os.makedirs(args.out, exist_ok=True)
    instr_header = ["address", "sass", "opcode", "stall_samples_all", "stall_samples_not_issued", "instructions_executed"] + ["stall_" + s for s in STALLS]
    ops = {}
    n_rows = 0
    with open(os.path.join(args.out, args.name + "-instructions.tsv"), "w") as f:
        f.write("\t".join(instr_header) + "\n")
        for r in rows:
            op = opcode_of(r[col["Source"]])
            if not op:
                continue
            all_s = num(r[col["Warp Stall Sampling (All Samples)"]])
            not_s = num(r[col["Warp Stall Sampling (Not-issued Samples)"]])
            ex = num(r[col["Instructions Executed"]])
            stalls = [num(r[col["stall_" + s]]) for s in STALLS]
            f.write("\t".join([r[col["Address"]], r[col["Source"]].strip(), op, "%d" % all_s, "%d" % not_s, "%d" % ex] + ["%d" % v for v in stalls]) + "\n")
            n_rows += 1
            acc = ops.setdefault(op, [0, 0.0, 0.0, 0.0] + [0.0] * len(STALLS))
            acc[0] += 1
            acc[1] += all_s
            acc[2] += not_s
            acc[3] += ex
            for i, v in enumerate(stalls):
                acc[4 + i] += v
    tot = [sum(a[i] for a in ops.values()) for i in range(len(next(iter(ops.values()))))] if ops else []
    op_header = ["opcode", "sass_lines", "stall_samples_all", "stall_samples_share_pct", "stall_samples_not_issued", "instructions_executed",
                 "instructions_executed_share_pct"] + ["stall_" + s for s in STALLS]
    with open(os.path.join(args.out, args.name + "-opcodes.tsv"), "w") as f:
        f.write("\t".join(op_header) + "\n")
        for op, a in sorted(ops.items(), key=lambda kv: (-kv[1][1], kv[0])):
            f.write("\t".join([op, "%d" % a[0], "%d" % a[1], "%.2f" % (100 * a[1] / tot[1] if tot[1] else 0), "%d" % a[2], "%d" % a[3],
                               "%.2f" % (100 * a[3] / tot[3] if tot[3] else 0)] + ["%d" % v for v in a[4:]]) + "\n")
        if tot:
            f.write("\t".join(["TOTAL", "%d" % tot[0], "%d" % tot[1], "100.00", "%d" % tot[2], "%d" % tot[3], "100.00"] + ["%d" % v for v in tot[4:]]) + "\n")
    imma = ops.get("IMMA")
    print("kernel: %s" % kernel[:120])
    print("sass lines with an opcode: %d; opcodes: %d; stall samples: %d; instructions executed: %d" % (n_rows, len(ops), tot[1] if tot else 0, tot[3] if tot else 0))
    if imma and imma[3]:
        print("executed instructions per IMMA: %.2f" % (tot[3] / imma[3]))


if __name__ == "__main__":
    main()
