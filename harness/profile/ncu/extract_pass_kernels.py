#!/usr/bin/env python3
"""Cut the ordered kernel launches of a few traced model passes out of the three old census nsys exports.

Input : {long,medium,short}.sqlite in --legacy DIR (default $DMK_LEGACY_TRACES; nsys exports of the old captures, read only; make them
        from .nsys-rep files with harness/profile/nsys/nsys_export.sh) and
        trace-passes.tsv of the census data set ($DMK_CENSUS_DATA, census_lib.data_file; pass start times and kernel counts).
Output: data/pass-kernels.tsv.gz, one row per kernel launch of the selected passes, in launch order.
        The planner (plan_ncu.py) reads only this file, so it does not need the 100 MB sqlite files.
A pass is the set of kernels that start at or after its start_ns and before the next pass of the same trace.
The kernel count of every selected pass is checked against n_kernels of trace-passes.tsv.

usage: extract_pass_kernels.py [--legacy DIR] [--out FILE]
"""
import argparse
import csv
import gzip
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "census"))
import census_lib  # noqa: E402

# (trace, pass_idx_in_trace) -> model id. These are the traced passes that stand in for the passes of a fresh server's first
# request (see plan_ncu.py MODEL_PASSES for the mapping and the reasons).
SELECT = {
    ("long", 0): "long-i0-N2048",
    ("long", 1): "long-i1-N1172",
    ("long", 4): "long-i4-N73",
    ("long", 8): "long-i8-N94",
    ("long", 12): "long-i12-N129",
    ("long", 13): "long-i13-N8",
    ("long", 14): "long-i14-N26",
    ("medium", 4): "medium-i4-N1288",
    ("medium", 5): "medium-i5-N129",
    ("medium", 6): "medium-i6-N8",
    ("medium", 7): "medium-i7-N26",
    ("short", 1): "short-i1-N120",
    ("short", 6): "short-i6-N86",
}
COLS = ["model", "pos", "func", "gx", "gy", "gz", "bx", "by", "bz", "regs", "smem_static", "smem_dyn", "dur_ns", "demangled"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy", default=os.environ.get("DMK_LEGACY_TRACES", ""), help="folder with long.sqlite, medium.sqlite, short.sqlite (default $DMK_LEGACY_TRACES)")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "pass-kernels.tsv.gz"))
    a = ap.parse_args()
    if not a.legacy:
        sys.exit("extract_pass_kernels: no legacy exports: use --legacy DIR or set DMK_LEGACY_TRACES")
    passes = {}
    with open(census_lib.data_file("trace-passes.tsv"), newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["row_kind"] == "pass":
                passes.setdefault(r["trace"], []).append(r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n_rows = 0
    with gzip.open(a.out, "wt", newline="", compresslevel=9) as out:
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        w.writerow(COLS)
        for trace in ("long", "medium", "short"):
            db = sqlite3.connect(f"file:{os.path.join(a.legacy, trace + '.sqlite')}?mode=ro", uri=True)
            rows = db.execute(
                "select k.start, s.value, d.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, k.registersPerThread, "
                "k.staticSharedMemory, k.dynamicSharedMemory, k.end - k.start "
                "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName join StringIds d on d.id = k.demangledName "
                "order by k.start, k.end").fetchall()
            starts = [int(p["start_ns"]) for p in passes[trace]]
            idx_of = {int(p["pass_idx_in_trace"]): i for i, p in enumerate(passes[trace])}
            for (tr, pidx), model in SELECT.items():
                if tr != trace:
                    continue
                i = idx_of[pidx]
                lo = starts[i]
                hi = starts[i + 1] if i + 1 < len(starts) else float("inf")
                ks = [r for r in rows if lo <= r[0] < hi]
                want = int(passes[trace][i]["n_kernels"])
                if len(ks) != want:
                    sys.exit(f"{model}: sliced {len(ks)} kernels, trace-passes.tsv says n_kernels={want}")
                for pos, r in enumerate(ks):
                    w.writerow([model, pos] + list(r[1:2]) + list(r[3:13]) + [r[2]])
                    n_rows += 1
                print(f"{model}: {len(ks)} kernels (trace-passes n_kernels {want}, n_tokens {passes[trace][i]['n_tokens']})")
    print(f"wrote {a.out}: {n_rows} rows, {os.path.getsize(a.out)} bytes")


if __name__ == "__main__":
    main()
