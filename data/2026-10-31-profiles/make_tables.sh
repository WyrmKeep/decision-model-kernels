#!/usr/bin/env bash
# make_tables.sh - make all derived tables and reports of this data set from the files in raw/ and, when you have them, from the capture files.
#
# Usage:   bash make_tables.sh [--captures DIR] [--census DIR] [--out DIR]
#   --captures DIR  folder with the capture files, in the layout of raw/:
#                     DIR/2-nsys/*.nsys-rep   DIR/3-leads/*.nsys-rep   DIR/4-ncu-smoke|4-ncu-core|4-ncu-extra/*.ncu-repz
#                   The repository does not have these files (see "Files not added" in README.md). Without this option the script
#                   makes only the tables of the decomposition (decomp, summary) and of the lead #30779 (leads-childpass). Use --out then.
#   --census DIR    folder of the census data set. Default: ../2026-10-30-shape-census
#   --out DIR       write DIR/tables and DIR/reports instead of tables/ and reports/ in this folder (to compare, for example
#                   `diff -r DIR/tables tables`). The raw files are always read from this folder.
#
# Tools: python3 (standard library). With --captures the script also needs the Nsight Systems command line tool (set DMK_NSYS or put nsys
# on PATH; DMK_NSYS_DEB can name a .deb to unpack) and the Nsight Compute command line tool (set DMK_NCU or put ncu on PATH).
# The script runs harness/profile/analyze_all.sh and then these two programs:
#   harness/profile/ncu/ncu_source.py   the per-instruction table of one MMQ kernel (needs the report 4-ncu-extra/p3-ncu-extra-long-full.ncu-repz)
#   make_summary.py                     the tables of the notebook entry (tables/summary/)
# The exit code is not 0 when a step of analyze_all.sh fails. A step that has no input is skipped, and the script says so.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROF="$HERE/../../harness/profile"
OUT="$HERE"
CAP=""
CENSUS="$HERE/../2026-10-30-shape-census"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --captures) CAP="$2"; shift 2 ;;
        --census) CENSUS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -z "$CAP" || -d "$CAP" ]] || { echo "make_tables.sh: --captures $CAP is not a folder" >&2; exit 2; }
[[ -d "$CENSUS" ]] || { echo "make_tables.sh: --census $CENSUS is not a folder" >&2; exit 2; }
if [[ -z "$CAP" && "$OUT" == "$HERE" ]]; then
    echo "make_tables.sh: without --captures the script cannot make the tables of Nsight Systems and Nsight Compute, and it would delete them in this folder. Use --out DIR." >&2
    exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# One results folder with the layout that analyze_all.sh reads: the raw files and the capture files side by side.
for s in 1-decomp 2-nsys 3-leads 4-ncu-smoke 4-ncu-core 4-ncu-extra; do
    mkdir -p "$WORK/res/$s"
    ln -s "$HERE/raw/$s"/* "$WORK/res/$s/"
    if [[ -n "$CAP" && -d "$CAP/$s" ]]; then
        for f in "$CAP/$s"/*.nsys-rep "$CAP/$s"/*.ncu-repz; do
            [[ -e "$f" ]] && ln -s "$(cd "$CAP/$s" && pwd)/$(basename "$f")" "$WORK/res/$s/"
        done
    fi
done

rc=0
bash "$PROF/analyze_all.sh" --census "$CENSUS" --no-mmq-launches "$WORK/res" "$WORK/out" > "$WORK/analyze_all.stdout" 2>&1 || rc=$?
grep -E '^== (done|skipped|failed)' -A3 "$WORK/analyze_all.stdout" | grep -v '^--' || true

rm -rf "$OUT/tables" "$OUT/reports"
for part in decomp nsys leads-nsys leads-kvwin leads-childpass ncu; do
    [[ -d "$WORK/out/$part" ]] || continue
    mkdir -p "$OUT/tables/$part"
    for f in "$WORK/out/$part"/*.tsv; do [[ -e "$f" ]] && cp "$f" "$OUT/tables/$part/"; done
    # A report names the work folder of this script and the run time of the analysis. Both change with each run, so the copy removes them.
    for f in "$WORK/out/$part"/*.txt; do
        [[ -e "$f" ]] || continue
        mkdir -p "$OUT/reports/$part"
        sed -e "s#$WORK/res/#raw/#g" -e "s#$WORK/out/#reports/#g" -e '/^wall time of this analysis: /d' "$f" > "$OUT/reports/$part/$(basename "$f")"
    done
done
# analyze_childpass.py writes no report file. The text that it prints is the report.
if [[ -s "$WORK/out/leads-childpass.out.txt" ]]; then
    mkdir -p "$OUT/reports/leads-childpass"
    sed -e "s#$WORK/res/#raw/#g" -e "s#$WORK/out/#reports/#g" "$WORK/out/leads-childpass.out.txt" > "$OUT/reports/leads-childpass/report.txt"
fi
mkdir -p "$OUT/tables"

EXTRA="$WORK/res/4-ncu-extra/p3-ncu-extra-long-full.ncu-repz"
if [[ -e "$EXTRA" ]]; then
    python3 "$PROF/ncu/ncu_source.py" --report "$EXTRA" --kernel-regex mul_mat_q --launch-skip 7 --name ffn_up_q4_0_N2048 \
        --out "$OUT/tables/ncu" || rc=1
else
    echo "make_tables.sh: skipped ncu_source.py, no p3-ncu-extra-long-full.ncu-repz (use --captures)"
fi

python3 "$HERE/make_summary.py" --tables "$OUT/tables" || rc=1
exit "$rc"
