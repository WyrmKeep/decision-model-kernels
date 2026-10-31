#!/usr/bin/env bash
# run_tests.sh - tests of the nsys package; CPU only, no GPU, no server, no network.
#   1. legacy mode reproduces the census tables trace-{kernels,passes,mmq-launches}.tsv byte for byte from the old exports
#   2. extra/prof/nsys_client.py against a fake HTTP server (all kinds + replay)
#   3. whole-server mode (analyze) and overhead on synthetic data built from the three real legacy captures
#   4. the time mapping client stamps -> nsys timeline, measured with a CPU-only nsys run
#   5. SIGTERM to nsys (as a plain process and as PID 1 of a PID namespace) writes the report and ends the profiled child
# Usage:  bash harness/profile/nsys/test/run_tests.sh      (any cwd; about 35 s)
# Environment (a step that needs a missing item prints SKIP and does not count as a failure):
#   DMK_LEGACY_TRACES  folder with the three old captures long, medium, short as .sqlite (or as .nsys-rep, then DMK_NSYS or nsys on PATH exports them)  [steps 1, 3]
#   DMK_CENSUS_DATA    census data set (default: data/2026-10-30-shape-census of this repository), holds the census tables             [steps 1, 3]
#   DMK_NSYS           path of the nsys command line tool (else `nsys` on PATH)                                                       [steps 4, 5]
#   DMK_BENCH_DIR      folder with cases.json (default: bench/ of this repository)                                                    [step 2]
#   DMK_TEST_OUT       keep the logs and small reference outputs in this folder (default: a temporary folder that is removed at the end)
# All files of the test are written to temporary folders (TMPDIR is honored); nothing is written into the repository.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS="$(cd "$HERE/.." && pwd)"
PROFILE="$(cd "$HERE/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
OUT="${DMK_TEST_OUT:-$TMP/out}"
SYN="$TMP/synth"
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1
rc=0
skipped=()

NSYS_TOOL="${DMK_NSYS:-}"
[[ -n "$NSYS_TOOL" ]] || NSYS_TOOL="$(command -v nsys || true)"

census_table() {   # path of a census table in the data set; empty when the data set does not have it
    python3 - "$PROFILE/census" "$1" <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, sys.argv[1])
import census_lib
print(census_lib.data_file(sys.argv[2]))
PYEOF
}

# the three old captures as .sqlite files in $TMP/legacy: from DMK_LEGACY_TRACES (.sqlite as they are, .nsys-rep exported with nsys)
LEG=""
legacy_ready() {
    [[ -n "$LEG" ]] && return 0
    local d="${DMK_LEGACY_TRACES:-}" n
    [[ -n "$d" && -d "$d" ]] || return 1
    mkdir -p "$TMP/legacy"
    for n in short medium long; do
        if [[ -s "$d/$n.sqlite" ]]; then
            ln -s "$(cd "$d" && pwd)/$n.sqlite" "$TMP/legacy/$n.sqlite"
        elif [[ -s "$d/$n.nsys-rep" && -n "$NSYS_TOOL" ]]; then
            ln -s "$(cd "$d" && pwd)/$n.nsys-rep" "$TMP/legacy/$n.nsys-rep"
        else
            return 1
        fi
    done
    if ls "$TMP/legacy"/*.nsys-rep >/dev/null 2>&1; then
        DMK_NSYS="$NSYS_TOOL" bash "$NS/nsys_export.sh" "$TMP/legacy" > "$OUT/export-legacy.txt" 2>&1 || return 1
    fi
    LEG="$TMP/legacy"
    return 0
}

echo "== 1. legacy mode: byte-exact reproduction of the census tables"
if ! legacy_ready; then
    echo "SKIP: DMK_LEGACY_TRACES not set or without long, medium, short as .sqlite (or .nsys-rep with nsys available)"; skipped+=(1)
elif [[ -z "$(census_table trace-kernels.tsv)" || -z "$(census_table trace-passes.tsv)" || -z "$(census_table trace-mmq-launches.tsv)" ]]; then
    echo "SKIP: the census tables trace-{kernels,passes,mmq-launches}.tsv are not in the census data set (DMK_CENSUS_DATA)"; skipped+=(1)
else
    LEGOUT="$TMP/legacy-out"
    python3 "$NS/nsys_census.py" legacy --in "$LEG" --out "$LEGOUT" > "$OUT/test1-legacy.stdout.txt" 2>&1 || { echo "FAIL: legacy mode exited non-zero"; rc=1; }
    {
        for f in trace-kernels.tsv trace-passes.tsv trace-mmq-launches.tsv; do
            ref="$(census_table "$f")"
            if cmp -s "$LEGOUT/$f" "$ref"; then echo "PASS legacy output $f is byte-identical to the census table ($(wc -c < "$LEGOUT/$f") bytes, sha256 $(sha256sum < "$LEGOUT/$f" | cut -c1-16))"
            else echo "FAIL legacy output $f differs from the census table"; fi
        done
        grep -E "VALIDATION PROBLEMS" "$OUT/test1-legacy.stdout.txt" | sed 's/^/legacy report: /'
    } | tee "$OUT/test1-legacy.txt"
    grep -q FAIL "$OUT/test1-legacy.txt" && rc=1
fi

echo; echo "== 2. nsys_client.py against the fake server"
python3 "$HERE/test_client.py" --out "$OUT/client" > "$OUT/test2-client.txt" 2>&1 || rc=1
grep -E "FAIL|client test" "$OUT/test2-client.txt" | tail -5; echo "($(grep -c '^PASS' "$OUT/test2-client.txt") checks passed)"

echo; echo "== 3. whole-server analysis and overhead on synthetic data"
if ! legacy_ready; then
    echo "SKIP: DMK_LEGACY_TRACES not set or without long, medium, short as .sqlite (or .nsys-rep with nsys available)"; skipped+=(3)
elif [[ -z "$(census_table trace-passes.tsv)" ]]; then
    echo "SKIP: the census tables are not in the census data set (DMK_CENSUS_DATA)"; skipped+=(3)
else
    python3 "$HERE/make_synthetic.py" --legacy "$LEG" --work "$SYN" > "$OUT/test3-make.txt" 2>&1 || rc=1
    python3 "$HERE/check_synthetic.py" --work "$SYN" --out "$OUT" > "$OUT/test3-synthetic.txt" 2>&1 || rc=1
    grep -E "FAIL|synthetic checks|explained" "$OUT/test3-synthetic.txt" | cut -c1-240; echo "($(grep -c '^PASS' "$OUT/test3-synthetic.txt") checks passed)"
fi

echo; echo "== 4. time mapping client stamps -> nsys timeline (CPU-only nsys run)"
python3 "$HERE/clock_check.py" ${NSYS_TOOL:+--nsys "$NSYS_TOOL"} > "$OUT/test4-clock.txt" 2>&1; crc=$?
grep -E "residual|clock mapping|SKIP" "$OUT/test4-clock.txt" | cut -c1-200
[[ $crc -eq 1 ]] && rc=1
[[ $crc -eq 2 ]] && skipped+=(4)

echo; echo "== 5. SIGTERM to nsys (ordinary process and PID 1 of a PID namespace): report written, profiled child ends"
bash "$HERE/sigterm_check.sh" ${NSYS_TOOL:+"$NSYS_TOOL"} > "$OUT/test5-sigterm.txt" 2>&1; src=$?
grep -E "launcher exit|PASS|FAIL|SKIP" "$OUT/test5-sigterm.txt" | cut -c1-200
[[ $src -eq 1 ]] && rc=1
[[ $src -eq 2 ]] && skipped+=(5)

echo
if [[ $rc -ne 0 ]]; then
    echo "SOME TESTS FAILED"
elif [[ ${#skipped[@]} -gt 0 ]]; then
    echo "NO FAILURE; steps skipped: ${skipped[*]} (see the SKIP lines)"
else
    echo "ALL TESTS PASSED"
fi
exit $rc
