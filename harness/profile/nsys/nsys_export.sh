#!/usr/bin/env bash
# nsys_export.sh - export the .nsys-rep reports of the whole-server nsys runs to SQLite (CPU only, no GPU, no network).
#
# Usage:   bash harness/profile/nsys/nsys_export.sh [--force] RESULTS_DIR [RUN ...]
#   RESULTS_DIR   directory with <run>.nsys-rep (a result directory of the schedule 2-nsys, for example)
#   RUN ...       run names (default: every <run>.nsys-rep in RESULTS_DIR)
#   --force       export again although <run>.sqlite exists and is newer than <run>.nsys-rep
# Output: RESULTS_DIR/<run>.sqlite next to each report (nothing else is written outside the temporary folder).
#
# nsys command line tool: $DMK_NSYS (path of the `nsys` executable), else `nsys` on PATH, else the .deb in $DMK_NSYS_DEB: it is
# unpacked with ar + tar into a temporary folder (honors TMPDIR) that is removed at the end. Nothing is installed system-wide.
# The reports are written by the nsys 2026.3 build on the GPU host; the exporter of the same release reads them.
set -euo pipefail

TMP=""
cleanup() { if [[ -n "$TMP" ]]; then rm -rf "$TMP"; fi; }
trap cleanup EXIT

FORCE=0
if [[ "${1:-}" == "--force" ]]; then FORCE=1; shift; fi
[[ $# -ge 1 ]] || { sed -n '2,12p' "${BASH_SOURCE[0]}" >&2; exit 2; }
RES="$(cd "$1" && pwd)"; shift

find_nsys() { find "$1" -type f -path '*/target-linux-x64/nsys' 2>/dev/null | sort | head -n 1; }

NSYS="${DMK_NSYS:-}"
[[ -n "$NSYS" ]] || NSYS="$(command -v nsys || true)"
if [[ -z "$NSYS" ]]; then
    DEB="${DMK_NSYS_DEB:-}"
    [[ -n "$DEB" ]] || { echo "[export] no nsys found: set DMK_NSYS to the nsys executable, put nsys on PATH, or set DMK_NSYS_DEB to its .deb" >&2; exit 1; }
    [[ -r "$DEB" ]] || { echo "[export] $DEB not readable" >&2; exit 1; }
    TMP="$(mktemp -d)"
    echo "[export] no nsys found, unpacking $DEB -> $TMP"
    ( cd "$TMP" && ar x "$(cd "$(dirname "$DEB")" && pwd)/$(basename "$DEB")" data.tar.gz )
    mkdir -p "$TMP/unpacked"
    tar -xzf "$TMP/data.tar.gz" -C "$TMP/unpacked"
    NSYS="$(find_nsys "$TMP/unpacked")"
fi
[[ -x "$NSYS" ]] || { echo "[export] nsys not executable: $NSYS" >&2; exit 1; }
echo "[export] nsys: $NSYS ($("$NSYS" --version | head -n 1))"

if [[ $# -ge 1 ]]; then
    RUNS=("$@")
else
    RUNS=()
    for f in "$RES"/*.nsys-rep; do
        [[ -e "$f" ]] && RUNS+=("$(basename "$f" .nsys-rep)")
    done
fi
[[ ${#RUNS[@]} -ge 1 ]] || { echo "[export] no .nsys-rep in $RES" >&2; exit 1; }

for run in "${RUNS[@]}"; do
    rep="$RES/$run.nsys-rep"
    out="$RES/$run.sqlite"
    [[ -s "$rep" ]] || { echo "[export] $rep missing or empty" >&2; exit 1; }
    if [[ $FORCE -eq 0 && -s "$out" && "$out" -nt "$rep" ]]; then
        echo "[export] $run: $out is newer than the report, skipped (--force to redo)"
        continue
    fi
    echo "[export] $run: $(du -h "$rep" | cut -f1) report -> $out"
    # the exporter draws a progress bar with carriage returns; keep only non-progress lines
    "$NSYS" export --type sqlite --force-overwrite=true --output "$out" "$rep" 2>&1 \
        | tr '\r' '\n' | grep -v -E '^\[[=]*[0-9]+%' | grep -v '^$' || true
    [[ -s "$out" ]] || { echo "[export] FAILED: $out missing" >&2; exit 1; }
    python3 - "$out" <<'PYEOF'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
def n(t):
    return db.execute("select count(*) from %s" % t).fetchone()[0] if t in tabs else "no table"
print("[export]   %s: kernels %s, runtime API %s, memcpy %s, OSRT_API %s, NVTX %s, %.1f MB" % (
    sys.argv[1].rsplit("/", 1)[-1], n("CUPTI_ACTIVITY_KIND_KERNEL"), n("CUPTI_ACTIVITY_KIND_RUNTIME"), n("CUPTI_ACTIVITY_KIND_MEMCPY"),
    n("OSRT_API"), n("NVTX_EVENTS"), __import__("os").path.getsize(sys.argv[1]) / 1e6))
PYEOF
done
echo "[export] done"
