#!/usr/bin/env bash
# sigterm_check.sh - checks with a CPU-only nsys run (no GPU) what `docker stop` does to a wrapped server:
# SIGTERM goes to PID 1 of the container = nsys; the report must be written and the profiled child must end.
# Variant 1: nsys as an ordinary process.  Variant 2: nsys as PID 1 of a new PID namespace (unshare -U -r --pid --fork --mount-proc),
# where signals from outside are only delivered if the process has installed a handler - the same rule as in a container.
# Usage: bash sigterm_check.sh [NSYS]      NSYS = path of the nsys executable (else $DMK_NSYS, else `nsys` on PATH)
#        exit 0 = report written in both variants, 2 = nsys not found (skipped)
set -uo pipefail
NSYS="${1:-${DMK_NSYS:-}}"
[[ -n "$NSYS" ]] || NSYS="$(command -v nsys || true)"
[[ -n "$NSYS" && -x "$NSYS" ]] || { echo "SKIP: no local nsys (give the path, set DMK_NSYS, or put nsys on PATH)"; exit 2; }
D="$(mktemp -d)"
trap 'rm -rf "$D"; pkill -f "sleep 1234" 2>/dev/null || true' EXIT
FLAGS=(profile --trace=cuda,osrt,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true)
rc=0

run_variant() {   # $1 = label, rest = launcher prefix
    local label="$1"; shift
    local out="$D/$label"
    "$@" "$NSYS" "${FLAGS[@]}" -o "$out" sleep 1234 > "$D/$label.log" 2>&1 &
    local lp=$!
    sleep 4
    local target=$lp
    if [[ "$label" == pid1 ]]; then target="$(pgrep -P "$lp" | head -n 1)"; fi   # the namespace's init, as seen from outside
    echo "[$label] sending SIGTERM to pid $target ($(ps -o comm= -p "$target" 2>/dev/null))"
    local t0; t0=$(date +%s.%N)
    kill -TERM "$target"
    wait "$lp"; local status=$?
    local dt; dt=$(echo "$(date +%s.%N) - $t0" | bc)
    local child="gone"; pgrep -f "sleep 1234" >/dev/null && child="STILL RUNNING"
    local rep="missing"; [[ -s "$out.nsys-rep" ]] && rep="$(stat -c %s "$out.nsys-rep") bytes"
    printf "[%s] launcher exit status %s after %.1f s; report: %s; profiled child: %s\n" "$label" "$status" "$dt" "$rep" "$child"
    if [[ "$rep" == missing || "$child" != gone ]]; then echo "[$label] FAIL"; rc=1; else echo "[$label] PASS"; fi
}

run_variant plain env
if unshare -U -r --pid --fork --mount-proc true 2>/dev/null; then
    run_variant pid1 unshare -U -r --pid --fork --mount-proc --
else
    echo "[pid1] SKIP: unshare of user + pid namespaces not possible here"
fi
exit $rc
