#!/usr/bin/env bash
# trace_export.sh - export the three .nsys-rep captures (long, medium, short) to SQLite with the Nsight Systems command line tool.
#
# Usage:   bash trace_export.sh [--in DIR] [--out DIR]       (CPU only, no GPU, no network, no sudo, nothing is installed)
#   --in DIR    the folder with long.nsys-rep, medium.nsys-rep and short.nsys-rep (default: $DMK_LEGACY_TRACES)
#   --out DIR   the folder for long.sqlite, medium.sqlite and short.sqlite (default: the --in folder). trace_census.py reads them.
#
# Environment:
#   DMK_NSYS       the nsys command line tool (the file target-linux-x64/nsys of Nsight Systems; the capture of the census was
#                  exported with version 2026.3.2). Required unless DMK_NSYS_DEB is set.
#   DMK_NSYS_DEB   optional: a Nsight Systems .deb file. If DMK_NSYS is not set, the .deb is unpacked into <out>/nsys_unpacked
#                  (created once, reused afterwards) and the nsys of the unpacked tree is used. A .deb is an ar archive:
#                  `ar` + `tar` do what `dpkg -x` does, so a host without dpkg works.
# Existing .sqlite files in the output folder are overwritten (--force-overwrite=true). The .nsys-rep files are never changed.
set -euo pipefail

IN="${DMK_LEGACY_TRACES:-}"
OUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --in) IN="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$IN" ]] || { echo "set --in DIR (or DMK_LEGACY_TRACES) to the folder with the .nsys-rep captures" >&2; exit 2; }
OUT="${OUT:-$IN}"
mkdir -p "$OUT"

NSYS="${DMK_NSYS:-}"
if [[ -z "$NSYS" ]]; then
    DEB="${DMK_NSYS_DEB:-}"
    [[ -n "$DEB" ]] || { echo "set DMK_NSYS to the nsys command line tool (or DMK_NSYS_DEB to a Nsight Systems .deb)" >&2; exit 2; }
    UNPACK="$OUT/nsys_unpacked"
    find_nsys() { find "$UNPACK" -type f -path '*/target-linux-x64/nsys' 2>/dev/null | sort | head -n 1; }
    NSYS="$(find_nsys || true)"
    if [[ -z "$NSYS" ]]; then
        echo "[export] unpacking $DEB -> $UNPACK"
        tmp="$(mktemp -d "$OUT/deb.XXXXXX")"
        trap 'rm -rf "$tmp"' EXIT
        ( cd "$tmp" && ar x "$DEB" data.tar.gz )
        mkdir -p "$UNPACK"
        tar -xzf "$tmp/data.tar.gz" -C "$UNPACK"
        rm -rf "$tmp"
        NSYS="$(find_nsys)"
    else
        echo "[export] nsys already unpacked"
    fi
fi
[[ -x "$NSYS" ]] || { echo "nsys not found or not executable: $NSYS" >&2; exit 2; }
echo "[export] nsys binary: $NSYS"
"$NSYS" --version

for t in long medium short; do
    rep="$IN/$t.nsys-rep"
    out="$OUT/$t.sqlite"
    [[ -f "$rep" ]] || { echo "[export] missing capture: $rep" >&2; exit 1; }
    echo "[export] $rep -> $out"
    # the exporter draws a progress bar with carriage returns; keep only non-progress lines
    "$NSYS" export --type sqlite --force-overwrite=true --output "$out" "$rep" 2>&1 \
        | tr '\r' '\n' | grep -v -E '^\[[=]*[0-9]+%' | grep -v '^$' || true
    [[ -s "$out" ]] || { echo "[export] FAILED: $out missing" >&2; exit 1; }
    ls -l "$out"
done
echo "[export] done"
