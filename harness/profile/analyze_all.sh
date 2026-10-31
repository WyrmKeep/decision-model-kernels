#!/usr/bin/env bash
# analyze_all.sh - run every analysis of the phase-3 profiling window on the results copied back from the GPU host.
# CPU only: nothing here touches the GPU. A step whose inputs are missing is skipped and listed at the end; a failing step does not stop the others.
#
# Usage (any cwd):  bash harness/profile/analyze_all.sh [--census DIR] [--no-mmq-launches] RESULTS_ROOT OUT_ROOT
#   RESULTS_ROOT  holds one directory per schedule that was run (1-decomp, 2-nsys, 3-leads, 4-ncu-smoke, 4-ncu-core, 4-ncu-extra)
#   OUT_ROOT      one sub-directory per part is written here (decomp, nsys, leads-nsys, leads-kvwin, leads-childpass, ncu)
#   --census DIR  folder of the census data set (exported as DMK_CENSUS_DATA; default: the data set of this repository, data/2026-10-30-shape-census)
#   --no-mmq-launches   do not write mmq-launches.tsv in the nsys steps (large file; the checks in summary.txt are still computed)
# Tools: DMK_NSYS = path of the nsys command line tool (else nsys on PATH; DMK_NSYS_DEB = a .deb to unpack), DMK_NCU = path of the ncu command line tool (else ncu on PATH).
# Steps (the order matters: 2 needs nothing, 3 needs the nsys exports of 2 and 3, 4 needs the passes table of 3):
#   decomp      analyze_decomp.py RESULTS_ROOT/1-decomp                  -> OUT/decomp/summary.txt
#   nsys        nsys_export.sh + nsys_census.py analyze + overhead       -> OUT/nsys/summary.txt, passes.tsv, overhead report
#   kvwin       nsys_export.sh RESULTS_ROOT/3-leads + analyze + analyze_kvwin -> OUT/leads-nsys/, OUT/leads-kvwin/
#   childpass   analyze_childpass.py on the DMK_TIMING server logs       -> OUT/leads-childpass/
#   ncu         analyze_ncu.py on 4-ncu-smoke / 4-ncu-core / 4-ncu-extra  -> OUT/ncu/sol.txt
# The summary.txt files are the entry points; nsys/NOTES.txt and ncu/NOTES.txt say what to read in them, COLUMNS.tsv in the same folders explains the columns.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P="$HERE"

usage() {   # print the header comment above and exit with $1
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}" >&2
    exit "${1:-2}"
}
CENSUS=""; MMQ=()
while (($#)); do
    case "$1" in
        --census) (($# >= 2)) || usage 2; CENSUS="$2"; shift 2 ;;
        --no-mmq-launches) MMQ=(--no-mmq-launches); shift ;;
        -h|--help) usage 0 ;;
        --) shift; break ;;
        -*) echo "analyze_all.sh: unknown option $1" >&2; usage 2 ;;
        *) break ;;
    esac
done
(($# == 2)) || usage 2
[[ -d "$1" ]] || { echo "analyze_all.sh: RESULTS_ROOT $1 is not a directory" >&2; exit 2; }
RES="$(cd "$1" && pwd)"
mkdir -p "$2"
OUT="$(cd "$2" && pwd)"
if [[ -n "$CENSUS" ]]; then
    [[ -d "$CENSUS" ]] || { echo "analyze_all.sh: --census $CENSUS is not a directory" >&2; exit 2; }
    export DMK_CENSUS_DATA="$(cd "$CENSUS" && pwd)"
fi
LOG="$OUT/analyze_all.log"
: > "$LOG"
DONE=(); SKIPPED=(); FAILED=()

say() { echo "$*" | tee -a "$LOG"; }

have() {   # have FILE...: all exist and are non-empty
    local f
    for f in "$@"; do [[ -s "$f" ]] || return 1; done
}

step() {   # step NAME CMD...: run, keep the output in OUT/NAME.out.txt (and in the log), record the result
    local name="$1" rc; shift
    say ""
    say "== $name: $*"
    "$@" > "$OUT/$name.out.txt" 2>&1; rc=$?
    cat "$OUT/$name.out.txt" >> "$LOG"
    if ((rc == 0)); then DONE+=("$name"); say "   ok -> $OUT/$name.out.txt"; else FAILED+=("$name (exit $rc)"); say "   FAILED (exit $rc), see $OUT/$name.out.txt"; fi
}

skip() { SKIPPED+=("$1: $2"); say ""; say "== $1: skipped, $2"; }

# ---------------------------------------------------------------- 1 decomposition
D="$RES/1-decomp"
if have "$D/p3-dec-timing.client.json" "$D/p3-dec-timing.server.log"; then
    ref=()
    have "$D/p3-dec-ref.client.json" && ref=(--ref p3-dec-ref) || ref=(--ref "")
    step decomp python3 "$P/decomp/analyze_decomp.py" "$D" --run p3-dec-timing "${ref[@]}" --out-dir "$OUT/decomp"
else
    skip decomp "no $D/p3-dec-timing.{client.json,server.log}"
fi

# ---------------------------------------------------------------- 2 nsys, one capture per request type
N="$RES/2-nsys"
if ls "$N"/*.nsys-rep >/dev/null 2>&1; then
    step nsys-export bash "$P/nsys/nsys_export.sh" "$N"
    step nsys-analyze python3 "$P/nsys/nsys_census.py" analyze --results "$N" --out "$OUT/nsys" "${MMQ[@]}"
    if have "$N/p3-nsys-ctl.client.json" "$OUT/nsys/passes.tsv"; then
        dec=()
        have "$D/p3-dec-ref.client.json" && dec=(--dec "$D/p3-dec-ref.client.json")
        step nsys-overhead python3 "$P/nsys/nsys_census.py" overhead --results "$N" --ctl p3-nsys-ctl --passes "$OUT/nsys/passes.tsv" --out "$OUT/nsys" "${dec[@]}"
    else
        skip nsys-overhead "no $N/p3-nsys-ctl.client.json or no $OUT/nsys/passes.tsv"
    fi
else
    skip nsys "no $N/*.nsys-rep"
fi

# ---------------------------------------------------------------- 3 leads
L="$RES/3-leads"
if ls "$L"/*.nsys-rep >/dev/null 2>&1; then
    step leads-export bash "$P/nsys/nsys_export.sh" "$L"
    runs=()
    for r in p3-lead-kvwin-nsys p3-lead-kvwin-nokvu-nsys; do have "$L/$r.client.json" "$L/$r.sqlite" && runs+=("$r"); done
    if ((${#runs[@]})); then
        step leads-nsys python3 "$P/nsys/nsys_census.py" analyze --results "$L" --runs "$(IFS=,; echo "${runs[*]}")" --out "$OUT/leads-nsys" "${MMQ[@]}"
    else
        skip leads-nsys "no <run>.client.json + <run>.sqlite in $L"
    fi
else
    skip leads-nsys "no $L/*.nsys-rep"
fi
if have "$OUT/leads-nsys/passes.tsv" "$L/p3-lead-kvwin-nsys.client.json"; then
    args=(--out "$OUT/leads-kvwin" --nsys "$OUT/leads-nsys/passes.tsv" --client "$L/p3-lead-kvwin-nsys.client.json" --run p3-lead-kvwin-nsys)
    have "$L/p3-lead-kvwin-nokvu-nsys.client.json" && grep -q "p3-lead-kvwin-nokvu-nsys" "$OUT/leads-nsys/passes.tsv" \
        && args+=(--nsys-nokvu "$OUT/leads-nsys/passes.tsv,$L/p3-lead-kvwin-nokvu-nsys.client.json,p3-lead-kvwin-nokvu-nsys")
    have "$L/p3-lead-kvwin-timing.server.log" "$L/p3-lead-kvwin-timing.client.json" \
        && args+=(--timing "kvu=$L/p3-lead-kvwin-timing.server.log,$L/p3-lead-kvwin-timing.client.json")
    have "$L/p3-lead-kvwin-nokvu-timing.server.log" "$L/p3-lead-kvwin-nokvu-timing.client.json" \
        && args+=(--timing "nokvu=$L/p3-lead-kvwin-nokvu-timing.server.log,$L/p3-lead-kvwin-nokvu-timing.client.json")
    step leads-kvwin python3 "$P/leads/analyze_kvwin.py" "${args[@]}"
else
    skip leads-kvwin "no $OUT/leads-nsys/passes.tsv or no $L/p3-lead-kvwin-nsys.client.json"
fi

# ---------------------------------------------------------------- every server log with DMK_TIMING lines
logs=(); clients=()
for pair in "$D/p3-dec-timing" "$L/p3-lead-kvwin-timing" "$L/p3-lead-kvwin-nokvu-timing"; do
    if have "$pair.server.log"; then logs+=("$pair.server.log"); have "$pair.client.json" && clients+=("$pair.client.json"); fi
done
if ((${#logs[@]})); then
    cl=(); ((${#clients[@]})) && cl=(--client "${clients[@]}")
    step leads-childpass python3 "$P/leads/analyze_childpass.py" "${logs[@]}" "${cl[@]}" --out "$OUT/leads-childpass"
else
    skip leads-childpass "no timing server log"
fi

# ---------------------------------------------------------------- 4 ncu
reps=()
for d in 4-ncu-smoke 4-ncu-core 4-ncu-extra; do { compgen -G "$RES/$d/*.ncu-rep" || compgen -G "$RES/$d/*.ncu-repz"; } >/dev/null && reps+=("$RES/$d"); done
if ((${#reps[@]})); then
    step ncu python3 "$P/ncu/analyze_ncu.py" "${reps[@]}" --out-dir "$OUT/ncu"
else
    skip ncu "no $RES/4-ncu-{smoke,core,extra}/*.ncu-rep or *.ncu-repz"
fi

# ---------------------------------------------------------------- summary
say ""
say "== done:    ${DONE[*]:-none}"
say "== skipped: ${#SKIPPED[@]}"; for s in "${SKIPPED[@]:-}"; do [[ -n "$s" ]] && say "   $s"; done
say "== failed:  ${#FAILED[@]}"; for s in "${FAILED[@]:-}"; do [[ -n "$s" ]] && say "   $s"; done
say "== output:  $OUT  (log: $LOG)"
((${#FAILED[@]} == 0))
