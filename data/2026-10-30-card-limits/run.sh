#!/usr/bin/env bash
# run.sh - run the microbenchmarks on a GPU host, one throwaway container per benchmark (needs docker and the NVIDIA runtime).
#
#   ./run.sh                    everything (mma, membw, cublaslt): the data set run took 256 s of benchmarks, about 6 minutes with the idle gaps
#   ./run.sh --only mma,membw   a subset: mma | membw | cublaslt | exhaustive (exhaustive = cuBLASLt algorithm enumeration, +5-15 min)
#   ./run.sh --quick            shorter timings, roughly 60 % of the time
#   ./run.sh --dry-run          print what would run, touch neither docker nor the GPU
#
# Each benchmark runs as `docker run --rm --name dmk-mb-<name> --runtime=nvidia ...` in the stock llama.cpp CUDA image with the
# binaries mounted read-only at /opt/mb. While they run, `nvidia-smi --query-gpu` (a read-only query) samples power, SM clock,
# memory clock, P-state, temperature, utilization, VRAM and throttle reasons every MB_SMI_MS (default 100) ms into smi.csv.
# The tools themselves also sample NVML (20 ms) and write <name>.series.tsv.
# NOTHING here changes clocks, power limits, persistence mode or any other GPU setting.
#
# Results (default $HERE/results/<UTC time>/):
#   env.txt                    kernel, driver, image, GPU state before/after (nvidia-smi -q), binary hashes; no host name
#   phases.tsv                 start/end epoch and exit code of every benchmark (to slice smi.csv)
#   smi.csv                    nvidia-smi samples (timestamp is host local time, see env.txt for the UTC offset)
#   <name>.tsv / .log / .series.tsv   benchmark rows (stdout) / progress (stderr) / NVML series
# Summarize with summarize.py (Python 3) on any host: python3 summarize.py --raw <results dir> --tables <dir> --summary <file>
#
# Environment (all optional):
#   MB_GPU            NVIDIA_VISIBLE_DEVICES value, a GPU index or UUID   (default 0)
#   MB_IMAGE          image to run in                (default ghcr.io/ggml-org/llama.cpp:server-cuda-b12004)
#   MB_BIN            directory with mb-mma, mb-membw, mb-cublaslt-i8   (default $HERE/bin)
#   MB_SHAPES         shape list for cublaslt        (default $HERE/shapes/qwen35-4b-mix8.tsv)
#   MB_OUT            results directory
#   MB_SMI_MS         nvidia-smi sampling interval in ms (default 100)
#   MB_SUSTAIN_S      length of the sustained int8 run in seconds (default 20)
#   MB_MIN_FREE_MIB   refuse to start with less free VRAM (default 2600)
#   MB_FORCE=1        start although the GPU is busy (utilization above 10 %) or a container that matches MB_BLOCKING_NAMES
#                     runs: the results are then not valid
#   MB_BLOCKING_NAMES regular expression for the names of containers that must not run during a measurement (default: none)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MB_GPU=${MB_GPU:-0}
MB_IMAGE=${MB_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda-b12004}
MB_BIN=${MB_BIN:-$HERE/bin}
MB_SHAPES=${MB_SHAPES:-$HERE/shapes/qwen35-4b-mix8.tsv}
MB_SMI_MS=${MB_SMI_MS:-100}
MB_SUSTAIN_S=${MB_SUSTAIN_S:-20}
MB_MIN_FREE_MIB=${MB_MIN_FREE_MIB:-2600}
MB_FORCE=${MB_FORCE:-0}
MB_BLOCKING_NAMES=${MB_BLOCKING_NAMES:-}
ONLY=mma,membw,cublaslt
QUICK=0
DRY=0
OUT=${MB_OUT:-}

die() { echo "run.sh: $*" >&2; exit 1; }
now() { local t=${EPOCHREALTIME:-}; if [[ -n $t ]]; then echo "${t/,/.}"; else date +%s.%N; fi; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --only) ONLY=${2:?--only needs a list}; shift 2 ;;
        --quick) QUICK=1; shift ;;
        --dry-run) DRY=1; shift ;;
        --out) OUT=${2:?--out needs a directory}; shift 2 ;;
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done
for w in ${ONLY//,/ }; do
    case $w in mma|membw|cublaslt|exhaustive) ;; *) die "--only: unknown benchmark '$w'" ;; esac
done
want() { [[ ,$ONLY, == *,$1,* ]]; }
[[ -n $OUT ]] || OUT=$HERE/results/$(date -u +%Y%m%dT%H%M%SZ)

[[ $MB_GPU =~ ^[A-Za-z0-9-]+$ ]] || die "bad MB_GPU"
for b in mb-mma mb-membw mb-cublaslt-i8; do
    [[ -x $MB_BIN/$b ]] || die "$MB_BIN/$b not found or not executable"
done
[[ -f $MB_SHAPES ]] || die "shape list $MB_SHAPES not found"

# durations (seconds per measurement) for the full and the quick run
if (( QUICK )); then
    MMA_S=1; SWEEP_S=0.5; ZERO_S=1.5; SUST=8; DRAM_S=0.8; L2_S=0.5; LT_S=0.1; LT_ALGOS=8; LT_RETIME=2
else
    MMA_S=2; SWEEP_S=1; ZERO_S=3; SUST=$MB_SUSTAIN_S; DRAM_S=1.5; L2_S=1; LT_S=0.25; LT_ALGOS=16; LT_RETIME=3
fi

# ------------------------------------------------------------------ docker invocation
docker_cmd() {  # docker_cmd <container-suffix> <binary> args...
    local name=dmk-mb-$1 bin=$2
    shift 2
    printf '%s\n' docker run --rm --name "$name" --runtime=nvidia \
        -e "NVIDIA_VISIBLE_DEVICES=$MB_GPU" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
        --pids-limit 512 --cpus 2 \
        -v "$MB_BIN":/opt/mb:ro -v "$(dirname "$MB_SHAPES")":/shapes:ro -v "$OUT":/out -w /opt/mb \
        --entrypoint "/opt/mb/$bin" --label dmk.owner=dm-kernels-microbench \
        "$MB_IMAGE" "$@"
}

PHASES=$OUT/phases.tsv
FAILS=0
run_bench() {  # run_bench <name> <binary> args...   -> $OUT/<name>.tsv, .log
    local name=$1 bin=$2
    shift 2
    local -a cmd
    mapfile -t cmd < <(docker_cmd "$name" "$bin" "$@")
    if (( DRY )); then
        printf '[dry-run] %s' "${cmd[0]}"; printf ' %s' "${cmd[@]:1}"; echo " > $OUT/$name.tsv 2> $OUT/$name.log"
        return 0
    fi
    docker rm -f "dmk-mb-$name" > /dev/null 2>&1   # only our own name
    echo "== $name: $bin $*"
    local t0 t1 rc
    t0=$(now)
    "${cmd[@]}" > "$OUT/$name.tsv" 2> "$OUT/$name.log"
    rc=$?
    t1=$(now)
    printf '%s\t%s\t%s\t%d\t%s %s\n' "$name" "$t0" "$t1" "$rc" "$bin" "$*" >> "$PHASES"
    if (( rc != 0 )); then
        FAILS=$((FAILS + 1))
        echo "   exit code $rc, last log lines:"; tail -n 5 "$OUT/$name.log" | sed 's/^/   | /'
    else
        echo "   done in $(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }') s"
    fi
    sleep 3   # idle gap: lets the clocks settle and gives the sampler a baseline between benchmarks
    return $rc
}

# ------------------------------------------------------------------ preflight, sampler
SMI_PID=
cleanup() {
    if [[ -n $SMI_PID ]] && kill -0 "$SMI_PID" 2> /dev/null; then
        kill -INT "$SMI_PID" 2> /dev/null; sleep 0.5; kill "$SMI_PID" 2> /dev/null
        wait "$SMI_PID" 2> /dev/null
    fi
    for n in mma_all mma_sweep mma_zero mma_sustained membw_all membw_sweep cublaslt cublaslt_exhaustive; do
        docker rm -f "dmk-mb-$n" > /dev/null 2>&1
    done
}

if (( DRY )); then
    echo "[dry-run] results would go to $OUT; GPU $MB_GPU; image $MB_IMAGE"
    PHASES=/dev/null
else
    command -v docker > /dev/null || die "docker not found"
    command -v nvidia-smi > /dev/null || die "nvidia-smi not found"
    if command -v flock > /dev/null; then
        exec 9> "$HERE/.run.lock"
        flock -n 9 || die "another run.sh is running (lock $HERE/.run.lock)"
    fi
    mkdir -p "$OUT" || die "cannot create $OUT"
    : > "$PHASES"
    trap cleanup EXIT
    trap 'exit 130' INT TERM

    smi() { nvidia-smi -i "$MB_GPU" "$@"; }
    smi_q() { smi -q -d CLOCK,POWER,PERFORMANCE | sed -E 's/^GPU [0-9A-Fa-f:.]+$/GPU (PCI address removed)/'; }
    q=$(smi --query-gpu=memory.free,memory.used,utilization.gpu,power.limit,pstate,clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader,nounits 2>&1) \
        || die "nvidia-smi query failed: $q"
    IFS=',' read -r free used util plimit pstate clk_sm clk_mem temp <<< "$q"
    echo "GPU before: free ${free// /} MiB, used ${used// /} MiB, util ${util// /} %, power limit ${plimit// /} W, ${pstate// /}, SM ${clk_sm// /} MHz, mem ${clk_mem// /} MHz, ${temp// /} C"
    (( ${free// /} >= MB_MIN_FREE_MIB )) || die "only ${free// /} MiB VRAM free, need $MB_MIN_FREE_MIB (set MB_MIN_FREE_MIB to override)"
    busy=0
    for i in 1 2 3; do
        u=$(smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
        (( u > 10 )) && busy=1
        sleep 0.5
    done
    if (( busy )) && [[ $MB_FORCE != 1 ]]; then
        smi --query-compute-apps=pid,process_name,used_memory --format=csv
        die "the GPU is busy (utilization above 10 %): something else is running; stop it or set MB_FORCE=1 (results would not be valid)"
    fi
    if [[ -n $MB_BLOCKING_NAMES ]]; then
        other=$(docker ps --format '{{.Names}}' 2> /dev/null | grep -E "$MB_BLOCKING_NAMES" | grep -vE '^dmk-mb-' | tr '\n' ' ')
        if [[ -n $other ]] && [[ $MB_FORCE != 1 ]]; then
            die "containers that match MB_BLOCKING_NAMES are running ($other): they would be disturbed and would disturb the measurement; wait for them or set MB_FORCE=1"
        fi
    fi

    # env.txt holds no host name, no PCI address, no process ID and no container names
    {
        echo "run.sh start:     $(date -u +%Y-%m-%dT%H:%M:%SZ)  (host local UTC offset $(date +%z))"
        echo "kernel:           $(uname -r)"
        echo "gpu selector:     $MB_GPU"
        echo "driver:           $(smi --query-gpu=driver_version,name,vbios_version --format=csv,noheader)"
        echo "image:            $MB_IMAGE  $(docker image inspect --format '{{.Id}} {{json .RepoDigests}}' "$MB_IMAGE" 2> /dev/null)"
        echo "only/quick:       $ONLY / $QUICK"
        echo "shapes:           $(basename "$MB_SHAPES")  ($(grep -vc '^#' "$MB_SHAPES") rows, sha256 $(sha256sum "$MB_SHAPES" | cut -d' ' -f1))"
        echo "containers:       $(docker ps -q | wc -l) running"
        echo "compute apps:     $(smi --query-compute-apps=process_name,used_memory --format=csv,noheader | tr '\n' ';')"
        echo
        echo "## binaries"
        (cd "$MB_BIN" && sha256sum mb-mma mb-membw mb-cublaslt-i8)
        echo
        echo "## nvidia-smi -q -d CLOCK,POWER,PERFORMANCE (before)"
        smi_q
    } > "$OUT/env.txt" 2>&1

    # sampler; fall back to a shorter field list when this driver rejects one of the names
    FIELDS=timestamp,power.draw,power.draw.instant,clocks.sm,clocks.mem,pstate,temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active
    if ! smi --query-gpu=$FIELDS --format=csv,nounits > /dev/null 2>&1; then
        FIELDS=timestamp,power.draw,clocks.sm,clocks.mem,pstate,temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active
        echo "note: power.draw.instant not accepted, sampling without it" | tee -a "$OUT/env.txt"
    fi
    echo "$FIELDS" > "$OUT/smi.fields"   # column order of smi.csv (the header names printed by nvidia-smi differ from the query names)
    SB=; command -v stdbuf > /dev/null && SB="stdbuf -oL"
    # shellcheck disable=SC2086
    $SB nvidia-smi -i "$MB_GPU" --query-gpu=$FIELDS --format=csv,nounits -lms "$MB_SMI_MS" > "$OUT/smi.csv" 2> "$OUT/smi.err" &
    SMI_PID=$!
    sleep 2
    kill -0 "$SMI_PID" 2> /dev/null || die "the nvidia-smi sampler died: $(cat "$OUT/smi.err")"
fi
(( DRY )) || mkdir -p "$OUT"

# ------------------------------------------------------------------ benchmarks
T_START=$(now)
peak=NA

if want mma; then
    run_bench mma_all mb-mma --mode all --seconds "$MMA_S" --series /out/mma_all.series.tsv
    run_bench mma_sweep mb-mma --mode sweep --sweep-seconds "$SWEEP_S" --series /out/mma_sweep.series.tsv
    run_bench mma_zero mb-mma --mode s8k32 --data zero --seconds "$ZERO_S" --series /out/mma_zero.series.tsv
    run_bench mma_sustained mb-mma --mode s8k32 --seconds "$SUST" --series /out/mma_sustained.series.tsv
fi
if (( ! DRY )); then
    # best measured s8k32 TOPS with random data over the "all", "sweep" and "sustained" runs (the sustained run
    # is the steady-state number, but the best short run is the right efficiency denominator for cuBLASLt)
    peak=$(cat "$OUT"/mma_all.tsv "$OUT"/mma_sweep.tsv 2> /dev/null | awk -F'\t' '
        $1 == "bench" { for (i = 1; i <= NF; i++) c[$i] = i; next }
        c["mode"] && $c["mode"] == "s8k32" && $c["data"] == "rand" && $c["tops"] + 0 > m { m = $c["tops"] + 0 }
        END { if (m > 0) printf "%.2f", m; else print "NA" }')
    echo "measured int8 peak (mma s8k32, random data, best of all/sweep): $peak TOPS"
fi
if want membw; then
    run_bench membw_all mb-membw --mode all --seconds "$DRAM_S" --l2-seconds "$L2_S" --series /out/membw_all.series.tsv
    run_bench membw_sweep mb-membw --mode sweep --series /out/membw_sweep.series.tsv
fi
peak_args=(); [[ $peak != NA ]] && peak_args=(--peak-tops "$peak")
if want cublaslt; then
    run_bench cublaslt mb-cublaslt-i8 --shapes "/shapes/$(basename "$MB_SHAPES")" --seconds "$LT_S" --max-algos "$LT_ALGOS" --retime "$LT_RETIME" \
        ${peak_args[@]+"${peak_args[@]}"} --series /out/cublaslt.series.tsv
fi
if want exhaustive; then
    run_bench cublaslt_exhaustive mb-cublaslt-i8 --shapes "/shapes/$(basename "$MB_SHAPES")" --exhaustive --n-only 8,32,128,512 \
        --seconds "$LT_S" --max-algos 8 --retime 3 --search-seconds 0.008 ${peak_args[@]+"${peak_args[@]}"} --series /out/cublaslt_exhaustive.series.tsv
fi
T_END=$(now)

if (( DRY )); then
    echo "[dry-run] done; nothing was executed"
    exit 0
fi

# ------------------------------------------------------------------ wrap-up
{
    echo
    echo "## nvidia-smi -q -d CLOCK,POWER,PERFORMANCE (after)"
    smi_q
    echo
    echo "run.sh end:       $(date -u +%Y-%m-%dT%H:%M:%SZ), $(awk -v a="$T_START" -v b="$T_END" 'BEGIN { printf "%.0f", b - a }') s of benchmarks, $FAILS failed"
} >> "$OUT/env.txt" 2>&1
cleanup
SMI_PID=
trap - EXIT
[[ -s $OUT/smi.csv ]] || echo "warning: smi.csv is empty (the NVML series in *.series.tsv still has the data)"

tabulate() { if command -v column > /dev/null; then column -t -s "$(printf '\t')"; else cat; fi; }
show() {  # show <tsv> <col,col,...>
    [[ -s $1 ]] || return 0
    awk -F'\t' -v want="$2" '
        BEGIN { n = split(want, w, ",") }
        $1 == "bench" { for (i = 1; i <= NF; i++) c[$i] = i; for (j = 1; j <= n; j++) printf "%s%s", w[j], j < n ? "\t" : "\n"; next }
        { for (j = 1; j <= n; j++) printf "%s%s", $(c[w[j]]), j < n ? "\t" : "\n" }' "$1" | tabulate
}
echo
echo "== headline (full tables: python3 summarize.py --raw $OUT --tables <dir> --summary <file>)"
echo "-- mma (all instruction shapes, random data)"; show "$OUT/mma_all.tsv" mode,warps_per_sm,tops,ops_per_clk_sm,eff_vs_nominal_pct,sm_mhz_mean,power_w_mean,swcap_pct
echo "-- mma sustained s8k32"; show "$OUT/mma_sustained.tsv" mode,seconds,tops,tops_last_half,sm_mhz_mean,sm_mhz_min,power_w_mean,power_w_max,swcap_pct
echo "-- memory"; show "$OUT/membw_all.tsv" test,variant,buf_mib,gbps,eff_vs_nominal_pct,mem_mhz_mean
n_lt=$(grep -c '^cublaslt_i8' "$OUT/cublaslt.tsv" 2> /dev/null); n_lt=${n_lt:-0}
echo "-- cublaslt: $n_lt rows, statuses: $(awk -F'\t' 'NR > 1 { s[$9]++ } END { for (k in s) printf "%s=%d ", k, s[k] }' "$OUT/cublaslt.tsv" 2> /dev/null)"
echo
echo "results in $OUT  ($FAILS benchmark(s) failed)"
exit $(( FAILS > 0 ? 1 : 0 ))
