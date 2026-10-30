#!/usr/bin/env bash
# build.sh - build the three microbenchmarks for sm_86 in a CUDA 12.8 builder container (no GPU needed).
#
#   ./build.sh                 builds into ./bin (mb-mma, mb-membw, mb-cublaslt-i8, MANIFEST.txt, build.log, ptxas.log)
#
# Environment:
#   MB_BUILDER_IMAGE   image with nvcc 12.8, g++, binutils (cuobjdump, c++filt) and the cuBLAS development files
#                      (default docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04; the data set was built in an image made from
#                      that one, with nvcc 12.8.93 and g++ 13.3.0)
#   MB_CONTAINER_CLI   podman or docker (default: podman if it is installed, else docker)
#
# Linking: the CUDA runtime is linked statically, libstdc++/libgcc statically, NVML is loaded with dlopen at run time
# (the container runtime provides libnvidia-ml.so.1). Only mb-cublaslt-i8 needs a CUDA library at run time:
# libcublasLt.so.12. The runtime image ghcr.io/ggml-org/llama.cpp:server-cuda-b12004 already contains
# libcublasLt.so.12.8.4.1 (in /usr/local/cuda/lib64, which is on its LD_LIBRARY_PATH), the same version as the builder, so
# nothing is shipped alongside and the binaries stay about 1.2 MB each.
#
# After the build, inside the same container (CPU only):
#   * --help, --dry-run and the "no GPU" error path of each binary are run,
#   * the SASS of every mma kernel is checked: the loop must contain only tensor-core instructions (IMMA/HMMA) plus
#     loop bookkeeping, no loads or stores,
#   * ldd and sha256 of the binaries go into bin/MANIFEST.txt.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ------------------------------------------------------------------ inside the container
if [[ "${1:-}" == "--inside" ]]; then
    cd /mb
    # a plain CUDA devel image can lack the host compiler (the data set was built in an image that had it)
    if ! command -v g++ > /dev/null; then apt-get update -qq && apt-get install -y -qq g++ binutils; fi
    out=/mb/bin
    mkdir -p "$out"
    rm -f "$out"/mb-* "$out/MANIFEST.txt" "$out/ptxas.log"

    nvflags=(-std=c++17 -O3 -gencode arch=compute_86,code=sm_86 -Xcompiler -Wall,-Wextra
             -Xcompiler -static-libstdc++,-static-libgcc -Xptxas -v)
    : > "$out/ptxas.log"
    echo "== nvcc mb-mma";          nvcc "${nvflags[@]}" -o "$out/mb-mma"          src/mma_peak.cu            2>> "$out/ptxas.log"
    echo "== nvcc mb-membw";        nvcc "${nvflags[@]}" -o "$out/mb-membw"        src/membw.cu               2>> "$out/ptxas.log"
    echo "== nvcc mb-cublaslt-i8";  nvcc "${nvflags[@]}" -o "$out/mb-cublaslt-i8"  src/cublaslt_i8.cu -lcublasLt 2>> "$out/ptxas.log"
    # compiler warnings (anything that is not a ptxas info line) must be empty
    if grep -v '^ptxas info' "$out/ptxas.log" | grep -qi 'warning\|error'; then
        echo "compiler warnings:"; grep -v '^ptxas info' "$out/ptxas.log"; exit 1
    fi
    if grep -q 'spill' "$out/ptxas.log" && grep 'spill' "$out/ptxas.log" | grep -vq ' 0 bytes spill stores, 0 bytes spill loads'; then
        echo "register spills in a kernel, see $out/ptxas.log"; exit 1
    fi

    echo "== smoke tests without a GPU"
    for b in mb-mma mb-membw mb-cublaslt-i8; do
        "$out/$b" --help > /tmp/help.txt 2>&1 || { echo "$b --help failed"; exit 1; }
        head -1 /tmp/help.txt
    done
    "$out/mb-cublaslt-i8" --dry-run --shapes shapes/qwen35-4b-mix8.tsv > /tmp/dry.tsv
    echo "dry-run rows: $(($(wc -l < /tmp/dry.tsv) - 1))"
    set +e
    "$out/mb-mma" --mode all > /dev/null 2> /tmp/nogpu.txt;   rc1=$?
    "$out/mb-membw" > /dev/null 2>> /tmp/nogpu.txt;           rc2=$?
    "$out/mb-cublaslt-i8" --shape 64,8,64 > /dev/null 2>> /tmp/nogpu.txt; rc3=$?
    set -e
    echo "no-GPU exit codes (3 = clean 'no CUDA device'): $rc1 $rc2 $rc3"
    [[ $rc1 == 3 && $rc2 == 3 && $rc3 == 3 ]] || { echo "unexpected exit codes"; cat /tmp/nogpu.txt; exit 1; }

    echo "== SASS check of the mma kernels"
    cuobjdump -sass "$out/mb-mma" 2> /dev/null | awk '
        /Function :/ { fn = $3; next }
        fn ~ /k_mma/ && /^[ \t]+\/\*[0-9a-f]+\*\// {
            ins = $0; sub(/^[ \t]+\/\*[0-9a-f]+\*\/[ \t]+/, "", ins); sub(/^@!?P[0-9T]+[ \t]+/, "", ins); split(ins, w, /[ .;]+/); op = w[1]
            if (op ~ /^(IMMA|HMMA)$/) mma[fn]++
            else if (op ~ /^(LDG|LDL|LDS|STG|STL|STS|LD|ST|ATOM|RED)$/) mem[fn" "op]++
        }
        END {
            for (f in mma) { n++; tot += mma[f] }
            printf "mma kernels: %d, tensor-core instructions in their SASS: %d\n", n, tot
            bad = 0
            for (k in mem) { split(k, a, " "); if (!(a[2] == "STG" && mem[k] == 1)) { printf "unexpected memory instruction %s in %s\n", a[2], a[1]; bad = 1 } }
            if (bad) exit 1; print "only the single sink store is a memory instruction: OK"
        }' | tee /tmp/sass-check.txt

    echo "== manifest"
    {
        echo "# microbench build manifest"
        echo "built:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "nvcc:   $(nvcc --version | tail -2 | tr '\n' ' ')"
        echo "gcc:    $(g++ --version | head -1)"
        echo "flags:  ${nvflags[*]}  (+ -lcublasLt for mb-cublaslt-i8)"
        echo "cublasLt in the builder: $(ls /usr/local/cuda/lib64/libcublasLt.so.12.* | head -1)"
        echo
        echo "## SASS check"; cat /tmp/sass-check.txt
        echo
        echo "## ptxas (registers per kernel, spills)"
        grep -E "Compiling entry|Used .* registers|spill" "$out/ptxas.log" | paste - - - | c++filt | awk -F'\t' '
            { n = $1; sub(/.*Compiling entry function ./, "", n); gsub(/\(anonymous namespace\)::/, "", n); sub(/^void /, "", n); sub(/\(.*/, "", n)
              regs = $3; sub(/.*Used /, "", regs); sub(/ registers.*/, "", regs)
              sp = $2; gsub(/[^0-9,]/, "", sp)
              printf "%-26s regs %-4s stack,spill-stores,spill-loads %s\n", n, regs, sp }'
        echo
        echo "## dynamic dependencies (ldd)"
        for b in mb-mma mb-membw mb-cublaslt-i8; do echo "-- $b"; ldd "$out/$b" | sed 's/(0x[0-9a-f]*)//'; done
        echo
        echo "## files (sha256, bytes, name)"
        (cd "$out" && for f in mb-mma mb-membw mb-cublaslt-i8; do echo "$(sha256sum "$f" | cut -d' ' -f1)  $(stat -c %s "$f")  $f"; done)
    } > "$out/MANIFEST.txt"
    exit 0
fi

# ------------------------------------------------------------------ host side
image=${MB_BUILDER_IMAGE:-docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04}
cli=${MB_CONTAINER_CLI:-}
if [[ -z $cli ]]; then
    if command -v podman > /dev/null; then cli=podman; else cli=docker; fi
fi
command -v "$cli" > /dev/null || { echo "$cli is not installed" >&2; exit 1; }
mkdir -p "$here/bin"
# the SELinux label option is for podman only
opts=()
[[ $cli == podman ]] && opts+=(--security-opt label=disable)
"$cli" run --rm ${opts[@]+"${opts[@]}"} -v "$here":/mb "$image" bash /mb/build.sh --inside 2>&1 | tee "$here/bin/build.log"
exit "${PIPESTATUS[0]}"
