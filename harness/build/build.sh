#!/usr/bin/env bash
# build.sh <src-dir> <out-dir> [extra cmake configure args...]
#
# Builds llama.cpp for sm_86 inside a podman builder image that mirrors the
# build stage of .devops/cuda.Dockerfile (CUDA 12.8.1 devel base pinned by digest,
# gcc-14, Ubuntu cmake). The source tree is mounted read-only. All output goes to
# <out-dir>:
#   build/       CMake build tree (kept between runs: a second run is incremental)
#   dist/        deployable files, same layout as /app in the stock server image
#   dist/MANIFEST.txt   hashes, flags, tool versions, source commit
#   build.log    full log of the last run
#
# Environment (all optional):
#   DMK_BUILDER_IMAGE  builder image tag      (default localhost/dmk-builder:cuda12.8.1-gcc14)
#   DMK_CUDA_ARCH      CMAKE_CUDA_ARCHITECTURES (default 86-real)
#   DMK_JOBS           parallel jobs          (default 16)
#   DMK_TARGETS        CMake targets to build (default: the list below; "all" builds everything)
#   DMK_CLEAN=1        delete <out-dir>/build before configuring
#   DMK_CCACHE=1       use ccache             (default off: see the note below; cache in $DMK_CCACHE_DIR)
#   DMK_CCACHE_DIR     ccache directory       (default <this dir>/ccache)
#
# Reproducibility (observed on b12004, 4 builds): every file in dist/ except libggml-cuda.so is byte-identical
# between builds of the same source. libggml-cuda.so changes hash from build to build (nvcc puts per-compilation
# IDs into symbol names), its SASS does not. Compare SASS, not hashes: harness/build/sass_compare.py.
#
# Extra cmake args come last on the command line, so they override the defaults,
# for example:  build.sh /path/to/llama.cpp out -DCMAKE_CUDA_ARCHITECTURES=86-real -DGGML_CUDA_FA=OFF
set -euo pipefail

DEFAULT_TARGETS="llama-server llama-app llama-bench llama-perplexity llama-tokenize test-backend-ops"

# ------------------------------------------------------------------ inside the container
if [[ "${1:-}" == "--inside" ]]; then
    shift
    src=/src out=/out
    jobs=${DMK_JOBS:-16}
    arch=${DMK_CUDA_ARCH:-86-real}
    targets=${DMK_TARGETS:-$DEFAULT_TARGETS}

    ccache_args=(-DGGML_CCACHE=OFF)
    if [[ "${DMK_CCACHE:-0}" == "1" ]]; then
        # ccache is off by default and meant for quick iteration in fresh out dirs. Observed on b12004: with CCACHE_BASEDIR
        # set, ccache rewrote __FILE__ strings and one kernel (flash_attn_tile, D=96) was scheduled differently. Without
        # CCACHE_BASEDIR the SASS of a cold-cache build matched an uncached build. Use uncached builds for baselines.
        export CCACHE_DIR=/ccache CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-20G}
        ccache --zero-stats >/dev/null
        ccache_args=(-DGGML_CCACHE=ON)
    fi

    [[ "${DMK_CLEAN:-0}" == "1" ]] && rm -rf "$out/build"

    # Same flags as .devops/cuda.Dockerfile at b12004, with these differences:
    #   CMAKE_CUDA_ARCHITECTURES=86-real     only the production GPU
    #   LLAMA_BUILD_TESTS=ON                 test-backend-ops
    #   CMAKE_BUILD_RPATH_USE_ORIGIN=ON      binaries find their libs next to themselves
    #   LLAMA_USE_PREBUILT_UI=OFF            no network access during the build; the UI is not used (--no-webui)
    cmake_args=(
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_NATIVE=OFF
        -DGGML_CUDA=ON
        -DGGML_BACKEND_DL=ON
        -DGGML_CPU_ALL_VARIANTS=ON
        -DLLAMA_BUILD_TESTS=ON
        -DCMAKE_CUDA_ARCHITECTURES="$arch"
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined
        -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
        -DLLAMA_USE_PREBUILT_UI=OFF
        "${ccache_args[@]}"
        "$@"
    )

    echo "== cmake configure: cmake -S $src -B $out/build ${cmake_args[*]}"
    t0=$(date +%s)
    cmake -S "$src" -B "$out/build" "${cmake_args[@]}"
    if [[ "$targets" == "all" ]]; then
        echo "== cmake build: all targets, -j$jobs"
        cmake --build "$out/build" --config Release -j"$jobs"
    else
        echo "== cmake build: $targets, -j$jobs"
        # shellcheck disable=SC2086
        cmake --build "$out/build" --config Release -j"$jobs" --target $targets
    fi
    t1=$(date +%s)

    # ---- dist: same layout as /app in the stock image (flat: binaries and libs together)
    rm -rf "$out/dist"
    mkdir -p "$out/dist"
    find "$out/build/bin" -maxdepth 1 -name '*.so*' -exec cp -P {} "$out/dist/" \;
    for b in $targets; do
        [[ "$b" == "all" ]] && continue
        exe=$b; [[ "$b" == "llama-app" ]] && exe=llama   # target llama-app writes bin/llama
        [[ -f "$out/build/bin/$exe" ]] && cp "$out/build/bin/$exe" "$out/dist/"
    done
    if [[ "$targets" == "all" ]]; then
        find "$out/build/bin" -maxdepth 1 -type f -executable ! -name '*.so*' -exec cp {} "$out/dist/" \;
    fi

    # ---- manifest
    m="$out/dist/MANIFEST.txt"
    {
        echo "# MANIFEST for ${DMK_OUT_NAME:-$(basename "$out")}"
        echo "built:            $(date -u +%Y-%m-%dT%H:%M:%SZ) (configure+build ${t0}..${t1}, $((t1 - t0)) s wall)"
        echo
        echo "## source"
        echo "src commit:       $(git -C "$src" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "src describe:     $(git -C "$src" describe --tags --always 2>/dev/null || echo unknown)"
        echo "src branch:       $(git -C "$src" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
        dirty=$(git -C "$src" status --porcelain 2>/dev/null | wc -l)
        echo "src dirty files:  $dirty"
        echo "src diff vs HEAD sha256: $(git -C "$src" diff HEAD 2>/dev/null | sha256sum | cut -d' ' -f1)  (hash of empty input is e3b0c442...)"
        echo
        echo "## toolchain"
        echo "base image tag:   ${DMK_BASE_TAG:-unknown}"
        echo "base image index digest:  ${DMK_BASE_INDEX:-unknown}"
        echo "base image amd64 digest:  ${DMK_BASE_AMD64:-unknown}"
        echo "builder image:    ${DMK_BUILDER_ID:-unknown}"
        echo "nvcc:             $(nvcc --version | tail -2 | tr '\n' ' ')"
        echo "gcc:              $(gcc-14 --version | head -1)"
        echo "g++:              $(g++-14 --version | head -1)"
        echo "cmake:            $(cmake --version | head -1)"
        echo "ccache:           $(ccache --version | head -1) (enabled: ${DMK_CCACHE:-0})"
        echo "packages:         $(dpkg-query -W -f='${Package}=${Version} ' cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 gcc-14 g++-14 cmake libssl-dev 2>/dev/null)"
        echo
        echo "## cmake configure"
        echo "CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX"
        echo "cmake -S <src> -B <out>/build ${cmake_args[*]}"
        echo "cmake --build <out>/build --config Release -j$jobs${targets:+ --target $targets}"
        grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_FLAGS|CMAKE_CXX_FLAGS|GGML_[A-Z0-9_]+|LLAMA_BUILD_[A-Z]+)(:[A-Z]+)?=' "$out/build/CMakeCache.txt" | sort | sed 's/^/cache: /'
        echo
        echo "## files (sha256, bytes, name)"
        (cd "$out/dist" && for f in $(ls | grep -v '^MANIFEST.txt$'); do
            if [[ -L "$f" ]]; then echo "symlink  $f -> $(readlink "$f")"
            else echo "$(sha256sum "$f" | cut -d' ' -f1)  $(stat -c %s "$f")  $f"; fi
        done)
    } > "$m"
    [[ "${DMK_CCACHE:-0}" == "1" ]] && { echo; echo "## ccache"; ccache --show-stats; } >> "$m"
    echo "== dist: $(ls "$out/dist" | wc -l) entries, manifest $m"
    exit 0
fi

# ------------------------------------------------------------------ host side
if [[ $# -lt 2 ]]; then
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    exit 2
fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
shift 2

[[ -f "$src/CMakeLists.txt" && -d "$src/ggml" ]] || { echo "build.sh: $src is not a llama.cpp tree" >&2; exit 1; }

image=${DMK_BUILDER_IMAGE:-localhost/dmk-builder:cuda12.8.1-gcc14}
if ! podman image exists "$image"; then
    echo "== builder image $image missing, building it from $here/Containerfile.builder"
    podman build -t "$image" -f "$here/Containerfile.builder" "$here"
fi

ccache_dir=${DMK_CCACHE_DIR:-$here/ccache}
mkdir -p "$ccache_dir"

lbl() { podman image inspect "$image" --format "{{ index .Config.Labels \"$1\" }}"; }

# A git worktree has a .git FILE that points at a gitdir outside $src. Mount that repository at the same path (read-only)
# so the manifest step (git rev-parse / status / diff) works inside the container. Plain clones need nothing.
gitmount=()
if [[ -f "$src/.git" ]]; then
    gd=$(sed -n 's/^gitdir: //p' "$src/.git")
    if [[ -n "$gd" && -f "$gd/commondir" ]]; then
        common=$(cd "$gd" && cd "$(cat commondir)" && pwd)
        gitmount=(-v "$common:$common:ro")
    fi
fi

# --security-opt label=disable: SELinux would otherwise block the bind mounts (and :Z would relabel the source tree)
podman run --rm --security-opt label=disable \
    -v "$src":/src:ro \
    "${gitmount[@]}" \
    -v "$out":/out \
    -v "$ccache_dir":/ccache \
    -v "$here/build.sh":/dmk/build.sh:ro \
    -e DMK_JOBS -e DMK_CUDA_ARCH -e DMK_TARGETS -e DMK_CLEAN -e DMK_CCACHE -e DMK_OUT_NAME="$(basename "$out")" \
    -e DMK_BASE_TAG="$(lbl dmk.base.tag)" \
    -e DMK_BASE_INDEX="$(lbl dmk.base.index)" \
    -e DMK_BASE_AMD64="$(lbl dmk.base.amd64)" \
    -e DMK_BUILDER_ID="$image $(podman image inspect "$image" --format '{{.Id}}')" \
    "$image" bash /dmk/build.sh --inside "$@" 2>&1 | tee "$out/build.log"
exit "${PIPESTATUS[0]}"
