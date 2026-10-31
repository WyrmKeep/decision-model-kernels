#!/usr/bin/env bash
# sass_check.sh <out folder> <baseline libggml-cuda.so> <new libggml-cuda.so> [name-regex]
#
# Compares the sm_86 SASS of the ablation build with the baseline build. No environment variable is set.
# It starts the builder image of harness/build with podman. In the container it extracts the cubins of both
# libraries with cuobjdump and runs harness/build/sass_compare.py. The two builds have the same SASS when
# names_sass_different, names_encoding_different, names_only_A, names_only_B and names_resusage_different
# of <out folder>/cmp/sass-compare.json are all 0.
#
# The script removes the cubins at the end. Set DMK_KEEP_CUBINS=1 to keep them in <out folder>/base and
# <out folder>/new. DMK_BUILDER_IMAGE sets the builder image (default localhost/dmk-builder:cuda12.8.1-gcc14).
# The default name-regex gives the kernels of the model; the script writes their SASS to <out folder>/cmp.
set -euo pipefail
usage="usage: sass_check.sh <out folder> <baseline libggml-cuda.so> <new libggml-cuda.so> [name-regex]"
out=$(realpath -m "${1:?$usage}")
base=$(realpath "${2:?$usage}")
new=$(realpath "${3:?$usage}")
keys=${4:-"mul_mat_q|quantize|gated_delta|gdn|flash_attn|rms_norm|k_get_rows|k_bin_bcast|unary|glu|ssm_conv|cpy|concat|l2_norm|rope"}
compare=$(cd "$(dirname "${BASH_SOURCE[0]}")/../build" && pwd)/sass_compare.py
image=${DMK_BUILDER_IMAGE:-localhost/dmk-builder:cuda12.8.1-gcc14}
mkdir -p "$out"
podman run --rm --security-opt label=disable \
    -v "$compare":/dmk/sass_compare.py:ro -v "$(dirname "$base")":/base:ro -v "$(dirname "$new")":/new:ro -v "$out":/out \
    -e BASE="/base/$(basename "$base")" -e NEW="/new/$(basename "$new")" -e KEYS="$keys" -e KEEP="${DMK_KEEP_CUBINS:-0}" \
    "$image" bash -c '
  set -euo pipefail
  rm -rf /out/base /out/new /out/cmp
  mkdir -p /out/base /out/new
  (cd /out/base && cuobjdump -arch sm_86 -xelf all "$BASE" >/out/base-extract.log) &
  (cd /out/new && cuobjdump -arch sm_86 -xelf all "$NEW" >/out/new-extract.log) &
  wait
  echo "cubins: base $(ls /out/base | wc -l)  new $(ls /out/new | wc -l)"
  python3 /dmk/sass_compare.py /out/base /out/new /out/cmp "$KEYS"
  if [ "$KEEP" != 1 ]; then rm -rf /out/base /out/new; fi'
