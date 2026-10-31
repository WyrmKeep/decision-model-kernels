#!/usr/bin/env bash
# run_all.sh - every step of the shape and launch census, with its inputs. CPU only: nothing here touches the GPU.
#
# Usage:   bash run_all.sh [--tables-only]
#   (no argument)    run all steps. Steps 1, 2 and T need private inputs (see the variables below).
#   --tables-only    run only steps 3, 4 and 5. They need only the data set, not the model, the tokenizer or the captures.
#                    (data/2026-10-30-shape-census/make_tables.sh does the same and can write to another folder.)
#
# The data set folder is $DMK_CENSUS_DATA (default: data/2026-10-30-shape-census of this repository). Each step writes there.
#
# Step  Program                              Needs                                               Writes
# ----  -----------------------------------  --------------------------------------------------  -----------------------------------------------
#  1    01_gguf_header.py                    DMK_GGUF: the mix8 GGUF file (only its first        raw/model-header/tensors.tsv, hparams.tsv,
#                                            16 MB are read); GGUF_PY: gguf-py of llama.cpp      layers.tsv
#  2    02_tokens.py                         DMK_GGUF, GGUF_PY, DMK_TOKENIZE: llama-tokenize     raw/tokens/token-counts.tsv, cases-plan.tsv,
#                                            of the b12004 build; bench/cases.json; the          token-check.tsv, ref-line-tokens.tsv
#                                            benchmark run of data/2026-10-30-window1
#  T    trace_export.sh + trace_census.py    DMK_LEGACY_TRACES: the folder with long, medium,    raw/traces/trace-kernels.tsv, trace-passes.tsv,
#                                            short as .nsys-rep; DMK_NSYS: the nsys command      trace-mmq-launches.tsv
#                                            line tool (or DMK_NSYS_DEB: a Nsight Systems .deb)
#  3    03_census.py                         the data set (raw/model-header, raw/tokens)         tables/census*.tsv, request-floors.tsv
#  4    04_crosscheck.py                     the data set (raw/*)                                tables/crosscheck-*.tsv
#  5    05_report.py                         the data set (raw/*, tables/)                       reports/census-report.txt
#
# Steps 1 and 2 run with `uv run --with ...` for numpy, pyyaml, tqdm, requests (gguf-py) and jinja2. The trace step exports the
# captures to SQLite in a temporary folder ($TMPDIR) that is removed at the end.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONDONTWRITEBYTECODE=1

REPO="$(cd ../../.. && pwd)"
DATA="${DMK_CENSUS_DATA:-$REPO/data/2026-10-30-shape-census}"
export DMK_CENSUS_DATA="$DATA"

TABLES_ONLY=0
case "${1:-}" in
    "") ;;
    --tables-only) TABLES_ONLY=1 ;;
    *) echo "usage: bash run_all.sh [--tables-only]" >&2; exit 2 ;;
esac

if [[ $TABLES_ONLY -eq 0 ]]; then
    : "${DMK_GGUF:?set DMK_GGUF to the mix8 GGUF file (steps 1 and 2)}"
    : "${GGUF_PY:?set GGUF_PY to the gguf-py directory of llama.cpp b12004 (steps 1 and 2)}"
    : "${DMK_TOKENIZE:?set DMK_TOKENIZE to the llama-tokenize program of the b12004 build (step 2)}"
    : "${DMK_LEGACY_TRACES:?set DMK_LEGACY_TRACES to the folder with the three .nsys-rep captures (step T)}"

    echo "== 1 GGUF header -> raw/model-header/tensors.tsv hparams.tsv layers.tsv"
    uv run --with numpy --with pyyaml --with tqdm --with requests python 01_gguf_header.py \
        --gguf "$DMK_GGUF" --gguf-py "$GGUF_PY" --out "$DATA/raw/model-header"

    echo "== 2 prompts and token counts -> raw/tokens/token-counts.tsv cases-plan.tsv token-check.tsv ref-line-tokens.tsv"
    uv run --with jinja2 --with numpy --with pyyaml --with tqdm --with requests python 02_tokens.py \
        --tokenize "$DMK_TOKENIZE" --gguf "$DMK_GGUF" --gguf-py "$GGUF_PY" --out "$DATA/raw/tokens"

    echo "== T trace export + trace census -> raw/traces/trace-*.tsv"
    sqlite_dir="$(mktemp -d "${TMPDIR:-/tmp}/census-sqlite.XXXXXX")"
    trap 'rm -rf "$sqlite_dir"' EXIT
    bash trace_export.sh --in "$DMK_LEGACY_TRACES" --out "$sqlite_dir"
    python3 trace_census.py --in "$sqlite_dir" --out "$DATA/raw/traces"
fi

echo "== 3 analytic census -> tables/census*.tsv request-floors.tsv"
python3 03_census.py --out "$DATA/tables"
echo "== 4 cross-check -> tables/crosscheck-*.tsv"
python3 04_crosscheck.py --out "$DATA/tables"
echo "== 5 report -> reports/census-report.txt"
python3 05_report.py --tables "$DATA/tables" --out "$DATA/reports"
