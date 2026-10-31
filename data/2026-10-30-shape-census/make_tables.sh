#!/usr/bin/env bash
# make_tables.sh - make all derived tables and the report of this data set from the files in raw/.
#
# Usage:   bash make_tables.sh [--out DIR]
#   no argument   write tables/ and reports/ in this folder
#   --out DIR     write DIR/tables and DIR/reports instead (to compare with the files of this folder, for example
#                 `diff -r DIR/tables tables`). The raw files are always read from this folder.
#
# The script needs only python3 (standard library). It does not need the model, the tokenizer or the Nsight Systems captures.
# It runs these programs of harness/profile/census:
#   03_census.py     raw/model-header, raw/tokens  ->  census*.tsv and request-floors.tsv
#   04_crosscheck.py raw/traces, raw/tokens        ->  crosscheck-*.tsv
#   05_report.py     raw/*, tables/                ->  reports/census-report.txt
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$HERE/../../harness/profile/census"
OUT="$HERE"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

export PYTHONDONTWRITEBYTECODE=1
export DMK_CENSUS_DATA="$HERE"
mkdir -p "$OUT/tables" "$OUT/reports"

python3 "$CODE/03_census.py" --out "$OUT/tables"
python3 "$CODE/04_crosscheck.py" --out "$OUT/tables"
python3 "$CODE/05_report.py" --tables "$OUT/tables" --out "$OUT/reports"
