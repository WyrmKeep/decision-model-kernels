#!/usr/bin/env bash
# Scan the files of a directory with the gate rules.
#
# Usage: scan-dir.sh [directory]      (default: the current directory)
#
# It runs `gitleaks dir` twice: with the default, public and private rules, then
# with the public and private rules alone (no default allowlists).
# Findings print as <file>:<line> <rule>. The text itself stays redacted.
# Exit code: 0 clean, 1 findings, 2 setup error.
set -u
export LC_ALL=C

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib.sh
. "$here/lib.sh"

case ${1:-} in -h|--help) sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;; esac
dir=${1:-.}
[ -d "$dir" ] || { echo "scan-dir: $dir is not a directory" >&2; exit 2; }
command -v "${DMK_GITLEAKS:-gitleaks}" >/dev/null 2>&1 || { echo "scan-dir: gitleaks is not installed or not in PATH (set DMK_GITLEAKS)" >&2; exit 2; }
# gitleaks reads .gitleaksignore from the scanned directory and would hide findings.
if [ -e "$dir/.gitleaksignore" ]; then
  echo "scan-dir: $dir/.gitleaksignore exists. It can hide findings. Remove it." >&2
  exit 2
fi

dmk_init_tmp || exit 2
trap dmk_cleanup EXIT
DMK_PUBLIC_CONFIG=$here/gitleaks.toml
dmk_require_private_rules || exit 2
dmk_build_configs || exit 2

dmk_scan_dir "$dir"
rc=$?
case $rc in
  0) dmk_say "scan-dir: $dir is clean"; exit 0 ;;
  1) exit 1 ;;
  *) exit 2 ;;
esac
