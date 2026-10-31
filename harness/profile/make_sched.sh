#!/usr/bin/env bash
# Write ready schedule files for harness/run_schedule.sh: copies the files of sched/ (next to this script) with the
# placeholder @REMOTE_DIR@ replaced by the work folder of the GPU host. Nothing is written into the repository tree
# unless you point --out into it.
# Usage: bash make_sched.sh [--remote-dir DIR] [--out OUTDIR] [FILE...]
#   --remote-dir DIR  work folder on the GPU host (the DMK_REMOTE_DIR of the harness env file; default: $DMK_REMOTE_DIR).
#                     It is needed only when a chosen file contains @REMOTE_DIR@ (nsys and ncu schedules).
#   --out OUTDIR      folder for the written files (default ./sched-out)
#   FILE...           schedule files: a name in sched/ (2-nsys.txt) or a path; default all of sched/*.txt
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
remote=${DMK_REMOTE_DIR:-}
out=./sched-out
files=()
while [ $# -gt 0 ]; do
	case $1 in
	--remote-dir) [ $# -ge 2 ] || { echo "make_sched.sh: --remote-dir needs a value" >&2; exit 2; }; remote=$2; shift 2 ;;
	--out) [ $# -ge 2 ] || { echo "make_sched.sh: --out needs a value" >&2; exit 2; }; out=$2; shift 2 ;;
	-h | --help) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; exit 0 ;;
	-*) echo "make_sched.sh: unknown option $1" >&2; exit 2 ;;
	*) files+=("$1"); shift ;;
	esac
done
if [ ${#files[@]} -eq 0 ]; then
	for f in "$here"/sched/*.txt; do files+=("$f"); done
fi
if [ -n "$remote" ]; then
	case $remote in
	*[\|\&\\\;\"\'\ ]*) echo "make_sched.sh: the remote dir must not contain spaces, quotes, | & \\ or ;: $remote" >&2; exit 2 ;;
	esac
fi
mkdir -p "$out"
for f in "${files[@]}"; do
	src=$f
	[ -f "$src" ] || src=$here/sched/$f
	[ -f "$src" ] || { echo "make_sched.sh: no such schedule file: $f" >&2; exit 2; }
	if grep -q '@REMOTE_DIR@' "$src" && [ -z "$remote" ]; then
		echo "make_sched.sh: $f needs @REMOTE_DIR@: use --remote-dir DIR or set DMK_REMOTE_DIR" >&2
		exit 2
	fi
	dest=$out/$(basename "$src")
	sed "s|@REMOTE_DIR@|${remote%/}|g" "$src" >"$dest"
	echo "wrote $dest"
done
