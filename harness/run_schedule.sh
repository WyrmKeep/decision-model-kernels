#!/usr/bin/env bash
# Run a schedule of experiments on the GPU host, one line at a time, in file order.
# Usage: DMK_ENV=<env file> run_schedule.sh <schedule file> <local results dir>
# Schedule lines:  bench  <run-name> <arm> [bench.py options]
#                  client <run-name> <arm> <script in /bench or /extra> [args]
#                  tool   <run-name> <arm> <llama command> [args]
#                  tool@<w> ...   the same as tool, under the wrapper $DMK_WRAP_<w>
#                  # comment
# A line may start with KEY=VALUE overrides of DMK_* settings for that line only,
# e.g.  DMK_MODEL=other.gguf bench r1 stock. A value with spaces goes in double
# quotes and must not contain quotes: DMK_SERVER_WRAP="/tools/x/nsys profile -o /runs/r1"
# DMK_EXTRA_DIR (optional, local) is copied to the host and mounted at /extra for clients.
# The env file sets DMK_HOST (ssh destination), DMK_BENCH_DIR (local benchmark dir),
# and the DMK_* variables that the remote scripts need (see harness.env.example).
set -euo pipefail
sched=${1:?usage: run_schedule.sh <schedule> <results dir>}
results=${2:?usage: run_schedule.sh <schedule> <results dir>}
# shellcheck disable=SC1090
source "${DMK_ENV:?set DMK_ENV to the env file}"
: "${DMK_HOST:?}" "${DMK_REMOTE_DIR:?}" "${DMK_BENCH_DIR:?}"
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$results"

ssh "$DMK_HOST" "mkdir -p $DMK_REMOTE_DIR/harness $DMK_REMOTE_DIR/bench $DMK_REMOTE_DIR/runs"
rsync -a --delete "$here/remote/" "$DMK_HOST:$DMK_REMOTE_DIR/harness/"
rsync -a --delete --exclude __pycache__ "$DMK_BENCH_DIR/" "$DMK_HOST:$DMK_REMOTE_DIR/bench/"
if [ -n "${DMK_EXTRA_DIR:-}" ]; then
	ssh "$DMK_HOST" "mkdir -p $DMK_REMOTE_DIR/extra"
	rsync -a --delete --exclude __pycache__ "$DMK_EXTRA_DIR/" "$DMK_HOST:$DMK_REMOTE_DIR/extra/"
fi
# The remote scripts read the DMK_* settings from run.env (host-only keys removed).
grep -v -E '^(export )?DMK_(HOST|BENCH_DIR|EXTRA_DIR)=' "$DMK_ENV" | ssh "$DMK_HOST" "cat > $DMK_REMOTE_DIR/harness/run.env"

fail=0
while read -r line; do
	[[ -z ${line// /} || $line == \#* ]] && continue
	envs=
	while [[ $line =~ ^(DMK_[A-Z0-9_]+)=(\"([^\"\']*)\"|[^[:space:]\"\']+)[[:space:]]+(.*)$ ]]; do
		key=${BASH_REMATCH[1]}
		val=${BASH_REMATCH[2]}
		[[ $val == \"*\" ]] && val=${BASH_REMATCH[3]}
		envs+="$key='$val' "
		line=${BASH_REMATCH[4]}
	done
	read -r kind name rest <<<"$line"
	echo "== $(date -u +%T) $envs$kind $name $rest"
	pre=
	case $kind in
	bench) script=run_bench.sh ;;
	client) script=run_client.sh ;;
	tool) script=run_tool.sh ;;
	tool@*)
		w=${kind#tool@}
		[[ $w =~ ^[A-Za-z0-9_]+$ ]] || { echo "bad wrapper name: $w" >&2; exit 2; }
		script=run_tool.sh
		pre="DMK_WRAP=\"\$DMK_WRAP_$w\""
		;;
	*) echo "bad line: $kind" >&2; exit 2 ;;
	esac
	if ! ssh -n "$DMK_HOST" "set -a; . $DMK_REMOTE_DIR/harness/run.env; set +a; $envs$pre bash $DMK_REMOTE_DIR/harness/$script $name $rest"; then
		echo "!! $name failed"
		fail=$((fail + 1))
	fi
	rsync -a "$DMK_HOST:$DMK_REMOTE_DIR/runs/$name.*" "$results/" 2>/dev/null || true
done <"$sched"
cp "$sched" "$results/"
echo "done, $fail failed"
exit $((fail > 0))
