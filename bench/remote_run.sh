#!/bin/bash
# Run one benchmark run on the machine that has the GPU. Fixed harness.
# Usage: remote_run.sh <run-name> [bench.py options]
#
# Steps: start a GPU sampler, run the launch script, wait until the server
# answers /health, run bench.py in a Python container on the host network, and
# clean up. Settings come from the environment or from the file BENCH_ENV_FILE.
# See bench.env.example for every variable.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
name=${1:?usage: remote_run.sh <run-name> [bench.py options]}
shift
[[ $name =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run name must match [A-Za-z0-9._-]+" >&2; exit 2; }

if [ -n "${BENCH_ENV_FILE:-}" ]; then
	set -a
	# shellcheck disable=SC1090
	. "$BENCH_ENV_FILE"
	set +a
fi

CLI=${CONTAINER_CLI:-docker}
SERVER=${SERVER_CONTAINER:-dmx-server}
CLIENT=${BENCH_CONTAINER:-dmx-bench}
GPU_ID=${GPU_ID:-0}
PY=${PY_IMAGE:-python:3.12-slim}
LAUNCH=${BENCH_LAUNCH:-$PWD/launch.sh}
RUNS=${BENCH_RUNS_DIR:-$PWD/runs}
HEALTH_URL=${HEALTH_URL:-http://127.0.0.1:${BENCH_PORT:-18080}/health}
HEALTH_WAIT_S=${HEALTH_WAIT_S:-300}
CLIENT_FLAGS=${CLIENT_FLAGS:-}
RESET_GPU_CLOCKS=${RESET_GPU_CLOCKS:-0}
export SERVER_CONTAINER=$SERVER

[ -f "$LAUNCH" ] || { echo "launch script not found: $LAUNCH (set BENCH_LAUNCH)" >&2; exit 2; }
mkdir -p "$RUNS"
out=$RUNS/$name

teardown() {
	if [ -f "$out.smi.pid" ]; then
		kill -- "-$(cat "$out.smi.pid")" 2>/dev/null
		rm -f "$out.smi.pid"
	fi
	"$CLI" logs "$SERVER" >"$out.server.log" 2>&1
	"$CLI" rm -f "$SERVER" "$CLIENT" >/dev/null 2>&1
	if [ "$RESET_GPU_CLOCKS" = 1 ]; then
		# Undo any clock lock that the launch script made (no-op when none was set).
		nvidia-smi -i "$GPU_ID" -rgc >/dev/null 2>&1
		nvidia-smi -i "$GPU_ID" -rmc >/dev/null 2>&1
	fi
	cat /proc/loadavg >"$out.load1"
}
trap teardown EXIT

"$CLI" rm -f "$SERVER" "$CLIENT" >/dev/null 2>&1
cat /proc/loadavg >"$out.load0"
setsid bash -c "nvidia-smi -i $GPU_ID --query-gpu=memory.used,power.draw,pstate,clocks.sm --format=csv,noheader,nounits -lms 250 | while IFS= read -r l; do echo \"\$(date +%s.%N), \$l\"; done" >"$out.smi" 2>&1 </dev/null &
echo $! >"$out.smi.pid"

t0=$(date +%s.%N)
if ! bash "$LAUNCH" >"$out.launch.log" 2>&1; then
	echo "LAUNCH FAILED"
	tail -n 30 "$out.launch.log"
	exit 3
fi
ok=0
for _ in $(seq 1 $((HEALTH_WAIT_S * 4))); do
	if curl -sf -m 2 "$HEALTH_URL" >/dev/null 2>&1; then
		ok=1
		break
	fi
	[ "$("$CLI" inspect -f '{{.State.Running}}' "$SERVER" 2>/dev/null)" = true ] || break
	sleep 0.25
done
if [ $ok != 1 ]; then
	echo "SERVER NOT HEALTHY"
	"$CLI" logs --tail 40 "$SERVER" 2>&1
	exit 3
fi
startup=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f", b - a }')
echo "startup_s=$startup"

# shellcheck disable=SC2086
"$CLI" run --rm --name "$CLIENT" --network host $CLIENT_FLAGS \
	-e BENCH_HOST -e BENCH_PORT \
	-v "$here:/bench:ro" -v "$RUNS:/runs" \
	"$PY" python -u /bench/bench.py --out "/runs/$name.json" --startup-s "$startup" "$@"
