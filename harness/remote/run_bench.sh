#!/bin/bash
# One benchmark run on the GPU host: GPU sampler, new server for <arm>, health wait,
# benchmark client in a Python container on the host network, teardown.
# Usage: run_bench.sh <run-name> <arm> [bench.py options]
# Env: as start_server.sh, plus DMK_PY_IMAGE (python image pinned by digest).
# Output: $DMK_REMOTE_DIR/runs/<run-name>.{json,smi,server.log,launch.log,load0,load1,meta}
set -uo pipefail
name=${1:?usage: run_bench.sh <run-name> <arm> [bench options]}
arm=${2:?usage: run_bench.sh <run-name> <arm> [bench options]}
shift 2
here=$(cd "$(dirname "$0")" && pwd)
: "${DMK_REMOTE_DIR:?}" "${DMK_GPU:?}" "${DMK_PY_IMAGE:?}"
port=${DMK_PORT:-18080}
runs=$DMK_REMOTE_DIR/runs
mkdir -p "$runs"
out=$runs/$name
[ ! -e "$out.json" ] || { echo "run $name exists" >&2; exit 2; }

teardown() {
	if [ -f "$out.smi.pid" ]; then
		kill -- "-$(cat "$out.smi.pid")" 2>/dev/null
		rm -f "$out.smi.pid"
	fi
	docker logs dmx-server >"$out.server.log" 2>&1
	docker rm -f dmx-server dmx-bench >/dev/null 2>&1
	cat /proc/loadavg >"$out.load1"
}
trap teardown EXIT

docker rm -f dmx-server dmx-bench >/dev/null 2>&1
cat /proc/loadavg >"$out.load0"
{
	echo "arm=$arm"
	echo "date_utc=$(date -u +%FT%TZ)"
	nvidia-smi -i "$DMK_GPU" --query-gpu=driver_version,power.limit,clocks_throttle_reasons.active --format=csv,noheader
} >"$out.meta" 2>&1
setsid bash -c "nvidia-smi -i $DMK_GPU --query-gpu=memory.used,power.draw,pstate,clocks.sm --format=csv,noheader,nounits -lms 250 | while IFS= read -r l; do echo \"\$(date +%s.%N), \$l\"; done" >"$out.smi" 2>&1 </dev/null &
echo $! >"$out.smi.pid"

t0=$(date +%s.%N)
if ! bash "$here/start_server.sh" "$arm" >"$out.launch.log" 2>&1; then
	echo "LAUNCH FAILED"
	tail -n 30 "$out.launch.log"
	exit 3
fi
ok=0
for _ in $(seq 1 1200); do
	if curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
		ok=1
		break
	fi
	[ "$(docker inspect -f '{{.State.Running}}' dmx-server 2>/dev/null)" = true ] || break
	sleep 0.25
done
if [ $ok != 1 ]; then
	echo "SERVER NOT HEALTHY"
	docker logs --tail 40 dmx-server 2>&1
	exit 3
fi
startup=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f", b - a }')
echo "startup_s=$startup"

docker run --rm --name dmx-bench --network host \
	-v "$DMK_REMOTE_DIR/bench:/bench:ro" -v "$runs:/runs" \
	"$DMK_PY_IMAGE" python -u /bench/bench.py --port "$port" --out "/runs/$name.json" \
	--startup-s "$startup" "$@"
