#!/bin/bash
# One client run on the GPU host: GPU sampler, new server for <arm>, health wait,
# a Python client script in a container on the host network, teardown.
# Usage: run_client.sh <run-name> <arm> <script> [script args]
#   <script> is a path inside the client container: /bench/... or /extra/...
#   "{name}" in the args becomes <run-name>; write outputs to /runs/{name}.*
# Env: as start_server.sh, plus DMK_PY_IMAGE (python image pinned by digest).
# Output: $DMK_REMOTE_DIR/runs/<run-name>.{client.out,smi,server.log,launch.log,load0,load1,meta}
set -uo pipefail
name=${1:?usage: run_client.sh <run-name> <arm> <script> [args]}
arm=${2:?usage}
script=${3:?usage}
shift 3
here=$(cd "$(dirname "$0")" && pwd)
: "${DMK_REMOTE_DIR:?}" "${DMK_GPU:?}" "${DMK_PY_IMAGE:?}"
port=${DMK_PORT:-18080}
runs=$DMK_REMOTE_DIR/runs
mkdir -p "$runs"
out=$runs/$name
[ ! -e "$out.client.out" ] || { echo "run $name exists" >&2; exit 2; }

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

args=()
for a in "$@"; do args+=("${a//\{name\}/$name}"); done
mkdir -p "$DMK_REMOTE_DIR/extra"
docker run --rm --name dmx-bench --network host \
	-v "$DMK_REMOTE_DIR/bench:/bench:ro" -v "$DMK_REMOTE_DIR/extra:/extra:ro" -v "$runs:/runs" \
	"$DMK_PY_IMAGE" python -u "$script" "${args[@]}" 2>&1 | tee "$out.client.out"
exit "${PIPESTATUS[0]}"
