#!/bin/bash
# Run one `llama <command>` (bench, perplexity, ...) of an arm on the GPU host, in a
# throwaway container, with the model directory and a work directory mounted.
# Usage: run_tool.sh <run-name> <arm> <command> [args...]
#   Paths in args: /models/<file> (read-only models), /work/<file> (read-write
#   $DMK_REMOTE_DIR/work), /shm/<file> (read-write host tmpfs dir $DMK_SHM_DIR).
# Output: $DMK_REMOTE_DIR/runs/<run-name>.{out,err,meta,smi}
set -uo pipefail
name=${1:?usage}; arm=${2:?usage}; cmd=${3:?usage}
shift 3
: "${DMK_REMOTE_DIR:?}" "${DMK_GPU:?}" "${DMK_IMAGE:?}" "${DMK_MODELS:?}"
runs=$DMK_REMOTE_DIR/runs
work=$DMK_REMOTE_DIR/work
shm=${DMK_SHM_DIR:-/dev/shm/dmk}
mkdir -p "$runs" "$work" "$shm"
out=$runs/$name
[ ! -e "$out.out" ] || { echo "run $name exists" >&2; exit 2; }

run=(docker run --rm --name dmx-tool --runtime=nvidia
	-e NVIDIA_VISIBLE_DEVICES="$DMK_GPU" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
	-v "$DMK_MODELS:/models:ro" -v "$work:/work" -v "$shm:/shm")
# Optional: a second read-only model directory (e.g. the Q8_0 reference) at /refmodels.
[ -n "${DMK_REF_MODELS:-}" ] && run+=(-v "$DMK_REF_MODELS:/refmodels:ro")
# Optional: run the tool under a wrapper (ncu, compute-sanitizer) from $DMK_REMOTE_DIR/tools,
# mounted at /tools. DMK_WRAP holds the wrapper command, e.g.
# "/tools/compute-sanitizer-12.8.93/compute-sanitizer --tool memcheck". DMK_TOOL_CAPS adds caps.
wrap=()
if [ -n "${DMK_WRAP:-}" ]; then
	run+=(-v "$DMK_REMOTE_DIR/tools:/tools:ro")
	# shellcheck disable=SC2206
	wrap=(${DMK_WRAP})
fi
[ -n "${DMK_TOOL_CAPS:-}" ] && run+=(--cap-add "$DMK_TOOL_CAPS")
case $arm in
stock) bin=/app/llama ;;
build:*)
	b=${arm#build:}
	[[ $b =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad build name: $b" >&2; exit 2; }
	dir=$DMK_REMOTE_DIR/builds/$b
	[ -d "$dir/dist" ] && dir=$dir/dist
	run+=(-v "$dir:/opt/dmk:ro" -w /opt/dmk -e LD_LIBRARY_PATH=/opt/dmk)
	bin=/opt/dmk/llama
	;;
*) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

cleanup() {
	[ -f "$out.smi.pid" ] && { kill -- "-$(cat "$out.smi.pid")" 2>/dev/null; rm -f "$out.smi.pid"; }
	docker rm -f dmx-tool >/dev/null 2>&1
}
trap cleanup EXIT
docker rm -f dmx-tool >/dev/null 2>&1
{
	echo "arm=$arm command=$cmd args=$*"
	echo "date_utc=$(date -u +%FT%TZ)"
	cat /proc/loadavg
	nvidia-smi -i "$DMK_GPU" --query-gpu=driver_version,power.limit --format=csv,noheader
} >"$out.meta" 2>&1
setsid bash -c "nvidia-smi -i $DMK_GPU --query-gpu=memory.used,power.draw,pstate,clocks.sm --format=csv,noheader,nounits -lms 250 | while IFS= read -r l; do echo \"\$(date +%s.%N), \$l\"; done" >"$out.smi" 2>&1 </dev/null &
echo $! >"$out.smi.pid"
if [ ${#wrap[@]} -gt 0 ]; then
	"${run[@]}" --entrypoint "${wrap[0]}" "$DMK_IMAGE" "${wrap[@]:1}" "$bin" "$cmd" "$@" >"$out.out" 2>"$out.err"
else
	"${run[@]}" --entrypoint "$bin" "$DMK_IMAGE" "$cmd" "$@" >"$out.out" 2>"$out.err"
fi
rc=$?
echo "rc=$rc" >>"$out.meta"
cat /proc/loadavg >>"$out.meta"
exit $rc
