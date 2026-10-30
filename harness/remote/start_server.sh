#!/bin/bash
# Start the experiment server container (dmx-server) on the GPU host for one arm.
# Usage: start_server.sh <arm> [extra llama-server args]
#   arm = stock            the image's own /app binaries
#   arm = build:<name>     binaries in $DMK_REMOTE_DIR/builds/<name>, mounted at /opt/dmk
# Env (from harness.env): DMK_REMOTE_DIR DMK_GPU DMK_IMAGE DMK_MODELS DMK_MODEL
#   optional: DMK_PORT (default 18080), DMK_SERVER_ARGS (default: the production flags),
#   DMK_CONTAINER (default dmx-server), DMK_DOCKER_EXTRA (extra docker run args)
set -euo pipefail
arm=${1:?usage: start_server.sh <arm> [extra server args]}
shift
: "${DMK_REMOTE_DIR:?}" "${DMK_GPU:?}" "${DMK_IMAGE:?}" "${DMK_MODELS:?}" "${DMK_MODEL:?}"
port=${DMK_PORT:-18080}
name=${DMK_CONTAINER:-dmx-server}
server_args=${DMK_SERVER_ARGS:--ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots}

run=(docker run -d --name "$name" --runtime=nvidia
	-e NVIDIA_VISIBLE_DEVICES="$DMK_GPU" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
	-v "$DMK_MODELS:/models:ro" -p "127.0.0.1:$port:8080")
case $arm in
stock) ;;
build:*)
	b=${arm#build:}
	[[ $b =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad build name: $b" >&2; exit 2; }
	dir=$DMK_REMOTE_DIR/builds/$b
	[ -d "$dir/dist" ] && dir=$dir/dist
	[ -x "$dir/llama-server" ] || { echo "no llama-server in $dir" >&2; exit 2; }
	run+=(-v "$dir:/opt/dmk:ro" -w /opt/dmk -e LD_LIBRARY_PATH=/opt/dmk --entrypoint /opt/dmk/llama-server)
	;;
*) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
# shellcheck disable=SC2206
extra_docker=(${DMK_DOCKER_EXTRA:-})
# shellcheck disable=SC2206
"${run[@]}" "${extra_docker[@]}" "$DMK_IMAGE" -m "/models/$DMK_MODEL" --host 0.0.0.0 --port 8080 $server_args "$@"
