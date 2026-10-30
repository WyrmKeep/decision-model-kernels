#!/bin/bash
# Example launch script for remote_run.sh. Copy it to launch.sh and edit it.
# Contract:
#   - start one container named $SERVER_CONTAINER (default dmx-server) and return;
#   - the server must answer http://127.0.0.1:$BENCH_PORT on this machine
#     (default port 18080) and must listen on the loopback address only;
#   - the harness waits for /health.
# The variables come from the environment or from bench.env (see bench.env.example).
set -euo pipefail
: "${IMAGE:?set IMAGE to the llama.cpp server image}"
: "${MODEL_DIR:?set MODEL_DIR to the folder that holds the model file}"
: "${MODEL:?set MODEL to the model file name}"
CLI=${CONTAINER_CLI:-docker}
NAME=${SERVER_CONTAINER:-dmx-server}
PORT=${BENCH_PORT:-18080}
GPU_FLAGS=${GPU_FLAGS:---gpus all}
SERVER_ARGS=${SERVER_ARGS:--ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots}

# shellcheck disable=SC2086
"$CLI" run -d --name "$NAME" $GPU_FLAGS \
	-v "$MODEL_DIR:/models:ro" -p "127.0.0.1:$PORT:8080" \
	"$IMAGE" \
	-m "/models/$MODEL" --host 0.0.0.0 --port 8080 $SERVER_ARGS
