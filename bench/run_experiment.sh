#!/usr/bin/env bash
# Run one benchmark run from your workstation, on a GPU machine reached by SSH.
# Fixed harness.
# Usage: ./run_experiment.sh <run-name> [bench.py options, e.g. --cold 3 or --passes 7]
#
# Steps: copy bench/ and the launch script to the GPU machine, run
# bench/remote_run.sh there, copy the run files back to runs/, and print the
# summary. Settings come from the environment (see bench.env.example):
#   GPU_HOST       SSH target of the GPU machine (required)
#   GPU_WORK_DIR   folder on the GPU machine for bench/, launch.sh and runs/ (required)
#   BENCH_ENV_FILE local file with more settings (optional, default ./bench.env if it exists)
#   LAUNCH_SCRIPT  local launch script (default ./launch.sh)
set -euo pipefail
cd "$(dirname "$0")"
name=${1:?usage: ./run_experiment.sh <run-name> [bench options]}
shift
[[ $name =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run name must match [A-Za-z0-9._-]+" >&2; exit 2; }
envfile=${BENCH_ENV_FILE:-}
[ -n "$envfile" ] || { [ ! -f ./bench.env ] || envfile=./bench.env; }
if [ -n "$envfile" ]; then
	set -a
	# shellcheck disable=SC1090
	. "$envfile"
	set +a
fi
: "${GPU_HOST:?set GPU_HOST to the SSH target of the GPU machine}"
: "${GPU_WORK_DIR:?set GPU_WORK_DIR to a folder on the GPU machine}"
launch=${LAUNCH_SCRIPT:-./launch.sh}
[ -f "$launch" ] || { echo "launch script not found: $launch" >&2; exit 2; }
runs=${LOCAL_RUNS_DIR:-./runs}
mkdir -p "$runs"
[ ! -e "$runs/$name.json" ] || { echo "run $name already exists" >&2; exit 2; }

ssh "$GPU_HOST" "mkdir -p '$GPU_WORK_DIR'"
rsync -a --delete --exclude __pycache__ --exclude runs ./ "$GPU_HOST:$GPU_WORK_DIR/bench/"
rsync -a "$launch" "$GPU_HOST:$GPU_WORK_DIR/launch.sh"
remote_env=""
if [ -n "$envfile" ]; then
	rsync -a "$envfile" "$GPU_HOST:$GPU_WORK_DIR/bench.env"
	remote_env="BENCH_ENV_FILE=$GPU_WORK_DIR/bench.env"
fi
rc=0
args=""
[ $# -eq 0 ] || args=$(printf '%q ' "$@")
ssh "$GPU_HOST" "cd '$GPU_WORK_DIR' && $remote_env bash bench/remote_run.sh $name $args" | tee "$runs/$name.out" || rc=$?
rsync -a "$GPU_HOST:$GPU_WORK_DIR/runs/$name.*" "$runs/" 2>/dev/null || true
cp "$launch" "$runs/$name.launch.sh"
[ -f "$runs/$name.json" ] || { echo "run $name produced no result (rc=$rc)" >&2; exit 3; }
python3 summarize.py "$runs/$name"
exit "$rc"
