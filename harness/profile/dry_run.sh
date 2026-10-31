#!/usr/bin/env bash
# Parse schedule files with the REAL parser of harness/run_schedule.sh and run nothing: ssh and rsync are replaced
# by printers. For every schedule line it prints the line as the harness sees it and, from the remote command that the
# harness would send, what the remote script receives (script name, the words of its argument list, the DMK_* values).
# Usage: bash dry_run.sh <schedule.txt> [<schedule.txt> ...]    (schedule files written by make_sched.sh)
# Env: DMK_ENV (optional) = the env file of the harness (harness/harness.env.example); without it a fake env file is
#      written to a temporary folder (host dry-run-host, remote dir /dry-run/remote, the repository bench and extra dirs).
# Needs nothing but bash. Does not touch a GPU host or the network.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_harness=$(cd "$here/.." && pwd)
[ $# -gt 0 ] || { echo "usage: bash dry_run.sh <schedule.txt> [<schedule.txt> ...]" >&2; exit 2; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"

if [ -z "${DMK_ENV:-}" ]; then
	cat >"$tmp/dry-run.env" <<EOF
DMK_HOST=dry-run-host
DMK_BENCH_DIR=$repo_harness/../bench
DMK_EXTRA_DIR=$here/extra
DMK_REMOTE_DIR=/dry-run/remote
DMK_GPU=GPU-dry-run
DMK_IMAGE=dry-run-image
DMK_PY_IMAGE=dry-run-py-image
DMK_MODELS=/dry-run/models
DMK_MODEL=dry-run.gguf
EOF
	DMK_ENV=$tmp/dry-run.env
fi
export DMK_ENV

cat >"$tmp/bin/rsync" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat >"$tmp/bin/ssh" <<'EOF'
#!/usr/bin/env bash
# fake ssh: the last argument is the remote command
cmd=${!#}
if [[ $cmd != *"harness/run_"* ]]; then
	[ -t 0 ] || cat >/dev/null
	exit 0
fi
set -a
# shellcheck disable=SC1090
. "$DMK_ENV"
set +a
body=${cmd#*set +a; }
eval 'bash() { local s=$1; shift; echo "   remote script: ${s##*/}"; local i=0 a; for a in "$@"; do echo "   arg[$i] = $a"; i=$((i+1)); done
  local v; for v in DMK_SERVER_WRAP DMK_DOCKER_EXTRA DMK_SERVER_ARGS DMK_TOOL_CAPS DMK_STOP_TIMEOUT DMK_WRAP DMK_MODEL DMK_PORT DMK_CONTAINER; do
    if [ -n "${!v:-}" ]; then echo "   env $v = ${!v}"; fi; done; }; '"$body"
EOF
chmod +x "$tmp/bin/ssh" "$tmp/bin/rsync"

for s in "$@"; do
	echo "######## $s"
	PATH="$tmp/bin:$PATH" bash "$repo_harness/run_schedule.sh" "$s" "$tmp/results-$(basename "$s" .txt)"
done
