#!/bin/sh
# Server wrapper for run_soak.sh: it starts the server in compute-sanitizer (memcheck).
# docker stop sends SIGTERM to this script. The script then stops the server only.
# The sanitizer then ends in the normal way and prints its summary.
# Usage in the container: san_wrap.sh <server program> <server flags>
# Env: DMK_SANITIZER is the path of compute-sanitizer in the container.
"$DMK_SANITIZER" --tool memcheck --error-exitcode 99 "$@" &
san=$!
stop() {
	for p in /proc/[0-9]*; do
		read -r c 2>/dev/null <"$p/comm"
		[ "$c" = llama-server ] && kill -TERM "${p#/proc/}"
	done
}
trap stop TERM INT
wait "$san"
rc=$?
while kill -0 "$san" 2>/dev/null; do
	wait "$san"
	rc=$?
done
exit "$rc"
