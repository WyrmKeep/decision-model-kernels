#!/bin/sh
# Run the server in compute-sanitizer (memcheck). When docker stop sends TERM, stop the server only.
# Then the sanitizer ends in the normal way and prints its summary.
/tools/compute-sanitizer-12.8.93/bin/compute-sanitizer --tool memcheck --error-exitcode 99 "$@" &
san=$!
stop() {
	for p in /proc/[0-9]*; do
		read -r c <"$p/comm" 2>/dev/null
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
