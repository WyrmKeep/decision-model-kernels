#!/bin/bash
# Soak test of one arm on the GPU host. One server stays up for a fixed time. The benchmark runs
# against it in passes, one after the other. A check runs at a fixed interval. A short
# compute-sanitizer run (memcheck) of the same arm comes before the soak and after it.
# Usage: run_soak.sh <run-name> <arm> <duration>
#   <duration> is a number of seconds, or numbers with the units s, m, h and d: 90m, 24h, 1h30m.
# Env: as start_server.sh, plus DMK_PY_IMAGE and
#   DMK_SOAK_SANITIZER  the path of compute-sanitizer in the server container, below /tools
#   optional: DMK_SOAK_INTERVAL (seconds between two checks, default 1800),
#   DMK_SOAK_BENCH_ARGS (bench.py flags for each pass, default none),
#   DMK_SOAK_PASS_TIMEOUT (longest time of one pass in seconds, default 1200),
#   DMK_SOAK_SAN_TIMEOUT (longest time of the client of one sanitizer phase, default 1800),
#   DMK_SOAK_SMI_MS (time between two samples of the GPU sampler, default 1000),
#   DMK_SOAK_MIN_UNALLOC_MIB (stop the soak if the btrfs folder of Docker has less unallocated
#   space, in MiB; default 0 = no limit).
# The production server must not run on the GPU. The script does not stop it.
# Output: the folder $DMK_REMOTE_DIR/runs/<run-name>/ with the files <run-name>.* and the files
#   of each pass, <run-name>-pNNNN.json and <run-name>-pNNNN.reference.json.
# Exit code: 0 if the run was complete and no pass failed. 1 if not. 2 for a wrong call. 3 if a
#   server did not start.
set -uo pipefail
name=${1:?usage: run_soak.sh <run-name> <arm> <duration>}
arm=${2:?usage: run_soak.sh <run-name> <arm> <duration>}
dur_arg=${3:?usage: run_soak.sh <run-name> <arm> <duration>}
here=$(cd "$(dirname "$0")" && pwd)
: "${DMK_REMOTE_DIR:?}" "${DMK_GPU:?}" "${DMK_PY_IMAGE:?}" "${DMK_SOAK_SANITIZER:?}"
port=${DMK_PORT:-18080}
interval=${DMK_SOAK_INTERVAL:-1800}
bench_args=${DMK_SOAK_BENCH_ARGS:-}
pass_timeout=${DMK_SOAK_PASS_TIMEOUT:-1200}
san_timeout=${DMK_SOAK_SAN_TIMEOUT:-1800}
smi_ms=${DMK_SOAK_SMI_MS:-1000}
min_unalloc=${DMK_SOAK_MIN_UNALLOC_MIB:-0}
[[ $name =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad run name: $name" >&2; exit 2; }
[[ $interval =~ ^[1-9][0-9]*$ && $pass_timeout =~ ^[1-9][0-9]*$ && $san_timeout =~ ^[1-9][0-9]*$ ]] ||
	{ echo "bad number in the settings" >&2; exit 2; }
[[ $smi_ms =~ ^[1-9][0-9]*$ && $min_unalloc =~ ^[0-9]+$ ]] || { echo "bad number in the settings" >&2; exit 2; }

# Seconds from a duration text.
dur_s() {
	local s=$1 total=0
	[[ $s =~ ^[0-9]+$ ]] && { echo $((10#$s)); return 0; }
	[[ $s =~ ^([0-9]+[smhd])+$ ]] || return 1
	while [[ $s =~ ^([0-9]+)([smhd])(.*)$ ]]; do
		n=$((10#${BASH_REMATCH[1]}))
		case ${BASH_REMATCH[2]} in
		m) n=$((n * 60)) ;;
		h) n=$((n * 3600)) ;;
		d) n=$((n * 86400)) ;;
		esac
		total=$((total + n))
		s=${BASH_REMATCH[3]}
	done
	echo "$total"
}
duration=$(dur_s "$dur_arg") && [ "$duration" -gt 0 ] || { echo "bad duration: $dur_arg" >&2; exit 2; }

case $arm in
stock) dir= ;;
build:*)
	b=${arm#build:}
	[[ $b =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad build name: $b" >&2; exit 2; }
	dir=$DMK_REMOTE_DIR/builds/$b
	[ -d "$dir/dist" ] && dir=$dir/dist
	[ -x "$dir/llama-server" ] || { echo "no llama-server in $dir" >&2; exit 2; }
	;;
*) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
san_bin=${DMK_SOAK_SANITIZER#/tools/}
[ "$san_bin" != "$DMK_SOAK_SANITIZER" ] && [ -x "$DMK_REMOTE_DIR/tools/$san_bin" ] ||
	{ echo "DMK_SOAK_SANITIZER must be a program in /tools: $DMK_SOAK_SANITIZER" >&2; exit 2; }
dmesg >/dev/null 2>&1 || { echo "dmesg is not readable" >&2; exit 2; }

runs=$DMK_REMOTE_DIR/runs
out=$runs/$name
mkdir -p "$runs"
[ ! -e "$out" ] || { echo "run $name exists" >&2; exit 2; }
mkdir "$out" || exit 2
echo $$ >"$out/$name.pid"
exec > >(tee -a "$out/$name.soak.log") 2>&1
say() { echo "== $(date -u +%FT%TZ) $*"; }

# One pattern for each line. The checker reads this file and finds the same lines.
cat >"$out/$name.error-patterns" <<'EOF'
[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ E 
CUDA error
CUDA_ERROR
cudaError
CUBLAS_STATUS_
GGML_ASSERT
ggml_abort
Aborted
abort\(\)
terminate called
Segmentation fault
SIGSEGV
core dumped
out of memory
failed to allocate
illegal memory access
unspecified launch failure
"code": ?5[0-9][0-9]
request: [A-Z]+ \S+ \S+ 5[0-9][0-9]\b
HTTP/1\.[01]\S* 5[0-9][0-9]
status[ =:]+5[0-9][0-9]\b
EOF

start_epoch=$(date +%s)
soak_start=$start_epoch
end_epoch=0
docker_root=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null)
abort_reason=
final_rc=0
check_pid=
events_pid=
bench_pid=

dmesg_lines() { # the Xid and NVRM lines after the baseline, with their time stamps
	dmesg 2>/dev/null | sed -n 's/^\[ *\([0-9][0-9.]*\)\] *\(.*\)$/\1 \2/p' |
		awk -v b="$dmesg_base" '$1 + 0 > b + 0' | grep -E 'Xid|NVRM'
}
dmesg_base=$(dmesg 2>/dev/null | sed -n 's/^\[ *\([0-9][0-9.]*\)\].*$/\1/p' | tail -n 1)
dmesg_base=${dmesg_base:-0}
num() { [[ $1 =~ ^-?[0-9]+(\.[0-9]+)?$ ]] && echo "$1" || echo null; }
unalloc_mib() { # the unallocated space of a btrfs folder of Docker, in MiB
	command -v btrfs >/dev/null 2>&1 && [ -n "$docker_root" ] || return 0
	btrfs filesystem usage -b "$docker_root" 2>/dev/null | awk '/Device unallocated:/ { printf "%d", $3 / 1048576 }'
}

# Copy the new lines of the log of the server container to the log file. Look for error lines.
harvest_log() {
	local last chunk=$out/.chunk n
	last=$(cat "$out/.log_last" 2>/dev/null || echo 1970-01-01T00:00:00.000000000Z)
	docker logs --timestamps --since "$last" dmx-server 2>&1 | awk -v last="$last" '$1 > last' >"$chunk"
	[ -s "$chunk" ] || { rm -f "$chunk"; return 0; }
	tail -n 1 "$chunk" | cut -d' ' -f1 >"$out/.log_last"
	gzip -c "$chunk" >>"$out/$name.server.log.gz"
	echo $(($(cat "$out/.log_lines" 2>/dev/null || echo 0) + $(wc -l <"$chunk"))) >"$out/.log_lines"
	grep -a -E -f "$out/$name.error-patterns" "$chunk" >"$chunk.err"
	n=$(wc -l <"$chunk.err")
	if [ "$n" -gt 0 ]; then
		cat "$chunk.err" >>"$out/$name.errors.log"
		echo $(($(cat "$out/.log_errors" 2>/dev/null || echo 0) + n)) >"$out/.log_errors"
	fi
	rm -f "$chunk" "$chunk.err"
}

# One check: the state of the server container, the GPU, dmesg and the log of the server.
# Tags: start, warm, periodic, end (the server runs) and final (no server).
do_check() {
	local tag=$1
	(
		flock -w 300 8 || exit 1
		local n now st running restarts exitc oom started health smi mem power pstate csm cmem temp
		local hits logerr cstate
		n=$(($(cat "$out/.check_n" 2>/dev/null || echo 0) + 1))
		echo "$n" >"$out/.check_n"
		now=$(date +%s)
		cstate=$(docker inspect -f '{{.State.Status}} {{.State.Running}} {{.RestartCount}} {{.State.ExitCode}} {{.State.OOMKilled}} {{.State.StartedAt}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' dmx-server 2>/dev/null)
		read -r st running restarts exitc oom started health <<<"${cstate:-missing false 0 0 false none none}"
		[ "$tag" = final ] || harvest_log
		smi=$(nvidia-smi -i "$DMK_GPU" --query-gpu=memory.used,power.draw,pstate,clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
		IFS=, read -r mem power pstate csm cmem temp <<<"$smi"
		hits=$(dmesg_lines | tee "$out/.dmesg.now" | wc -l)
		[ "$hits" -gt 0 ] && cp "$out/.dmesg.now" "$out/$name.dmesg.log"
		logerr=$(cat "$out/.log_errors" 2>/dev/null || echo 0)
		printf '{"n":%d,"tag":"%s","t_utc":"%s","elapsed_s":%d,"state":"%s","running":%s,"restarts":%s,"exit_code":%s,"oom":%s,"started_at":"%s","health":"%s","mem_mib":%s,"power_w":%s,"pstate":"%s","clock_sm":%s,"clock_mem":%s,"temp_c":%s,"load1":%s,"unalloc_mib":%s,"passes_done":%d,"passes_failed":%d,"dmesg_hits":%d,"log_error_hits":%d,"log_lines":%d}\n' \
			"$n" "$tag" "$(date -u +%FT%TZ)" "$((now - soak_start))" "$st" "$running" "$(num "$restarts")" "$(num "$exitc")" "$oom" "$started" "$health" \
			"$(num "$mem")" "$(num "$power")" "${pstate:-}" "$(num "$csm")" "$(num "$cmem")" "$(num "$temp")" \
			"$(num "$(cut -d' ' -f1 /proc/loadavg)")" "$(num "$(unalloc_mib)")" \
			"$(cat "$out/.passes_done" 2>/dev/null || echo 0)" "$(cat "$out/.passes_failed" 2>/dev/null || echo 0)" \
			"$hits" "$logerr" "$(cat "$out/.log_lines" 2>/dev/null || echo 0)" >>"$out/$name.checks.jsonl"
		say "check $n ($tag): server=$st restarts=$restarts mem=${mem}MiB power=${power}W ${pstate} dmesg_hits=$hits log_error_hits=$logerr"
		if [ "$min_unalloc" -gt 0 ] && [ "$tag" != final ]; then
			local ua
			ua=$(unalloc_mib)
			if [ -n "$ua" ] && [ "$ua" -lt "$min_unalloc" ]; then
				echo "unallocated space ${ua} MiB is below the limit $min_unalloc MiB" >"$out/.abort"
			fi
		fi
	) 8>"$out/.check.lock"
}

check_loop() {
	local next=$((soak_start + interval)) now
	while [ -e "$out/.running" ]; do
		sleep 5
		now=$(date +%s)
		if [ "$now" -ge "$next" ]; then
			do_check periodic
			next=$((next + interval))
			[ "$next" -le "$now" ] && next=$((now + interval))
		fi
	done
}

cleanup() {
	trap '' INT TERM HUP
	rm -f "$out/.running"
	[ -n "$check_pid" ] && kill "$check_pid" 2>/dev/null
	[ -n "$events_pid" ] && kill "$events_pid" 2>/dev/null
	[ -n "$bench_pid" ] && kill "$bench_pid" 2>/dev/null
	if [ -f "$out/$name.smi.pid" ]; then
		kill -- "-$(cat "$out/$name.smi.pid")" 2>/dev/null
		rm -f "$out/$name.smi.pid"
	fi
	if docker inspect dmx-server >/dev/null 2>&1; then
		harvest_log
		docker rm -f dmx-server >/dev/null 2>&1
	fi
	docker rm -f dmx-bench >/dev/null 2>&1
	cat /proc/loadavg >"$out/$name.load1"
	{
		[ -n "$abort_reason" ] && echo "reason=$abort_reason"
		echo "ended_utc=$(date -u +%FT%TZ)"
		echo "passes=$(cat "$out/.passes_done" 2>/dev/null || echo 0)"
		echo "passes_failed=$(cat "$out/.passes_failed" 2>/dev/null || echo 0)"
		echo "rc=$final_rc"
	} >>"$out/$name.status"
	rm -f "$out/$name.pid" "$out/.chunk" "$out/.chunk.err" "$out/.dmesg.now" "$out/.check.lock"
	say "soak $name ended rc=$final_rc ${abort_reason:+reason=$abort_reason}"
	sleep 1
}
trap cleanup EXIT
trap 'final_rc=130; abort_reason=${abort_reason:-signal}; exit 130' INT TERM HUP

wait_health() { # <seconds>
	local i
	for ((i = 0; i < $1 * 4; i++)); do
		curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
		[ "$(docker inspect -f '{{.State.Running}}' dmx-server 2>/dev/null)" = true ] || return 1
		sleep 0.25
	done
	return 1
}

# One sanitizer phase: the server of the arm in compute-sanitizer, one request of each type.
sanitizer_phase() { # <label>
	local label=$1 pre=$out/$name.san-$1 crc=3 sexit= summary=
	say "sanitizer phase $label"
	docker rm -f dmx-server dmx-bench >/dev/null 2>&1
	if DMK_SERVER_WRAP=/dmk/san_wrap.sh \
		DMK_DOCKER_EXTRA="${DMK_DOCKER_EXTRA:-} -v $DMK_REMOTE_DIR/harness/san_wrap.sh:/dmk/san_wrap.sh:ro -e DMK_SANITIZER=$DMK_SOAK_SANITIZER" \
		bash "$here/start_server.sh" "$arm" >"$pre.launch.log" 2>&1 && wait_health 600; then
		timeout "$san_timeout" docker run --rm --name dmx-bench --network host \
			-v "$DMK_REMOTE_DIR/bench:/bench:ro" -v "$DMK_REMOTE_DIR/harness:/harness:ro" -v "$out:/runs" \
			"$DMK_PY_IMAGE" python -u /harness/san_client.py "$port" "/runs/$name.san-$label.answers.json" \
			>"$pre.client.out" 2>&1 &
		bench_pid=$!
		wait "$bench_pid"
		crc=$?
		bench_pid=
		docker rm -f dmx-bench >/dev/null 2>&1
	else
		say "!! the sanitizer server did not start"
		docker logs --tail 40 dmx-server >"$pre.client.out" 2>&1
	fi
	docker stop -t 300 dmx-server >/dev/null 2>&1
	sexit=$(docker inspect -f '{{.State.ExitCode}}' dmx-server 2>/dev/null)
	docker logs dmx-server >"$pre.server.log" 2>&1
	docker rm -f dmx-server >/dev/null 2>&1
	summary=$(grep -a -E '^=+ ERROR SUMMARY' "$pre.server.log" | tail -n 1)
	{
		echo "label=$label"
		echo "client_rc=$crc"
		echo "server_exit=${sexit:-unknown}"
		echo "summary=${summary:-none}"
	} >"$pre.result"
	say "sanitizer phase $label: client_rc=$crc server_exit=${sexit:-unknown} ${summary:-no summary}"
	[ "$crc" = 0 ] && [ "$sexit" = 0 ] && [[ $summary =~ ^=+\ ERROR\ SUMMARY:\ 0\ errors$ ]]
}

# ---- start
say "soak $name open: arm=$arm duration=${duration}s interval=${interval}s"
docker rm -f dmx-server dmx-bench >/dev/null 2>&1
cat /proc/loadavg >"$out/$name.load0"
{
	echo "arm=$arm"
	echo "date_utc=$(date -u +%FT%TZ)"
	echo "duration_s=$duration"
	echo "interval_s=$interval"
	echo "bench_args=$bench_args"
	echo "server_args=${DMK_SERVER_ARGS:-default}"
	echo "image=${DMK_IMAGE:-}"
	echo "py_image=$DMK_PY_IMAGE"
	echo "kernel=$(uname -r)"
	echo "dmesg_base=$dmesg_base"
	echo "dmesg_xid_nvrm_before=$(dmesg 2>/dev/null | grep -c -E 'Xid|NVRM')"
	nvidia-smi -i "$DMK_GPU" --query-gpu=driver_version,power.limit,clocks_throttle_reasons.active --format=csv,noheader
} >"$out/$name.meta" 2>&1
[ -n "$dir" ] && (cd "$dir" && sha256sum llama-server lib*.so* 2>/dev/null) >"$out/$name.build.sha256"
setsid bash -c "nvidia-smi -i $DMK_GPU --query-gpu=memory.used,power.draw,pstate,clocks.sm --format=csv,noheader,nounits -lms $smi_ms | while IFS= read -r l; do echo \"\$(date +%s.%N), \$l\"; done" >"$out/$name.smi" 2>&1 </dev/null &
echo $! >"$out/$name.smi.pid"
echo "started_utc=$(date -u +%FT%TZ)" >"$out/$name.status"

san_start_ok=0
san_end_ok=0
sanitizer_phase start && san_start_ok=1

# ---- soak
docker rm -f dmx-server dmx-bench >/dev/null 2>&1
touch "$out/.running"
docker events --since "$(date +%s)" --filter container=dmx-server --filter type=container \
	--filter event=die --filter event=kill --filter event=oom --filter event=restart --filter event=stop \
	--format '{{.Time}} {{.Action}}' >"$out/$name.events.log" 2>/dev/null &
events_pid=$!
t0=$(date +%s.%N)
if ! DMK_DOCKER_EXTRA="${DMK_DOCKER_EXTRA:-} --log-opt max-size=50m --log-opt max-file=3" \
	bash "$here/start_server.sh" "$arm" >"$out/$name.launch.log" 2>&1; then
	say "LAUNCH FAILED"
	tail -n 30 "$out/$name.launch.log"
	abort_reason=launch-failed
	final_rc=3
	exit 3
fi
if ! wait_health 300; then
	say "SERVER NOT HEALTHY"
	docker logs --tail 40 dmx-server 2>&1
	abort_reason=not-healthy
	final_rc=3
	exit 3
fi
echo "startup_s=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f", b - a }')" | tee -a "$out/$name.meta"
soak_start=$(date +%s)
end_epoch=$((soak_start + duration))
echo "soak_start_utc=$(date -u +%FT%TZ)" >>"$out/$name.status"
do_check start
check_loop &
check_pid=$!

pass_n=0
fails_in_row=0
while [ "$(date +%s)" -lt "$end_epoch" ]; do
	if [ -e "$out/.abort" ]; then
		abort_reason=$(cat "$out/.abort")
		say "!! stop: $abort_reason"
		break
	fi
	if [ "$(docker inspect -f '{{.State.Running}}' dmx-server 2>/dev/null)" != true ]; then
		abort_reason=server-not-running
		say "!! the server container does not run: stop the passes"
		break
	fi
	pass_n=$((pass_n + 1))
	pname=$(printf '%s-p%04d' "$name" "$pass_n")
	t_start=$(date -u +%FT%TZ)
	docker rm -f dmx-bench >/dev/null 2>&1
	# shellcheck disable=SC2086
	timeout "$pass_timeout" docker run --rm --name dmx-bench --network host \
		-v "$DMK_REMOTE_DIR/bench:/bench:ro" -v "$out:/runs" \
		"$DMK_PY_IMAGE" python -u /bench/bench.py --port "$port" --out "/runs/$pname.json" \
		--write-reference "$pname" $bench_args >"$out/.pass.out" 2>&1 &
	bench_pid=$!
	wait "$bench_pid"
	rc=$?
	bench_pid=
	[ "$rc" = 124 ] && docker rm -f dmx-bench >/dev/null 2>&1
	{ echo "pass $pass_n rc=$rc"; cat "$out/.pass.out"; } >>"$out/$name.bench.out"
	printf '%d\t%d\t%s\t%s\t%s\n' "$pass_n" "$rc" "$t_start" "$(date -u +%FT%TZ)" "$pname" >>"$out/$name.passes"
	echo "$pass_n" >"$out/.passes_done"
	if [ "$rc" = 0 ]; then
		fails_in_row=0
	else
		fails_in_row=$((fails_in_row + 1))
		echo $(($(cat "$out/.passes_failed" 2>/dev/null || echo 0) + 1)) >"$out/.passes_failed"
	fi
	say "pass $pass_n rc=$rc $(tail -n 1 "$out/.pass.out" | cut -c1-200)"
	[ "$pass_n" = 1 ] && do_check warm
	if [ "$fails_in_row" -ge 3 ]; then
		abort_reason=three-passes-failed
		say "!! three passes failed one after the other: stop"
		break
	fi
done
rm -f "$out/.running"
kill "$check_pid" 2>/dev/null
check_pid=
do_check end
kill "$events_pid" 2>/dev/null
events_pid=
say "soak time ended: $pass_n passes"
docker stop -t 60 dmx-server >/dev/null 2>&1
harvest_log
docker rm -f dmx-server >/dev/null 2>&1

# ---- end
sanitizer_phase end && san_end_ok=1
do_check final
{
	echo "sanitizer_start_ok=$san_start_ok"
	echo "sanitizer_end_ok=$san_end_ok"
} >>"$out/$name.status"
if [ -n "$abort_reason" ] || [ "$san_start_ok$san_end_ok" != 11 ] || [ "$(cat "$out/.passes_failed" 2>/dev/null || echo 0)" != 0 ]; then
	final_rc=1
fi
exit "$final_rc"
