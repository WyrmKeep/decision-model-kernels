#!/usr/bin/env python3
"""Static checks of schedule files for harness/run_schedule.sh (runs nothing; python3 stdlib).

The real parser is exercised by `bash dry_run.sh <schedule>`; this script adds what a parser cannot know:
  * no `@REMOTE_DIR@` placeholder is left (write the files with `bash make_sched.sh --remote-dir DIR` first);
  * the line parses with the same regexes as run_schedule.sh (KEY=VALUE overrides, value in double quotes without quotes inside);
  * run names are unique inside the files given and not longer than 80 characters, only [A-Za-z0-9._-];
  * `client` lines: the script exists under the extra dir (/extra/..., default: extra/ next to this script) or the
    bench dir (/bench/..., default: ../../bench of the repository), the arm is `stock` or `build:<name>` (with --builds DIR:
    DIR/<name>/dist must exist), every /runs/ path in the arguments contains {name};
  * the arguments (everything after the run name) have no shell metacharacters: the harness pastes them unquoted into the
    remote shell command, so `;|&<>()$`*?[]!"'\\` would be interpreted by the remote shell;
  * DMK_SERVER_WRAP: every /runs/<x> in it starts with the run name (the harness only replaces {name} in the arguments);
    an nsys wrapper needs the target-linux-x64 mount in DMK_DOCKER_EXTRA; an ncu wrapper needs DMK_TOOL_CAPS=SYS_ADMIN and
    `--clock-control none` (never change clocks);
  * DMK_DOCKER_EXTRA --env-file paths: with --builds DIR, the part after `/builds/` must exist below DIR.
Usage: python3 check_sched.py [--extra DIR] [--bench DIR] [--builds DIR] <schedule.txt> [...]
  --extra DIR   local copy of the /extra mount (default: <this script's folder>/extra)
  --bench DIR   local copy of the /bench mount (default: <repository>/bench; DMK_BENCH_DIR overrides the default)
  --builds DIR  local folder with one sub folder per build arm (<name>/dist); without it the arm check is skipped
Exit status 1 on any error; warnings are printed with WARN.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLACEHOLDER = "@REMOTE_DIR@"
ENV_RE = re.compile(r'^(DMK_[A-Z0-9_]+)=("([^"\']*)"|[^\s"\']+)\s+(.*)$')
META = re.compile(r"[;|&<>()$`*?\[\]!\"'\\#]")

errors = 0
names = {}
EXTRA_DIR = os.path.join(HERE, "extra")
BENCH_DIR = os.environ.get("DMK_BENCH_DIR") or os.path.join(HERE, "..", "..", "bench")
BUILDS_DIR = None


def err(msg):
    global errors
    errors += 1
    print("  ERROR", msg)


def warn(msg):
    print("  WARN ", msg)


def check_line(fn, no, line):
    envs = {}
    if PLACEHOLDER in line:
        err(f"{fn}:{no}: unresolved {PLACEHOLDER} (write the schedule with make_sched.sh --remote-dir DIR first)")
    while True:
        m = ENV_RE.match(line)
        if not m:
            break
        val = m.group(3) if m.group(2).startswith('"') else m.group(2)
        envs[m.group(1)] = val
        line = m.group(4)
    parts = line.split(None, 2)
    if len(parts) < 2:
        err(f"{fn}:{no}: no kind/name in {line!r}")
        return
    kind, name = parts[0], parts[1]
    rest = parts[2] if len(parts) > 2 else ""
    where = f"{fn}:{no} {name}"
    if re.match(r"^DMK_[A-Z0-9_]+=", line):
        err(f"{where}: an override is not parsed (quotes inside the value, or a value that starts with a quote and has spaces unquoted): {line[:100]!r}")
    if not (kind in ("bench", "client", "tool") or re.match(r"^tool@[A-Za-z0-9_]+$", kind)):
        err(f"{where}: bad kind {kind!r}")
    if not re.match(r"^[A-Za-z0-9._-]{1,80}$", name):
        err(f"{where}: bad run name {name!r}")
    if name in names:
        err(f"{where}: run name already used in {names[name]}")
    names[name] = f"{fn}:{no}"
    if kind == "bench" or kind == "client" or kind.startswith("tool"):
        arm_and_script = rest.split(None, 2)
        arm = arm_and_script[0] if arm_and_script else ""
        if arm != "stock":
            m = re.match(r"^build:([A-Za-z0-9._-]+)$", arm)
            if not m:
                err(f"{where}: bad arm {arm!r}")
            elif BUILDS_DIR and not os.path.isdir(os.path.join(BUILDS_DIR, m.group(1), "dist")):
                warn(f"{where}: no local build dir {os.path.join(BUILDS_DIR, m.group(1), 'dist')} (the arm must exist on the host under $DMK_REMOTE_DIR/builds/)")
    after_arm = rest.split(None, 1)[1] if len(rest.split(None, 1)) > 1 else ""
    if META.search(after_arm):
        err(f"{where}: shell metacharacter in the arguments {META.search(after_arm).group(0)!r}: {after_arm[:100]!r}")
    if kind == "client":
        script = after_arm.split()[0] if after_arm.split() else ""
        if script.startswith("/extra/"):
            if not os.path.isfile(os.path.join(EXTRA_DIR, script[len("/extra/"):])):
                err(f"{where}: client script {script} not in {EXTRA_DIR}")
        elif script.startswith("/bench/"):
            if not os.path.isfile(os.path.join(BENCH_DIR, script[len("/bench/"):])):
                warn(f"{where}: {script} not found in the bench dir {BENCH_DIR}")
        else:
            err(f"{where}: client script {script!r} must start with /extra/ or /bench/")
        for tok in after_arm.split():
            if tok.startswith("/runs/") and "{name}" not in tok:
                err(f"{where}: {tok} lacks {{name}}")
    wrap = envs.get("DMK_SERVER_WRAP", "")
    extra = envs.get("DMK_DOCKER_EXTRA", "")
    if wrap and kind != "client":
        err(f"{where}: DMK_SERVER_WRAP only works for client lines (start_server.sh)")
    for m in re.finditer(r"/runs/([^\s]*)", wrap):
        if not m.group(1).startswith(name):
            err(f"{where}: DMK_SERVER_WRAP writes /runs/{m.group(1)}, which does not start with the run name {name}")
    if "nsys" in wrap:
        if "/opt/nsight-systems/target-linux-x64/nsys" not in wrap and "/target-linux-x64/nsys" not in wrap:
            err(f"{where}: nsys must be called as .../target-linux-x64/nsys (it only works from a directory of that name)")
        if "target-linux-x64:ro" not in extra and "/tools/nsys/target-linux-x64" not in wrap:
            err(f"{where}: no mount of the nsys directory as .../target-linux-x64 in DMK_DOCKER_EXTRA")
    if "ncu" in wrap.split("/")[-1] or "/ncu " in wrap or wrap.strip().endswith("/ncu"):
        if envs.get("DMK_TOOL_CAPS") != "SYS_ADMIN":
            err(f"{where}: ncu needs DMK_TOOL_CAPS=SYS_ADMIN")
        if "--clock-control none" not in wrap:
            err(f"{where}: ncu wrapper without `--clock-control none` would lock the GPU clocks")
        if "--target-processes" not in wrap:
            warn(f"{where}: no --target-processes (default application-only is fine for llama-server)")
    for m in re.finditer(r"--env-file=(\S+)", extra):
        p = m.group(1)
        if BUILDS_DIR and "/builds/" in p:
            local = os.path.join(BUILDS_DIR, p.split("/builds/", 1)[1])
            if not os.path.isfile(local):
                err(f"{where}: env file {p} not found locally as {local}")
    if wrap and "-o " in wrap and not re.search(r"-o /runs/", wrap) and "--output" not in wrap:
        warn(f"{where}: profiler output not under /runs/ (it would be lost with the container)")
    shown = " ".join(f"{k}={v!r}" for k, v in envs.items())
    print(f"  ok   {name:40s} {kind:8s} {shown[:150]}")


def main():
    global EXTRA_DIR, BENCH_DIR, BUILDS_DIR
    ap = argparse.ArgumentParser(description="Static checks of schedule files for harness/run_schedule.sh")
    ap.add_argument("--extra", default=EXTRA_DIR, help="local copy of the /extra mount (default: extra/ next to this script)")
    ap.add_argument("--bench", default=BENCH_DIR, help="local copy of the /bench mount (default: <repository>/bench)")
    ap.add_argument("--builds", help="folder with one sub folder <name>/dist per build arm; without it the arm check is skipped")
    ap.add_argument("schedules", nargs="+", help="schedule files written by make_sched.sh")
    a = ap.parse_args()
    EXTRA_DIR, BENCH_DIR, BUILDS_DIR = a.extra, a.bench, a.builds
    for fn in a.schedules:
        print(f"== {fn}")
        for no, raw in enumerate(open(fn), 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            check_line(os.path.basename(fn), no, line)
    print(f"{errors} error(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
