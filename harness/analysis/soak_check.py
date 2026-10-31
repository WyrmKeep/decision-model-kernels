#!/usr/bin/env python3
"""Check the run folder of harness/remote/run_soak.sh and print PASS or FAIL for each criterion.

Usage: soak_check.py RUN_FOLDER [--reference GLOB ...] [--slowdown-pct 2] [--mem-mib 64] [--window 10]

Criteria:
  (a) each pass has all guards true
  (b) the answers of each pass are bit-identical to the answers of the first pass
      (b2: also to each file of --reference, if you give it)
  (c) no Xid line and no NVRM line in dmesg after the start
  (d) no restart of the server container, and the server never exited
  (e) no HTTP 5xx line, CUDA error line or abort line in the log of the server or in a pass
  (f) the sanitizer (memcheck) runs at the start and at the end have 0 errors
  (g) lat_ms of the last passes is at most --slowdown-pct above the first passes
  (h) memory.used of the last check is at most --mem-mib away from the check after the first pass
  (i) the run is complete and the checks ran at the interval
The exit code is 0 only if each criterion passes. It uses the Python standard library only.
"""
import argparse
import glob
import gzip
import json
import os
import re
import statistics
import sys

sys.dont_write_bytecode = True  # the import below must not make a __pycache__ folder in the repository
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answers_compare import compare  # noqa: E402

SERVER_TAGS = ("start", "warm", "periodic", "end")
results = []


def report(key, ok, text, details=()):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  ({key}) {text}")
    for d in details:
        print(f"        {d}")


def info(text):
    print(f"info  {text}")


def read_kv(path):
    kv = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                kv[k] = v
    return kv


def read_json(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def differ(x, y):
    """Return a short text about two different answer sets."""
    try:
        c = compare(x, y)
        return f"max_abs_dp={c['max_abs_dp']:.6g}, top flips={c['top_flips'] or 'none'}"
    except (KeyError, TypeError):
        return "the sets of cases or questions are different"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--reference", action="append", default=[], metavar="GLOB",
                    help="answer files that pass 1 must equal (b2); the option can repeat")
    ap.add_argument("--slowdown-pct", type=float, default=2.0)
    ap.add_argument("--mem-mib", type=float, default=64.0)
    ap.add_argument("--window", type=int, default=10, help="number of passes at the start and at the end")
    args = ap.parse_args()

    folder = args.folder
    names = [os.path.basename(p)[:-len(".checks.jsonl")] for p in glob.glob(os.path.join(folder, "*.checks.jsonl"))]
    if len(names) != 1:
        sys.exit(f"soak_check: need exactly one <name>.checks.jsonl in {folder}, found {len(names)}")
    name = names[0]

    def path(suffix):
        return os.path.join(folder, f"{name}{suffix}")

    meta, status = read_kv(path(".meta")), read_kv(path(".status"))
    duration = float(meta.get("duration_s", 0))
    interval = float(meta.get("interval_s", 0))
    checks = []
    for line in open(path(".checks.jsonl")):
        try:
            checks.append(json.loads(line))
        except ValueError:
            pass
    server_checks = [c for c in checks if c["tag"] in SERVER_TAGS]

    passes = []  # (number, rc, pass name)
    if os.path.exists(path(".passes")):
        for line in open(path(".passes")):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 5:
                passes.append((int(f[0]), int(f[1]), f[4]))
    print(f"soak_check: run {name}, arm {meta.get('arm')}, duration {duration:.0f} s, interval {interval:.0f} s, "
          f"{len(passes)} passes, {len(checks)} checks")

    # (a) guards of each pass
    bad, data = [], {}
    for n, rc, pname in passes:
        d = read_json(os.path.join(folder, f"{pname}.json"))
        data[n] = d
        if d is None:
            bad.append(f"pass {n}: no result file (rc={rc})")
        elif not isinstance(d.get("guard"), dict) or not all(d["guard"].values()) or d.get("guard_pass") is not True:
            bad.append(f"pass {n}: guards {d.get('guard')} guard_pass={d.get('guard_pass')}")
        elif rc != 0:
            bad.append(f"pass {n}: the client gave exit code {rc}")
    report("a", bool(passes) and not bad,
           f"guards: {len(passes) - len(bad)} of {len(passes)} passes have all guards true", bad[:8])

    # (b) answers equal to pass 1
    refs = {}
    for n, _, pname in passes:
        r = read_json(os.path.join(folder, f"{pname}.reference.json"))
        if r is not None:
            refs[n] = r["answers"]
    bad = []
    if not passes or passes[0][0] not in refs:
        bad.append("pass 1 has no answer file")
    else:
        first = refs[passes[0][0]]
        for n, _, _ in passes[1:]:
            if n not in refs:
                bad.append(f"pass {n}: no answer file")
            elif refs[n] != first:
                bad.append(f"pass {n}: {differ(first, refs[n])}")
    decisions = sum(len(q) for q in refs[passes[0][0]].values()) if refs and passes[0][0] in refs else 0
    report("b", bool(passes) and not bad,
           f"answers: {len(refs)} of {len(passes)} passes compared with pass 1 ({decisions} decisions), "
           f"{len(bad)} differ", bad[:8])
    if args.reference:
        files = sorted(f for g in args.reference for f in glob.glob(g))
        bad = []
        if not files:
            bad.append("the pattern gives no file")
        elif not passes or passes[0][0] not in refs:
            bad.append("pass 1 has no answer file")
        for f in files:
            r = read_json(f)
            if r is None:
                bad.append(f"{os.path.basename(f)}: not readable")
            elif refs and refs.get(passes[0][0]) != r["answers"]:
                bad.append(f"{os.path.basename(f)}: {differ(refs[passes[0][0]], r['answers'])}")
        report("b2", not bad, f"answers of pass 1 against {len(files)} reference files: {len(bad)} differ", bad[:8])

    # (c) Xid and NVRM lines
    lines = open(path(".dmesg.log"), errors="replace").read().splitlines() if os.path.exists(path(".dmesg.log")) else []
    hits = max([c.get("dmesg_hits", 0) for c in checks] + [len(lines)])
    report("c", bool(checks) and hits == 0,
           f"dmesg: {hits} Xid or NVRM lines after the start (baseline {meta.get('dmesg_base')}, "
           f"{meta.get('dmesg_xid_nvrm_before')} such lines before)", lines[:5])

    # (d) the server container
    bad = []
    for c in server_checks:
        if c["state"] != "running" or c["running"] is not True:
            bad.append(f"check {c['n']} ({c['tag']}): state {c['state']}")
        if c["restarts"] != 0 or c["exit_code"] != 0 or c["oom"]:
            bad.append(f"check {c['n']} ({c['tag']}): restarts={c['restarts']} exit_code={c['exit_code']} oom={c['oom']}")
    starts = {c["started_at"] for c in server_checks}
    if len(starts) > 1:
        bad.append(f"the start time of the container changed: {sorted(starts)}")
    events = open(path(".events.log"), errors="replace").read().splitlines() if os.path.exists(path(".events.log")) else []
    events = [e for e in events if re.match(r"^\d+ (die|kill|oom|restart|stop)\b", e)]
    if events:
        bad.append(f"container events (die, kill, oom, restart, stop): {events[:5]}")
    if not server_checks:
        bad.append("no check of a running server")
    if status.get("reason") in ("server-not-running", "launch-failed", "not-healthy"):
        bad.append(f"the run ended: {status['reason']}")
    report("d", not bad, f"server: {len(server_checks)} checks, restarts "
           f"{max([c['restarts'] for c in server_checks] or [0])}, container events {len(events)}", bad[:8])

    # (e) error lines
    patterns = []
    if os.path.exists(path(".error-patterns")):
        patterns = [re.compile(p) for p in open(path(".error-patterns")).read().split("\n") if p]
    scanned, found = 0, []
    log = path(".server.log.gz")
    if os.path.exists(log):
        with gzip.open(log, "rt", errors="replace") as f:
            for line in f:
                scanned += 1
                if any(p.search(line) for p in patterns):
                    found.append(line.rstrip("\n")[:200])
    errs = open(path(".errors.log"), errors="replace").read().splitlines() if os.path.exists(path(".errors.log")) else []
    client = [f"pass {n}: {e}"[:200] for n, d in data.items() if d for e in d.get("errors", [])
              if re.search(r"HTTP 5\d\d", str(e))]
    problems = []
    if not os.path.exists(log) or not patterns:
        problems.append("the log of the server or the list of patterns is missing")
    report("e", not problems and not found and not errs and not client,
           f"log: {scanned} server log lines scanned, {len(found)} error lines, {len(errs)} found during the run, "
           f"{len(client)} HTTP 5xx in the passes", problems + found[:5] + errs[:3] + client[:3])

    # (f) sanitizer runs
    for label in ("start", "end"):
        res = read_kv(path(f".san-{label}.result"))
        ans = read_json(path(f".san-{label}.answers.json")) or {}
        ok_requests = sum(1 for v in ans.values() if v.get("ok"))
        summary = res.get("summary", "none")
        ok = (res.get("client_rc") == "0" and res.get("server_exit") == "0" and ok_requests == len(ans) > 0
              and re.match(r"^=+ ERROR SUMMARY: 0 errors$", summary) is not None)
        report("f", ok, f"memcheck {label}: {summary}; server exit {res.get('server_exit')}, "
               f"client exit {res.get('client_rc')}, {ok_requests} of {len(ans)} requests ok")
        if ans and 1 in refs:
            same = all(v.get("answers") == refs[1].get(v["id"]) for v in ans.values() if v.get("ok"))
            info(f"memcheck {label}: the answers {'equal' if same else 'do not equal'} the answers of pass 1")

    # (g) latency
    lat = [d["seq"]["lat_ms"] for n, rc, _ in passes if rc == 0 and (d := data.get(n)) and d.get("seq", {}).get("lat_ms")]
    k = min(args.window, len(lat) // 2)
    if k < 2:
        report("g", False, f"latency: {len(lat)} passes, too few to compare")
    else:
        a, b = statistics.fmean(lat[:k]), statistics.fmean(lat[-k:])
        change = 100.0 * (b / a - 1.0)
        q = [statistics.fmean(x) for x in (lat[i * len(lat) // 4:(i + 1) * len(lat) // 4] for i in range(4)) if x]
        report("g", change <= args.slowdown_pct,
               f"latency: first {k} passes {a:.2f} ms, last {k} passes {b:.2f} ms, change {change:+.2f} % "
               f"(limit +{args.slowdown_pct:g} %)",
               [f"mean lat_ms of each quarter: {', '.join(f'{x:.2f}' for x in q)}"])

    # (h) memory
    warm = next((c for c in server_checks if c["tag"] == "warm"), None)
    last = next((c for c in reversed(server_checks) if c["tag"] in ("end", "periodic")), None)
    if warm is None or last is None or warm is last or warm["mem_mib"] is None or last["mem_mib"] is None:
        report("h", False, "memory: no check after the first pass and a later check")
    else:
        diff = last["mem_mib"] - warm["mem_mib"]
        mems = [c["mem_mib"] for c in server_checks if c["mem_mib"] is not None]
        report("h", abs(diff) <= args.mem_mib,
               f"memory: after pass 1 {warm['mem_mib']:g} MiB, last check {last['mem_mib']:g} MiB, "
               f"difference {diff:+g} MiB (limit {args.mem_mib:g} MiB)",
               [f"range over all {len(mems)} checks: {min(mems):g} to {max(mems):g} MiB"])

    # (i) the run is complete and the checks ran at the interval
    bad = []
    if "rc" not in status:
        bad.append("the status file has no end line: the script did not end in a normal way")
    if status.get("reason"):
        bad.append(f"the script stopped early: {status['reason']}")
    if status.get("sanitizer_end_ok") is None:
        bad.append("the end sanitizer phase did not run")
    end = next((c for c in reversed(server_checks) if c["tag"] == "end"), None)
    if end is None:
        bad.append("no check at the end of the soak")
    elif end["elapsed_s"] < duration:
        bad.append(f"the soak ran {end['elapsed_s']} s, less than {duration:.0f} s")
    gaps = [b["elapsed_s"] - a["elapsed_s"] for a, b in zip(server_checks, server_checks[1:])]
    if gaps and max(gaps) > 2 * interval + 60:
        bad.append(f"a gap of {max(gaps)} s between two checks (interval {interval:.0f} s)")
    unalloc = [c["unalloc_mib"] for c in checks if c.get("unalloc_mib") is not None]
    report("i", not bad, f"run: {sum(1 for c in checks if c['tag'] == 'periodic')} periodic checks, "
           f"longest gap {max(gaps) if gaps else 0} s", bad)
    if unalloc:
        info(f"docker unallocated space: {min(unalloc)} to {max(unalloc)} MiB")
    sm = [c["clock_sm"] for c in server_checks if c.get("clock_sm") is not None]
    pw = [c["power_w"] for c in server_checks if c.get("power_w") is not None]
    tp = [c["temp_c"] for c in server_checks if c.get("temp_c") is not None]
    if sm and pw and tp:
        info(f"gpu at the checks: sm clock {min(sm):g} to {max(sm):g} MHz, power {min(pw):g} to {max(pw):g} W, "
             f"temperature {min(tp):g} to {max(tp):g} C")

    ok = all(results)
    print(f"RESULT: {'PASS' if ok else 'FAIL'} ({sum(results)} of {len(results)} criteria pass)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
