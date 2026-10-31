#!/usr/bin/env python3
"""Verifies on this host, with a CPU-only nsys run (no GPU, no server), the time mapping that `nsys_census.py analyze` uses:
    CLOCK_MONOTONIC_ns = TARGET_INFO_SESSION_START_TIME.systemClockNs + t_nsys      (default, client stamps t_*_mono_ns)
    epoch_ns           = TARGET_INFO_SESSION_START_TIME.utcEpochNs     + t_nsys      (--clock epoch, client stamp t_send_epoch_ns)
A child python process sleeps 0.3 s, takes stamps A (epoch, CLOCK_MONOTONIC, CLOCK_MONOTONIC_RAW), sleeps 0.777 s, takes stamps B;
nsys traces the sleeps as clock_nanosleep OS runtime calls: the second sleep starts at A and ends at B on the nsys timeline.
The residual = stamp - (session start clock + nsys time of the same instant); it must be below 1 ms (observed: tens of microseconds).
CLOCK_MONOTONIC_RAW is checked only to show that it is NOT the nsys clock.
Usage: python3 clock_check.py [--nsys PATH]   (else $DMK_NSYS, else `nsys` on PATH); exit 0 = verified, 2 = no local nsys found (skipped)."""
import argparse
import json
import os
import sqlite3
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHILD = r"""
import json, time
C = time.clock_gettime_ns
r = {}
time.sleep(0.3)
r["A"] = [time.time_ns(), C(time.CLOCK_MONOTONIC), C(time.CLOCK_MONOTONIC_RAW)]
time.sleep(0.777)
r["B"] = [time.time_ns(), C(time.CLOCK_MONOTONIC), C(time.CLOCK_MONOTONIC_RAW)]
print("STAMPS " + json.dumps(r))
"""


def find_nsys(arg):
    cands = [arg, os.environ.get("DMK_NSYS"), shutil.which("nsys")]
    for c in cands:
        if c and os.access(c, os.X_OK):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsys", default=None)
    args = ap.parse_args()
    nsys = find_nsys(args.nsys)
    if not nsys:
        print("SKIP: no local nsys (use --nsys PATH, set DMK_NSYS, or put nsys on PATH)")
        sys.exit(2)
    with tempfile.TemporaryDirectory() as d:
        child = Path(d, "child.py")
        child.write_text(CHILD)
        base = Path(d, "clk")
        p = subprocess.run([nsys, "profile", "--trace=osrt", "--sample=none", "--cpuctxsw=none", "--force-overwrite=true", "-o", str(base),
                            sys.executable, str(child)], capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("STAMPS ")]
        assert line, ("child output missing", p.stdout[-500:], p.stderr[-500:])
        st = json.loads(line[0][len("STAMPS "):])
        subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite=true", "--output", str(base) + ".sqlite", str(base) + ".nsys-rep"],
                       capture_output=True, text=True, check=True)
        db = sqlite3.connect(str(base) + ".sqlite")
        utc, sysclk = db.execute("select utcEpochNs, systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()
        sleeps = db.execute("select o.start, o.end from OSRT_API o join StringIds s on s.id=o.nameId where s.value like '%nanosleep%' "
                            "and o.end-o.start > 700000000 and o.end-o.start < 900000000 order by o.start").fetchall()
        assert sleeps, "the 0.777 s clock_nanosleep call was not traced"
        s0, e0 = sleeps[0]
    print("nsys: %s" % nsys)
    print("session start: utcEpochNs=%d systemClockNs=%d; traced sleep (0.777 s requested): %.6f s" % (utc, sysclk, (e0 - s0) / 1e9))
    ok = True
    for name, idx, base_clock, instant in (("epoch", 0, utc, "A"), ("epoch", 0, utc, "B"), ("CLOCK_MONOTONIC", 1, sysclk, "A"),
                                           ("CLOCK_MONOTONIC", 1, sysclk, "B"), ("CLOCK_MONOTONIC_RAW", 2, sysclk, "A"), ("CLOCK_MONOTONIC_RAW", 2, sysclk, "B")):
        t = s0 if instant == "A" else e0
        res = st[instant][idx] - (base_clock + t)
        verdict = ""
        if name != "CLOCK_MONOTONIC_RAW":
            good = abs(res) < 1_000_000
            ok &= good
            verdict = "ok" if good else "FAIL"
        else:
            verdict = "(not the nsys clock)" if abs(res) > 1_000_000 else "(happens to be close on this host)"
        print("  stamp %s %-20s residual %+10.1f us  %s" % (instant, name, res / 1e3, verdict))
    print("clock mapping:", "VERIFIED (|residual| < 1 ms for epoch and CLOCK_MONOTONIC)" if ok else "NOT VERIFIED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
