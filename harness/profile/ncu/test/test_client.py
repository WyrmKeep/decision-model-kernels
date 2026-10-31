#!/usr/bin/env python3
"""Test of extra/prof/ncu_client.py against fake servers on localhost (no GPU, no docker):
  A  answers the request                       -> outcome "answered", exit 0
  B  resets the connection on the POST and stops listening 2 s later (what ncu --kill + container exit look like)
                                               -> outcome "server-killed", wall_ms_to_gone >= 2000, exit 0
  C  nothing listens                           -> outcome "connect-failed", exit 1
Cases come from the benchmark file: $DMK_BENCH_DIR/cases.json, else bench/cases.json of this repository.
The client JSON files and client-test.txt go to a temporary folder ($DMK_TEST_OUT keeps them there).
usage: test_client.py
"""
import json
import os
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CLIENT = os.path.join(HERE, "..", "..", "extra", "prof", "ncu_client.py")
CASES = os.path.join(os.environ.get("DMK_BENCH_DIR") or os.path.join(R, "bench"), "cases.json")
OUTDIR = None   # set in main()


def serve(port, mode, stop_after=2.0):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(5)
    srv.settimeout(0.2)
    state = {"post_seen": None}

    def loop():
        t_end = None
        while True:
            if t_end is not None and time.monotonic() > t_end:
                break
            try:
                c, _ = srv.accept()
            except socket.timeout:
                continue
            c.settimeout(2)
            try:
                data = c.recv(65536)
            except OSError:
                c.close()
                continue
            if data.startswith(b"GET /health"):
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            elif data.startswith(b"POST"):
                # drain the body (Content-Length) so the client has really sent it
                head, _, rest = data.partition(b"\r\n\r\n")
                n = int([h for h in head.split(b"\r\n") if h.lower().startswith(b"content-length")][0].split(b":")[1])
                while len(rest) < n:
                    rest += c.recv(65536)
                state["post_seen"] = n
                if mode == "answer":
                    body = json.dumps({"usage": {"input_tokens": 158}, "answers": []}).encode()
                    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(body) + body)
                else:
                    c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")  # RST
                    t_end = time.monotonic() + stop_after
            c.close()
        srv.close()

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th, state


def run_client(port, out):
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, CLIENT, "--kind", "short1q", "--case", "check03", "--out", out, "--port", str(port), "--cases", CASES,
                        "--gone-timeout", "20", "--timeout", "30"], capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip(), time.monotonic() - t0


def main():
    global OUTDIR
    OUTDIR = os.environ.get("DMK_TEST_OUT") or tempfile.mkdtemp(prefix="ncu-client-test-")
    os.makedirs(OUTDIR, exist_ok=True)
    lines, ok = [], True
    for name, port, mode, want_exit, want_outcome in (("A answered", 18581, "answer", 0, "answered"),
                                                       ("B reset then gone", 18582, "reset", 0, "server-killed"),
                                                       ("C nothing listens", 18583, None, 1, "connect-failed")):
        out = os.path.join(OUTDIR, f"client-test-{name.split()[0]}.json")
        th = None
        if mode:
            th, state = serve(port, mode)
        rc, so, se, dt = run_client(port, out)
        rec = json.load(open(out))
        good = rc == want_exit and rec["outcome"] == want_outcome
        if name.startswith("B"):
            good = good and rec.get("wall_ms_to_gone", 0) >= 1900 and "t_gone_mono_ns" in rec and state["post_seen"] == rec["body_bytes"]
        if name.startswith("A"):
            good = good and rec.get("input_tokens") == 158 and state["post_seen"] == rec["body_bytes"]
        ok &= good
        keys = ("outcome", "wall_ms", "wall_ms_to_loss", "wall_ms_to_gone", "input_tokens", "error", "health_status", "ref", "body_bytes")
        lines.append(f"{'PASS' if good else 'FAIL'} {name}: exit {rc} (want {want_exit}), {dt:.1f} s wall; " +
                     ", ".join(f"{k}={rec[k]!r}" for k in keys if k in rec))
        if se:
            lines.append("  stderr: " + se[-300:])
        if th:
            th.join(timeout=5)
    with open(os.path.join(OUTDIR, "client-test.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not os.environ.get("DMK_TEST_OUT"):
        shutil.rmtree(OUTDIR, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
