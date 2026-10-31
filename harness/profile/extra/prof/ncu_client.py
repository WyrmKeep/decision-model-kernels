#!/usr/bin/env python3
"""One-request client of the ncu runs (phase 3, package 4). Python 3.12 stdlib + common.py.

The server runs under `ncu --kill yes -k <regex> -s <skip> -c <count>`: ncu profiles a window of kernel launches (each one is replayed
many times, so the request takes minutes) and terminates the server after the last profiled kernel. That is the normal end of a run:
the connection of this client is reset and the container stops once ncu has written the report.

  ncu_client.py --kind long3q --case long00-storage --out /runs/<name>.client.json

Behaviour
  * builds the request like the benchmark and the census: the state gets the first line "Ref: <tag("seq", 0, case id)>", which is the
    Ref line of the shape census (harness/profile/census) for its token counts (prefix 3221 / 1288 / 86 tokens, check03 158), so the passes of the fresh
    server's first request are the ones plan_ncu.py predicted;
  * sends ONE request with a long HTTP timeout (default 3600 s) and never retries (a retry would be a second request);
  * connection reset / closed / refused after the send -> the server was killed by ncu: wait until the port refuses connections
    (container gone, i.e. ncu has finished writing the report) for at most --gone-timeout seconds, then exit 0;
  * an answer (HTTP 200) means ncu did not stop the app: the window was not complete (or the skip is too large). The client records
    it and exits 0; the harness teardown (`docker stop -t $DMK_STOP_TIMEOUT`) then ends the server and ncu writes what it has;
  * a timeout or an HTTP error status -> exit 1.
The JSON written to --out has the CLOCK_MONOTONIC stamps (same clock as the server's DMK_TIMING lines), the epoch stamp and the wall times.
"""
import argparse
import http.client
import json
import socket
import sys
import time

import common


def port_open(host, port, timeout=1.0):
    """True if a TCP connection is accepted. With docker -p the proxy also accepts while nothing listens inside the container, so
    this is only used together with a refusal test: 'gone' = connection refused."""
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except ConnectionRefusedError:
        return False
    except OSError:
        return True
    finally:
        s.close()


def wait_gone(host, port, timeout_s, need=3):
    """Poll until `need` consecutive connection attempts are refused. Returns the CLOCK_MONOTONIC ns of the first refusal of the run
    that ended the wait, or None on timeout."""
    deadline = time.monotonic() + timeout_s
    first = None
    refused = 0
    while time.monotonic() < deadline:
        if port_open(host, port):
            refused, first = 0, None
        else:
            refused += 1
            if first is None:
                first = common.mono_ns()
            if refused >= need:
                return first
        time.sleep(0.5)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=sorted(common.KINDS))
    ap.add_argument("--case", help="case id (default: the representative case of the kind)")
    ap.add_argument("--out", required=True, help="JSON file for the stamps, /runs/{name}.client.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--timeout", type=float, default=3600.0, help="HTTP timeout of the request, seconds")
    ap.add_argument("--gone-timeout", type=float, default=300.0, help="how long to wait for the container to disappear after the reset")
    ap.add_argument("--cases", default=common.CASES_PATH)
    a = ap.parse_args()

    cases = common.load_cases(a.cases)
    case_id = a.case or common.REP_CASE[a.kind]
    case = common.case_by_id(cases, case_id)
    if case["kind"] != a.kind:
        sys.exit(f"case {case_id} is of kind {case['kind']}, not {a.kind}")
    ref = common.tag("seq", 0, case_id)
    data = common.body(case, ref)
    rec = {"kind": a.kind, "case": case_id, "ref": ref, "body_bytes": len(data), "n_questions": len(case["questions"]), "timeout_s": a.timeout,
           "python": sys.version.split()[0]}

    try:
        status, raw = common.Client(a.host, a.port).get("/health")
    except OSError as e:
        status = f"{type(e).__name__}"
    rec["health_status"] = status
    print(f"health {status}; sending {case_id} ({a.kind}, {len(data)} bytes, ref {ref}) as the only request", flush=True)

    conn = http.client.HTTPConnection(a.host, a.port, timeout=a.timeout)
    try:
        conn.connect()
    except OSError as e:
        rec.update(outcome="connect-failed", error=f"{type(e).__name__}: {e}")
        common.write_json(a.out, rec)
        print(json.dumps({"outcome": rec["outcome"], "error": rec["error"]}), flush=True)
        sys.exit(1)
    rec["t_send_epoch_ns"] = common.epoch_ns()
    rec["t_send_mono_ns"] = t0 = common.mono_ns()
    exit_code = 0
    try:
        conn.request("POST", "/v1/systemone", data, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        rec["t_header_mono_ns"] = common.mono_ns()
        body = resp.read()
        rec["t_recv_mono_ns"] = t1 = common.mono_ns()
        rec["http_status"] = resp.status
        rec["wall_ms"] = (t1 - t0) / 1e6
        if resp.status == 200:
            j = json.loads(body)
            rec["input_tokens"] = j.get("usage", {}).get("input_tokens")
            rec["outcome"] = "answered"
            rec["note"] = ("the server answered: ncu did not stop the app, so the window was not complete (skip too large or fewer "
                           "matching launches than skip + count); the harness stops the server and ncu writes the partial report")
        else:
            rec["outcome"] = "http-error"
            rec["body"] = body[:300].decode("utf-8", "replace")
            exit_code = 1
    except (ConnectionError, http.client.RemoteDisconnected, http.client.IncompleteRead, http.client.BadStatusLine) as e:
        rec["t_lost_mono_ns"] = tl = common.mono_ns()
        rec["wall_ms_to_loss"] = (tl - t0) / 1e6
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["outcome"] = "server-killed"
        gone = wait_gone(a.host, a.port, a.gone_timeout)
        if gone is None:
            rec["outcome"] = "server-killed-container-still-up"
            rec["note"] = f"the port still accepted connections {a.gone_timeout:.0f} s after the reset"
        else:
            rec["t_gone_mono_ns"] = gone
            rec["wall_ms_to_gone"] = (gone - t0) / 1e6
    except (socket.timeout, TimeoutError, OSError) as e:
        rec["t_lost_mono_ns"] = common.mono_ns()
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["outcome"] = "timeout" if isinstance(e, (socket.timeout, TimeoutError)) else "error"
        exit_code = 1
    finally:
        conn.close()
    common.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in rec if k in ("outcome", "wall_ms", "wall_ms_to_loss", "wall_ms_to_gone", "input_tokens", "error", "note")}),
          flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
