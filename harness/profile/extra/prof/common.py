"""Shared helpers of the phase-3 profiling clients (python 3.12 stdlib only).

The clients run inside python:3.12-slim on the GPU host with host networking (harness `client` lines): the server is
http://127.0.0.1:18080, the benchmark cases are /bench/cases.json (the harness mounts DMK_BENCH_DIR there), client scripts
are /extra/prof/<file>.py (the harness mounts DMK_EXTRA_DIR there, so this module is importable: the script directory is on
sys.path), reports go to /runs/{name}.* (the harness replaces {name} in the script arguments and copies runs/<name>.* back).

Clocks: `mono_ns()` is CLOCK_MONOTONIC. The server's `DMK_TIMING` lines (`t=` seconds, std::chrono::steady_clock) use the
same clock on the same host kernel, so client send/receive stamps and server stamps are directly comparable. `epoch_ns()`
is the wall clock (time.time_ns()), recorded next to it to map requests to Nsight Systems timestamps (UTC based).

Request bodies are built exactly like the benchmark client (`bench/bench.py`): the state gets a first line
"Ref: <10 hex>" that makes every request unique, so no cached prompt can answer it.
"""
import hashlib
import http.client
import json
import time

CASES_PATH = "/bench/cases.json"

# request types of the benchmark (size, number of questions) and their representative case (the one used in
# harness/profile/census): median token count of its type
KINDS = {"short1q": ("short", 1), "short2q": ("short", 2), "medium3q": ("medium", 3), "long3q": ("long", 3)}
REP_CASE = {"short1q": "check03", "short2q": "triage09", "medium3q": "medium01-memory", "long3q": "long00-storage"}


def mono_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def epoch_ns():
    return time.time_ns()


def kind_of(case):
    for k, (size, nq) in KINDS.items():
        if case["size"] == size and len(case["questions"]) == nq:
            return k
    raise ValueError(f"case {case['id']}: no request type for size={case['size']} questions={len(case['questions'])}")


def load_cases(path=CASES_PATH):
    """All 48 cases, each with an added "kind" (short1q / short2q / medium3q / long3q)."""
    cases = json.load(open(path))
    for c in cases:
        c["kind"] = kind_of(c)
    return cases


def cases_of(cases, kind):
    return [c for c in cases if c["kind"] == kind]


def rep_case(cases, kind):
    """The representative case of a request type (REP_CASE)."""
    for c in cases:
        if c["id"] == REP_CASE[kind]:
            return c
    raise KeyError(REP_CASE[kind])


def case_by_id(cases, cid):
    for c in cases:
        if c["id"] == cid:
            return c
    raise KeyError(cid)


def tag(*parts):
    """10 hex characters from the parts: a deterministic unique Ref value."""
    return hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:10]


def body(case, ref):
    return json.dumps({"state": f"Ref: {ref}\n{case['state']}", "questions": case["questions"]}).encode()


class Client:
    """One keep-alive HTTP connection to /v1/systemone (the benchmark uses one connection per client, too)."""

    def __init__(self, host="127.0.0.1", port=18080, timeout=3600):
        self.host, self.port, self.timeout = host, port, timeout
        self.conn = None
        self.reconnects = 0

    def post(self, data, path="/v1/systemone"):
        """POST; returns a dict with the client stamps of the request. Raises on a non-200 status."""
        for attempt in (0, 1):
            if self.conn is None:
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            try:
                e0 = epoch_ns()
                t0 = mono_ns()
                self.conn.request("POST", path, data, {"Content-Type": "application/json"})
                resp = self.conn.getresponse()
                t_hdr = mono_ns()
                raw = resp.read()
                t1 = mono_ns()
            except (http.client.HTTPException, OSError):
                self.conn.close()
                self.conn = None
                self.reconnects += 1
                if attempt:
                    raise
                continue
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {raw[:300]!r}")
            j = json.loads(raw)
            return {"t_send_mono_ns": t0, "t_header_mono_ns": t_hdr, "t_recv_mono_ns": t1, "t_send_epoch_ns": e0,
                    "wall_ms": (t1 - t0) / 1e6, "input_tokens": j.get("usage", {}).get("input_tokens"),
                    "answers": j.get("answers")}
        raise RuntimeError("unreachable")

    def get(self, path):
        c = http.client.HTTPConnection(self.host, self.port, timeout=60)
        c.request("GET", path)
        r = c.getresponse()
        raw = r.read()
        c.close()
        return r.status, raw


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
