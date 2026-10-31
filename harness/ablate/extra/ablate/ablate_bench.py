#!/usr/bin/env python3
"""Lean latency client for the timing-only ablation build (phase 3). Answers are ignored.

Runs inside python:3.12-slim on the GPU host with host networking, stdlib only, like bench.py.
Same requests as the benchmark's sequential phase (bench.py "seq"): one request at a time, every request
state starts with a unique "Ref:" line so no earlier request can answer it, passes shuffled with random.Random(pass).
No quality phase, no throughput phase, no reference comparison: ablated runs give meaningless answers.

Usage (inside the harness client container, see run_client.sh):
  python -u /extra/ablate/ablate_bench.py --cases /bench/cases.json --size all --passes 5 \
      --out /runs/{name}.json [--warmup 6] [--linger-s 3]

  --size short|medium|long|all   which request sizes to send (the case "size" field)
  --warmup N                     N untimed requests first (default 6: the first N cases of the chosen size, spread)
  --linger-s S                   sleep S seconds after the last request, so the server's idle dump of DMK_OPS
                                 (DMK_ABLATE_SYNC=1, one second idle) reaches the container log before teardown
Output JSON: meta, by_size {mean, p50, p95, min, max, n}, pass_means_ms per size, per-request list
[{id, size, pass, ms, input_tokens, n_questions}], errors. Prints one summary line.
"""
import argparse
import hashlib
import http.client
import json
import random
import statistics
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=18080)
ap.add_argument("--cases", default="/bench/cases.json")
ap.add_argument("--size", choices=["short", "medium", "long", "all"], default="all")
ap.add_argument("--passes", type=int, default=5)
ap.add_argument("--warmup", type=int, default=6)
ap.add_argument("--linger-s", type=float, default=0.0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

SIZES = ("short", "medium", "long")
cases = [c for c in json.load(open(args.cases)) if args.size == "all" or c["size"] == args.size]
if not cases:
    sys.exit(f"no cases of size {args.size}")


def tag(*parts):
    return hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:10]


def body(case, ref):
    return json.dumps({"state": f"Ref: {ref}\n{case['state']}", "questions": case["questions"]}).encode()


class Client:
    def __init__(self):
        self.conn = None
        self.reconnects = 0

    def post(self, data):
        for attempt in (0, 1):
            if self.conn is None:
                self.conn = http.client.HTTPConnection(args.host, args.port, timeout=300)
            try:
                t0 = time.perf_counter()
                self.conn.request("POST", "/v1/systemone", data, {"Content-Type": "application/json"})
                resp = self.conn.getresponse()
                raw = resp.read()
                ms = (time.perf_counter() - t0) * 1000.0
            except (http.client.HTTPException, OSError):
                self.conn.close()
                self.conn = None
                self.reconnects += 1
                if attempt:
                    raise
                continue
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {raw[:300]!r}")
            return json.loads(raw), ms
        raise RuntimeError("unreachable")


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def stats(xs):
    return {"n": len(xs), "mean": statistics.fmean(xs), "p50": pct(xs, 50), "p95": pct(xs, 95),
            "min": min(xs), "max": max(xs)}


client = Client()
errors, requests = [], []
t_start = time.time()

# warmup: spread over the chosen cases so every size in the set sees a few shapes first
step = max(1, len(cases) // max(1, args.warmup))
for i, c in enumerate(cases[::step][:args.warmup]):
    try:
        client.post(body(c, tag("warmup", i)))
    except Exception as e:  # noqa: BLE001
        errors.append(f"warmup {c['id']}: {e}")

order = list(range(len(cases)))
pass_means = {s: [] for s in SIZES}
t_seq = time.time()
for p in range(args.passes):
    random.Random(p).shuffle(order)
    this_pass = {s: [] for s in SIZES}
    for i in order:
        c = cases[i]
        try:
            r, ms = client.post(body(c, tag("seq", p, c["id"])))
        except Exception as e:  # noqa: BLE001
            errors.append(f"seq {p} {c['id']}: {e}")
            continue
        this_pass[c["size"]].append(ms)
        requests.append({"id": c["id"], "size": c["size"], "pass": p, "ms": ms,
                         "input_tokens": r.get("usage", {}).get("input_tokens"),
                         "n_questions": len(c["questions"])})
    for s in SIZES:
        if this_pass[s]:
            pass_means[s].append(statistics.fmean(this_pass[s]))
t_end = time.time()

by_size = {}
for s in SIZES:
    ms = [r["ms"] for r in requests if r["size"] == s]
    if ms:
        by_size[s] = stats(ms)
        by_size[s]["pass_means_ms"] = pass_means[s]
result = {"meta": {"size": args.size, "passes": args.passes, "warmup": args.warmup, "cases": len(cases),
                   "started": t_start, "seq_started": t_seq, "seq_s": t_end - t_seq,
                   "reconnects": client.reconnects},
          "by_size": by_size, "requests": requests, "errors": errors}
json.dump(result, open(args.out, "w"), indent=1)

print("seq_s=%.1f " % (t_end - t_seq) + " ".join(f"{s}={by_size[s]['mean']:.1f}ms(n={by_size[s]['n']})" for s in by_size)
      + f" errors={len(errors)}")
if args.linger_s > 0:
    time.sleep(args.linger_s)
sys.exit(0 if not errors else 1)
