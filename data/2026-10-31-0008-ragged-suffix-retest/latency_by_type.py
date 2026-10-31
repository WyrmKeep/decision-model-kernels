#!/usr/bin/env python3
"""Latency of the requests of the private benchmark for each request type.

The method is the method of the sequential phase of bench.py: the same warm-up, the same
order of the cases in each pass, and the same tags (a first line "Ref: <tag>") that stop the
cache of the prompt. The result is the mean latency for each type of request, in all passes.
The types are: short with one question, short with two questions, medium, and long.

Usage: latency_by_type.py <port> [passes]
"""
import hashlib
import http.client
import json
import random
import statistics
import sys
import time

port = int(sys.argv[1])
passes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
cases = json.load(open("/bench/cases.json"))


def tag(*parts):
    return hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:10]


def body(case, ref):
    return json.dumps({"state": f"Ref: {ref}\n{case['state']}", "questions": case["questions"]}).encode()


def kind(case):
    return f"short{len(case['questions'])}q" if case["size"] == "short" else case["size"]


conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)


def post(data):
    t0 = time.perf_counter()
    conn.request("POST", "/v1/systemone", data, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    ms = (time.perf_counter() - t0) * 1000.0
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {raw[:300]!r}")
    return ms


for i, c in enumerate(cases[:2] + cases[-1:]):
    post(body(c, tag("warmup", i)))

order = list(range(len(cases)))
by_kind = {}
for p in range(passes):
    random.Random(p).shuffle(order)
    for i in order:
        c = cases[i]
        by_kind.setdefault(kind(c), []).append(post(body(c, tag("seq", p, c["id"]))))

result = {k: {"n": len(v), "mean_ms": statistics.fmean(v), "median_ms": statistics.median(v)} for k, v in sorted(by_kind.items())}
for k, v in result.items():
    print(f"{k:8s} n={v['n']:4d} mean_ms={v['mean_ms']:.3f} median_ms={v['median_ms']:.3f}", flush=True)
print("RESULT " + json.dumps(result), flush=True)
