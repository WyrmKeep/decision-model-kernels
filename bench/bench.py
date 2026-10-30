#!/usr/bin/env python3
"""Decision-model benchmark client. Fixed harness: do not change it to improve a result.

Uses the Python standard library only. It sends requests to
POST http://HOST:PORT/v1/systemone of a running llama-server. HOST and PORT
come from --host and --port, or from BENCH_HOST and BENCH_PORT.

Phases:
  warmup   3 requests, not counted
  quality  every case once with its raw state: accuracy against labels and
           agreement with the reference answers
  seq      --passes timed passes over every case, one request at a time.
           Each request's state starts with a unique "Ref:" line so no
           request can be answered from a cache of an earlier identical one.
  tput     every case x3 (unique Ref lines), --conc requests in flight
  cold     optional: --cold N single short requests, each after 30 s idle

Primary metric: lat_ms = median over seq passes of the mean request latency.

reference.json and baseline.json can hold the marker {"status": "not-measured"}.
The client then skips the agreement metrics and the quality guard.
"""
import argparse
import concurrent.futures as cf
import hashlib
import http.client
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--host", default=os.environ.get("BENCH_HOST", "127.0.0.1"))
ap.add_argument("--port", type=int, default=int(os.environ.get("BENCH_PORT", "18080")))
ap.add_argument("--cases", default=os.path.join(HERE, "cases.json"))
ap.add_argument("--reference", default=os.path.join(HERE, "reference.json"))
ap.add_argument("--baseline", default=os.path.join(HERE, "baseline.json"))
ap.add_argument("--out", required=True)
ap.add_argument("--passes", type=int, default=5)
ap.add_argument("--conc", type=int, default=8)
ap.add_argument("--cold", type=int, default=0)
ap.add_argument("--startup-s", type=float, default=None)
ap.add_argument("--write-reference", default=None, metavar="LABEL",
                help="save the quality-pass answers as LABEL.reference.json in the folder of --out")
args = ap.parse_args()

cases = json.load(open(args.cases))
SIZES = ("short", "medium", "long")


def load_optional(path, what):
    """Return the JSON of an optional file, or None when it is missing or not measured yet."""
    try:
        data = json.load(open(path))
    except FileNotFoundError:
        return None
    if data.get("status") == "not-measured":
        print(f"note: {what} is not measured yet ({path}): its checks are skipped", file=sys.stderr)
        return None
    return data


def tag(*parts):
    return hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:10]


def body(case, ref=None):
    state = case["state"] if ref is None else f"Ref: {ref}\n{case['state']}"
    return json.dumps({"state": state, "questions": case["questions"]}).encode()


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
                self.conn.request("POST", "/v1/systemone", data,
                                  {"Content-Type": "application/json"})
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
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def stats(xs):
    return {"n": len(xs), "mean": statistics.fmean(xs), "p50": pct(xs, 50),
            "p95": pct(xs, 95), "max": max(xs)}


def top(ans):
    if ans["type"] == "noul":
        return ans["noul"] >= 0.5
    if ans["type"] == "choice":
        return ans["choice"]
    return round(ans["score"])


def dp(a, b):
    if a["type"] == "noul":
        return abs(a["noul"] - b["noul"])
    pa, pb = a["probabilities"], b["probabilities"]
    return max(abs(pa[k] - pb.get(k, 0.0)) for k in pa)


errors = []
result = {"meta": {"started": time.time(), "passes": args.passes, "conc": args.conc,
                   "startup_s": args.startup_s, "cases": len(cases),
                   "decisions": sum(len(c["questions"]) for c in cases)},
          "phases": {}}
client = Client()


def phase(name):
    result["phases"][name] = [time.time(), None]


def end(name):
    result["phases"][name][1] = time.time()


# warmup
phase("warmup")
for i, c in enumerate(cases[:2] + cases[-1:]):
    try:
        client.post(body(c, tag("warmup", i)))
    except Exception as e:  # noqa: BLE001
        errors.append(f"warmup {c['id']}: {e}")
end("warmup")

# quality
phase("quality")
answers, tokens, model = {}, {s: [] for s in SIZES}, None
for c in cases:
    try:
        r, ms = client.post(body(c))
    except Exception as e:  # noqa: BLE001
        errors.append(f"quality {c['id']}: {e}")
        continue
    model = r.get("model", model)
    missing = set(c["questions"]) - set(r.get("answers", {}))
    if missing:
        errors.append(f"quality {c['id']}: missing answers {sorted(missing)}")
        continue
    answers[c["id"]] = r["answers"]
    tokens[c["size"]].append(r.get("usage", {}).get("input_tokens", 0))
end("quality")

correct = {s: [0, 0] for s in SIZES}
for c in cases:
    a = answers.get(c["id"])
    for qid, want in c["expect"].items():
        correct[c["size"]][1] += 1
        if a is not None and top(a[qid]) == want:
            correct[c["size"]][0] += 1
quality = {"model": model,
           "correct": sum(v[0] for v in correct.values()),
           "total": sum(v[1] for v in correct.values()),
           "correct_by_size": {s: f"{v[0]}/{v[1]}" for s, v in correct.items()},
           "input_tokens_mean": {s: (statistics.fmean(v) if v else None) for s, v in tokens.items()}}
ref = load_optional(args.reference, "the reference")
if ref is not None:
    agree, n, dps, flips = 0, 0, [], []
    for c in cases:
        for qid in c["questions"]:
            n += 1
            a = answers.get(c["id"], {}).get(qid)
            b = ref["answers"][c["id"]][qid]
            if a is None:
                continue
            if top(a) == top(b):
                agree += 1
            else:
                flips.append(f"{c['id']}.{qid}")
            dps.append(dp(a, b))
    quality.update({"agree": agree / n, "agree_n": f"{agree}/{n}", "flips": flips,
                    "mean_dp": statistics.fmean(dps) if dps else None,
                    "max_dp": max(dps) if dps else None,
                    "reference_model": ref.get("label")})
result["quality"] = quality
if args.write_reference:
    ref_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"{args.write_reference}.reference.json")
    json.dump({"label": args.write_reference, "answers": answers}, open(ref_path, "w"), indent=1)

# sequential latency
phase("seq")
order = list(range(len(cases)))
pass_means, by_size, all_ms, nondet = [], {s: [] for s in SIZES}, [], 0.0
for p in range(args.passes):
    random.Random(p).shuffle(order)
    ms_pass = []
    for i in order:
        c = cases[i]
        try:
            r, ms = client.post(body(c, tag("seq", p, c["id"])))
        except Exception as e:  # noqa: BLE001
            errors.append(f"seq {p} {c['id']}: {e}")
            continue
        ms_pass.append(ms)
        by_size[c["size"]].append(ms)
    if ms_pass:
        pass_means.append(statistics.fmean(ms_pass))
        all_ms += ms_pass
end("seq")
seq = {"lat_ms": statistics.median(pass_means) if pass_means else None,
       "pass_means": pass_means,
       "spread_pct": (100.0 * (max(pass_means) - min(pass_means)) / statistics.median(pass_means)
                      if pass_means else None),
       "all": stats(all_ms) if all_ms else None,
       "by_size": {s: stats(v) for s, v in by_size.items() if v},
       "reconnects": client.reconnects}
result["seq"] = seq

# concurrent throughput
phase("tput")
jobs = [(k, c) for k in range(3) for c in cases]
random.Random(42).shuffle(jobs)
local = __import__("threading").local()


def work(job):
    k, c = job
    if not hasattr(local, "client"):
        local.client = Client()
    _, ms = local.client.post(body(c, tag("tput", k, c["id"])))
    return ms, len(c["questions"])


t0 = time.perf_counter()
lat, dec = [], 0
with cf.ThreadPoolExecutor(max_workers=args.conc) as ex:
    for fut in [ex.submit(work, j) for j in jobs]:
        try:
            ms, nq = fut.result()
            lat.append(ms)
            dec += nq
        except Exception as e:  # noqa: BLE001
            errors.append(f"tput: {e}")
wall = time.perf_counter() - t0
end("tput")
result["tput"] = {"conc": args.conc, "requests": len(lat), "wall_s": wall,
                  "rps": len(lat) / wall, "decisions_per_s": dec / wall,
                  "lat": stats(lat) if lat else None}

# cold requests after idle
if args.cold:
    phase("cold")
    cold = []
    for i in range(args.cold):
        time.sleep(30)
        try:
            _, ms = Client().post(body(cases[0], tag("cold", i)))
            cold.append(ms)
        except Exception as e:  # noqa: BLE001
            errors.append(f"cold {i}: {e}")
    end("cold")
    result["cold_ms"] = cold

# guard
result["errors"] = errors
guard = {"errors": len(errors) == 0}
base = load_optional(args.baseline, "the baseline")
if base is not None and "agree" in quality:
    guard["correct"] = quality["correct"] >= base["correct"] - 1
    guard["agree"] = quality["agree"] >= base["agree"] - 0.02
    guard["mean_dp"] = quality["mean_dp"] <= base["mean_dp"] + 0.01
result["guard"] = guard
# The verdict exists only when the quality guards ran: a measured baseline and a measured reference.
result["guard_pass"] = all(guard.values()) if base is not None and "agree" in quality else None
json.dump(result, open(args.out, "w"), indent=1)

bs = seq["by_size"]
print(f"lat_ms={seq['lat_ms']:.1f} spread={seq['spread_pct']:.1f}% "
      f"short={bs['short']['mean']:.1f} medium={bs['medium']['mean']:.1f} long={bs['long']['mean']:.1f} "
      f"p95={seq['all']['p95']:.1f} rps@{args.conc}={result['tput']['rps']:.2f} "
      f"correct={quality['correct']}/{quality['total']} "
      f"agree={quality.get('agree_n')} mean_dp={quality.get('mean_dp')} "
      f"errors={len(errors)} guard={result['guard_pass']}"
      if seq["lat_ms"] is not None else f"FAILED errors={errors[:5]}")
sys.exit(0 if not errors else 1)
