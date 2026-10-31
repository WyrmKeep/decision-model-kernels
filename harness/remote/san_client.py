#!/usr/bin/env python3
"""Send one request of each type to a running server. run_soak.sh uses it for the sanitizer phase.

Usage: san_client.py <port> <out json> [<cases json>]
The default cases file is /bench/cases.json. The script takes the first case of each type:
the short cases with the same number of questions, the first medium case and the first long case.
It sends the cases one at a time, with no unique line, as the quality pass of bench.py does.
The out file has the status and the answers of each request.
The exit code is 0 if each request has the status 200 and an answer for each question.
It uses the Python standard library only.
"""
import http.client
import json
import sys
import time

port, out = int(sys.argv[1]), sys.argv[2]
cases = json.load(open(sys.argv[3] if len(sys.argv) > 3 else "/bench/cases.json"))
pick = {}
for c in cases:
    key = f"short{len(c['questions'])}q" if c["size"] == "short" else c["size"]
    pick.setdefault(key, c)
rank = {"short": 0, "medium": 1, "long": 2}
keys = sorted(pick, key=lambda k: (rank.get(pick[k]["size"], 9), k))
res, bad = {}, 0
for key in keys:
    c = pick[key]
    body = json.dumps({"state": c["state"], "questions": c["questions"]}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3000)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/systemone", body, {"Content-Type": "application/json"})
    r = conn.getresponse()
    raw = r.read()
    ms = (time.perf_counter() - t0) * 1000
    answers = json.loads(raw).get("answers") if r.status == 200 else None
    ok = r.status == 200 and answers is not None and set(c["questions"]) <= set(answers)
    bad += 0 if ok else 1
    print(key, c["id"], r.status, f"{ms:.0f} ms", "ok" if ok else "BAD", flush=True)
    res[key] = {"id": c["id"], "status": r.status, "ok": ok,
                "answers": answers if ok else raw.decode(errors="replace")[:300]}
json.dump(res, open(out, "w"), indent=1, sort_keys=True)
print(f"requests={len(keys)} bad={bad}")
sys.exit(1 if bad else 0)
