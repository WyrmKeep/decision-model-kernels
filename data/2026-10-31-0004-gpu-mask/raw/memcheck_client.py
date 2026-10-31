# One request of each type (short with one question, short with two, medium, long) to a server.
import http.client, json, sys, time

port, out = int(sys.argv[1]), sys.argv[2]
cases = json.load(open("/bench/cases.json"))
pick = {}
for c in cases:
    key = "short%dq" % len(c["questions"]) if c["size"] == "short" else c["size"]
    pick.setdefault(key, c)
res = {}
for key in ("short1q", "short2q", "medium", "long"):
    c = pick[key]
    body = json.dumps({"state": c["state"], "questions": c["questions"]}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3000)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/systemone", body, {"Content-Type": "application/json"})
    r = conn.getresponse()
    raw = r.read()
    ms = (time.perf_counter() - t0) * 1000
    print(key, c["id"], r.status, "%.0f ms" % ms, flush=True)
    res[key] = {"id": c["id"], "status": r.status, "ms": ms, "answers": json.loads(raw).get("answers") if r.status == 200 else raw.decode()[:300]}
json.dump(res, open(out, "w"), indent=1)
