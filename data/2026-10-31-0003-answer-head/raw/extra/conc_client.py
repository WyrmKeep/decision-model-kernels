# Send every case of the benchmark once, with C requests in flight, and write the answers by case id.
# Usage: concurrent.py <port> <out json> <C>
import concurrent.futures as cf
import http.client
import json
import sys
import time

port, out, conc = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
cases = json.load(open("/bench/cases.json"))


def one(c):
    body = json.dumps({"state": c["state"], "questions": c["questions"]}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3000)
    conn.request("POST", "/v1/systemone", body, {"Content-Type": "application/json"})
    r = conn.getresponse()
    raw = r.read()
    ans = json.loads(raw).get("answers") if r.status == 200 else raw.decode()[:300]
    return c["id"], r.status, ans


res = {}
t0 = time.perf_counter()
with cf.ThreadPoolExecutor(conc) as ex:
    for cid, status, ans in ex.map(one, cases):
        res[cid] = {"status": status, "answers": ans}
print("%d cases, %d with status 200, %.1f s, %d requests in flight" % (
    len(res), sum(1 for v in res.values() if v["status"] == 200), time.perf_counter() - t0, conc), flush=True)
json.dump(res, open(out, "w"), indent=1, sort_keys=True)
