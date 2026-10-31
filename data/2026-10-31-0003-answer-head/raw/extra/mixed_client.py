# Mixed load on one server: the cases of the decision benchmark go one at a time in one thread,
# while other threads send completions (POST /completion, n_predict 16) and chat requests
# (POST /v1/chat/completions, max_tokens 16). The decision answers go to <answers json> in the format of
# conc_client.py. The status, the text and the times of the other requests go to <others json>.
# The exit code is 1 if one request gives a status other than 200, or no text, or no answer at all.
# Usage: mixed_client.py <port> <answers json> <others json> [<completions> <chats> <completion threads>]
import http.client
import json
import sys
import threading
import time

port, out_answers, out_others = int(sys.argv[1]), sys.argv[2], sys.argv[3]
n_comp = int(sys.argv[4]) if len(sys.argv) > 4 else 20
n_chat = int(sys.argv[5]) if len(sys.argv) > 5 else 5
n_threads = int(sys.argv[6]) if len(sys.argv) > 6 else 4
GAP = 0.5  # seconds between two requests of one thread, so the other requests spread over the decision requests
cases = json.load(open("/bench/cases.json"))
t0 = time.perf_counter()
lock = threading.Lock()


def post(path, obj):
    """Return (status, parsed body or the first 300 characters of the body or the error text, start, end)."""
    start = time.perf_counter() - t0
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3000)
        conn.request("POST", path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
        r = conn.getresponse()
        raw = r.read()
        try:
            body = json.loads(raw) if r.status == 200 else raw.decode(errors="replace")[:300]
        except ValueError:
            body = raw.decode(errors="replace")[:300]
        return r.status, body, start, time.perf_counter() - t0
    except Exception as e:  # the server stopped or the connection broke
        return "error", "%s: %s" % (type(e).__name__, e), start, time.perf_counter() - t0


decisions = {}
others = []


def decision_thread():
    for c in cases:
        status, body, start, end = post("/v1/systemone", {"state": c["state"], "questions": c["questions"]})
        answers = body.get("answers") if status == 200 and isinstance(body, dict) else body
        with lock:
            decisions[c["id"]] = {"status": status, "answers": answers, "start": start, "end": end}


def other_thread(kind, count, tid):
    for i in range(count):
        n = tid * 1000 + i
        if kind == "completion":
            status, body, start, end = post("/completion", {
                "prompt": "Request %d. Write one short sentence about a parcel that arrives late." % n,
                "n_predict": 16, "temperature": 0})
            text = body.get("content", "") if status == 200 and isinstance(body, dict) else ""
        else:
            status, body, start, end = post("/v1/chat/completions", {
                "messages": [{"role": "user", "content": "Request %d. Say hello in five words." % n}],
                "max_tokens": 16, "temperature": 0})
            text = ""
            if status == 200 and isinstance(body, dict):
                m = body["choices"][0]["message"]
                text = m.get("content") or m.get("reasoning_content") or ""
        with lock:
            others.append({"kind": kind, "thread": tid, "index": i, "status": status,
                           "text": text, "non_empty": bool(text.strip()), "start": start, "end": end,
                           "error": None if status == 200 else body})
        time.sleep(GAP)


threads = [threading.Thread(target=decision_thread)]
per = [n_comp // n_threads + (1 if t < n_comp % n_threads else 0) for t in range(n_threads)]
threads += [threading.Thread(target=other_thread, args=("completion", per[t], t)) for t in range(n_threads)]
threads += [threading.Thread(target=other_thread, args=("chat", n_chat, n_threads))]
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.perf_counter() - t0

others.sort(key=lambda o: (o["kind"], o["thread"], o["index"]))
spans = [(o["start"], o["end"]) for o in others]
overlap = sum(1 for d in decisions.values() if any(s < d["end"] and e > d["start"] for s, e in spans))
json.dump({cid: {"status": v["status"], "answers": v["answers"]} for cid, v in decisions.items()},
          open(out_answers, "w"), indent=1, sort_keys=True)
json.dump({"requests": others, "decision_requests": {cid: {"start": v["start"], "end": v["end"]} for cid, v in sorted(decisions.items())}},
          open(out_others, "w"), indent=1)
bad_dec = sum(1 for v in decisions.values() if v["status"] != 200)
bad_other = sum(1 for o in others if o["status"] != 200 or not o["non_empty"])
print("%d decision requests, %d with status 200, %d of them overlapped another request" % (
    len(decisions), len(decisions) - bad_dec, overlap), flush=True)
for kind in ("completion", "chat"):
    rs = [o for o in others if o["kind"] == kind]
    print("%s: %d requests, %d with status 200, %d with text" % (
        kind, len(rs), sum(1 for o in rs if o["status"] == 200), sum(1 for o in rs if o["non_empty"])), flush=True)
print("%.1f s" % wall, flush=True)
sys.exit(1 if bad_dec or bad_other or len(decisions) != len(cases) else 0)
