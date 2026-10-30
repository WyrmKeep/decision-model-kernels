#!/usr/bin/env python3
"""Run the 32-case smoke test against a decision server and score it.

Usage:
    python3 dectest/run_dectest.py --url http://127.0.0.1:18080 --out answers.json
    python3 dectest/run_dectest.py --compare answers-a.json answers-b.json

The first form sends each case of dectest/cases.json to POST /v1/systemone and
prints how many of the 52 decisions match the labels. It saves the answers as
{case id: {question id: answer}}.
The second form compares the top answers of two saved answer files.
The URL can also come from the environment variable BENCH_URL.
"""
import argparse
import http.client
import json
import os
import pathlib
import sys
import urllib.parse

HERE = pathlib.Path(__file__).resolve().parent


def wire(questions):
    """Cases use the type "bool". The server calls it "noul"."""
    out = {}
    for qid, q in questions.items():
        q = dict(q)
        if q["type"] == "bool":
            q["type"] = "noul"
        out[qid] = q
    return out


def top(ans):
    """The top answer: True or False, a choice label, or a rounded score."""
    if "noul" in ans or "bool" in ans:
        return ans.get("noul", ans.get("bool")) >= 0.5
    if "choice" in ans:
        return ans["choice"]
    return round(ans["score"])


def post(url, case):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=300)
    body = json.dumps({"state": case["state"], "questions": wire(case["questions"])}).encode()
    conn.request("POST", "/v1/systemone", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {raw[:300]!r}")
    return json.loads(raw)["answers"]


def run(args):
    cases = json.load(open(args.cases))
    answers, ok, total, by_task, wrong = {}, 0, 0, {}, []
    for c in cases:
        answers[c["id"]] = post(args.url, c)
        for qid, want in c["expect"].items():
            total += 1
            hit = top(answers[c["id"]][qid]) == want
            ok += hit
            t = by_task.setdefault(c["task"], [0, 0])
            t[0] += hit
            t[1] += 1
            if not hit:
                wrong.append(f"{c['id']}.{qid}")
    print(f"correct {ok}/{total}")
    for task, (a, b) in by_task.items():
        print(f"  {task:8s} {a}/{b}")
    print("misses:", ", ".join(wrong) if wrong else "none")
    if args.out:
        json.dump(answers, open(args.out, "w"), indent=1)
        print("answers saved to", args.out)


def compare(path_a, path_b):
    a, b = json.load(open(path_a)), json.load(open(path_b))
    same, total, diff = 0, 0, []
    for cid in a:
        for qid in a[cid]:
            total += 1
            if cid in b and qid in b[cid] and top(a[cid][qid]) == top(b[cid][qid]):
                same += 1
            else:
                diff.append(f"{cid}.{qid}")
    print(f"same top answer {same}/{total}")
    print("different:", ", ".join(diff) if diff else "none")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=os.environ.get("BENCH_URL", "http://127.0.0.1:18080"))
    ap.add_argument("--cases", default=str(HERE / "cases.json"))
    ap.add_argument("--out", default=None, help="save the answers to this file")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two saved answer files")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    sys.exit(main())
