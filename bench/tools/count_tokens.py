#!/usr/bin/env python3
"""Count the tokens of each case with the tokenizer of a running llama-server.

Usage:
    python3 tools/count_tokens.py cases.json --url http://127.0.0.1:8080
    python3 tools/count_tokens.py cases.json --compare other-cases.json --url ...

The tool calls POST /tokenize of the server. Start the server on the CPU with
the model file (no GPU is needed), for example:
    llama-server -m model.gguf --host 127.0.0.1 --port 8080 -ngl 0 -c 512 --no-warmup

For each case it prints the tokens of the state and of each question. The text
of a question is its instructions, then one "key: value" line for each criterion.
With --compare it prints the difference to a second file and lists every case
that differs. The exit code is 1 when any difference exists.
"""
import argparse
import json
import sys
import urllib.request


def make_counter(url):
    cache = {}

    def count(text):
        if text not in cache:
            body = json.dumps({"content": text, "add_special": False}).encode()
            req = urllib.request.Request(url.rstrip("/") + "/tokenize", body,
                                         {"Content-Type": "application/json"})
            cache[text] = len(json.load(urllib.request.urlopen(req, timeout=120))["tokens"])
        return cache[text]

    return count


def question_text(q):
    lines = [q["instructions"]]
    lines += [f"{k}: {v}" for k, v in (q.get("criteria") or {}).items()]
    return "\n".join(lines)


def count_file(path, count):
    out = {}
    for c in json.load(open(path)):
        out[c["id"]] = {"state": count(c["state"]),
                        "questions": {k: count(question_text(q)) for k, q in c["questions"].items()}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases")
    ap.add_argument("--url", required=True, help="base URL of a running llama-server")
    ap.add_argument("--compare", default=None, help="second cases file with the same case ids")
    ap.add_argument("--json", default=None, help="write the counts (and differences) to this file")
    args = ap.parse_args()

    count = make_counter(args.url)
    a = count_file(args.cases, count)
    b = count_file(args.compare, count) if args.compare else None
    if b is not None and list(a) != list(b):
        sys.exit("the two files do not have the same case ids in the same order")

    print(f"{'case':20s} {'state':>6s}  questions")
    for cid, row in a.items():
        print(f"{cid:20s} {row['state']:6d}  " + " ".join(f"{k}={v}" for k, v in row["questions"].items()))
    print(f"total state tokens: {sum(r['state'] for r in a.values())}")

    diffs = {}
    if b is not None:
        for cid in a:
            d_state = b[cid]["state"] - a[cid]["state"]
            d_q = {k: b[cid]["questions"].get(k, 0) - v for k, v in a[cid]["questions"].items()}
            if d_state or any(d_q.values()):
                diffs[cid] = {"state": d_state, "questions": d_q}
        print(f"cases with a different token count: {len(diffs)}")
        for cid, d in diffs.items():
            print(f"  {cid}: state {d['state']:+d}, questions {d['questions']}")
    if args.json:
        json.dump({"counts": a, "compare": b, "differences": diffs}, open(args.json, "w"), indent=1)
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
