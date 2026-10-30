#!/usr/bin/env python3
"""Compare saved quality-pass answers (bench.py --write-reference files).

Usage: answers_compare.py --a 'runs/p1-bench-b*-S.reference.json' --b 'runs/p1-bench-b*-D.reference.json'
Reports, as JSON:
  - self-repeat: whether all files of one arm hold exactly the same values;
  - cross-arm: whether the first file of each arm holds exactly the same values, the
    largest absolute probability difference over all decisions and all files, and
    the number of decisions whose top answer differs.
"""
import argparse
import glob
import json


def probs(ans):
    if ans["type"] == "noul":
        return {"yes": ans["noul"]}
    if "probabilities" in ans:
        return dict(ans["probabilities"])
    return {"score": ans.get("score")}


def top(ans):
    if ans["type"] == "noul":
        return ans["noul"] >= 0.5
    if ans["type"] == "choice":
        return ans["choice"]
    return round(ans["score"])


def load(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files for {pattern}")
    return files, [json.load(open(f))["answers"] for f in files]


def compare(x, y):
    max_dp, flips, n = 0.0, [], 0
    for cid, qs in x.items():
        for qid, a in qs.items():
            b = y[cid][qid]
            n += 1
            pa, pb = probs(a), probs(b)
            for k in set(pa) | set(pb):
                max_dp = max(max_dp, abs((pa.get(k) or 0.0) - (pb.get(k) or 0.0)))
            if top(a) != top(b):
                flips.append(f"{cid}.{qid}")
    return {"decisions": n, "identical": x == y, "max_abs_dp": max_dp, "top_flips": flips}


ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
args = ap.parse_args()
fa, A = load(args.a)
fb, B = load(args.b)
out = {"files_a": fa, "files_b": fb,
       "self_repeat_a": all(x == A[0] for x in A),
       "self_repeat_b": all(x == B[0] for x in B),
       "self_a_max_dp": max(compare(A[0], x)["max_abs_dp"] for x in A),
       "self_b_max_dp": max(compare(B[0], x)["max_abs_dp"] for x in B)}
pairs = [compare(x, y) for x in A for y in B]
out["cross_identical_all"] = all(p["identical"] for p in pairs)
out["cross_max_abs_dp"] = max(p["max_abs_dp"] for p in pairs)
out["cross_top_flips"] = sorted({f for p in pairs for f in p["top_flips"]})
out["decisions"] = pairs[0]["decisions"]
print(json.dumps(out, indent=1))
