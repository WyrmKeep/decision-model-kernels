#!/usr/bin/env python3
"""Test of extra/prof/nsys_client.py against fake_server.py (no GPU): all kinds + replay.
Checks: JSON schema and field names, request counts, case order, unique Refs per file and server-side, identical Refs and
identical case sequences between each nsys-run kind and its segment of the replay run, monotonic stamps, exit status.
Usage: python3 test_client.py [--cases PATH] [--out DIR]   (--cases default: $DMK_BENCH_DIR/cases.json, else bench/cases.json of this repository;
--out default: a new temporary folder, which is printed)"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE = HERE.parents[1]
CLIENT = PROFILE / "extra" / "prof" / "nsys_client.py"
FIELDS = ["idx", "segment", "kind", "case", "phase", "ref", "t_send_mono_ns", "t_header_mono_ns", "t_recv_mono_ns",
          "t_send_epoch_ns", "wall_ms", "input_tokens"]
DEFAULT_CASES = str(Path(os.environ.get("DMK_BENCH_DIR") or HERE.parents[3] / "bench") / "cases.json")


def run(kind, port, cases, outdir, extra=()):
    log = outdir / f"server-{kind}.jsonl"
    srv = subprocess.Popen([sys.executable, str(HERE / "fake_server.py"), "--port", str(port), "--log", str(log)],
                           stdout=subprocess.PIPE, text=True)
    srv.stdout.readline()
    out = outdir / f"{kind}.client.json"
    p = subprocess.run([sys.executable, str(CLIENT), "--kind", kind, "--n", "10", "--out", str(out), "--cases", cases,
                        "--port", str(port), *extra], capture_output=True, text=True, cwd=str(CLIENT.parent))
    (outdir / f"{kind}.stdout.txt").write_text(p.stdout + p.stderr)
    srv.terminate()
    srv.wait()
    assert p.returncode == 0, (kind, p.returncode, p.stdout, p.stderr)
    refs_server = [json.loads(l)["ref"].replace("Ref: ", "", 1) for l in log.read_text().splitlines()]
    return json.loads(out.read_text()), refs_server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    outdir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="nsys-client-test-"))
    outdir.mkdir(parents=True, exist_ok=True)
    cases = json.load(open(args.cases))
    ids = [c["id"] for c in cases]
    res = {}
    port = 18110
    for kind in ["short1q", "short2q", "medium3q", "long3q", "all48", "replay"]:
        port += 1
        res[kind] = run(kind, port, args.cases, outdir)
    fails = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    exp_counts = {"short1q": 12, "short2q": 12, "medium3q": 12, "long3q": 12, "all48": 50, "replay": 12 * 4 + 50}
    for kind, (j, refs_server) in res.items():
        reqs = j["requests"]
        check(set(j) == {"meta", "requests", "errors"}, f"{kind}: top-level keys meta/requests/errors")
        check(all(list(r) == FIELDS for r in reqs), f"{kind}: every request has exactly the fields {FIELDS}")
        check(len(reqs) == exp_counts[kind] and not j["errors"], f"{kind}: {len(reqs)} requests (expected {exp_counts[kind]}), no errors")
        check([r["idx"] for r in reqs] == list(range(len(reqs))), f"{kind}: idx = 0..{len(reqs) - 1}")
        check(len({r["ref"] for r in reqs}) == len(reqs), f"{kind}: all Refs unique")
        check(sorted(refs_server) == sorted(r["ref"] for r in reqs), f"{kind}: server saw exactly the client's Refs")
        check(all(r["t_send_mono_ns"] < r["t_header_mono_ns"] <= r["t_recv_mono_ns"] for r in reqs), f"{kind}: send < header <= recv")
        check(all(a["t_recv_mono_ns"] <= b["t_send_mono_ns"] for a, b in zip(reqs, reqs[1:])), f"{kind}: requests strictly sequential")
        check(all(r["input_tokens"] and r["wall_ms"] > 0 for r in reqs), f"{kind}: input_tokens and wall_ms set")
        check(all(r["case"] in ids for r in reqs), f"{kind}: case ids exist in cases.json")
        for seg in j["meta"]["segments"]:
            rs = [r for r in reqs if r["segment"] == seg["name"]]
            check(seg["first_idx"] == rs[0]["idx"] and seg["last_idx"] == rs[-1]["idx"] and seg["n_warmup"] + seg["n_timed"] == len(rs),
                  f"{kind}: meta.segments[{seg['name']}] matches the requests")
    # per-kind sequence
    rep = {"short1q": "check03", "short2q": "triage09", "medium3q": "medium01-memory", "long3q": "long00-storage"}
    for kind in ["short1q", "short2q", "medium3q", "long3q"]:
        reqs = res[kind][0]["requests"]
        timed = [r for r in reqs if r["phase"] == "timed"]
        warm = [r for r in reqs if r["phase"] == "warmup"]
        check([r["phase"] for r in reqs] == ["warmup"] * 2 + ["timed"] * 10, f"{kind}: 2 warmup then 10 timed")
        check(timed[0]["case"] == rep[kind], f"{kind}: first timed request is the representative case {rep[kind]}")
        check(all(r["kind"] == kind for r in reqs), f"{kind}: all cases are of the kind")
        cyc = [c["id"] for c in cases if c["kind"] == kind] if "kind" in cases[0] else None
        if cyc:
            i0 = cyc.index(rep[kind])
            check([r["case"] for r in timed] == [cyc[(i0 + i) % len(cyc)] for i in range(10)], f"{kind}: timed cases cycle in cases.json order")
            check([r["case"] for r in warm] == [cyc[(i0 - 2 + i) % len(cyc)] for i in range(2)], f"{kind}: warm-up cases = the 2 before the rep case")
    a = res["all48"][0]["requests"]
    check([r["case"] for r in a if r["phase"] == "timed"] == ids, "all48: timed cases = all 48 in cases.json order")
    check([r["case"] for r in a if r["phase"] == "warmup"] == ["check03", "triage09"], "all48: warm-up = rep short1q, rep short2q")
    # replay equals the segments
    rp = res["replay"][0]["requests"]
    for seg in ["short1q", "short2q", "medium3q", "long3q", "all48"]:
        mine = [(r["phase"], r["case"], r["ref"]) for r in res[seg][0]["requests"]]
        theirs = [(r["phase"], r["case"], r["ref"]) for r in rp if r["segment"] == seg]
        check(mine == theirs, f"replay segment {seg}: same phases, cases and Refs as the nsys run of {seg}")
    check([s["name"] for s in res["replay"][0]["meta"]["segments"]] == ["short1q", "short2q", "medium3q", "long3q", "all48"],
          "replay: segments in schedule order")
    print("\nclient test:", "ALL PASS" if not fails else f"{len(fails)} FAIL")
    (outdir / "result.txt").write_text("ALL PASS\n" if not fails else "\n".join(fails) + "\n")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
