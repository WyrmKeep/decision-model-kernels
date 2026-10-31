#!/usr/bin/env python3
"""Synthetic inputs for the nsys_census.py tests (no GPU). Built from the three real legacy exports
(--legacy DIR with {short,medium,long}.sqlite, default $DMK_LEGACY_TRACES): their request windows are re-created as client JSONs with stamps
that follow the verified mapping (CLOCK_MONOTONIC_ns = systemClockNs + t_nsys, epoch_ns = utcEpochNs + t_nsys + 20 us).

Writes to --work (default <tmp>/nsys-synth, tmp = $TMPDIR or /tmp):
  legacy-{short,medium,long}.{sqlite,client.json}   every request, first request of each run = warm-up (kinds short2q / medium3q / long3q)
  tail-short.*          only requests 3..11 in the client JSON (3 requests of the capture are not requests of the client)
  shift-short.*         client stamps shifted by +3 ms (wrong clock): window assignment must fail, order assignment must work
  osrt-short.*          sqlite copy with a synthetic OSRT_API table (known wait calls in the host gaps) -> expected.osrt.json
  stray-short.*         sqlite copy with a model-load-like H2D copy before the first pass and 3 stray kernels 200 ms after the last pass
  syn-ctl.client.json   control run: the three legacy runs' requests with the injected differences below subtracted from wall_ms
  syn-dec.client.json   decomposition-client-like JSON (phases unique / cycle) for the overhead --dec test
  expected.json         injected numbers for the checks"""
import argparse
import importlib.util
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = None   # folder with the legacy exports, set in main()
KIND = {"short": "short2q", "medium": "medium3q", "long": "long3q"}
INJECT_MS = {"short": 2.5, "medium": 4.0, "long": 6.0}    # difference nsys - ctl per request injected into syn-ctl
EPS_MS = 0.2                                               # +-eps alternating noise on the control walls


def load_nc():
    spec = importlib.util.spec_from_file_location("nsys_census", HERE.parent / "nsys_census.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def legacy_requests(nc, name):
    t = nc.load_trace(name, SRC / (name + ".sqlite"))
    prob = []
    ps = nc.segment(t, prob)
    for P in ps:
        nc.classify(P, prob, name)
    n = nc.requests_of(ps)
    reqs = []
    for r in range(n):
        g = [P for P in ps if P["request"] == r]
        reqs.append({"first_api": g[0]["win_start"], "last_end": max(k["end"] for P in g for k in P["ks"]),
                     "n_tokens": sum(P["n_tokens"] for P in g), "case": nc.CASES[name][r % len(nc.CASES[name])]})
    return t, ps, reqs


def client_json(nc, t, name, reqs, first_idx=0, shift_ns=0, warm=1, tag="legacy"):
    out = []
    for j, r in enumerate(reqs):
        send = r["first_api"] - 1_000_000
        recv = r["last_end"] + 500_000
        out.append({"idx": j, "segment": KIND[name], "kind": KIND[name], "case": r["case"],
                    "phase": "warmup" if j < warm else "timed", "ref": "syn-%s-%s-%03d" % (tag, name, first_idx + j),
                    "t_send_mono_ns": t.sys_clock_ns + send + shift_ns, "t_header_mono_ns": t.sys_clock_ns + recv - 100_000 + shift_ns,
                    "t_recv_mono_ns": t.sys_clock_ns + recv + shift_ns, "t_send_epoch_ns": t.utc_epoch_ns + send + 20_000 + shift_ns,
                    "wall_ms": (recv - send) / 1e6, "input_tokens": r["n_tokens"]})
    return {"meta": {"script": "make_synthetic.py", "synthetic": True, "name": name}, "requests": out, "errors": []}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1) + "\n")


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def copy_db(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    shutil.copyfile(src, dst)


def make_osrt(nc, t, ps, dst):
    """synthetic OSRT_API: in the largest CUDA-API-free gap before every pass (except the first): wait call over 60 % of it
    (pthread_cond_wait) + a 5 us send; a decoy call on another thread covering the whole gap. Returns the expected wait ns per pass."""
    tid = collections_most_common_tid(t)
    api = [a for a in t.api if a["tid"] == tid]
    db = sqlite3.connect(str(dst))
    nid = db.execute("select max(id) from StringIds").fetchone()[0] + 1
    names = {"pthread_cond_wait": nid, "send": nid + 1, "epoll_wait": nid + 2}
    for k, v in names.items():
        db.execute("insert into StringIds (id, value) values (?, ?)", (v, k))
    db.execute("create table OSRT_API (start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER, nameId INTEGER, "
               "returnValue INTEGER, nestingLevel INTEGER, callchainId INTEGER, argumentsId INTEGER)")
    expected = {}
    for i in range(1, len(ps)):
        Q, P = ps[i - 1], ps[i]
        t0 = max(k["end"] for k in Q["ks"])
        tl = min(a["start"] for a in P["api"] if a["name"].startswith("cudaLaunchKernel"))
        calls = sorted((a for a in api if a["end"] > t0 and a["start"] < tl), key=lambda a: a["start"])
        gaps, cur = [], t0
        for c in calls:
            if c["start"] > cur:
                gaps.append((cur, c["start"]))
            cur = max(cur, c["end"])
        if tl > cur:
            gaps.append((cur, tl))
        gaps = [g for g in gaps if g[1] - g[0] >= 200_000]
        if not gaps:
            continue
        gs, ge = max(gaps, key=lambda g: g[1] - g[0])
        L = ge - gs
        ws, we = gs + L // 10, gs + 7 * L // 10
        db.execute("insert into OSRT_API values (?, ?, 1, ?, ?, 0, 0, 0, 0)", (ws, we, tid, names["pthread_cond_wait"]))
        db.execute("insert into OSRT_API values (?, ?, 1, ?, ?, 0, 0, 0, 0)", (gs + 8 * L // 10, gs + 8 * L // 10 + 5000, tid, names["send"]))
        db.execute("insert into OSRT_API values (?, ?, 1, ?, ?, 0, 0, 0, 0)", (gs, ge, tid + 1, names["epoll_wait"]))   # other thread
        expected[str(P["idx"])] = {"wait_ns": we - ws, "n_calls": 2, "gap_ns": L}
    db.commit()
    db.close()
    return expected


def collections_most_common_tid(t):
    import collections
    c = collections.Counter(a["tid"] for a in t.api if a["name"].startswith("cudaLaunchKernel"))
    return c.most_common(1)[0][0]


def make_stray(t, dst):
    db = sqlite3.connect(str(dst))
    cols = [r[1] for r in db.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    rows = db.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL order by start desc limit 3").fetchall()
    for r in rows:
        d = dict(zip(cols, r))
        d["start"] += 200_000_000
        d["end"] += 200_000_000
        d["correlationId"] += 50_000_000
        db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL (%s) values (%s)" % (",".join(cols), ",".join("?" * len(cols))), [d[c] for c in cols])
    mcols = [r[1] for r in db.execute("pragma table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")]
    r = db.execute("select * from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 order by start limit 1").fetchone()
    d = dict(zip(mcols, r))
    first_k = t.K[0]["start"]
    d["start"], d["end"], d["bytes"], d["correlationId"] = first_k - 100_000_000, first_k - 50_000_000, 104_857_600, 99_999_999
    db.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY (%s) values (%s)" % (",".join(mcols), ",".join("?" * len(mcols))), [d[c] for c in mcols])
    db.commit()
    db.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "nsys-synth"))
    ap.add_argument("--legacy", default=os.environ.get("DMK_LEGACY_TRACES", ""), help="folder with short.sqlite, medium.sqlite, long.sqlite (default $DMK_LEGACY_TRACES)")
    args = ap.parse_args()
    if not args.legacy:
        raise SystemExit("make_synthetic: no legacy exports: use --legacy DIR or set DMK_LEGACY_TRACES")
    global SRC
    SRC = Path(args.legacy)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    nc = load_nc()
    exp = {"inject_ms": INJECT_MS, "eps_ms": EPS_MS}
    ctl_reqs, dec_reqs = [], []
    for name in ("short", "medium", "long"):
        t, ps, reqs = legacy_requests(nc, name)
        link(SRC / (name + ".sqlite"), work / ("legacy-%s.sqlite" % name))
        cj = client_json(nc, t, name, reqs)
        dump(work / ("legacy-%s.client.json" % name), cj)
        exp["n_requests_" + name] = len(reqs)
        exp["n_passes_" + name] = len(ps)
        # control run: same Refs, wall = nsys wall - injected difference +- eps (alternating)
        for j, r in enumerate(cj["requests"]):
            c = dict(r)
            c["idx"] = len(ctl_reqs)
            c["wall_ms"] = round(r["wall_ms"] - INJECT_MS[name] + (EPS_MS if j % 2 == 0 else -EPS_MS), 6)
            ctl_reqs.append(c)
            if r["phase"] == "timed" and j < 6:
                dec_reqs.append({"idx": len(dec_reqs), "kind": r["kind"], "case": r["case"], "phase": "unique" if j % 2 else "cycle",
                                 "pair": None, "ref": "syn-dec-%s-%d" % (name, j), "wall_ms": r["wall_ms"] - INJECT_MS[name] - 0.5,
                                 "t_send_mono_ns": 0, "t_recv_mono_ns": 1, "input_tokens": r["input_tokens"]})
        if name == "short":
            dump(work / "tail-short.client.json", client_json(nc, t, name, reqs[3:], first_idx=3, warm=0, tag="legacy"))
            link(SRC / "short.sqlite", work / "tail-short.sqlite")
            dump(work / "shift-short.client.json", client_json(nc, t, name, reqs, shift_ns=3_000_000))
            link(SRC / "short.sqlite", work / "shift-short.sqlite")
            copy_db(SRC / "short.sqlite", work / "osrt-short.sqlite")
            dump(work / "osrt-short.client.json", cj)
            exp["osrt"] = make_osrt(nc, t, ps, work / "osrt-short.sqlite")
            copy_db(SRC / "short.sqlite", work / "stray-short.sqlite")
            make_stray(t, work / "stray-short.sqlite")
            dump(work / "stray-short.client.json", cj)
    dump(work / "syn-ctl.client.json", {"meta": {"synthetic": True}, "requests": ctl_reqs, "errors": []})
    dump(work / "syn-dec.client.json", {"meta": {"synthetic": True}, "requests": dec_reqs, "errors": []})
    dump(work / "expected.json", exp)
    print("synthetic data in", work, "-", sorted(p.name for p in work.iterdir() if p.suffix in (".sqlite", ".json"))[:40])


if __name__ == "__main__":
    main()
