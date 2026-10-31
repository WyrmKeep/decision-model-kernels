#!/usr/bin/env python3
"""Upstream lead #30779 (child slot extra decode pass): does any request type get more passes than the census predicts?

Input : one or more server logs (`<run>.server.log` with DMK_TIMING lines and the default slot lines), optionally the client JSON of
        the same run (--client, same order) that names the case of every request and gives the client wall time.
Output: tables (markdown) and the two decision lines at the end. python3 stdlib + census_lib (read-only import).

Which counter answers which question (see NOTES.txt):
  * ubatch lines  (`DMK_TIMING ubatch`, one per model pass = one llama_decode micro-batch)   -> passes of a request, their sizes
  * decode lines  (`DMK_TIMING decode`, one per llama_decode call = one update_slots() iteration that has tokens) -> "batches"
  An "extra decode pass" in the text of PR #30779 is an extra batch = an extra llama_decode call. It may or may not add a ubatch:
  the census predicts the same number of ubatches for every slot placement, so a child slot below the parent costs a decode CALL
  (host round trip, extra sync) but no extra model pass; the sizes of the passes differ.
Usage : analyze_childpass.py LOG [LOG ...] [--client JSON [JSON ...]] [--out DIR]
        analyze_childpass.py --self-test
"""
import argparse
import bisect
import json
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict

import leadlib as LL

L = LL.L
KINDS = ["short1q", "short2q", "medium3q", "long3q"]


# ------------------------------------------------------------------------------------------ records
def match_client(requests, client):
    """Client record of every server request: by time (client send <= server request start, server end <= client receive: the client and the server stamp CLOCK_MONOTONIC,
    an INFERENCE checked by clock_check), so that requests of other clients and failed requests do not shift the match; by order when the client has no stamps or fewer than 90 %
    of the server requests match by time. Returns (list of client records or None, how)."""
    if not client:
        return [None] * len(requests), "none"
    if all("t_send_mono_ns" in c and "t_recv_mono_ns" in c for c in client):
        order = sorted(range(len(client)), key=lambda i: client[i]["t_send_mono_ns"])
        ts = [client[i]["t_send_mono_ns"] / 1e9 for i in order]
        used, out = set(), []
        for r in requests:
            j = bisect.bisect_right(ts, r["t_start"] + 1e-3) - 1
            c = client[order[j]] if j >= 0 else None
            if c is not None and id(c) not in used and c["t_recv_mono_ns"] / 1e9 >= r["t"] - 1e-3:
                used.add(id(c))
                out.append(c)
            else:
                out.append(None)
        if sum(c is not None for c in out) >= 0.9 * len(requests):
            return out, "time"
    return [client[k] if k < len(client) else None for k in range(len(requests))], "order"


def build_records(parsed, client=None, label=""):
    recs = []
    matched, how = match_client(parsed["requests"], client)
    n_un = sum(1 for c in matched if c is None) if client else 0
    if client and (how == "order" and len(client) != len(parsed["requests"]) or n_un):
        print(f"WARN {label}: {len(parsed['requests'])} request lines in the log, {len(client)} client requests, matched by {how}, {n_un} server requests without a client record "
              "(their type comes from the log)", file=sys.stderr)
    for r in parsed["requests"]:
        k = r["k"]
        c = matched[k]
        slots = r.get("slots")
        slots_bad = bool(slots) and len(slots) != int(r["n_questions"])      # slot lines not aligned with this request (a missing or extra launch line): placement unknown
        rec = {"run": label, "k": k, "n_q": int(r["n_questions"]), "kind": (c or {}).get("kind") or LL.request_kind(r), "case": (c or {}).get("case"),
               "phase": (c or {}).get("phase"), "slots": None if slots_bad else slots, "slots_bad": slots_bad, "select": r.get("select"), "ub": LL.obs_ubatches(r),
               "n_dec": len(r["decodes"]), "dec": [(int(d["n_tokens"]), int(d["n_ubatches"])) for d in r["decodes"]],
               "total_us": r["total_us"], "wait_us": r["wait_us"], "client_wall_ms": (c or {}).get("wall_ms"),
               "t_start": r["t_start"], "t_end": r["t"], "client": c, "kv_unified": parsed.get("kv_unified")}
        recs.append(rec)
    return recs


def clock_check(recs):
    """INFERENCE check of the context file: client send <= server request start and server end <= client receive (same clock)."""
    n = ok = 0
    lead, tail = [], []
    for r in recs:
        c = r["client"]
        if not c or "t_send_mono_ns" not in c:
            continue
        n += 1
        a = r["t_start"] - c["t_send_mono_ns"] / 1e9
        b = c["t_recv_mono_ns"] / 1e9 - r["t_end"]
        lead.append(a * 1e3)
        tail.append(b * 1e3)
        ok += a >= -1e-4 and b >= -1e-4
    return n, ok, lead, tail


def shared_of(rec):
    """Shared prefix tokens seen by the server = tokens of the ubatches without an output row (multi-question requests only)."""
    return sum(n for n, s, o in rec["ub"] if o == 0) if rec["n_q"] > 1 else 0


def candidates(rec):
    """Suffix tuples to try: the plan of the case when the client JSON names it, else every distinct tuple of the plan."""
    nq = rec["n_q"]
    if nq == 1:
        return [(sum(n for n, s, o in rec["ub"]),)]
    plan = LL.case_plan()
    if rec["case"] in plan and plan[rec["case"]]["n_questions"] == nq:
        return [tuple(plan[rec["case"]]["suffix"])]
    return LL.suffix_candidates(nq)


def match(rec, placements=None):
    """Compare the observed ubatches with the census default model (unified split, sequential = false: --kv-unified) and the sequential-split model (sequential = true: --no-kv-unified,
    and what the first census assumed) for the observed placement."""
    shared = shared_of(rec)
    ubs = rec["ub"]
    if not ubs or (rec["n_q"] > 1 and shared == 0):
        return {"unified": None, "seq": None}      # no ubatch lines, or no prefix pass for a multi-question request (the whole prompt came from the cache): no model applies
    slots_list = [rec["slots"]] if rec["slots"] else ([(0,)] if rec["n_q"] == 1 else L.arrangements(rec["n_q"]))
    out = {"unified": None, "seq": None}
    for mode in LL.MODES:
        found = None
        for slots in slots_list:
            for suf in candidates(rec):
                if rec["n_q"] > 1 and len(slots) != len(suf):
                    continue
                passes = LL.simulate(shared, suf, slots, mode)
                ok = LL.ub_key(passes) == ubs
                score = (ok, -abs(len(passes) - len(ubs)))
                if found is None or score > found[0]:
                    found = (score, slots, suf, passes)
        out[mode] = {"ok": found[0][0], "slots": found[1], "suffix": found[2], "passes": found[3]} if found else None
    return out


def pseudo_passes(rec):
    """Pass dicts rebuilt from the observed ubatches alone (requests that no model reproduces): prefix ubatches without an output
    at their running positions, suffix ubatches at the shared prefix [INFERENCE: later passes of a sequence start further right; only the
    attention part of the roofline floor depends on it]."""
    shared = shared_of(rec)
    out, pos, it = [], 0, 0
    for n, s, o in rec["ub"]:
        if rec["n_q"] > 1 and o == 0:
            out.append({"kind": "prefix", "seqs": [(pos, n)], "slots": [0], "questions": [0], "n_tokens": n, "n_out": 0, "iteration": it})
            pos += n
        else:
            kind = "single" if rec["n_q"] == 1 else "suffix"
            out.append({"kind": kind, "seqs": [(shared, n // s)] * s, "slots": list(range(s)), "questions": list(range(s)), "n_tokens": n, "n_out": o, "iteration": it})
        it += 1
    return out


def fixed_of(rec, suffix):
    return LL.simulate(shared_of(rec), suffix, rec["slots"] or tuple(range(rec["n_q"])), "fixed")


def child_below(rec):
    s = rec["slots"]
    return bool(s) and len(s) > 1 and any(x < s[0] for x in s[1:])


def reproduced_passes(r):
    """Pass list of the model that reproduces the observed ubatches: the model expected for the run's cache mode first (unified for --kv-unified, sequential for --no-kv-unified), then the other."""
    order = ("seq", "unified") if r.get("expected_mode") == "seq" else ("unified", "seq")
    for mode in order:
        if r["m"][mode] and r["m"][mode]["ok"]:
            return r["m"][mode]["passes"]
    return None


EXCLUDED = []       # (run, request index, reason): requests the pass-structure models cannot describe (listed in the report)


def exclusion(rec):
    if not rec["ub"]:
        return "no complete ubatch line in the request"
    if rec["n_q"] > 1 and shared_of(rec) == 0:
        return "multi-question request without a prefix pass (the whole shared prefix came from the cache)"
    return None


def analyse(recs):
    """Adds the model comparison to every record (in place; requests no model can describe are removed and listed in EXCLUDED) and returns the records."""
    EXCLUDED.clear()
    kept = []
    for r in recs:
        why = exclusion(r)
        if why:
            EXCLUDED.append((r["run"], r["k"], why))
            continue
        kept.append(r)
    recs[:] = kept
    for r in recs:
        m = match(r)
        r["m"] = m
        cen = m["unified"]                  # the census default (sequential = false) = the reference prediction of the decisions
        r["unified_ok"] = bool(cen and cen["ok"])
        r["seq_ok"] = bool(m["seq"] and m["seq"]["ok"])
        r["expected_mode"] = "seq" if r.get("kv_unified") is False else "unified"      # source: split_equal(n_ubatch, !unified); --no-kv-unified -> n_stream > 1 -> sequential = true
        r["expected_ok"] = r["unified_ok"] if r["expected_mode"] == "unified" else r["seq_ok"]
        suffix = (cen or m["seq"])["suffix"]
        r["suffix"] = suffix
        r["pred_ub"] = len(cen["passes"]) if cen else None
        r["pred_dec"] = LL.n_decodes(cen["passes"]) if cen else None
        fx = fixed_of(r, suffix)
        r["fixed_ub"] = len(fx)
        r["fixed_dec"] = LL.n_decodes(fx)
        r["fixed_keys"] = LL.ub_key(fx)
        r["below"] = child_below(r)
        obs_struct = reproduced_passes(r)
        r["structure"] = LL.structure(obs_struct) if obs_struct else "?"
        r["describe"] = LL.describe(obs_struct) if obs_struct else "~" + LL.describe(pseudo_passes(r))
        if obs_struct is None:
            obs_struct = pseudo_passes(r)
        r["more_ub"] = r["pred_ub"] is not None and len(r["ub"]) > r["pred_ub"]
        r["less_ub"] = r["pred_ub"] is not None and len(r["ub"]) < r["pred_ub"]
        r["dec_vs_fixed"] = r["n_dec"] - r["fixed_dec"]
        r["sizes_vs_fixed"] = r["ub"] != r["fixed_keys"]
        r["floor_obs"] = LL.pass_floors(obs_struct)
        r["floor_fixed"] = LL.pass_floors(fx)
        # decode structure: (tokens, ubatches) per llama_decode against the iterations of the matching model
        ref = reproduced_passes(r)
        if ref:
            by_it = defaultdict(lambda: [0, 0])
            for p in ref:
                by_it[p["iteration"]][0] += p["n_tokens"]
                by_it[p["iteration"]][1] += 1
            r["dec_match"] = [tuple(v) for _, v in sorted(by_it.items())] == r["dec"]
        else:
            r["dec_match"] = None
    return recs


# ------------------------------------------------------------------------------------------ printing
def table(title, headers, rows, source):
    print(f"\n**{title}**\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")
    print(f"\nSource: {source}")


def f1(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else None


def by_kind(recs):
    d = defaultdict(list)
    for r in recs:
        d[r["kind"]].append(r)
    return {k: d[k] for k in KINDS if k in d}


def report(recs, parsed_list, out=None):
    res = {}
    bk = by_kind(recs)
    print(f"# Lead #30779: child slot decode passes ({len(recs)} requests, {len(parsed_list)} log(s))")
    if EXCLUDED:
        print(f"\nExcluded from the tables: {len(EXCLUDED)} request(s) the models cannot describe: " + "; ".join(f"{a}#{k}: {w}" for a, k, w in EXCLUDED[:6]) + (" ..." if len(EXCLUDED) > 6 else "") + ".")
    n_clock, ok_clock, lead, tail = clock_check(recs)
    if n_clock:
        print(f"\nClock check (context file INFERENCE: client and server use the same CLOCK_MONOTONIC): client send <= server request start and "
              f"server request end <= client receive in {ok_clock} of {n_clock} requests; client send -> server start median {f1(statistics.median(lead), 2)} ms, "
              f"server end -> client receive median {f1(statistics.median(tail), 2)} ms.")
    sel = Counter(r["select"] for r in recs)
    print(f"\nSlot selection of the parent task: {dict(sel)}. Requests without slot lines: {sum(1 for r in recs if not r['slots'])}.")

    # (i) ubatches per request type vs census
    rows = []
    for k, rs in bk.items():
        ob = [len(r["ub"]) for r in rs]
        pr = [r["pred_ub"] for r in rs if r["pred_ub"] is not None]
        rows.append([k, len(rs), sum(r["unified_ok"] for r in rs), sum(r["seq_ok"] for r in rs), sum(r["unified_ok"] or r["seq_ok"] for r in rs),
                     sum(1 for r in rs if not (r["unified_ok"] or r["seq_ok"])), f1(mean(ob), 2), f1(mean(pr), 2), sum(r["more_ub"] for r in rs),
                     sum(r["less_ub"] for r in rs)])
    table("Table 1: observed ubatches (model passes) against the census prediction for the observed slot placement",
          ["request type", "requests", "census (unified split) exact (ubatch tokens, seqs, outputs)", "sequential-split model exact", "either model exact", "neither",
           "ubatches observed, mean", "ubatches census, mean", "requests with MORE ubatches than census", "requests with FEWER"], rows,
          "DMK_TIMING ubatch lines (n_tokens, n_seqs, n_outputs) per request; census = census_lib.simulate_request with its default split (split_equal sequential = false, the split of "
          "--kv-unified: llama-memory-hybrid.cpp:83-90), sequential-split = the same flow with sequential = true (leadlib.simulate mode seq; the split of --no-kv-unified); "
          "placement from the slot lines (parent is_child = 0, children in task order); shared prefix = tokens of the "
          "ubatches without an output; suffix lengths from the case plan (census table cases-plan.tsv, ref variant) of the client JSON case, else every distinct plan tuple.")
    res["more_ub"] = sum(r["more_ub"] for r in recs)
    res["neither_list"] = [(r["run"], r["k"], r["kind"], r["slots"], r["ub"]) for r in recs if not (r["unified_ok"] or r["seq_ok"])]
    # expected model per run (cache mode from the load_model line of the log)
    rows = []
    for run in sorted({r["run"] for r in recs}):
        rs = [r for r in recs if r["run"] == run]
        kvu = rs[0]["kv_unified"]
        rows.append([run, {True: "true", False: "false", None: "not in the log"}[kvu], rs[0]["expected_mode"] + (" (assumed)" if kvu is None else ""), len(rs),
                     sum(r["expected_ok"] for r in rs), sum(r["unified_ok"] for r in rs), sum(r["seq_ok"] for r in rs)])
    table("Table 1b: split model expected for the cache mode of each log, and how many requests it reproduces",
          ["run", "kv_unified in the log", "expected split model", "requests", "reproduced by the expected model", "reproduced by the unified split", "reproduced by the sequential split"], rows,
          "the `load_model: initializing ... kv_unified = '...'` line of each server log; expected = unified split for kv_unified = true, sequential split for false "
          "(src/llama-memory-hybrid.cpp:83-90: split_equal(n_ubatch, !unified); --no-kv-unified gives n_stream > 1, unified = false, sequential = true) [INFERENCE from source]; a log without the line is taken as unified.")
    res["expected_mismatch"] = [(r["run"], r["k"], r["kind"], r["slots"]) for r in recs if not r["expected_ok"]]
    # models differ only where?
    diff = [r for r in recs if r["unified_ok"] != r["seq_ok"]]
    if diff:
        print(f"\nThe two split models disagree in {len(diff)} requests: " + ", ".join(
            f"{k}: {sum(1 for r in diff if r['kind'] == k and r['unified_ok'])} unified-only / {sum(1 for r in diff if r['kind'] == k and r['seq_ok'])} sequential-only"
            for k in bk if any(r["kind"] == k for r in diff)) + ". Placements: " + ", ".join(sorted({str(r['slots']) for r in diff})))
    if res["neither_list"]:
        print("\nRequests that neither model reproduces (run, request, type, slots, observed ubatches (tokens, seqs, outputs)); their structure in the tables is rebuilt from the "
              "observed ubatches and marked ~:")
        for x in res["neither_list"][:12]:
            print("  ", x)

    # (ii) decode calls
    rows = []
    for k, rs in bk.items():
        nd = [r["n_dec"] for r in rs]
        rows.append([k, len(rs), f"{min(nd)}-{max(nd)}", f1(mean(nd), 2), f1(mean([r["pred_dec"] for r in rs]), 2), f1(mean([r["fixed_dec"] for r in rs]), 2),
                     sum(1 for r in rs if r["pred_dec"] is not None and r["n_dec"] == r["pred_dec"]), sum(1 for r in rs if r["n_dec"] == r["fixed_dec"]),
                     sum(1 for r in rs if r["dec_vs_fixed"] > 0), sum(1 for r in rs if r["dec_match"])])
    table("Table 2: llama_decode calls (DMK_TIMING decode lines) per request: observed, census model, fixed server (parents visited before children)",
          ["request type", "requests", "decode calls observed, min-max", "observed, mean", "census, mean", "fixed server, mean", "requests observed = census",
           "requests observed = fixed", "requests with MORE decodes than fixed", "requests whose (tokens, ubatches) per decode match the model"], rows,
          "decode lines per request; census prediction = number of update_slots iterations of simulate_request; fixed = simulate_request with the parent slot forced to the lowest id "
          "(all children in the parent's suffix batch).")

    # (iii) child below parent: counts and cost
    rows = []
    for k, rs in bk.items():
        for flag in (True, False):
            sub = [r for r in rs if r["below"] == flag]
            if not sub or (k == "short1q" and flag):
                continue
            st = Counter(r["describe"] for r in sub)
            rows.append([k, "yes" if flag else "no", len(sub), f1(mean([r["total_us"] / 1e3 for r in sub]), 2), f1(sd([r["total_us"] / 1e3 for r in sub]), 2),
                         f1(mean([r["wait_us"] / 1e3 for r in sub]), 2), f1(mean([r["client_wall_ms"] for r in sub]), 2),
                         f1(mean([r["floor_obs"][0] for r in sub]), 1), f1(mean([r["floor_obs"][1] for r in sub]), 1),
                         f1(mean([r["n_dec"] for r in sub]), 2), "; ".join(f"{s} x{n}" for s, n in st.most_common(4))])
    table("Table 3: requests with a child slot id below the parent's (#30779 situation) and without",
          ["request type", "child below parent", "requests", "request total_us, mean (ms)", "sd (ms)", "request wait_us, mean (ms)", "client wall, mean (ms)",
           "floor max(weight, int8) of the observed passes, mean (ms)", "serial roofline floor, mean (ms)", "decode calls, mean", "pass structures (census labels, count)"], rows,
          "request lines (total_us, wait_us), client JSON wall_ms when given, census pass_costs floors of the matching structure; placement and structure differ between the groups, so the "
          "difference of the means mixes the extra decode call with the different pass sizes: compare with the difference of the floors (same table).")
    cost_rows = []
    for k, rs in bk.items():
        a = [r for r in rs if r["below"]]
        b = [r for r in rs if not r["below"]]
        if not a or not b:
            continue
        d_wall = mean([r["total_us"] / 1e3 for r in a]) - mean([r["total_us"] / 1e3 for r in b])
        d_floor = mean([r["floor_obs"][1] for r in a]) - mean([r["floor_obs"][1] for r in b])
        se = ((sd([r["total_us"] / 1e3 for r in a]) or 0) ** 2 / len(a) + (sd([r["total_us"] / 1e3 for r in b]) or 0) ** 2 / len(b)) ** 0.5
        cost_rows.append([k, len(a), len(b), f1(d_wall, 2), f1(se, 2), f1(d_floor, 2), f1(d_wall - d_floor, 2)])
    if cost_rows:
        table("Table 4: cost of the #30779 situation per request type (difference with - without)",
              ["request type", "requests with", "requests without", "request total_us difference (ms)", "standard error of the difference (ms)",
               "serial roofline floor difference (ms)", "total_us difference minus floor difference (ms)"], cost_rows,
              "Table 3 means; a value of the last column inside 2 standard errors of zero means the structure difference explains the wall difference.")

    # (iv) fixed ordering
    rows = []
    for k, rs in bk.items():
        for slots in sorted({r["slots"] for r in rs if r["slots"] and r["below"]}, key=str):
            sub = [r for r in rs if r["slots"] == slots]
            r0 = sub[0]
            fx = fixed_of(r0, r0["suffix"])
            rows.append([k, str(slots), len(sub), r0["describe"], len(r0["ub"]), r0["n_dec"], LL.describe(fx), len(fx), LL.n_decodes(fx),
                         f1(mean([r["floor_obs"][0] for r in sub]), 1), f1(mean([r["floor_fixed"][0] for r in sub]), 1),
                         f1(mean([r["floor_obs"][1] for r in sub]), 1), f1(mean([r["floor_fixed"][1] for r in sub]), 1)])
    if rows:
        table("Table 5: fixed ordering (parent slot forced to the lowest id) against the observed placement, placements with a child below the parent",
              ["request type", "slots (Q0, Q1, ...)", "requests", "observed passes", "ubatches", "decodes", "fixed passes", "ubatches", "decodes",
               "floor max(weight, int8) observed (ms)", "floor fixed (ms)", "serial roofline observed (ms)", "serial roofline fixed (ms)"], rows,
              "census_lib.simulate_request / pass_costs; P = prefix pass, S<n>x<t> = n sequences of t tokens, F = whole prompt in one pass.")

    # decisions
    below = [r for r in recs if r["below"]]
    n_more = sum(r["more_ub"] for r in recs)
    types_more = sorted({r["kind"] for r in recs if r["more_ub"]})
    extra_dec = [r for r in below if r["dec_vs_fixed"] > 0]
    size_change = [r for r in below if r["sizes_vs_fixed"]]
    unexpl = [r for r in recs if not (r["unified_ok"] or r["seq_ok"])]
    print("\n## Decisions\n")
    print("Threshold: the counts are deterministic structure counts (no timing), so one request is evidence; the census matched 20 of 20 traced requests. "
          "'More passes' = observed ubatches > census prediction (default split = unified) for the observed slot placement.")
    ans1 = "yes" if n_more else "no"
    print(f"does any request type get more passes (ubatches) than the census predicts: {ans1}  "
          f"({n_more} of {len(recs)} requests have more ubatches than predicted; types: {types_more or '-'}; census (unified split) exact in {sum(r['unified_ok'] for r in recs)}, "
          f"sequential-split model exact in {sum(r['seq_ok'] for r in recs)}, neither in {len(unexpl)} requests; the model expected for each log's cache mode: "
          f"{len(recs) - len(res['expected_mismatch'])} of {len(recs)} requests)")
    ans2 = "yes" if (extra_dec or size_change) else "no"
    print(f"does #30779 add a decode call / change pass sizes: {ans2}  "
          f"({len(below)} requests have a child slot below the parent; {len(extra_dec)} of them issue more llama_decode calls than the fixed order "
          f"(by type: {dict(Counter(r['kind'] for r in extra_dec)) or '-'}), {len(size_change)} have different ubatch sizes than the fixed order "
          f"(by type: {dict(Counter(r['kind'] for r in size_change)) or '-'}); ubatch COUNT differs from the fixed order in "
          f"{sum(1 for r in below if len(r['ub']) != r['fixed_ub'])} of them)")
    res.update({"n_requests": len(recs), "n_below": len(below), "extra_dec": len(extra_dec), "size_change": len(size_change), "ans1": ans1, "ans2": ans2,
                "n_more": n_more, "unified_ok": sum(r["unified_ok"] for r in recs), "seq_ok": sum(r["seq_ok"] for r in recs), "neither": len(unexpl)})
    if out:
        os.makedirs(out, exist_ok=True)
        cols = ["run", "k", "kind", "case", "phase", "n_q", "slots", "select", "structure", "describe", "below", "unified_ok", "seq_ok", "expected_mode", "expected_ok", "n_dec", "pred_dec", "fixed_dec",
                "pred_ub", "fixed_ub", "total_us", "wait_us", "client_wall_ms", "more_ub", "dec_vs_fixed", "sizes_vs_fixed"]
        with open(os.path.join(out, "childpass-requests.tsv"), "w") as f:
            f.write("\t".join(cols[:6] + ["n_ubatches_obs"] + cols[6:]) + "\n")
            for r in recs:
                vals = [r[c] if c in r else "" for c in cols[:6]] + [len(r["ub"])] + [r[c] for c in cols[6:]]
                f.write("\t".join("" if v is None else str(v) for v in vals) + "\n")
    return res


def run(logs, clients=None, out=None):
    recs, parsed_list = [], []
    for i, lg in enumerate(logs):
        parsed = LL.parse_server_log(lg)
        parsed_list.append(parsed)
        client = LL.read_client_json(clients[i]) if clients and i < len(clients) else None
        if parsed.get("incomplete"):
            print(f"WARN {lg}: {len(parsed['incomplete'])} incomplete DMK_TIMING lines skipped (first: line {parsed['incomplete'][0][0]}, {parsed['incomplete'][0][1]}, missing {parsed['incomplete'][0][2]}): "
                  "the request they belong to can show a missing ubatch/decode/sync", file=sys.stderr)
        recs += build_records(parsed, client, os.path.basename(lg).replace(".server.log", ""))
    analyse(recs)
    return report(recs, parsed_list, out)


# ------------------------------------------------------------------------------------------ self test
def self_test():
    """Synthetic data in a temporary folder (TMPDIR) that is removed afterwards."""
    with tempfile.TemporaryDirectory(prefix="childpass-selftest-") as tmp:
        return _self_test_in(tmp)


def _self_test_in(tmp):
    import make_synthetic as MS
    MS.generate(tmp)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        ok &= bool(cond)

    for scen in ("childpass-census", "childpass-seq", "childpass-extra"):
        base = os.path.join(tmp, "syn-" + scen)
        truth = json.load(open(base + ".truth.json"))["requests"]
        print(f"\n===== self test: {scen} ({len(truth)} planted requests)")
        parsed = LL.parse_server_log(base + ".server.log")
        client = LL.read_client_json(base + ".json")
        recs = analyse(build_records(parsed, client, scen))
        check("request count", len(recs) == len(truth), f"{len(recs)} vs {len(truth)}")
        check("slot placement recovered", all(tuple(t["slots"]) == r["slots"] for t, r in zip(truth, recs)))
        check("child-below-parent flag", all(t["child_below_parent"] == r["below"] for t, r in zip(truth, recs)),
              f"{sum(r['below'] for r in recs)} planted-below requests")
        check("observed ubatch count = planted", all(len(r["ub"]) == t["planted_ubatches"] for t, r in zip(truth, recs)))
        check("observed decode count = planted", all(r["n_dec"] == t["planted_decodes"] for t, r in zip(truth, recs)))
        check("fixed-order decode count = planted", all(r["fixed_dec"] == t["fixed_decodes"] for t, r in zip(truth, recs)))
        check("census prediction (ubatches, decodes) = planted census", all(r["pred_ub"] == t["census_ubatches"] and r["pred_dec"] == t["census_decodes"]
                                                                          for t, r in zip(truth, recs)))
        check("census (unified split) exact = planted_equals_census", all(r["unified_ok"] == t["planted_equals_census"] for t, r in zip(truth, recs)))
        check("kv_unified read from the log (false only in the seq scenario)", all(r["kv_unified"] == (scen != "childpass-seq") for r in recs))
        check("expected model exact exactly where planted (seq scenario: every request; census and extra: the census-equal requests)",
              all(r["expected_ok"] for r in recs) if scen == "childpass-seq" else all(r["expected_ok"] == t["planted_equals_census"] for t, r in zip(truth, recs)),
              f"{sum(r['expected_ok'] for r in recs)}/{len(recs)}")
        out = os.path.join(tmp, "out-" + scen)
        sys.stdout.flush()
        res = report(recs, [parsed], out)
        exp_more = sum(1 for t in truth if t["planted_ubatches"] > t["census_ubatches"])
        check("n_more (requests with more ubatches than census)", res["n_more"] == exp_more, f"{res['n_more']} vs planted {exp_more}")
        exp1 = "yes" if exp_more else "no"
        check("decision 1 = " + exp1, res["ans1"] == exp1)
        exp_dec = sum(1 for t in truth if t["child_below_parent"] and t["kind"] != "short1q" and t["planted_decodes"] > t["fixed_decodes"])
        check("requests with more decodes than fixed", res["extra_dec"] == exp_dec, f"{res['extra_dec']} vs planted {exp_dec}")
        check("decision 2 = yes", res["ans2"] == "yes")
        if scen == "childpass-seq":
            seq_only = [r for r in recs if r["seq_ok"] and not r["unified_ok"]]
            check("sequential-split-only matches are exactly the 2q placement (0, 2)", seq_only and all(r["kind"] == "short2q" and r["slots"] == (0, 2) for r in seq_only),
                  f"{len(seq_only)} requests")
            check("every request is explained by one of the models", res["neither"] == 0)
            check("every request is sequential-split exact (expected model of --no-kv-unified)", res["seq_ok"] == len(recs), f"{res['seq_ok']}/{len(recs)}")
        if scen == "childpass-census":
            check("every request is census (unified split) exact", res["unified_ok"] == len(recs), f"{res['unified_ok']}/{len(recs)}")
    # the clean census log must give the answer 'no' to question 1; a log that matches no model must be flagged
    print("\nself test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="*", help="server log(s) with DMK_TIMING lines")
    ap.add_argument("--client", nargs="*", help="client JSON(s), same order as the logs")
    ap.add_argument("--out", help="directory for childpass-requests.tsv")
    ap.add_argument("--self-test", action="store_true", help="run on synthetic logs with planted answers (make_synthetic.py)")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if not a.logs:
        ap.error("give a server log or --self-test")
    run(a.logs, a.client, a.out)


if __name__ == "__main__":
    main()
