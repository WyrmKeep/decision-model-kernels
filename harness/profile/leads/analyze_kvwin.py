#!/usr/bin/env python3
"""Upstream lead #28495 (flash attention K/V window with --kv-unified): is the flash-attention time of a request wider than needed because
cells of other (idle or released) sequences sit in the shared pool?  python3 stdlib + census_lib (read-only import).

Three kinds of evidence, each optional except the first (it is the existing data and runs by default):
  1. old traces   : trace-passes.tsv of the census data set (+ crosscheck-passes.tsv for the window the model needs): FA kernel ms per pass against the
                    window seen (n_kv_padded of the mask copy) and the window needed; model-free fit  fa_ms = a + b * N * n_kv  on the passes that
                    read their whole window (N < 1024), prediction at the needed window = possible gain if the window were exact.
  2. kvwin nsys   : --nsys PASSES.tsv --client KVWIN.json [--run NAME]   (passes.tsv of package 2, client JSON of kvwin_client.py)
                    T1 (fresh server) / T2 (behind the stale cells of other long requests) / T3 / T4 per pass: window seen, needed, FA ms, GPU busy, wall.
                    --nsys-nokvu PASSES.tsv,CLIENT.json[,RUN]: the same for the --no-kv-unified run (tables and T/T1 ratios; no FA fit).
  3. timing runs  : --timing LABEL=LOG[,CLIENT.json] (repeatable; LABEL kvu or nokvu): pass time without a profiler from the DMK_TIMING lines
                    (sync since_first_submit_us / wait_us per request), slot placement, pass structure, T1 vs T2 vs T3, kv-unified vs no-kv-unified.
Definitions (see NOTES.txt): window seen = n_kv_padded (mask H2D bytes / (2 x n_tokens), padded to 256); window needed = the sequence's own cells
padded to 256 (census_lib.attn_cost); effective window = what the kernel reads (implied by the fit).
Decision lines are printed at the end; the threshold is 2 x the run-to-run spread (defined in the text of each rule).
Usage : analyze_kvwin.py [--out DIR] [--nsys PASSES.tsv --client JSON [--run NAME]] [--nsys-nokvu PASSES.tsv,CLIENT.json[,RUN]]
                         [--timing LABEL=LOG[,CLIENT] ...] [--no-old]
        analyze_kvwin.py --self-test
"""
import argparse
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict

import leadlib as LL
import analyze_childpass as CP
from analyze_childpass import table, f1, mean, sd

L = LL.L
FA_LAYERS = 8
OLD_PASSES = "trace-passes.tsv"          # tables of the census data set, found with census_lib.data_file()
OLD_CROSS = "crosscheck-passes.tsv"
MIN_SPREAD = 0.02          # floor of the run-to-run spread used by the decision rules (2 %)


# ------------------------------------------------------------------------------------------ small statistics
def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    st = sum((y - my) ** 2 for y in ys)
    ss = sum(r * r for r in res)
    return {"a": a, "b": b, "r2": 1 - ss / st if st else float("nan"), "n": n, "rmse": (ss / n) ** 0.5, "res": res}


def rel_sd(xs):
    """Run-to-run spread = sample standard deviation / mean of repeated measurements (needs >= 2 values)."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = statistics.mean(xs)
    return statistics.stdev(xs) / m if m else None


def pct(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def fnum(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


# ------------------------------------------------------------------------------------------ pass rows
def old_rows():
    """One dict per pass of the three old traces: window seen from trace-passes.tsv, window needed from crosscheck-passes.tsv."""
    cc = {(r["trace"], r["request_idx"], r["pass_idx_in_request"]): r for r in LL.read_tsv(L.data_file(OLD_CROSS))}
    rows = []
    start = defaultdict(int)
    for r in LL.read_tsv(L.data_file(OLD_PASSES)):
        if r["row_kind"] != "pass":
            continue
        n = int(r["n_tokens"])
        key = (r["trace"], r["request_idx"])
        c = cc.get((r["trace"], r["request_idx"], r["pass_idx_in_request"]))
        is_prefix = r["pass_kind"].startswith("prefix")
        st = start[key] if is_prefix else None
        if is_prefix:
            start[key] += n
        needed = int(c["n_kv_window_model"]) if c else None
        eff = LL.eff_front(st, n) if (n >= LL.SCAN_N and st is not None) else needed
        rows.append({"src": "old", "trace": r["trace"], "req": int(r["request_idx"]), "j": int(r["pass_idx_in_request"]), "phase": r["trace"], "kind": r["pass_kind"],
                     "N": n, "seen": int(r["n_kv_padded"]), "needed": needed, "eff": eff, "pool": None, "fa": float(r["t_cat_flash_ms"]),
                     "busy": float(r["gpu_busy_ms"]), "wall": float(r["wall_ms"]), "case": r["case_id_inferred"], "n_fa": int(r["n_flash_attn"] or FA_LAYERS)})
    return rows


def census_match(case, obs):
    """Pass list of the census model (unified split first, then the sequential split) that reproduces the observed (n_tokens, n_outputs) per pass, with the number of matching placements."""
    plan = LL.case_plan().get(case)
    if not plan:
        return None, 0, None
    nq = plan["n_questions"]
    shared = sum(n for n, o in obs if o == 0) if nq > 1 else 0
    suffix = plan["suffix"] if nq > 1 else [sum(n for n, o in obs)]
    for mode in LL.MODES:
        hits = []
        for slots in (L.arrangements(nq) if nq > 1 else [(0,)]):
            ps = LL.simulate(shared, suffix, slots, mode)
            if [(p["n_tokens"], p["n_out"]) for p in ps] == obs:
                hits.append((slots, ps))
        if hits:
            return hits[0][1], len(hits), (mode, hits[0][0])
    return None, 0, None


def nsys_rows(path, client, run=None):
    rows = LL.read_tsv(path)
    rows = [r for r in rows if r.get("row_kind", "pass") == "pass"]
    runs = sorted({r.get("run") or r.get("trace") for r in rows})
    if run is None and len(runs) > 1:
        sys.exit(f"{path} has several runs {runs}: give --run")
    if run is None:
        run = runs[0]
    by = defaultdict(list)
    for r in rows:
        if (r.get("run") or r.get("trace")) != run:
            continue
        if "client_idx" in r:
            if r["client_idx"] in ("", None):          # a pass outside every client request window (server start, model load warm-up): not part of the sequence
                continue
            ci = int(r["client_idx"])
        else:
            ci = int(r["request_idx"])
        by[ci].append(r)
    cmap = {c.get("idx", i): c for i, c in enumerate(client)}
    out = []
    for ci in sorted(by):
        rs = sorted(by[ci], key=lambda r: int(r["pass_idx_in_request"]))
        if any(r.get("n_kv_padded") in ("", None) for r in rs):
            print(f"WARN: client request {ci}: n_kv_padded not derived for {sum(1 for r in rs if r.get('n_kv_padded') in ('', None))} of {len(rs)} passes: request left out", file=sys.stderr)
            continue
        c = cmap.get(ci, {})
        case = c.get("case") or rs[0].get("case") or rs[0].get("case_id_inferred")
        obs = [(int(r["n_tokens"]), int(r["n_outputs"] or 0)) for r in rs]
        sim, nmatch, how = census_match(case, obs)
        wins = LL.pass_windows(sim) if sim else None
        for j, r in enumerate(rs):
            w = wins[j] if wins else {"needed_seq": None, "needed_pool": None, "eff_front": None}
            out.append({"src": "nsys", "trace": run, "req": ci, "j": j, "phase": c.get("phase", "?"), "kind": r["pass_kind"], "N": int(r["n_tokens"]),
                        "seen": int(r["n_kv_padded"]), "needed": w["needed_seq"], "eff": w["eff_front"], "pool": w["needed_pool"], "fa": float(r["t_cat_flash_ms"]),
                        "busy": float(r["gpu_busy_ms"]), "wall": float(r["wall_ms"]), "case": case, "n_fa": int(r.get("n_flash_attn") or FA_LAYERS),
                        "mod_ms": float(r.get("module_load_ms") or 0), "match": nmatch, "model": how[0] if how else None, "slots": how[1] if how else None,
                        "describe": LL.describe(sim) if sim else "?"})
    return out, run


# ------------------------------------------------------------------------------------------ FA model
def fit_fa(rows):
    """fa_ms = a + b * N * seen over the passes that read their whole window (N < 1024: no mask scan)."""
    small = [r for r in rows if r["N"] < LL.SCAN_N]
    return linfit([r["N"] * r["seen"] for r in small], [r["fa"] for r in small]), small


def annotate(rows, fit):
    """Adds the fit-based columns to every pass row. Scan passes (N >= 1024) are predicted at the needed window with the same law applied to the
    cells a query tile reads when the own cells sit at the front (eff_front); the model's constant error on those passes is calibrated with the pass
    that is closest to the model (smallest observed / model ratio, c_scan), so the gain is a lower bound."""
    a, b = fit["a"], fit["b"]
    raw = {}
    for i, r in enumerate(rows):
        r["w_impl"] = (r["fa"] - a) / (b * r["N"]) if b > 0 else None            # window the kernel appears to read (cells)
        small = r["N"] < LL.SCAN_N
        r["need_eff"] = r["needed"] if small else r["eff"]
        if r["need_eff"]:
            raw[i] = a + b * r["N"] * r["need_eff"]
    ratios = [rows[i]["fa"] / v for i, v in raw.items() if rows[i]["N"] >= LL.SCAN_N and v]
    fit["c_scan"] = min(ratios) if ratios else 1.0
    for i, r in enumerate(rows):
        small = r["N"] < LL.SCAN_N
        r["pred_seen"] = a + b * r["N"] * r["seen"] if small else None
        r["pred_need"] = (raw[i] if small else raw[i] * fit["c_scan"]) if i in raw else None
        if i not in raw:
            r["gain"] = None
        elif small:
            r["gain"] = b * r["N"] * max(0, r["seen"] - r["needed"])                # model difference: no residual noise
        else:
            r["gain"] = max(0.0, r["fa"] - r["pred_need"])
    return rows


def identical_groups(rows, key=lambda r: (r["trace"], r["kind"], r["N"], r["seen"]), nmin=3):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return {k: v for k, v in g.items() if len(v) >= nmin}


def spreads(rows):
    """Run-to-run spread of FA time and of GPU busy time: relative standard deviation among passes that are identical (same trace, kind, N, window), median over groups."""
    gs = identical_groups(rows)
    fa = [rel_sd([r["fa"] for r in v]) for v in gs.values()]
    bu = [rel_sd([r["busy"] for r in v]) for v in gs.values()]
    fa = [x for x in fa if x is not None]
    bu = [x for x in bu if x is not None]
    return (statistics.median(fa) if fa else None, statistics.median(bu) if bu else None, len(gs))


# ------------------------------------------------------------------------------------------ report: regression part (old traces or any pass list)
def report_regression(rows, title, out=None, tag="old"):
    res = {}
    fit, small = fit_fa(rows)
    annotate(rows, fit)
    print(f"\n## {title}\n")
    print(f"Passes: {len(rows)} ({len(small)} with N < {LL.SCAN_N}: the kernel reads the whole window; {len(rows) - len(small)} with N >= {LL.SCAN_N}: the mask scan "
          f"`flash_attn_mask_to_KV_max` skips masked tail blocks, not interior ones).")
    print(f"\nFit on the N < {LL.SCAN_N} passes:  FA kernel time of the 8 layers (ms) = a + b x N x n_kv  with  a = {fit['a']:.3f} ms,  "
          f"b = {fit['b'] * 1e6:.3f} x 10^-6 ms per (token x cell) ({fit['b'] * 1e6 / FA_LAYERS * 1e3:.2f} ns per token x cell and layer),  "
          f"R^2 = {fit['r2']:.4f},  rmse = {fit['rmse']:.3f} ms,  n = {fit['n']}.")
    rr = sorted(fit["res"])
    print(f"Residuals (observed - fit, ms): min {rr[0]:.3f}, p10 {rr[int(0.1 * (len(rr) - 1))]:.3f}, median {statistics.median(rr):.3f}, "
          f"p90 {rr[int(0.9 * (len(rr) - 1))]:.3f}, max {rr[-1]:.3f}.")
    worst = sorted(small, key=lambda r: -abs(r["fa"] - (fit["a"] + fit["b"] * r["N"] * r["seen"])))[:5]
    print("Largest residuals: " + "; ".join(f"{r['trace']}/{r['req']}/{r['j']} N={r['N']} n_kv={r['seen']} observed {r['fa']:.3f} fit {r['pred_seen']:.3f}" for r in worst))
    res["fit"] = {k: fit[k] for k in ("a", "b", "r2", "rmse", "n")}

    # equal-N groups
    g = defaultdict(list)
    for r in small:
        g[(r["kind"].split("-")[0], r["N"])].append(r)
    erows = []
    eff_max = 0.0
    for key in sorted(g, key=lambda k: (k[0], k[1])):
        n = key[1]
        ws = sorted({r["seen"] for r in g[key]})
        if len(ws) < 2:
            continue
        lo = [r["fa"] for r in g[key] if r["seen"] == ws[0]]
        hi = [r["fa"] for r in g[key] if r["seen"] == ws[-1]]
        flo, fhi = statistics.median(lo), statistics.median(hi)
        slope = (fhi - flo) / (ws[-1] - ws[0]) / n * 1e6
        erows.append([key[0], n, len(g[key]), "/".join(map(str, ws)), f"{flo:.3f} (n={len(lo)})", f"{fhi:.3f} (n={len(hi)})", f"{fhi / flo:.2f}", f"{ws[-1] / ws[0]:.2f}", f"{slope:.2f}"])
        eff_max = max(eff_max, fhi / flo - 1)
    if erows:
        table(f"Table A1: passes of equal kind and N with different windows (N < {LL.SCAN_N}), FA kernel time of the 8 layers",
              ["pass kind", "N (tokens)", "passes", "windows seen (cells)", "FA at smallest window, median (ms)", "FA at largest window, median (ms)", "FA ratio", "window ratio",
               "slope (10^-6 ms per token x cell)"], erows, f"{os.path.basename(OLD_PASSES) if tag == 'old' else 'passes TSV'}; t_cat_flash_ms (flash kernels incl. mask scan and fix-up) per pass.")
    res["max_equal_n_fa_effect"] = eff_max

    # large passes
    lrows = []
    for r in rows:
        if r["N"] < LL.SCAN_N:
            continue
        lrows.append([f"{r['trace']}/{r['req']}/{r['j']}", r["kind"], r["N"], r["seen"], r["needed"], f"{r['eff']:.0f}" if r["eff"] else "-", f"{r['w_impl']:.0f}",
                      f"{r['fa']:.2f}", f"{r['pred_need']:.2f}" if r["pred_need"] else "-", f"{r['gain']:.2f}" if r["gain"] is not None else "-", f"{r['busy']:.1f}"])
    if lrows:
        table(f"Table A2: passes with N >= {LL.SCAN_N} (mask scan active): window seen, needed, effective (implied by the fit)",
              ["trace/request/pass", "kind", "N (tokens)", "window seen (cells)", "window needed, own cells padded (cells)", "mean cells read per tile if own cells at the front (cells)",
               "window implied by the fit, (FA - a)/(b N) (cells)", "FA observed (ms)", "FA predicted at the needed window (ms)", "FA above prediction (ms)", "GPU busy (ms)"], lrows,
              "observed from the trace; needed = crosscheck window model; implied window = (FA - a) / (b x N) with the fit above (cells the kernel appears to read, on average over the query tiles).")

    # equal-N scan passes: does the pass time follow the FA time?
    gn = defaultdict(list)
    for r in rows:
        if r["N"] >= LL.SCAN_N:
            gn[r["N"]].append(r)
    for n, rs in gn.items():
        if len(rs) >= 4 and max(r["fa"] for r in rs) > 1.5 * min(r["fa"] for r in rs):
            f2 = linfit([r["fa"] for r in rs], [r["busy"] for r in rs])
            arows = [[f"{r['trace']}/{r['req']}/{r['j']}", r["seen"], f"{r['fa']:.2f}", f"{r['busy']:.1f}", f"{r['busy'] - r['fa']:.1f}"] for r in sorted(rs, key=lambda r: r["fa"])]
            table(f"Table A3: the {len(rs)} passes of N = {n} (same shape, different windows): does the pass time follow the FA time?",
                  ["trace/request/pass", "window seen (cells)", "FA (ms)", "GPU busy (ms)", "GPU busy minus FA (ms)"], arows,
                  f"{os.path.basename(OLD_PASSES)}; fit busy = c + s x FA over these passes: s = {f2['b']:.2f} ms per ms, R^2 = {f2['r2']:.3f}, n = {f2['n']} [INFERENCE: four points, different cases].")
            res["busy_vs_fa_slope_%d" % n] = f2["b"]
            res["busy_minus_fa_%d" % n] = (min(r["busy"] - r["fa"] for r in rs), max(r["busy"] - r["fa"] for r in rs))
    # totals per trace (request type)
    trows = []
    for tr in sorted({r["trace"] for r in rows}):
        rs = [r for r in rows if r["trace"] == tr]
        reqs = sorted({r["req"] for r in rs})
        fa = sum(r["fa"] for r in rs)
        busy = sum(r["busy"] for r in rs)
        wall = sum(r["wall"] for r in rs)
        gain = sum(r["gain"] or 0 for r in rs)
        gain_small = sum(r["gain"] or 0 for r in rs if r["N"] < LL.SCAN_N)
        trows.append([tr, len(reqs), len(rs), f"{fa / len(reqs):.2f}", f"{busy / len(reqs):.1f}", f"{100 * fa / busy:.1f}", f"{gain_small / len(reqs):.2f}",
                      f"{(gain - gain_small) / len(reqs):.2f}", f"{gain / len(reqs):.2f}", f"{100 * gain / busy:.1f}", f"{100 * gain / wall:.1f}"])
        res.setdefault("gain_pct_busy", {})[tr] = 100 * gain / busy
        res.setdefault("gain_ms_per_request", {})[tr] = gain / len(reqs)
    table("Table A4: possible gain if the window were exact (per request, summed over the passes)",
          ["trace / request type", "requests", "passes", "FA kernel time per request (ms)", "GPU busy per request (ms)", "FA share of GPU busy (%)",
           "gain, passes N<1024: b x N x (seen - needed) (ms)", f"gain, passes N>=1024: FA above prediction (ms)", "total possible gain (ms)", "gain / GPU busy (%)", "gain / pass wall (%)"], trows,
          f"FA ms, busy and wall from the pass rows; gain per pass: N < 1024 = b x N x (seen - needed) from the fit; N >= 1024 = FA above c_scan x prediction at the front-of-pool window "
          f"(c_scan = {fit['c_scan']:.2f}, the smallest observed / model ratio of the scan passes, so the gain is a lower bound); never negative. "
          "Under nsys the kernel times include profiler overhead [INFERENCE: small for kernels]. Per-pass rows: the TSV written with --out.")
    sp_fa, sp_bu, ng = spreads(rows)
    res["spread_fa"], res["spread_busy"] = sp_fa, sp_bu
    print(f"\nRun-to-run spread among identical passes (same trace, kind, N, window; {ng} groups of >= 3): FA time {pct(sp_fa)} %, GPU busy {pct(sp_bu)} % "
          f"(relative standard deviation, median over the groups).")
    if out:
        cols = ["src", "trace", "req", "j", "phase", "kind", "case", "N", "seen", "needed", "pool", "eff", "need_eff", "fa", "busy", "wall", "w_impl", "pred_seen", "pred_need", "gain"]
        with open(os.path.join(out, f"kvwin-{tag}-passes.tsv"), "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join("" if r.get(c) is None else (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")
    return res, fit


def decide_old(res, rows, fit):
    """Decision rules on the old traces."""
    print("\n### Decision from the old traces\n")
    sp_fa = max(res["spread_fa"] or 0, MIN_SPREAD)
    thr_fa = 2 * sp_fa
    eff = res["max_equal_n_fa_effect"]
    # FA effect: largest equal-N ratio among N < 1024 passes; and, for the scan passes, the equal-N ratio of the N = 2048 passes
    big = [r for r in rows if r["N"] == 2048]
    ratio2048 = (max(r["fa"] for r in big) / min(r["fa"] for r in big) - 1) if len(big) >= 2 else None
    fa_yes = eff > thr_fa
    print(f"FA kernel: largest FA increase between passes of equal N with different windows: +{100 * eff:.0f} % (N < 1024) and "
          f"{'+' + format(100 * ratio2048, '.0f') + ' %' if ratio2048 is not None else 'n/a'} between the 2048-token passes; threshold = 2 x spread = 2 x {pct(sp_fa)} % = {pct(thr_fa)} %.")
    # pass time
    gains = res["gain_pct_busy"]
    gmax = max(gains.values()) / 100
    sp_bu = max(res["spread_busy"] or 0, MIN_SPREAD)
    thr_bu = 2 * sp_bu
    bu_yes = gmax > thr_bu
    print(f"Pass time: possible gain from an exact window = {', '.join(f'{k} {v:.1f} %' for k, v in gains.items())} of GPU busy time "
          f"({', '.join(f'{k} {v:.1f} ms' for k, v in res['gain_ms_per_request'].items())} per request); threshold = 2 x spread of the busy time of identical passes "
          f"= 2 x {pct(sp_bu)} % = {pct(thr_bu)} %.")
    return ("yes" if fa_yes else "no", eff, thr_fa), ("yes" if bu_yes else "no", gmax, thr_bu)


# ------------------------------------------------------------------------------------------ nsys kvwin run
def report_nsys(rows, client, run, out=None, kvu=True):
    """kvu=False: the --no-kv-unified run: Tables K1-K3 only (the FA law fitted on kv-unified passes does not hold: with n_stream > 1 the mask scan is on for every pass)."""
    print(f"\n## K/V test, nsys run {run}" + ("" if kvu else " (--no-kv-unified)") + "\n")
    if kvu:
        fit, small = fit_fa(rows)
        annotate(rows, fit)
    else:
        fit = None
        for r in rows:
            r["gain"] = r["pred_need"] = None
    cmap = {c.get("idx", i): c for i, c in enumerate(client)}
    phases = [p for p in dict.fromkeys(r["phase"] for r in rows)]
    tphases = [p for p in phases if p.startswith("T")]
    print(f"Requests with passes: {len({r['req'] for r in rows})}; phases: {', '.join(f'{p} ({len({r['req'] for r in rows if r['phase'] == p})})' for p in phases)}. " +
          (f"Fit on all N < {LL.SCAN_N} passes of the run: a = {fit['a']:.3f} ms, b = {fit['b'] * 1e6:.3f} x 10^-6 ms per (token x cell), R^2 = {fit['r2']:.4f}, n = {fit['n']}."
           if kvu else "No FA fit and no gain estimate for this run (the K/V window of a sequence is its own stream here; compare the T phases directly)."))
    nomatch = [r["req"] for r in rows if r["describe"] == "?" and r["j"] == 0]
    if nomatch:
        print(f"WARN: no census pass structure matches the passes of requests {sorted(set(nomatch))}: window needed unknown for them.")
    prow = []
    first = {}
    for r in rows:
        if r["phase"] in tphases:
            first.setdefault(r["phase"], r["req"])
    t1_req = first.get("T1")
    base = {r["j"]: r for r in rows if r["req"] == t1_req}
    for p in tphases:
        for r in [x for x in rows if x["req"] == first[p]]:
            b = base.get(r["j"])
            same = b is not None and b["N"] == r["N"]
            prow.append([p, r["req"], r["j"], r["kind"], r["N"], r["seen"], r["needed"], r["pool"], f"{r['seen'] / r['needed']:.2f}" if r["needed"] else "-",
                         f"{r['fa']:.3f}", f"{r['fa'] / r['n_fa']:.4f}", f"{r['fa'] / b['fa']:.2f}" if same else "n/a", f"{r['busy']:.2f}", f"{r['busy'] / b['busy']:.3f}" if same else "n/a",
                         f"{r['wall']:.2f}", f"{r['mod_ms']:.1f}" if r["mod_ms"] else "0"])
    table("Table K1: per pass of the target request (first request of every phase T1, T2, ...): window seen vs needed, FA kernel time, pass time",
          ["phase", "client request", "pass", "pass kind", "N (tokens)", "window seen (cells)", "window needed, own cells (cells)", "window needed, compact unified pool (cells)",
           "seen / needed", "FA, 8 layers (ms)", "FA per layer (ms)", "FA / T1 pass of equal N", "GPU busy (ms)", "busy / T1 pass of equal N", "pass wall (ms)", "module load in the pass (ms)"],
          prow, "passes TSV (t_cat_flash_ms = flash kernels incl. mask scan and fix-up; gpu_busy_ms; wall_ms; module_load_ms), phases from the client JSON; needed = census_lib cell positions of the "
                "pass structure that reproduces the observed passes (leadlib.pass_windows). T1 includes one-time lazy module loads [INFERENCE]: see the last column.")
    # per request totals
    qrows = []
    sums = {}
    for ci in sorted({r["req"] for r in rows}):
        rs = [r for r in rows if r["req"] == ci]
        c = cmap.get(ci, {})
        ph = rs[0]["phase"]
        sums[ci] = {"phase": ph, "fa": sum(r["fa"] for r in rs), "busy": sum(r["busy"] for r in rs), "wall": sum(r["wall"] for r in rs),
                    "seen": max(r["seen"] for r in rs), "needed": max((r["needed"] or 0) for r in rs), "n": len(rs), "desc": rs[0]["describe"],
                    "fa_al": None, "client_wall": c.get("wall_ms"), "ns": [r["N"] for r in rs], "gain": sum(r["gain"] or 0 for r in rs), "slots": rs[0]["slots"],
                    "seen_sum": sum(r["seen"] for r in rs)}
    for ci, s in sums.items():
        qrows.append([ci, s["phase"], cmap.get(ci, {}).get("case", "-"), s["desc"], s["n"], f"{s['fa']:.2f}", f"{s['busy']:.1f}", f"{s['wall']:.1f}",
                      f1(s["client_wall"], 1), s["seen"], s["needed"]] + ([f"{s['gain']:.2f}"] if kvu else []))
    table("Table K2: per request: sums over the passes", ["client request", "phase", "case", "pass structure (census labels)", "passes", "FA kernel time (ms)", "GPU busy (ms)",
                                                          "pass wall (ms)", "client wall (ms)", "largest window seen (cells)", "largest window needed (cells)"] +
          (["possible gain from an exact window (ms)"] if kvu else []),
          qrows, "passes TSV; client JSON wall_ms" + ("; gain as in Table A4." if kvu else "."))
    # aligned comparison per phase (same pass index and N as T1)
    arows = []
    al = {}
    for p in tphases:
        reqs = [ci for ci, s in sums.items() if s["phase"] == p]
        for ci in reqs:
            rs = [r for r in rows if r["req"] == ci]
            idx = [r for r in rs if r["j"] in base and base[r["j"]]["N"] == r["N"]]
            fa_t = sum(r["fa"] for r in idx)
            fa_1 = sum(base[r["j"]]["fa"] for r in idx)
            bu_t = sum(r["busy"] for r in idx)
            bu_1 = sum(base[r["j"]]["busy"] for r in idx)
            wa_t = sum(r["wall"] for r in idx)
            wa_1 = sum(base[r["j"]]["wall"] for r in idx)
            al[ci] = (p, len(idx), len(rs), fa_t / fa_1 if fa_1 else None, bu_t / bu_1 if bu_1 else None, wa_t / wa_1 if wa_1 else None, sums[ci]["seen_sum"])
            arows.append([ci, p, f"{len(idx)} of {len(rs)}", f"{fa_t:.2f}", f"{fa_1:.2f}", f"{fa_t / fa_1:.3f}" if fa_1 else "-", f"{bu_t:.1f}", f"{bu_1:.1f}", f"{bu_t / bu_1:.3f}" if bu_1 else "-",
                          f"{wa_t / wa_1:.3f}" if wa_1 else "-"])
    table("Table K3: T phases against T1 over the passes that have the same index and the same N as in T1",
          ["client request", "phase", "passes compared", "FA (ms)", "FA of the same T1 passes (ms)", "FA ratio", "GPU busy (ms)", "GPU busy of the same T1 passes (ms)", "GPU busy ratio",
           "pass wall ratio"], arows, "Table K1 rows; the slot placement (hence the suffix pass sizes) differs between requests, so only passes with equal index and N are paired.")
    return fit, sums, al, base, first


def decide_nsys(sums, al, first, sp_fa_old, sp_bu_old, label="nsys K/V run"):
    """T2..Tn against T1 (aligned passes), threshold 2 x run-to-run spread; spread from the T phases of this run that have equal window sums."""
    print(f"\n### Decision from the {label}\n")
    t = {p: ci for p, ci in first.items()}
    if "T1" not in t:
        print("no T1 request in the run")
        return None, None
    ks = [p for p in t if p != "T1"]
    fa_ratio = [al[t[p]][3] for p in ks if al.get(t[p]) and al[t[p]][3]]
    bu_ratio = [al[t[p]][4] for p in ks if al.get(t[p]) and al[t[p]][4]]
    wa_ratio = [al[t[p]][5] for p in ks if al.get(t[p]) and al[t[p]][5]]
    # run spread: T phases with the same window sum (and the same structure) as another T phase >= T2
    sp_fa = rel_sd(fa_ratio) if len(fa_ratio) >= 2 and len({sums[t[p]]["seen_sum"] for p in ks}) == 1 else None
    sp_bu = rel_sd(bu_ratio) if len(bu_ratio) >= 2 and len({sums[t[p]]["seen_sum"] for p in ks}) == 1 else None
    sp_fa_used = sp_fa if sp_fa is not None else sp_fa_old                           # the repeats of this run when they have equal windows, else the old traces
    sp_bu_used = sp_bu if sp_bu is not None else sp_bu_old
    thr_fa = 2 * max(sp_fa_used or 0, MIN_SPREAD)
    thr_bu = 2 * max(sp_bu_used or 0, MIN_SPREAD)
    e_fa = (statistics.mean(fa_ratio) - 1) if fa_ratio else None
    e_bu = (statistics.mean(bu_ratio) - 1) if bu_ratio else None
    e_wa = (statistics.mean(wa_ratio) - 1) if wa_ratio else None
    print(f"windows seen, sum over the passes: " + ", ".join(f"{p} {sums[t[p]]['seen_sum']}" for p in t) + " cells; needed (largest per request): " +
          ", ".join(f"{p} {sums[t[p]]['needed']}" for p in t) + " cells.")
    src_fa = "repeats of this run (T2..Tn, equal windows)" if sp_fa is not None else "old traces"
    src_bu = "repeats of this run (T2..Tn, equal windows)" if sp_bu is not None else "old traces"
    print(f"FA kernel, aligned passes: " + ", ".join(f"{p}/T1 = {al[t[p]][3]:.3f}" for p in ks if al.get(t[p]) and al[t[p]][3]) +
          f"; mean effect {pct(e_fa)} %; threshold = 2 x max(spread {pct(sp_fa_used)} % from the {src_fa}, floor {pct(MIN_SPREAD)} %) = {pct(thr_fa)} %.")
    print(f"GPU busy, aligned passes: " + ", ".join(f"{p}/T1 = {al[t[p]][4]:.3f}" for p in ks if al.get(t[p]) and al[t[p]][4]) +
          f"; mean effect {pct(e_bu)} %; pass wall effect {pct(e_wa)} %; threshold = 2 x max(spread {pct(sp_bu_used)} % from the {src_bu}, floor {pct(MIN_SPREAD)} %) = {pct(thr_bu)} %.")
    return ("yes" if (e_fa is not None and e_fa > thr_fa) else "no", e_fa, thr_fa), ("yes" if (e_bu is not None and e_bu > thr_bu) else "no", e_bu, thr_bu)


# ------------------------------------------------------------------------------------------ timing runs
def timing_records(label, log, client_path):
    parsed = LL.parse_server_log(log)
    client = LL.read_client_json(client_path) if client_path else None
    if parsed.get("incomplete"):
        print(f"WARN {log}: {len(parsed['incomplete'])} incomplete DMK_TIMING lines skipped (first: line {parsed['incomplete'][0][0]}, {parsed['incomplete'][0][1]}, missing {parsed['incomplete'][0][2]})", file=sys.stderr)
    recs = CP.analyse(CP.build_records(parsed, client, label))
    for r, pr in zip(recs, parsed["requests"]):
        r["syncs"] = pr["syncs"]
        r["ubs"] = pr["ubatches"]
        r["sync_wait_ms"] = sum(s["wait_us"] for s in pr["syncs"]) / 1e3
        r["gpu_side_ms"] = sum(s["since_first_submit_us"] for s in pr["syncs"]) / 1e3
        r["req_total_ms"] = r["total_us"] / 1e3
        r["req_wait_ms"] = r["wait_us"] / 1e3
        r["groups"] = [(int(s["passes"]), int(s["tokens"]), round(s["since_first_submit_us"] / 1e3, 2)) for s in pr["syncs"]]
        # structure floor with the window needed (census) as the normaliser for slot placements
        ps = CP.reproduced_passes(r) or CP.pseudo_passes(r)
        r["floor_ms"] = LL.pass_floors(ps)[1]
        r["n_syncs"] = len(pr["syncs"])
    return recs


def report_timing(label, recs, client_given):
    print(f"\n## Timing run `{label}` (DMK_TIMING, no profiler)\n")
    print(f"Requests: {len(recs)}; parent slot selection {dict(Counter(r['select'] for r in recs))}; cache mode in the log kv_unified = {recs[0]['kv_unified'] if recs else None} "
          f"(expected split: {recs[0]['expected_mode'] if recs else None}); pass structure: census (unified split) exact in {sum(r['unified_ok'] for r in recs)}, "
          f"sequential-split model exact in {sum(r['seq_ok'] for r in recs)}, expected model exact in {sum(r['expected_ok'] for r in recs)}, neither in {sum(1 for r in recs if not (r['unified_ok'] or r['seq_ok']))} of {len(recs)} requests; "
          f"llama_decode calls per request type: " + ", ".join(f"{k} {min(x['n_dec'] for x in v)}-{max(x['n_dec'] for x in v)}" for k, v in CP.by_kind(recs).items()) + ".")
    trs = [r for r in recs if r["phase"] and (r["phase"].startswith("T") or r["phase"] == "other")]
    rows = []
    for r in trs:
        rows.append([r["k"], r["phase"], r["case"], str(r["slots"]), r["describe"], len(r["ub"]), r["n_dec"], r["n_syncs"],
                     "; ".join(f"{p}p/{t}t/{ms}ms" for p, t, ms in r["groups"]), f"{r['gpu_side_ms']:.2f}", f"{r['sync_wait_ms']:.2f}", f"{r['req_wait_ms']:.2f}",
                     f"{r['req_total_ms']:.2f}", f1(r["client_wall_ms"], 2), f"{r['floor_ms']:.1f}", f"{r['gpu_side_ms'] / r['floor_ms']:.3f}"])
    table(f"Table T1 ({label}): per request of the K/V sequence (T phases and the others)",
          ["request", "phase", "case", "slots (Q0, Q1, Q2)", "pass structure", "ubatches", "llama_decode calls", "sync groups", "sync groups (passes / tokens / since_first_submit)",
           "GPU-side time: sum of sync since_first_submit_us (ms)", "sum of sync wait_us (ms)", "request wait_us (ms)", "request total_us (ms)", "client wall (ms)",
           "serial roofline floor of the structure (ms)", "GPU-side time / floor"], rows,
          "DMK_TIMING sync / decode / ubatch / request lines and slot lines of the server log; since_first_submit_us runs from the start of the first pending ubatch to the end of the wait for "
          "the GPU (all pending passes together: the logits read is the only synchronise of a request, so prefix passes and the first suffix batch share one group); floor = census_lib pass_costs "
          "of the matching structure (window needed).")
    return trs


def decide_timing(label, trs):
    t = {}
    for r in trs:
        t.setdefault(r["phase"], r)
    if "T1" not in t:
        print(f"[{label}] no T1 request")
        return None
    ks = [p for p in t if p.startswith("T") and p != "T1"]
    ratio = {p: (t[p]["gpu_side_ms"] / t[p]["floor_ms"]) / (t["T1"]["gpu_side_ms"] / t["T1"]["floor_ms"]) for p in ks}
    raw = {p: t[p]["gpu_side_ms"] / t["T1"]["gpu_side_ms"] for p in ks}
    sp = rel_sd([t[p]["gpu_side_ms"] / t[p]["floor_ms"] for p in ks]) if len(ks) >= 2 else None
    thr = 2 * max(sp or 0, MIN_SPREAD)
    e = statistics.mean(ratio.values()) - 1 if ratio else None
    print(f"[{label}] GPU-side time / structure floor, Tk / T1: " + ", ".join(f"{p} {v:.3f} (raw time ratio {raw[p]:.3f}, structure {t[p]['describe']} vs {t['T1']['describe']})" for p, v in ratio.items()) +
          f"; mean effect {pct(e)} %; threshold = 2 x max(spread of T2..Tn {pct(sp)} %, floor {pct(MIN_SPREAD)} %) = {pct(thr)} %.")
    return ("yes" if (e is not None and e > thr) else "no", e, thr)


def compare_timing(runs):
    """kv-unified against no-kv-unified, T phases side by side."""
    if len(runs) < 2:
        return
    labels = list(runs)
    base_l = labels[0]
    rows = []
    for ph in dict.fromkeys(r["phase"] for l in labels for r in runs[l] if r["phase"] and r["phase"].startswith("T")):
        row = [ph]
        vals = {}
        for l in labels:
            rs = [r for r in runs[l] if r["phase"] == ph]
            if rs:
                r = rs[0]
                vals[l] = r
                row += [r["describe"], f"{r['gpu_side_ms']:.2f}", f"{r['req_total_ms']:.2f}", f1(r["client_wall_ms"], 2)]
            else:
                row += ["-", "-", "-", "-"]
        a, b = vals.get(labels[0]), vals.get(labels[1])
        row.append(f"{b['gpu_side_ms'] / a['gpu_side_ms']:.3f}" if a and b else "-")
        rows.append(row)
    hdr = ["phase"]
    for l in labels:
        hdr += [f"{l}: pass structure", f"{l}: GPU-side time (ms)", f"{l}: request total_us (ms)", f"{l}: client wall (ms)"]
    hdr.append(f"{labels[1]} / {labels[0]} GPU-side time")
    table("Table T2: kv-unified against --no-kv-unified (the upstream workaround), same client sequence", hdr, rows,
          "Table T1 rows of both runs; with --no-kv-unified every slot is its own stream (n_ctx_slot = 5461 cells) and the window of a sequence is its own cells; a child's prefix is copied "
          "across streams (llama_kv_cache::seq_cp -> buffer copy after llama_synchronize) [source]. Pass structures can differ between the runs (slot placement, split mode).")


# ------------------------------------------------------------------------------------------ main flow
class Tee:
    def __init__(self, *fs):
        self.fs = fs

    def write(self, s):
        for f in self.fs:
            f.write(s)

    def flush(self):
        for f in self.fs:
            f.flush()


def analyse_all(args):
    out = args.out
    if out:
        os.makedirs(out, exist_ok=True)
    decisions = {}
    sp_fa_old = sp_bu_old = None
    print("# Lead #28495: flash-attention K/V window with --kv-unified")
    if not args.no_old:
        rows = old_rows()
        res, fit = report_regression(rows, "1. Existing evidence: the three old Nsight Systems traces (long, medium, short: 72 passes)", out, "old")
        d_fa, d_bu = decide_old(res, rows, fit)
        decisions["old"] = (d_fa, d_bu)
        sp_fa_old, sp_bu_old = res["spread_fa"], res["spread_busy"]
    def write_pass_tsv(name, rws):
        if not out:
            return
        cols = ["trace", "req", "j", "phase", "kind", "case", "N", "seen", "needed", "pool", "eff", "fa", "busy", "wall", "w_impl", "pred_need", "gain", "describe"]
        with open(os.path.join(out, name), "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rws:
                f.write("\t".join("" if r.get(c) is None else (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")

    if args.nsys:
        client = LL.read_client_json(args.client) if args.client else []
        rows_n, run = nsys_rows(args.nsys, client, args.run)
        fit, sums, al, base, first = report_nsys(rows_n, client, run, out)
        d_fa, d_bu = decide_nsys(sums, al, first, sp_fa_old, sp_bu_old)
        decisions["nsys"] = (d_fa, d_bu)
        write_pass_tsv("kvwin-nsys-passes.tsv", rows_n)
    if args.nsys_nokvu:
        parts = args.nsys_nokvu.split(",")
        client = LL.read_client_json(parts[1]) if len(parts) > 1 and parts[1] else []
        rows_k, run = nsys_rows(parts[0], client, parts[2] if len(parts) > 2 else None)
        fit, sums, al, base, first = report_nsys(rows_k, client, run, out, kvu=False)
        d_fa, d_bu = decide_nsys(sums, al, first, sp_fa_old, sp_bu_old, label="nsys --no-kv-unified run")
        decisions["nsys-nokvu"] = (d_fa, d_bu)
        write_pass_tsv("kvwin-nsys-nokvu-passes.tsv", rows_k)
    runs = {}
    for spec in args.timing or []:
        label, rest = spec.split("=", 1)
        log, _, cj = rest.partition(",")
        recs = timing_records(label, log, cj or None)
        trs = report_timing(label, recs, bool(cj))
        runs[label] = recs
        decisions["timing:" + label] = decide_timing(label, trs)
    compare_timing(runs)
    print("\n## Decisions\n")
    print("Thresholds: an effect counts if it exceeds 2 x the run-to-run spread (relative standard deviation = stdev / mean of repeated identical measurements; floor 2 %). "
          "Spreads come from the repeated T phases of the run when they have equal windows, else from identical passes of the old traces (printed above).")
    for src, v in decisions.items():
        if src == "old":
            (fa, e, th), (bu, g, th2) = v
            print(f"[old traces]   window effect on the FA kernel: {fa} (largest equal-N FA increase {100 * e:.0f} %, threshold {100 * th:.1f} %); "
                  f"window effect on pass time: {bu} (possible gain {100 * g:.1f} % of GPU busy, threshold {100 * th2:.1f} %)")
        elif src in ("nsys", "nsys-nokvu"):
            fa, bu = v
            if fa:
                print(f"[{'kvwin nsys' if src == 'nsys' else 'nsys --no-kv-unified'}]   window effect on the FA kernel: {fa[0]} (mean T2..Tn / T1 effect {pct(fa[1])} %, threshold {pct(fa[2])} %); "
                      f"window effect on pass time: {bu[0]} (GPU busy effect {pct(bu[1])} %, threshold {pct(bu[2])} %)")
        elif v:
            print(f"[{src}]   window effect on pass time: {v[0]} (GPU-side time / floor, mean Tk / T1 effect {pct(v[1])} %, threshold {pct(v[2])} %)")
    best_fa = decisions["nsys"][0] if "nsys" in decisions and decisions["nsys"][0] else decisions["old"][0] if "old" in decisions else None
    src_fa = "kvwin nsys run" if "nsys" in decisions and decisions["nsys"][0] else "old traces"
    best_bu_t = decisions.get("timing:kvu") or next((v for k, v in decisions.items() if k.startswith("timing:") and v), None)
    best_bu = best_bu_t if best_bu_t else (decisions["nsys"][1] if "nsys" in decisions and decisions["nsys"][1] else decisions["old"][1] if "old" in decisions else None)
    src_bu = "kvwin timing run" if best_bu_t else ("kvwin nsys run" if "nsys" in decisions and decisions["nsys"][1] else "old traces")
    print("\nFinal (the most direct source available: nsys run for the FA kernel, timing run for the pass time, else the old traces):")
    if best_fa:
        print(f"window effect on the FA kernel: {best_fa[0]}   (effect {pct(best_fa[1], 0)} %, threshold {pct(best_fa[2], 1)} %; source: {src_fa})")
    if best_bu:
        print(f"window effect on pass time: {best_bu[0]}   (effect {pct(best_bu[1])} %, threshold {pct(best_bu[2], 1)} %; source: {src_bu})")
    return decisions


def self_test():
    """Synthetic data in a temporary folder (TMPDIR) that is removed afterwards."""
    with tempfile.TemporaryDirectory(prefix="kvwin-selftest-") as tmp:
        return _self_test_in(tmp)


def _self_test_in(tmp):
    import make_synthetic as MS
    MS.generate(tmp)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        ok &= bool(cond)

    for scen in ("kvwin-growth", "kvwin-strong", "kvwin-null"):
        base = os.path.join(tmp, "syn-" + scen)
        truth = json.load(open(base + ".truth.json"))
        growth, planted = truth["growth"], truth["planted_effect_T2_T4_vs_T1"]
        print(f"\n===== self test: {scen} (planted window growth: {growth}; FA law x{truth['fa_scale']}; planted effect T2/T4 vs T1: FA {pct(planted['fa'])} %, pass time {pct(planted['pass_time'])} %)")
        client = LL.read_client_json(base + ".json")
        rows, run = nsys_rows(base + ".passes.tsv", client)
        check("phases recovered", [r["phase"] for r in rows if r["j"] == 0][:3] == ["warmup", "warmup", "T1"])
        check("census structure matches every request", all(r["describe"] != "?" for r in rows if r["j"] == 0))
        sink = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            fit, sums, al, base_rows, first = report_nsys(rows, client, run, tmp)
            d_fa, d_bu = decide_nsys(sums, al, first, 0.045, 0.044)           # old-trace spreads (printed by the old-data run)
        finally:
            sys.stdout.close()
            sys.stdout = sink
        check("fit recovers a (0.48 ms) within 0.1 ms", abs(fit["a"] - MS.FA_A) < 0.1, f"a = {fit['a']:.3f}")
        check("fit recovers b within 3 %", abs(fit["b"] / (MS.FA_B * truth["fa_scale"]) - 1) < 0.03, f"b = {fit['b']:.3e} vs {MS.FA_B * truth['fa_scale']:.3e}")
        tr = {r["idx"]: r for r in truth["requests"]}
        fa_sums = {ci: s["fa"] for ci, s in sums.items()}
        check("per-request FA sums equal the planted sums", all(abs(fa_sums[i] - tr[i]["fa_ms_sum"]) < 1e-3 for i in fa_sums))
        check(f"FA decision = {truth['expect']['fa_effect']}", d_fa and d_fa[0] == truth["expect"]["fa_effect"], f"effect {pct(d_fa[1]) if d_fa else None} % thr {pct(d_fa[2]) if d_fa else None} %")
        want = truth["expect"]["pass_time_effect"]
        rule = "yes" if d_bu[1] > d_bu[2] else "no"
        check(f"pass-time decision = {want if want != 'any' else 'the rule (effect > threshold)'}", d_bu and d_bu[0] == (rule if want == "any" else want),
              f"effect {pct(d_bu[1]) if d_bu else None} % thr {pct(d_bu[2]) if d_bu else None} %")
        check("recovered FA effect within 20 % (relative) of the planted one" if growth else "recovered FA effect below 3 %",
              abs(d_fa[1] - planted["fa"]) < 0.2 * abs(planted["fa"]) if growth else abs(d_fa[1]) < 0.03, f"{pct(d_fa[1])} % vs planted {pct(planted['fa'])} %")
        check("recovered pass-time effect within 1.5 points of the planted one", abs(d_bu[1] - planted["pass_time"]) < 0.015, f"{pct(d_bu[1])} % vs planted {pct(planted['pass_time'])} %")
        # window seen/needed planted: T1 seen = needed + 256 (warm-up cells)
        t1 = [r for r in rows if r["phase"] == "T1"]
        check("T1 window seen = needed + 256 for N < 1024", all(r["seen"] - r["needed"] == 256 for r in t1 if r["N"] < 1024))
        if growth:
            t2 = [r for r in rows if r["phase"] == "T2"]
            check("T2 window seen = needed + 3328", all(r["seen"] - r["needed"] == 3328 for r in t2 if r["N"] < 1024))
        # timing log
        recs = timing_records(scen, base + ".server.log", base + ".json")
        sink = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            trs = report_timing(scen, recs, True)
            dec = decide_timing(scen, trs)
        finally:
            sys.stdout.close()
            sys.stdout = sink
        check("timing run: every request is reproduced by the model expected for its cache mode", all(r["expected_ok"] for r in recs))
        rule = "yes" if dec[1] > dec[2] else "no"
        check(f"timing decision = {want if want != 'any' else 'the rule (effect > threshold)'}", dec and dec[0] == (rule if want == "any" else want),
              f"effect {pct(dec[1]) if dec else None} % thr {pct(dec[2]) if dec else None} %")
        gpu = {r["k"]: r["gpu_side_ms"] for r in recs}
        dev = max(abs(gpu[i] / tr[i]["gpu_ms_sum"] - 1) for i in gpu if tr[i]["phase"].startswith("T"))
        check("sync GPU-side time within 3 % of the planted GPU time of the request", dev < 0.03, "max dev %.2f %%" % (100 * dev))
    base = os.path.join(tmp, "syn-kvwin-nokvu")
    recs = timing_records("nokvu", base + ".server.log", base + ".json")
    print("\n===== self test: kvwin-nokvu (timing log with a synchronise before the suffix of every multi-question request)")
    multi = [r for r in recs if r["n_q"] > 1 and any(p.startswith("P") for p in r["describe"].split("+"))]
    check("the prefix passes form their own sync group (first group = number of P passes)",
          all(r["groups"][0][0] == sum(1 for p in r["describe"].split("+") if p.startswith("P")) for r in multi),
          f"groups of T1: {[r for r in recs if r['phase'] == 'T1'][0]['groups']}")
    check("T1 of the no-kv-unified log has two sync groups (prefix passes, then suffix passes)", [r for r in recs if r["phase"] == "T1"][0]["n_syncs"] == 2)
    print("\nself test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nsys", help="passes TSV of the kvwin nsys run (package 2 schema)")
    ap.add_argument("--client", help="client JSON of the kvwin run (kvwin_client.py)")
    ap.add_argument("--run", help="run name in the passes TSV (when it has several)")
    ap.add_argument("--timing", action="append", help="LABEL=LOG[,CLIENT.json] of a timing run; repeat for kvu and nokvu")
    ap.add_argument("--nsys-nokvu", help="PASSES.tsv,CLIENT.json[,RUN] of the --no-kv-unified nsys run (Tables K1-K3 and the T/T1 decision only)")
    ap.add_argument("--no-old", action="store_true", help="skip the old-trace regression")
    ap.add_argument("--out", help="directory for the TSVs and report.txt")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        rep = open(os.path.join(a.out, "report.txt"), "w")
        sys.stdout = Tee(sys.__stdout__, rep)
    analyse_all(a)


if __name__ == "__main__":
    main()
