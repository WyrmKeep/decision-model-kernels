#!/usr/bin/env python3
"""Analysis of ncu reports of the phase-3 runs (python 3 stdlib; calls the LOCAL ncu only to turn a .ncu-rep / .ncu-repz into CSV).

  analyze_ncu.py <report.ncu-rep | report.ncu-repz | raw.csv | dir> [...] [--ncu PATH] [--out-dir DIR] [--plan-dir DIR]
  analyze_ncu.py --self-test

Every report goes through `ncu --import <rep> --csv --page raw` (a temporary file, HOME in a temporary directory so ncu does not create
~/Documents/NVIDIA Nsight Compute; default units, i.e. auto-scaled with 6 decimals, converted to ns / byte / Hz here; the same code reads the
`--print-units base` format with thousands separators). A directory means all *.ncu-rep and *.ncu-repz in it (ncu 2026.3 writes compressed
reports with the extension .ncu-repz). A .csv is read as such (tests).
The run name of a report is its file name without .ncu-rep / .ncu-repz; plan/<run>.json (plan_ncu.py --write) gives the planned window.

Outputs in --out-dir (default: <first input>/analysis or next to the first input file):
  kernels.tsv    one row per profiled launch: class, window label, alignment, duration, SM clock, DRAM/L2 traffic, pipes, occupancy, stalls
  classes.tsv    one row per (run, class): the same columns averaged over the launches of the class
  sol.tsv        speed of light per class: measured time, binding limit, time at the limit, gap per launch / per pass, share of the pass
  sol.txt        sol.tsv as tables (Markdown syntax) + the alignment check + planned-vs-hit classes + per-pass coverage
  alignment.tsv  one row per run: planned launches, observed launches, shift, status
A report with 0 launches, a missing metric (empty cell), several reports in one call: all handled; nothing raises on missing data.
"""
import argparse
import collections
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ncu_cfg as cfg  # noqa: E402

TYPE_NAME = {2: "Q4_0", 8: "Q8_0"}
FIXED_COLS = ("ID", "Kernel Name", "Block Size", "Grid Size")


# ----------------------------------------------------------------------------------------------------------------- csv parsing
_PREFIX = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
_TIME = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_FLOAT = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def unit_factor(unit):
    """Factor to the base unit of the analysis: time -> ns, byte / byte/s / hz (any K, M, G prefix, decimal) -> 1."""
    u = (unit or "").strip()
    if u in _TIME:
        return _TIME[u]
    m = re.fullmatch(r"([KMGTkmgt]?)(byte|hz)(/s|/block)?", u, re.I)
    if m:
        return _PREFIX[m.group(1).upper()]
    return 1.0


def num(cell, unit=""):
    """Cell -> float in base units, None for empty / n/a / not a number."""
    c = (cell or "").strip()
    if c == "" or c.lower() in ("n/a", "nan", "-"):
        return None
    if _THOUSANDS.match(c):
        c = c.replace(",", "")
    if not _FLOAT.match(c):
        return None
    return float(c) * unit_factor(unit)


def parse_dims(s):
    m = re.findall(r"\d+", s or "")
    return tuple(int(x) for x in m[:3]) if len(m) >= 3 else None


def parse_csv(text):
    """ncu raw-page CSV -> list of kernel dicts (id, name, grid, block, metrics{name: float|None}, extra_cols). Lines before the header
    (==PROF== / ==WARNING== messages) are skipped; no header -> no kernels."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.lstrip('"').startswith("ID")), None)
    if start is None:
        return []
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    if len(rows) < 2:
        return []
    header = rows[0]
    col = {h: i for i, h in enumerate(header)}
    if "Kernel Name" not in col:
        return []
    body = rows[1:]
    units = [""] * len(header)
    if body and (body[0][:1] == [""] or not body[0][0].strip().isdigit()):
        units = body[0] + [""] * (len(header) - len(body[0]))
        body = body[1:]
    want = {}
    for key, name in cfg.METRICS.items():
        if name in col:
            want[key] = col[name]
    stall_cols = {}
    for h, i in col.items():
        m = re.fullmatch(r"smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio", h)
        if m and m.group(1) not in ("selected", "not_selected"):
            stall_cols[m.group(1)] = i
    out = []
    for r in body:
        if len(r) < len(header) or not r[0].strip():
            continue
        g = lambda name: r[col[name]] if name in col else ""  # noqa: E731
        k = {"id": int(g("ID")) if g("ID").strip().isdigit() else len(out), "name": g("Kernel Name"), "block": parse_dims(g("Block Size")),
             "grid": parse_dims(g("Grid Size")), "device": g("Device"), "cc": g("CC"),
             "m": {key: num(r[i], units[i]) for key, i in want.items()},
             "stalls": {s: num(r[i], units[i]) for s, i in stall_cols.items()}}
        out.append(k)
    return out


def ncu_csv(ncu, rep):
    """Run ncu --import on a report; returns the CSV text."""
    home = tempfile.mkdtemp(prefix="ncu-home-")
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tf:
        path = tf.name
    try:
        with open(path, "w") as out:
            p = subprocess.run([ncu, "--import", rep, "--csv", "--page", "raw", "--print-kernel-base", "demangled"], stdout=out, stderr=subprocess.PIPE, text=True,
                               env=dict(os.environ, HOME=home))
        if p.returncode != 0:
            sys.stderr.write(f"warning: ncu --import {rep} exited {p.returncode}: {p.stderr[-300:]}\n")
        with open(path, errors="replace") as f:
            return f.read()
    finally:
        os.unlink(path)
        try:
            for root, dirs, files in os.walk(home, topdown=False):
                for n in files:
                    os.unlink(os.path.join(root, n))
                for n in dirs:
                    os.rmdir(os.path.join(root, n))
            os.rmdir(home)
        except OSError:
            pass


# ----------------------------------------------------------------------------------------------------------------- names
def split_top(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_name(name):
    """Demangled kernel name -> (function without namespace/template/parameters, template arguments as ints where they are numbers)."""
    s = re.sub(r"^\s*(?:void|static|__global__)\s+", "", name.strip())
    m = re.match(r"((?:[A-Za-z_]\w*\s*::\s*)*)([A-Za-z_]\w*)\s*(.*)$", s, re.S)
    if not m:
        return name, []
    func, rest = m.group(2), m.group(3)
    targs = []
    if rest.startswith("<"):
        depth, end = 0, None
        for i, ch in enumerate(rest):
            depth += ch == "<"
            depth -= ch == ">"
            if depth == 0:
                end = i
                break
        for a in split_top(rest[1:end] if end else rest[1:]):
            a = re.sub(r"^\(\s*[\w: ]+\)\s*", "", a)
            if a in ("true", "false"):
                targs.append(int(a == "true"))
            elif re.fullmatch(r"-?\d+[uUlL]*", a):
                targs.append(int(re.sub(r"[uUlL]+$", "", a)))
            else:
                targs.append(a)
    return func, targs


def kernel_features(k):
    func, t = parse_name(k["name"])
    k["func"], k["targs"] = func, t
    k["J"] = k["wtype"] = k["need_check"] = None
    if func.startswith("mul_mat_q") and len(t) >= 3 and all(isinstance(x, int) for x in t[:3]):
        k["wtype"], k["J"], k["need_check"] = TYPE_NAME.get(t[0], f"type{t[0]}"), t[1], t[2]


def generic_class(k):
    g = k["grid"] or ("?", "?", "?")
    f = k["func"]
    if f == "mul_mat_q" and k["J"] is not None:
        return f"mmq {k['wtype']} J={k['J']} chk={k['need_check']} grid=({g[0]},{g[1]},{g[2]})"
    if f == "mul_mat_q_stream_k_fixup" and k["J"] is not None:
        return f"mmq-fixup {k['wtype']} J={k['J']} chk={k['need_check']} grid=({g[0]},{g[1]},{g[2]})"
    return f"{f} grid=({g[0]},{g[1]},{g[2]})"


# ----------------------------------------------------------------------------------------------------------------- alignment
def e_matches(k, e):
    """observed kernel k vs expected record e: function, and the grid/block where the plan is certain, and J/type of an MMQ."""
    if k["func"] != e["func"]:
        return False
    if e.get("grid_exact") and k["grid"] and k["block"]:
        if tuple(k["grid"]) != (e["gx"], e["gy"], e["gz"]) or tuple(k["block"][:2]) != (e["bx"], e["by"]):
            return False
    if k["func"] == "mul_mat_q" and e.get("J") not in ("", None) and k["J"] is not None:
        if k["J"] != e["J"] or k["wtype"] != e["wtype"]:
            return False
    return True


def c_matches(k, c):
    """observed vs context entry [rel, func, gx, gy, gz, bx, by, grid_exact, J, wtype]."""
    e = {"func": c[1], "gx": c[2], "gy": c[3], "gz": c[4], "bx": c[5], "by": c[6], "grid_exact": bool(c[7]), "J": c[8], "wtype": c[9]}
    return e_matches(k, e)


def align(kernels, plan):
    """Find the shift s (observed launch i <-> planned filter-stream index skip + s + i, context of +-CTX launches in the plan) that matches best.
    Returns dict(kind, status, shift, matched, n_obs, n_plan); kind is ALIGNED / PARTIAL / SHIFTED / MISALIGNED / EMPTY."""
    n_obs, n_plan = len(kernels), len(plan["expected"])
    res = dict(n_obs=n_obs, n_plan=n_plan, shift=0, matched=0, kind="EMPTY", status="EMPTY (no launches in the report)")
    if n_obs == 0:
        return res
    ctx = {c[0]: c for c in plan.get("context", [])}
    best = None
    for s in range(-40, 41):
        m = sum(1 for i, k in enumerate(kernels) if (i + s) in ctx and c_matches(k, ctx[i + s]))
        key = (m, -abs(s))
        if best is None or key > best[0]:
            best = (key, s, m)
    _, s, m = best
    res.update(shift=s, matched=m)
    w = plan["windows"][0]
    if s == 0 and m == n_obs and n_obs == n_plan:
        res.update(kind="ALIGNED", status="ALIGNED")
    elif s == 0 and m == n_obs:
        res.update(kind="PARTIAL", status=f"PARTIAL ({n_obs} of {n_plan} planned launches profiled)")
    elif s == 0 and m >= 0.9 * n_obs and n_obs >= 3:
        res.update(kind="PARTIAL", status=f"PARTIAL ({n_obs - m} of {n_obs} profiled launches differ from the plan; class of those from the kernel name)")
    elif m >= 0.9 * n_obs and n_obs >= 3:
        res.update(kind="SHIFTED", status=f"SHIFTED by {s:+d} launches of the filter stream; correct with plan_ncu.py --shift {w['filter']}={w['shift'] - s} "
                                          f"(skip {w['skip']} -> {w['skip'] - s})")
    else:
        res.update(kind="MISALIGNED", status=f"MISALIGNED (best {m} of {n_obs} launches match, at shift {s:+d}); classes are named from the kernel names only")
    return res


# ----------------------------------------------------------------------------------------------------------------- per kernel numbers
def mget(k, key):
    return k["m"].get(key)


def stall_shares(k):
    st = {s: v for s, v in k["stalls"].items() if v is not None}
    tot = sum(st.values())
    if tot <= 0:
        return []
    return sorted(((s, v / tot) for s, v in st.items()), key=lambda x: -x[1])


def kernel_row(run, k, e, window, status):
    dur = mget(k, "duration_ns")
    rd, wr = mget(k, "dram_rd"), mget(k, "dram_wr")
    dram = (rd or 0) + (wr or 0) if (rd is not None or wr is not None) else None
    if dram is None and mget(k, "dram_bps") is not None and dur:
        dram = mget(k, "dram_bps") * dur / 1e9
    hz = mget(k, "sm_hz") or mget(k, "sm_hz_alt")
    top = stall_shares(k)
    r = {
        "run": run, "id": k["id"], "window": window, "class": k["cls"], "align": status,
        "function": k["func"], "grid": "x".join(map(str, k["grid"])) if k["grid"] else "", "block": "x".join(map(str, k["block"])) if k["block"] else "",
        "duration_us": None if dur is None else dur / 1e3,
        "nsys_proxy_us": None if e is None else e["dur_ns_nsys_proxy"] / 1e3,
        "sm_clock_mhz": None if hz is None else hz / 1e6,
        "sm_cycles_k": None if (hz is None or dur is None) else hz * dur / 1e9 / 1e3,
        "dram_clock_mhz": None if mget(k, "dram_hz") is None else mget(k, "dram_hz") / 1e6,
        "dram_read_mb": None if rd is None else rd / 1e6, "dram_write_mb": None if wr is None else wr / 1e6,
        "dram_gb_s": None if (dram is None or not dur) else dram / dur,
        "dram_pct_of_422": None if (dram is None or not dur) else 100 * dram / dur / cfg.LIM_DRAM_GBPS,
        "ncu_dram_pct_peak": mget(k, "dram_pct"),
        "l2_mb": None if mget(k, "l2_bytes") is None else mget(k, "l2_bytes") / 1e6,
        "l2_hit_pct": mget(k, "l2_hit"), "l1_hit_pct": mget(k, "l1_hit"),
        "imma_pipe_pct": mget(k, "imma_pct"), "hmma_pipe_pct": mget(k, "hmma_pct"),
        "imma_inst_pct": mget(k, "imma_inst_pct"), "hmma_inst_pct": mget(k, "hmma_inst_pct"),
        "imma_inst_m": None if mget(k, "imma_inst") is None else mget(k, "imma_inst") / 1e6,
        "int8_gops": None if mget(k, "int8_ops") is None else mget(k, "int8_ops") / 1e9,
        "inst_m": None if mget(k, "inst") is None else mget(k, "inst") / 1e6,
        "issue_active_pct": mget(k, "issue_pct"), "sm_throughput_pct": mget(k, "sm_thr_pct"), "warps_active_pct": mget(k, "warps_pct"),
        "warps_max_pct": mget(k, "warps_max_pct"), "latency_cycles": mget(k, "latency"),
        "regs": mget(k, "regs"), "smem_static_b": mget(k, "smem_static"), "smem_dyn_b": mget(k, "smem_dyn"), "waves_per_sm": mget(k, "waves"),
        "occ_limit_regs": mget(k, "occ_lim_regs"), "occ_limit_smem": mget(k, "occ_lim_smem"), "occ_limit_warps": mget(k, "occ_lim_warps"),
        "occ_limit_blocks": mget(k, "occ_lim_blocks"), "passes": mget(k, "passes"),
        "stall_top3": "; ".join(f"{s} {100 * v:.0f}%" for s, v in top[:3]),
    }

    for s in ("long_scoreboard", "short_scoreboard", "barrier", "mio_throttle", "math_pipe_throttle", "wait", "no_instruction", "lg_throttle", "dispatch_stall"):
        v = k["stalls"].get(s)
        tot = sum(x for x in k["stalls"].values() if x is not None)
        r["stall_" + s + "_pct"] = None if (v is None or tot <= 0) else 100 * v / tot
    return r


def sol_numbers(k, e):
    """Time at the binding limit of one launch. Returns dict(bound, floor_us, parts{limit: us}, source) or None when the duration is missing."""
    dur = mget(k, "duration_ns")
    if dur is None:
        return None
    parts = {}
    src = "model bytes"
    bytes_model = e.get("bytes_model") if e else ""
    if bytes_model not in ("", None):
        parts["DRAM"] = bytes_model / (cfg.LIM_DRAM_GBPS * 1e9) * 1e6
    else:
        rd, wr = mget(k, "dram_rd"), mget(k, "dram_wr")
        if rd is not None or wr is not None:
            parts["DRAM"] = ((rd or 0) + (wr or 0)) / (cfg.LIM_DRAM_GBPS * 1e9) * 1e6
            src = "measured DRAM bytes (no byte model for this kernel)"
    for key, limit, label in (("ops_int8", cfg.LIM_INT8_TOPS * 1e12, "INT8"), ("flops_fp16", cfg.LIM_FP16_TFLOPS * 1e12, "FP16"),
                              ("flops_fp32", cfg.LIM_FP32_TFLOPS * 1e12, "FP32")):
        v = e.get(key) if e else ""
        if v not in ("", None):
            parts[label] = v / limit * 1e6
    parts["launch"] = cfg.LAUNCH_FLOOR_US
    bound = max(parts, key=parts.get)
    return {"bound": bound, "floor_us": parts[bound], "parts": parts, "source": src}


# ----------------------------------------------------------------------------------------------------------------- analysis of one run
def analyze_run(run, kernels, plan):
    for k in kernels:
        kernel_features(k)
    if plan:
        al = align(kernels, plan)
    else:
        al = dict(n_obs=len(kernels), n_plan=0, shift=0, matched=0, kind="NOPLAN" if kernels else "EMPTY",
                  status="no plan for this run" if kernels else "EMPTY (no launches in the report)")
    shift = al["shift"]
    ok = plan is not None and al["kind"] in ("ALIGNED", "PARTIAL", "SHIFTED")
    krows = []
    for i, k in enumerate(kernels):
        e = None
        if ok:
            j = i + shift
            if 0 <= j < len(plan["expected"]):
                cand = plan["expected"][j]
                if e_matches(k, cand):
                    e = cand
        k["cls"] = e["cls"] if e else generic_class(k)
        window = e["window"] if e else ""
        krows.append((k, e, kernel_row(run, k, e, window, al["kind"] if e else ("unplanned" if plan else "no plan"))))
        k["e"] = e
    return al, krows


AVG_COLS = ["duration_us", "nsys_proxy_us", "sm_clock_mhz", "sm_cycles_k", "dram_clock_mhz", "dram_read_mb", "dram_write_mb", "dram_gb_s", "dram_pct_of_422",
            "ncu_dram_pct_peak", "l2_mb", "l2_hit_pct", "l1_hit_pct", "imma_pipe_pct", "hmma_pipe_pct", "imma_inst_pct", "hmma_inst_pct", "imma_inst_m",
            "int8_gops", "inst_m", "issue_active_pct", "sm_throughput_pct", "warps_active_pct", "warps_max_pct", "latency_cycles", "regs", "smem_static_b",
            "smem_dyn_b", "waves_per_sm", "occ_limit_regs", "occ_limit_smem", "occ_limit_warps", "occ_limit_blocks", "passes",
            "stall_long_scoreboard_pct", "stall_short_scoreboard_pct", "stall_barrier_pct", "stall_mio_throttle_pct", "stall_math_pipe_throttle_pct",
            "stall_wait_pct", "stall_no_instruction_pct", "stall_lg_throttle_pct", "stall_dispatch_stall_pct"]
KERNEL_COLS = ["run", "id", "window", "class", "align", "function", "grid", "block", "duration_us", "nsys_proxy_us", "sm_clock_mhz", "sm_cycles_k",
               "dram_clock_mhz", "dram_read_mb", "dram_write_mb", "dram_gb_s", "dram_pct_of_422", "ncu_dram_pct_peak", "l2_mb", "l2_hit_pct", "l1_hit_pct",
               "imma_pipe_pct", "hmma_pipe_pct", "imma_inst_pct", "hmma_inst_pct", "imma_inst_m", "int8_gops", "inst_m", "issue_active_pct",
               "sm_throughput_pct", "warps_active_pct", "warps_max_pct", "latency_cycles", "regs", "smem_static_b", "smem_dyn_b", "waves_per_sm",
               "occ_limit_regs", "occ_limit_smem", "occ_limit_warps", "occ_limit_blocks", "passes", "stall_top3"] + \
              [c for c in AVG_COLS if c.startswith("stall_") and c != "stall_top3"]
HEAD_UNITS = {"duration_us": "duration, us", "nsys_proxy_us": "nsys duration of the model pass kernel, us", "sm_clock_mhz": "SM clock, MHz",
              "sm_cycles_k": "SM cycles, thousand", "dram_clock_mhz": "DRAM clock, MHz", "dram_read_mb": "DRAM read, MB", "dram_write_mb": "DRAM write, MB",
              "dram_gb_s": "DRAM throughput, GB/s", "dram_pct_of_422": "DRAM throughput, % of 422 GB/s", "ncu_dram_pct_peak": "ncu DRAM throughput, % of peak",
              "l2_mb": "L2 traffic, MB", "l2_hit_pct": "L2 sector hit, %", "l1_hit_pct": "L1 sector hit, %", "imma_pipe_pct": "IMMA pipe active, %",
              "hmma_pipe_pct": "HMMA pipe active, %", "imma_inst_pct": "IMMA instructions, % of peak", "hmma_inst_pct": "HMMA instructions, % of peak",
              "imma_inst_m": "IMMA instructions, million", "int8_gops": "int8 tensor ops, G", "inst_m": "instructions executed, million",
              "issue_active_pct": "issue slots active, %", "sm_throughput_pct": "SM throughput, % of peak", "warps_active_pct": "achieved warps, % of peak",
              "warps_max_pct": "theoretical warps, %", "latency_cycles": "warp latency per instruction, cycles", "regs": "registers per thread",
              "smem_static_b": "static smem, B", "smem_dyn_b": "dynamic smem, B", "waves_per_sm": "waves per SM", "passes": "replay passes"}


def head_name(c):
    return c + (f" [{HEAD_UNITS[c]}]" if c in HEAD_UNITS else "")


def fmt(v, nd=3):
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:
            return ""
        if abs(v) >= 1e6:
            return f"{v:.6g}"
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.0f}"
    return str(v)


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return sum(xs) / len(xs) if xs else None


def class_rows(krows_by_run):
    out = []
    for run, krows in krows_by_run.items():
        groups = collections.OrderedDict()
        for k, e, r in krows:
            groups.setdefault(k["cls"], []).append((k, e, r))
        for cls, items in groups.items():
            row = {"run": run, "class": cls, "n": len(items), "window": items[0][2]["window"], "function": items[0][2]["function"], "grid": items[0][2]["grid"],
                   "block": items[0][2]["block"], "align": items[0][2]["align"]}
            for c in AVG_COLS:
                row[c] = mean([it[2].get(c) for it in items])
            durs = [it[2]["duration_us"] for it in items if it[2]["duration_us"] is not None]
            row["duration_min_us"] = min(durs) if durs else None
            row["duration_max_us"] = max(durs) if durs else None
            tops = collections.Counter()
            for it in items:
                for s, v in stall_shares(it[0]):
                    tops[s] += v / len(items)
            row["stall_top3"] = "; ".join(f"{s} {100 * v:.0f}%" for s, v in tops.most_common(3))
            out.append(row)
    return out


def sol_rows(krows_by_run, plans):
    """Speed of light per class (kernels with a plan record only carry model inputs; others use the measured DRAM bytes)."""
    out = []
    for run, krows in krows_by_run.items():
        groups = collections.OrderedDict()
        for k, e, r in krows:
            groups.setdefault(k["cls"], []).append((k, e, r))
        plan = plans.get(run)
        for cls, items in groups.items():
            k0, e0, _ = items[0]
            sols = [sol_numbers(k, e) for k, e, _ in items]
            sols = [s for s in sols if s]
            if not sols:
                continue
            dur = mean([it[2]["duration_us"] for it in items])
            floor = mean([s["floor_us"] for s in sols])
            bounds = collections.Counter(s["bound"] for s in sols)
            per_pass = e0["per_pass"] if e0 else None
            useful = e0.get("ops_int8") if e0 else ""
            row = {
                "run": run, "class": cls, "n": len(items), "launches_per_pass": per_pass, "pass_k": e0["pass_k"] if e0 else None,
                "n_tokens": e0["n_tokens"] if e0 else None,
                "duration_us": dur, "nsys_proxy_us": mean([it[2]["nsys_proxy_us"] for it in items]),
                "bound": bounds.most_common(1)[0][0], "floor_us": floor, "pct_of_limit": 100 * floor / dur if dur else None,
                "gap_us": dur - floor if dur is not None else None,
                "ms_per_pass": None if per_pass is None else per_pass * dur / 1e3,
                "gap_ms_per_pass": None if per_pass is None else per_pass * (dur - floor) / 1e3,
                "floor_parts": "; ".join(f"{k}={v:.1f}" for k, v in sorted(sols[0]["parts"].items(), key=lambda x: -x[1])),
                "floor_source": sols[0]["source"],
                "dram_gb_s": mean([it[2]["dram_gb_s"] for it in items]),
                "dram_pct_of_422": mean([it[2]["dram_pct_of_422"] for it in items]),
                "useful_tops": None if (useful in ("", None) or not dur) else useful / (dur * 1e-6) / 1e12,
                "useful_tops_pct": None if (useful in ("", None) or not dur) else 100 * useful / (dur * 1e-6) / 1e12 / cfg.LIM_INT8_TOPS,
                "imma_pipe_pct": mean([it[2]["imma_pipe_pct"] for it in items]),
                "padded_over_useful": None, "padded_expected": None,
                "l2_hit_pct": mean([it[2]["l2_hit_pct"] for it in items]),
                "sm_clock_mhz": mean([it[2]["sm_clock_mhz"] for it in items]),
                "stall_top3": "",
            }
            g = mean([it[2]["int8_gops"] for it in items])
            if g and useful not in ("", None):
                row["padded_over_useful"] = g * 1e9 / useful
            if e0 and e0.get("padded_ops") not in ("", None) and useful not in ("", None, 0):
                row["padded_expected"] = e0["padded_ops"] / useful
            tops = collections.Counter()
            for it in items:
                for s, v in stall_shares(it[0]):
                    tops[s] += v / len(items)
            row["stall_top3"] = "; ".join(f"{s} {100 * v:.0f}%" for s, v in tops.most_common(3))
            if plan and e0:
                row["pass_nsys_ms"] = plan["passes"][e0["pass_k"]]["nsys_ms"]
                row["share_of_pass_pct"] = 100 * row["ms_per_pass"] / row["pass_nsys_ms"] if row["pass_nsys_ms"] else None
                row["case"] = plan["case"]
                row["kind"] = plan["kind"]
            else:
                row["pass_nsys_ms"] = row["share_of_pass_pct"] = row["case"] = row["kind"] = None
            out.append(row)
    return out


SOL_COLS = ["run", "case", "class", "n", "pass_k", "n_tokens", "launches_per_pass", "duration_us", "nsys_proxy_us", "bound", "floor_us", "pct_of_limit", "gap_us",
            "ms_per_pass", "gap_ms_per_pass", "pass_nsys_ms", "share_of_pass_pct", "dram_gb_s", "dram_pct_of_422", "useful_tops", "useful_tops_pct",
            "padded_over_useful", "padded_expected", "imma_pipe_pct", "l2_hit_pct", "sm_clock_mhz", "stall_top3", "floor_parts", "floor_source"]
SOL_HEAD = {"pass_k": "pass of the request", "n_tokens": "N, tokens of the pass", "launches_per_pass": "launches per pass", "duration_us": "measured duration, us",
            "nsys_proxy_us": "nsys duration of the traced pass, us", "floor_us": "time at the limit, us", "pct_of_limit": "% of limit (floor / measured)",
            "gap_us": "gap per launch, us", "ms_per_pass": "measured, ms per pass", "gap_ms_per_pass": "gap, ms per pass",
            "pass_nsys_ms": "GPU time of the traced pass (nsys), ms", "share_of_pass_pct": "share of the pass, %", "dram_gb_s": "DRAM, GB/s",
            "dram_pct_of_422": "DRAM, % of 422 GB/s", "useful_tops": "useful int8, TOP/s", "useful_tops_pct": "useful int8, % of 149.3 TOP/s",
            "padded_over_useful": "executed / useful int8 ops (sm__ops_path_tensor_src_int8)", "padded_expected": "padded / useful ops expected from the tile grid (plan)",
            "imma_pipe_pct": "IMMA pipe active, %", "l2_hit_pct": "L2 hit, %", "sm_clock_mhz": "SM clock, MHz",
            "floor_parts": "limit times, us (largest = binding)", "stall_top3": "top 3 stall reasons, % of the summed stall ratios (selected / not_selected excluded)"}


# ----------------------------------------------------------------------------------------------------------------- output
def write_tsv(path, cols, rows, head=lambda c: c):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow([head(c) for c in cols])
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in cols])


def md_table(cols, rows, head, fmts=None):
    out = ["| " + " | ".join(head(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c), 1 if c in ("duration_us", "floor_us", "gap_us", "nsys_proxy_us") else 2) or "n/a" for c in cols) + " |")
    return "\n".join(out)


def write_md(path, inputs, alignments, krows_by_run, srows, plans, errors):
    L = []
    L.append("# Speed of light per kernel class\n")
    L.append(f"Reports: {', '.join(os.path.basename(i) for i in inputs) or '(none)'}\n")
    L.append("Limits of the RTX 3060 Ti used for the floors (card measurements quoted in the task context, not re-measured here):\n")
    L.append("| limit | value |\n|---|---|")
    L.append(f"| DRAM read bandwidth, GB/s | {cfg.LIM_DRAM_GBPS} |")
    L.append(f"| int8 tensor (mma.sync, dense), TOP/s | {cfg.LIM_INT8_TOPS} |")
    L.append(f"| fp16 tensor with fp32 accumulate, TFLOP/s | {cfg.LIM_FP16_TFLOPS} |")
    L.append(f"| fp32 CUDA cores, TFLOP/s | {cfg.LIM_FP32_TFLOPS} (census_lib.FP32_TFLOPS) |")
    L.append(f"| launch floor, us | {cfg.LAUNCH_FLOOR_US} (shortest kernel of the 13 nsys-traced passes: 0.992 us) |")
    L.append("\nSource: ncu_cfg.py.\n")
    L.append("Time at the limit = max(model bytes / DRAM bandwidth, useful int8 ops / int8 peak, fp16 flops / fp16 peak, fp32 flops / fp32 peak, launch floor). "
             "Model bytes = all tensors of the kernel from/to DRAM (weights + q8_1 input + f32 output for a matmul): cold-cache numbers, which is what "
             "`--cache-control all` measures. The binding limit is the one with the largest time; kernels without a byte model use the measured DRAM bytes. "
             "'Gap' = measured - time at the limit; 'per pass' multiplies it with the launches of the class in one pass (census). "
             "A kernel that cannot reach its floor for structural reasons (e.g. gdn_single: 32 CTAs, serial chunk loop) still shows the gap; read the stall columns.\n")
    L.append("## Alignment of the windows\n")
    arows = []
    for run, al in alignments.items():
        arows.append({"run": run, "planned": al["n_plan"], "observed": al["n_obs"], "matched": al["matched"], "shift": al["shift"], "status": al["status"]})
    L.append(md_table(["run", "planned", "observed", "matched", "shift", "status"], arows,
                      lambda c: {"planned": "planned launches", "observed": "profiled launches", "matched": "launches matching the plan", "shift": "shift, launches"}.get(c, c)))
    L.append("\nSource: analyze_ncu.py align(): observed launch i vs the planned filter stream at index skip + shift + i (function; grid/block and MMQ type/J where the plan is certain).\n")
    L.append("## Planned vs hit classes\n")
    prow = []
    for run, krows in krows_by_run.items():
        plan = plans.get(run)
        hit = collections.Counter(k["cls"] for k, e, r in krows)
        planned = collections.Counter(e["cls"] for e in plan["expected"]) if plan else collections.Counter()
        for cls in list(planned) + [c for c in hit if c not in planned]:
            prow.append({"run": run, "class": cls, "planned": planned.get(cls, 0), "hit": hit.get(cls, 0), "ok": "ok" if planned.get(cls, 0) == hit.get(cls, 0) else "DIFFERS"})
    L.append(md_table(["run", "class", "planned", "hit", "ok"], prow, lambda c: {"planned": "planned launches", "hit": "profiled launches"}.get(c, c)) if prow else "(no kernels)")
    L.append("\nSource: plan/<run>.json (expected) vs the report (class of a launch = plan class when the launch matches its planned kernel, else the kernel name and grid).\n")
    L.append("## Speed of light per class\n")
    cols = ["run", "class", "launches_per_pass", "duration_us", "nsys_proxy_us", "bound", "floor_us", "pct_of_limit", "gap_us", "gap_ms_per_pass", "share_of_pass_pct",
            "dram_pct_of_422", "useful_tops_pct", "imma_pipe_pct", "stall_top3"]
    L.append(md_table(cols, srows, lambda c: SOL_HEAD.get(c, c)) if srows else "(no kernels in the reports)")
    L.append("\nSource: analyze_ncu.py from the ncu raw page (duration, DRAM bytes, pipe utilisation, stall ratios) and plan/<run>.json (launches per pass, model bytes and "
             "ops from census_lib). `ncu / nsys` differences come from the replay (caches flushed, clocks not locked).\n")
    L.append("## Coverage of the passes by the profiled classes\n")
    cov = collections.OrderedDict()
    seen = set()
    for r in srows:
        if r["pass_k"] is None or r["case"] is None:
            continue
        key = (r["case"], r["class"])
        if key in seen:
            continue                       # the same class measured in two runs (std and full set): count it once
        seen.add(key)
        c = cov.setdefault((r["case"], r["pass_k"]), {"case": r["case"], "pass": r["pass_k"], "n": r["n_tokens"], "pass_ms": r["pass_nsys_ms"], "ms": 0.0, "gap": 0.0})
        c["ms"] += r["ms_per_pass"] or 0
        c["gap"] += r["gap_ms_per_pass"] or 0
    crow = []
    for c in cov.values():
        crow.append({"case": c["case"], "pass": f"P{c['pass']}", "n": c["n"], "pass_ms": c["pass_ms"], "ms": c["ms"], "cov": 100 * c["ms"] / c["pass_ms"] if c["pass_ms"] else None,
                     "gap": c["gap"], "gapp": 100 * c["gap"] / c["pass_ms"] if c["pass_ms"] else None})
    L.append(md_table(["case", "pass", "n", "pass_ms", "ms", "cov", "gap", "gapp"], crow,
                      lambda c: {"n": "N, tokens", "pass_ms": "GPU time of the traced pass (nsys), ms", "ms": "profiled classes, ms per pass (launches x measured)",
                                 "cov": "coverage of the pass, %", "gap": "gap to the limits of these classes, ms per pass", "gapp": "gap, % of the pass"}.get(c, c)) if crow else "(none)")
    L.append("\nSource: sum over the classes of this table (each class once per case); pass GPU time = sum of the nsys kernel durations of the traced pass that models it "
             "(plan/<run>.json passes).\n")
    if errors:
        L.append("## Problems\n")
        L.extend(f"- {e}" for e in errors)
        L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def load_plan(run, plan_dir):
    p = os.path.join(plan_dir, run + ".json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def collect_inputs(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith((".ncu-rep", ".ncu-repz")))
        else:
            out.append(p)
    return out


def run_analysis(inputs, ncu, out_dir, plan_dir):
    os.makedirs(out_dir, exist_ok=True)
    errors = []
    krows_by_run, alignments, plans = collections.OrderedDict(), collections.OrderedDict(), {}
    for p in inputs:
        base = re.sub(r"\.(ncu-repz?|csv)$", "", os.path.basename(p))
        run, n = base, 2
        while run in krows_by_run:           # the same report name in two directories
            run, n = f"{base}#{n}", n + 1
        try:
            text = ncu_csv(ncu, p) if p.endswith((".ncu-rep", ".ncu-repz")) else open(p, errors="replace").read()
            kernels = parse_csv(text)
        except Exception as ex:  # a broken report must not stop the others
            errors.append(f"{p}: {type(ex).__name__}: {ex}")
            kernels = []
        plan = load_plan(base, plan_dir)
        plans[run] = plan
        al, krows = analyze_run(run, kernels, plan)
        alignments[run] = al
        krows_by_run[run] = krows
        if not kernels:
            errors.append(f"{run}: no kernels in the report")
    all_rows = [r for krows in krows_by_run.values() for _, _, r in krows]
    write_tsv(os.path.join(out_dir, "kernels.tsv"), KERNEL_COLS, all_rows, head_name)
    crows = class_rows(krows_by_run)
    ccols = ["run", "class", "n", "window", "function", "grid", "block", "align", "duration_min_us", "duration_max_us"] + AVG_COLS + ["stall_top3"]
    write_tsv(os.path.join(out_dir, "classes.tsv"), list(dict.fromkeys(ccols)), crows, head_name)
    srows = sol_rows(krows_by_run, plans)
    write_tsv(os.path.join(out_dir, "sol.tsv"), SOL_COLS, srows, lambda c: c + (f" [{SOL_HEAD[c]}]" if c in SOL_HEAD else ""))
    write_md(os.path.join(out_dir, "sol.txt"), inputs, alignments, krows_by_run, srows, plans, errors)
    arows = [{"run": r, "planned_launches": a["n_plan"], "profiled_launches": a["n_obs"], "matching": a["matched"], "shift": a["shift"], "status": a["status"]} for r, a in alignments.items()]
    write_tsv(os.path.join(out_dir, "alignment.tsv"), ["run", "planned_launches", "profiled_launches", "matching", "shift", "status"], arows)
    return dict(kernels=all_rows, classes=crows, sol=srows, alignments=alignments, errors=errors)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help=".ncu-rep / .ncu-repz files, .csv files (ncu --import --csv --page raw) or directories of reports")
    ap.add_argument("--ncu", default=cfg.LOCAL_NCU or "ncu", help="local ncu command line tool (default: $DMK_NCU, else ncu on PATH)")
    ap.add_argument("--out-dir")
    ap.add_argument("--plan-dir", default=os.path.join(HERE, "plan"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        import selftest_ncu
        sys.exit(selftest_ncu.main())
    inputs = collect_inputs(a.inputs)
    if not inputs:
        ap.error("no input reports")
    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(inputs[0])), "analysis")
    res = run_analysis(inputs, a.ncu, out_dir, a.plan_dir)
    n = len(res["kernels"])
    print(f"{len(inputs)} report(s), {n} profiled launches, {len(res['classes'])} classes -> {out_dir}/{{kernels.tsv,classes.tsv,sol.tsv,sol.txt,alignment.tsv}}")
    for run, al in res["alignments"].items():
        print(f"  {run}: {al['status']}")
    for e in res["errors"]:
        print(f"  note: {e}")


if __name__ == "__main__":
    main()
