#!/usr/bin/env python3
"""Make the derived tables and summary.md of this data set from the raw files. Python 3 standard library only.

    python3 summarize.py --raw raw --tables tables --summary summary.md

Reads raw/{env-public.txt, phases.tsv, smi.csv, smi.fields, mma_*.tsv, membw_*.tsv, cublaslt*.tsv, *.series.tsv}.
Writes one TSV file for each derived table in the tables folder, and the Markdown summary.

cuBLASLt layouts: only a layout whose self-check says OK in every row of cublaslt.tsv ("verified") can be the best
layout of a shape. The layout col32 fails its self-check, so its times are in the tables as "not verified" and
are never used for a best time or a sum.
"""
import argparse
import csv
import datetime as dt
import math
import os
import re
import statistics
import sys

BYTES_PER_WEIGHT = {"Q4_0": 18 / 32, "Q8_0": 34 / 32, "F16": 2.0, "BF16": 2.0, "F32": 4.0}


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def f(x, d=1):
    v = fnum(x)
    return "NA" if v is None else ("%.*f" % (d, v))


def read_tsv(path):
    """Rows of a tool TSV as dicts (header = first line that starts with 'bench'); [] if the file is missing."""
    rows, header = [], None
    if not os.path.isfile(path):
        return rows
    with open(path, newline="") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cells = line.split("\t")
            if cells[0] == "bench":
                header = cells
                continue
            if header and len(cells) == len(header):
                rows.append(dict(zip(header, cells)))
    return rows


def write_tsv(path, header, rows):
    with open(path, "w", newline="") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(c) for c in r) + "\n")


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join("-" if str(c) == "NA" else str(c) for c in r) + " |")
    return "\n".join(out)


def pctl(vals, q):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


# ---------------------------------------------------------------------------------------------- nvidia-smi samples
def utc_offset(env_text):
    m = re.search(r"UTC offset ([+-])(\d\d)(\d\d)", env_text)
    if not m:
        return dt.timezone.utc
    sign = -1 if m.group(1) == "-" else 1
    return dt.timezone(sign * dt.timedelta(hours=int(m.group(2)), minutes=int(m.group(3))))


def read_smi(raw, tz):
    path = os.path.join(raw, "smi.csv")
    samples = []
    if not os.path.isfile(path):
        return samples
    with open(os.path.join(raw, "smi.fields")) as fh:
        fields = fh.read().strip().split(",")
    with open(path, newline="") as fh:
        for row in csv.reader(fh, skipinitialspace=True):
            if len(row) != len(fields) or row[0].strip().lower().startswith("timestamp"):
                continue
            d = dict(zip(fields, row))
            try:
                t = dt.datetime.strptime(d["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=tz).timestamp()
            except ValueError:
                continue
            reasons = d.get("clocks_throttle_reasons.active", d.get("clocks_event_reasons.active", "0x0")).strip()
            try:
                reasons = int(reasons, 16)
            except ValueError:
                reasons = 0
            samples.append({"t": t, "sm": fnum(d.get("clocks.sm")), "mem": fnum(d.get("clocks.mem")), "pw": fnum(d.get("power.draw")),
                            "pwi": fnum(d.get("power.draw.instant")), "temp": fnum(d.get("temperature.gpu")),
                            "pstate": d.get("pstate", "").strip(), "util": fnum(d.get("utilization.gpu")),
                            "vram": fnum(d.get("memory.used")), "reasons": reasons})
    return samples


def read_phases(raw):
    ph = []
    path = os.path.join(raw, "phases.tsv")
    if os.path.isfile(path):
        for line in open(path):
            c = line.rstrip("\n").split("\t")
            if len(c) >= 4 and fnum(c[1]) is not None and fnum(c[2]) is not None:
                ph.append((c[0], float(c[1]), float(c[2]), c[3]))
    return ph


# ---------------------------------------------------------------------------------------------- tables
def build(raw, tables):
    T = {}   # table name -> (header, rows)
    info = {}

    def rd(name):
        return read_tsv(os.path.join(raw, name))

    # 1. tensor-core rates
    runs = [("all", rd("mma_all.tsv")), ("zero", rd("mma_zero.tsv")), ("sustained", rd("mma_sustained.tsv"))]
    rows = []
    for run, rs in runs:
        for r in rs:
            rows.append([run, r["mode"], r["data"], r["ilp"], r["warps_per_sm"], f(r["seconds"], 1), f(r["tops"], 2), f(r["tops_last_half"], 2),
                         f(r["ops_per_clk_sm"], 0), r["nominal_ops_per_clk_sm"], f(r["eff_vs_nominal_pct"], 1), f(r["sm_mhz_mean"], 0),
                         f(r["power_w_mean"], 0), f(r["power_w_max"], 0), f(r["swcap_pct"], 0)])
            if run == "all" and r["mode"] == "s8k32":
                info["peak_short"] = fnum(r["tops"])
            if run == "sustained":
                info["peak_sustained"] = r
            if run == "zero":
                info["zero"] = r
    T["tensor_rates"] = (["run", "mode", "data", "ilp", "warps_per_sm", "seconds", "tops", "tops_last_half", "ops_per_clk_sm",
                          "nominal_ops_per_clk_sm", "pct_of_nominal", "sm_mhz_mean", "power_w_mean", "power_w_max", "swcap_pct"], rows)
    info["mma_all"] = rd("mma_all.tsv")

    # 2. s8k32 sweep
    sw = rd("mma_sweep.tsv")
    T["s8k32_sweep"] = (["ilp", "warps_per_sm", "tops", "ops_per_clk_sm", "sm_mhz_mean", "power_w_mean"],
                        [[r["ilp"], r["warps_per_sm"], f(r["tops"], 2), f(r["ops_per_clk_sm"], 0), f(r["sm_mhz_mean"], 0), f(r["power_w_mean"], 0)]
                         for r in sorted(sw, key=lambda r: (int(r["ilp"]), int(r["warps_per_sm"])))])
    info["sweep"] = sw

    # 3. sustained run, 2 s buckets of the NVML series
    series = os.path.join(raw, "mma_sustained.series.tsv")
    rows = []
    if os.path.isfile(series):
        pts = [x for x in csv.DictReader(open(series), delimiter="\t") if x["phase"].endswith("measure")]
        if pts:
            t0 = float(pts[0]["t_s"])
            buckets = {}
            for x in pts:
                buckets.setdefault(int((float(x["t_s"]) - t0) // 2), []).append(x)
            for k, v in sorted(buckets.items()):
                pw = [fnum(x["power_w"]) for x in v if fnum(x["power_w"]) is not None]
                pwi = [fnum(x["power_inst_w"]) for x in v if fnum(x["power_inst_w"]) is not None]
                rows.append([2 * k, f(statistics.mean(float(x["sm_mhz"]) for x in v), 0), f(statistics.mean(float(x["mem_mhz"]) for x in v), 0),
                             f(statistics.mean(pw) if pw else None, 0), f(statistics.mean(pwi) if pwi else None, 0), f(max(float(x["temp_c"]) for x in v), 0)])
    T["sustained_2s"] = (["t_start_s", "sm_mhz_mean", "mem_mhz_mean", "power_w_mean", "power_inst_w_mean", "temp_c_max"], rows)

    # 4. memory
    mem = rd("membw_all.tsv")
    T["dram_l2_bandwidth"] = (["test", "variant", "buf_mib", "gbps", "gbps_median_launch", "nominal_gbps", "pct_of_nominal", "mem_mhz_mean", "sm_mhz_mean", "power_w_mean"],
                              [[r["test"], r["variant"], f(r["buf_mib"], 2), f(r["gbps"], 1), f(r["gbps_median_launch"], 1), f(r["nominal_gbps"], 1),
                                f(r["eff_vs_nominal_pct"], 1), f(r["mem_mhz_mean"], 0), f(r["sm_mhz_mean"], 0), f(r["power_w_mean"], 0)] for r in mem])
    info["mem"] = mem
    reads = [fnum(r["gbps"]) for r in mem if r["test"] == "dram_read" and fnum(r["gbps"])]
    info["dram_read_gbps"] = max(reads) if reads else None
    ms = rd("membw_sweep.tsv")
    T["dram_read_sweep"] = (["blocks_per_sm", "unroll", "gbps"],
                            [[r["blocks_per_sm"], r["unroll"], f(r["gbps"], 1)] for r in sorted(ms, key=lambda r: (int(r["blocks_per_sm"]), int(r["unroll"])))])
    info["mem_sweep"] = ms

    # 5. cuBLASLt
    lt = rd("cublaslt.tsv")
    info["lt"] = lt
    checks = {}
    for r in lt:
        checks.setdefault(r["variant"], set()).add(r["check"])
    verified = {v for v, c in checks.items() if c == {"OK"}}
    variants = sorted(checks)
    info["lt_checks"] = {v: sorted(c) for v, c in checks.items()}
    info["verified"] = sorted(verified)
    ok = [r for r in lt if r["status"] == "OK" and fnum(r["time_us"])]
    peak = info.get("peak_short")
    key = lambda r: (r["label"], int(r["M"]), int(r["N"]), int(r["K"]))
    by_shape = {}
    for r in ok:
        by_shape.setdefault(key(r), {})[r["variant"]] = r

    def best_of(cands, allowed):
        c = [r for v, r in cands.items() if v in allowed]
        return min(c, key=lambda r: fnum(r["time_us"])) if c else None

    unver = [v for v in variants if v not in verified]
    hdr = ["label", "M", "N", "K", "count", "wtype", "best_verified_variant", "time_us", "tops", "pct_of_mma_peak", "a_read_gbps", "sm_mhz_mean",
           "pct_of_nominal_at_run_clock", "algo"]
    hdr += ["us_" + v + ("" if v in verified else "_not_verified") for v in variants]
    hdr += ["best_incl_unverified_variant", "best_incl_unverified_us"]
    rows, best_ver = [], {}
    for k in sorted(by_shape, key=lambda k: (k[0], k[2])):
        cands = by_shape[k]
        b = best_of(cands, verified)
        a = best_of(cands, set(variants))
        if b is None:
            continue
        best_ver[k] = b
        rows.append([k[0], k[1], k[2], k[3], b["count"], b["wtype"], b["variant"], f(b["time_us"], 2), f(b["tops"], 2),
                     f(100 * fnum(b["tops"]) / peak, 1) if peak else "NA", f(b["a_read_gbps"], 0), f(b["sm_mhz_mean"], 0),
                     f(b["eff_vs_nominal_pct"], 1), b["algo"]]
                    + [f(cands[v]["time_us"], 2) if v in cands else "NA" for v in variants] + [a["variant"], f(a["time_us"], 2)])
    T["cublaslt_best_verified"] = (hdr, rows)
    info["lt_best"] = best_ver
    info["lt_unver"] = unver

    # model-level sums
    dram = info["dram_read_gbps"]
    classes = sorted({(k[0], k[1], k[3], b["wtype"], int(b["count"])) for k, b in best_ver.items() if b["count"].isdigit() and int(b["count"]) > 0})
    ns = sorted({k[2] for k, b in best_ver.items() if b["count"].isdigit() and int(b["count"]) > 0})
    allow_sets = {"verified": verified, "col": {"col"}, "all": set(variants)}
    rows, sums = [], {}
    for n in ns:
        res = {}
        for name, allowed in allow_sets.items():
            tot = tot_nohead = 0.0
            ops = 0.0
            complete = True
            nclass = 0
            for (label, m, kk, wt, cnt) in classes:
                cands = by_shape.get((label, m, n, kk))
                b = best_of(cands, allowed) if cands else None
                if b is None:
                    if label != "output_head":
                        complete = False
                    else:
                        res.setdefault(name, {})["head_missing"] = True
                    continue
                t = cnt * fnum(b["time_us"]) / 1000.0
                tot += t
                if label != "output_head":
                    tot_nohead += t
                    ops += cnt * 2.0 * m * n * kk
            res.setdefault(name, {}).update(total=tot, nohead=tot_nohead if complete else None, ops=ops)
            if res[name].get("head_missing") or not complete:
                res[name]["total"] = None
        nb_all = sum(m * kk * cnt * BYTES_PER_WEIGHT.get(wt, 1.0) for (label, m, kk, wt, cnt) in classes)
        nb_nohead = sum(m * kk * cnt * BYTES_PER_WEIGHT.get(wt, 1.0) for (label, m, kk, wt, cnt) in classes if label != "output_head")
        floor_all = nb_all / (dram * 1e9) * 1000 if dram else None
        floor_nohead = nb_nohead / (dram * 1e9) * 1000 if dram else None
        v = res["verified"]
        rows.append([n, f(v["total"], 2), f(v["nohead"], 2), f(res["col"]["nohead"], 2), f(res["all"]["nohead"], 2), f(floor_nohead, 2), f(floor_all, 2),
                     f(v["nohead"] / floor_nohead, 2) if v["nohead"] and floor_nohead else "NA",
                     f(v["ops"] / (v["nohead"] * 1e-3) / 1e12, 1) if v["nohead"] else "NA"])
        sums[n] = dict(total=v["total"], nohead=v["nohead"], col=res["col"]["nohead"], allv=res["all"]["nohead"], floor_nohead=floor_nohead,
                       floor_all=floor_all, tops=(v["ops"] / (v["nohead"] * 1e-3) / 1e12) if v["nohead"] else None)
    T["cublaslt_model_sums"] = (["n_tokens", "ms_with_head", "ms_without_head", "ms_without_head_col_only", "ms_without_head_incl_not_verified",
                                 "gguf_dram_floor_without_head_ms", "gguf_dram_floor_with_head_ms", "ratio_to_floor_without_head", "tops_without_head"], rows)
    info["sums"] = sums

    # 6. clocks and power per phase from nvidia-smi
    env = os.path.join(raw, "env-public.txt")
    tz = utc_offset(open(env).read() if os.path.isfile(env) else "")
    samples = read_smi(raw, tz)
    rows = []
    for name, t0, t1, rc in read_phases(raw):
        w = [s for s in samples if t0 <= s["t"] <= t1 and (s["util"] or 0) >= 50]
        ps = {}
        for s in w:
            ps[s["pstate"]] = ps.get(s["pstate"], 0) + 1
        cap = 100.0 * sum(1 for s in w if s["reasons"] & 0x4) / len(w) if w else None
        vals = lambda k: [s[k] for s in w if s[k] is not None]
        rows.append([name, f(t1 - t0, 0), rc, len(w), f(pctl(vals("sm"), .5), 0), f(pctl(vals("sm"), .05), 0), f(max(vals("sm")) if w else None, 0),
                     f(pctl(vals("mem"), .5), 0), f(statistics.mean(vals("pw")) if vals("pw") else None, 0), f(max(vals("pw")) if vals("pw") else None, 0),
                     f(statistics.mean(vals("pwi")) if vals("pwi") else None, 0), f(max(vals("pwi")) if vals("pwi") else None, 0),
                     f(max(vals("temp")) if w else None, 0), "/".join("%s:%d" % kv for kv in sorted(ps.items(), key=lambda kv: -kv[1])), f(cap, 0),
                     f(max(vals("vram")) if w else None, 0)])
    T["phase_clocks_power"] = (["phase", "seconds", "exit_code", "busy_samples", "sm_mhz_median", "sm_mhz_p05", "sm_mhz_max", "mem_mhz_median",
                                "power_w_mean", "power_w_max", "power_inst_w_mean", "power_inst_w_max", "temp_c_max", "pstates", "swcap_pct", "vram_used_mib_max"], rows)
    idle = [s for s in samples if all(not (t0 <= s["t"] <= t1) for _, t0, t1, _ in read_phases(raw))]
    info["idle"] = idle
    info["smi_n"] = len(samples)

    os.makedirs(tables, exist_ok=True)
    for name, (hdr, rws) in T.items():
        write_tsv(os.path.join(tables, name + ".tsv"), hdr, rws)
    return T, info


# ---------------------------------------------------------------------------------------------- summary.md
def code(x):
    return "`%s`" % x


def summary(T, info):
    o = []
    o.append("# Summary of the limits of the GPU\n")
    o.append("This file shows the numbers of the run on 2026-10-30. The script `summarize.py` made it from the files in `raw/`. "
             "The folder `tables/` has the same numbers as TSV files. The README of the data set gives the columns and the method.\n")

    mma = info["mma_all"]
    o.append("## Tensor core rate\n")
    o.append("Each mode is one `mma.sync` instruction in a loop. The loop has no access to memory. Each mode runs for 2 s. "
             "The nominal rate is the highest rate that you can calculate from the architecture of the GPU. We did not measure it.\n")
    o.append(md(["Mode", "TOPS", "Rate (operations for each clock for each SM)", "Nominal rate (operations for each clock for each SM)",
                 "Percent of the nominal rate", "SM clock (MHz)", "Power (W)"],
                [[code(r["mode"]), f(r["tops"], 2), f(r["ops_per_clk_sm"], 0), r["nominal_ops_per_clk_sm"], f(r["eff_vs_nominal_pct"], 1),
                  f(r["sm_mhz_mean"], 0), f(r["power_w_mean"], 0)] for r in mma]))
    o.append("")
    o.append("The unit TOPS is 10^12 operations each second. A multiply-add counts as 2 operations. "
             "The modes `f16f32`, `f16f16` and `bf16f32` use floating point values. For these modes the column has the unit TFLOPS.\n")
    s = info.get("peak_sustained")
    z = info.get("zero")
    if s:
        o.append("### Run of 20 s\n")
        o.append("The mode `s8k32` runs for 20 s. The rate is %s TOPS for the full time and %s TOPS in the last 10 s. "
                 "The mean SM clock is %s MHz. The power of the GPU is %s W (mean) and %s W (highest). "
                 "The power cap is active in %s percent of the samples. The highest temperature is %s °C.\n"
                 % (f(s["tops"], 2), f(s["tops_last_half"], 2), f(s["sm_mhz_mean"], 0), f(s["power_w_mean"], 0), f(s["power_w_max"], 0),
                    f(s["swcap_pct"], 0), f(s["temp_c_max"], 0)))
    if z:
        r0 = [r for r in mma if r["mode"] == "s8k32"][0]
        o.append("### Operands that are all zero\n")
        o.append("With operands that are all zero, the rate is %s TOPS at %s W. With random operands, the rate is %s TOPS at %s W. "
                 "The power changes with the data.\n" % (f(z["tops"], 2), f(z["power_w_mean"], 0), f(r0["tops"], 2), f(r0["power_w_mean"], 0)))
    sw = info["sweep"]
    if sw:
        ilps = sorted({int(r["ilp"]) for r in sw})
        warps = sorted({int(r["warps_per_sm"]) for r in sw})
        tab = {(int(r["ilp"]), int(r["warps_per_sm"])): r for r in sw}
        o.append("### Number of warps\n")
        o.append("The table shows the rate of `s8k32` in TOPS. A row is the value of `ilp`. This is the number of mma instructions that a warp keeps in flight. "
                 "These instructions do not use the results of each other. A column is the number of warps of each SM. "
                 "The sign - shows a pair that the run did not use.\n")
        o.append(md(["`ilp` \\ warps"] + [str(w) for w in warps], [[str(i)] + [f(tab[(i, w)]["tops"], 1) if (i, w) in tab else "-" for w in warps] for i in ilps]))
        o.append("")

    mem = info["mem"]
    o.append("## Memory bandwidth\n")
    o.append("The unit GB/s is 10^9 bytes each second. A copy counts the bytes that it reads and the bytes that it writes. "
             "The nominal rate of the DRAM is the memory clock times 2 times the width of the memory bus.\n")
    dram = [r for r in mem if r["test"].startswith("dram")]
    o.append(md(["Test", "Method", "Buffer (MiB)", "GB/s", "Nominal rate (GB/s)", "Percent of the nominal rate", "Memory clock (MHz)"],
                [[code(r["test"]), code(r["variant"]), f(r["buf_mib"], 0), f(r["gbps"], 1), f(r["nominal_gbps"], 1), f(r["eff_vs_nominal_pct"], 1), f(r["mem_mhz_mean"], 0)] for r in dram]))
    o.append("")
    l2 = [r for r in mem if r["test"] == "l2_read"]
    if l2:
        o.append("### L2 cache\n")
        o.append("The program reads a buffer many times with `ld.global.cg`. The L2 cache of this GPU holds 3 MiB. "
                 "A buffer of 12 MiB or more has the rate of the DRAM.\n")
        o.append(md(["Buffer (MiB)", "GB/s"], [[f(r["buf_mib"], 2), f(r["gbps"], 0)] for r in l2]))
        o.append("")
    ms = info["mem_sweep"]
    if ms:
        bps = sorted({int(r["blocks_per_sm"]) for r in ms})
        un = sorted({int(r["unroll"]) for r in ms})
        tab = {(int(r["blocks_per_sm"]), int(r["unroll"])): r for r in ms}
        o.append("### Number of loads in flight\n")
        o.append("The table shows the DRAM read rate in GB/s. A row is the number of thread blocks of each SM. A column is the number of loads in flight for each thread.\n")
        o.append(md(["Blocks \\ loads"] + [str(u) for u in un], [[str(b)] + [f(tab[(b, u)]["gbps"], 0) if (b, u) in tab else "-" for u in un] for b in bps]))
        o.append("")

    o.append("## cuBLASLt int8 GEMM\n")
    o.append("The test multiplies a matrix of int8 weights with a matrix of int8 inputs. The weights have the shapes of the model. "
             "`N` is the number of tokens.\n")
    o.append("Each weight has 1 byte, and the GEMM has no scales. Q4_0 has 0.5625 bytes for each weight, and Q8_0 has 1.0625 bytes. "
             "Thus the rates are a reference. They are not the rates of MMQ.\n")
    chk = info["lt_checks"]
    o.append("### Layouts and check\n")
    txt = ("The program checks each layout on a problem of 64 by 48 by 128 against the result of the CPU. "
           "Only a layout that has the correct result can give a best time. ")
    for v in info["lt_unver"]:
        txt += ("The layout `%s` does not have the correct result: the check gives `%s`. Thus this file does not use its times. "
                % (v, ", ".join(chk[v])))
    if info["lt_unver"]:
        txt += "The README gives the cause."
    o.append(txt.rstrip() + "\n")
    st = {}
    for r in info["lt"]:
        st.setdefault(r["variant"], {}).setdefault(r["status"], 0)
        st[r["variant"]][r["status"]] += 1
    o.append(md(["Layout", "Check", "Rows with a time", "Rows that the memory limit removes", "Used for the best time"],
                [[code(v), code(", ".join(chk[v])), st[v].get("OK", 0), st[v].get("SKIP_MEM", 0), "yes" if v in info["verified"] else "no"] for v in sorted(chk)]))
    o.append("")
    o.append("### Best time of each shape\n")
    o.append("Only the layouts that have the correct result are in this table. The column Percent of the highest rate is the rate of the shape divided by the `s8k32` rate of the first table (%s TOPS). "
             "The column Percent of the nominal rate uses the mean SM clock of the run of the shape. "
             "The column Read rate is 1 byte for each weight divided by the time.\n" % f(info.get("peak_short"), 2))
    best = info["lt_best"]
    rows = []
    for k in sorted(best, key=lambda k: (k[0], k[2])):
        b = best[k]
        rows.append([code(k[0]), k[1], k[2], k[3], code(b["variant"]), f(b["time_us"], 1), f(b["tops"], 1), f(100 * fnum(b["tops"]) / info["peak_short"], 1),
                     f(b["sm_mhz_mean"], 0), f(b["eff_vs_nominal_pct"], 1), f(b["a_read_gbps"], 0)])
    o.append(md(["Label", "`M`", "`N`", "`K`", "Layout", "Time (µs)", "TOPS", "Percent of the highest rate", "SM clock (MHz)", "Percent of the nominal rate", "Read rate (GB/s)"], rows))
    o.append("")
    o.append("The output head with `N` = 2048 has no row. This shape uses more than 1800 MiB of memory, thus the test does not run it.\n")

    o.append("### Sum for the model\n")
    o.append("For each `N`, the table gives the sum of one time for each group of weights. All weights of a group have the same shape. "
             "The time of a group is the best time of one GEMM, multiplied by the number of weights in the group. "
             "The sum has no attention, no recurrent layer and no other operation.\n")
    o.append("The weight-read floor is the number of GGUF weight bytes divided by the best DRAM read rate (%s GB/s). "
             "The column Ratio is the sum without the output head divided by the weight-read floor without the output head.\n" % f(info["dram_read_gbps"], 1))
    sums = info["sums"]
    rows = []
    for n in sorted(sums, key=int):
        v = sums[n]
        rows.append([n, f(v["total"], 2), f(v["nohead"], 2), f(v["col"], 2), f(v["allv"], 2), f(v["floor_nohead"], 2),
                     f(v["nohead"] / v["floor_nohead"], 2) if v["nohead"] and v["floor_nohead"] else "-", f(v["tops"], 1)])
    o.append(md(["`N`", "Sum with the output head (ms)", "Sum without the output head (ms)", "Same, layout `col` only (ms)", "Same, with `col32` (not correct) (ms)",
                 "Weight-read floor without the output head (ms)", "Ratio", "TOPS"], rows))
    o.append("")
    o.append("The weight-read floor with the output head is %s ms.\n" % f(sums[min(sums, key=int)]["floor_all"], 2))

    ph = T["phase_clocks_power"][1]
    o.append("## Clocks and power of each phase\n")
    o.append("`nvidia-smi` made %d samples, 10 each second. The table uses only the samples with a GPU use of 50 percent or more. "
             "The column Power cap is the percent of these samples in which the power cap is active.\n" % info["smi_n"])
    o.append(md(["Phase", "Seconds", "SM clock median (MHz)", "SM clock highest (MHz)", "Memory clock (MHz)", "Power mean (W)", "Power highest (W)", "Temperature highest (°C)",
                 "P-states", "Power cap (percent)"],
                [[code(p[0]), p[1], p[4], p[6], p[7], p[8], p[9], p[12], code(p[13]), p[14]] for p in ph]))
    o.append("")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--summary", default="summary.md")
    a = ap.parse_args()
    if not os.path.isdir(a.raw):
        sys.exit("not a directory: " + a.raw)
    T, info = build(a.raw, a.tables)
    with open(a.summary, "w") as fh:
        fh.write(summary(T, info))
    print("wrote %d tables to %s and %s" % (len(T), a.tables, a.summary))


if __name__ == "__main__":
    main()
