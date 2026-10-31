#!/usr/bin/env python3
"""trace_census.py - kernel / pass / MMQ-launch census of the three old nsys captures (long, medium, short).

Usage:   python3 trace_census.py [--in DIR] [--out DIR] [--old-export DIR]
         (after `bash trace_export.sh`; CPU only, stdlib only)

Inputs:
    --in DIR          the folder with {long,medium,short}.sqlite: the exports of the .nsys-rep captures made by trace_export.sh
                      (default: $DMK_LEGACY_TRACES). The census used nsys 2026.3.2 exports.
    --old-export DIR  optional cross-check only (opened read-only): a second set of {long,medium,short}.sqlite, exported earlier
                      from the same captures. The program prints whether the kernel and memcpy counts and the kernel time are equal.
Outputs (into --out; default: raw/traces of the census data set, $DMK_CENSUS_DATA):
    trace-kernels.tsv        one row per (trace, kernel_short, grid, block, smem, regs)
    trace-passes.tsv         one row per model pass (row_kind=pass), per request (request), per trace (trace_total)
    trace-mmq-launches.tsv   ordered main mul_mat_q / mul_mat_vec_q launches of one representative pass per
                             distinct (trace, pass_kind, n_tokens)
A human-readable summary (validation results, kernel category shares, tables of the pass classes) is printed to stdout.

Pass segmentation rule (single compute stream, no CUDA graphs in the captured window):
    every llama_decode starts with a burst of host-to-device input copies (cudaMemcpyAsync + sync, 9 or 10 copies:
    3 x n_tokens*10240 B, small index tensors, the KQ mask, [out_ids]).  The first kernel that starts after a copy of
    that burst has finished is the first kernel of the pass (always rms_norm_f32<1024,...> with grid.x == n_tokens).
    A pass = all kernels from that anchor up to (not including) the next anchor.  Validation checks are printed
    (every pass: anchor = rms_norm_f32<1024> with grid.x == first-copy-bytes/10240; 24 GDN + 8 flash-attention main launches;
    65 (suffix) / 63 (prefix) rms_norm_f32<1024> launches; cudaLaunchKernel API calls == kernels; one kernel stream; equal
    (kind, n_tokens) passes have identical kernel-name sequences).

Conventions:
    * times: *_ns raw nsys timestamps, *_us / *_ms as named; wall = first kernel start .. last kernel end of the pass;
      gpu_busy = union of kernel intervals; gap_total = wall - busy (in-pass gaps between consecutive kernels, includes the
      device-to-device memcpy records that sit between kernels of some passes, see gap_total_excl_d2d_ms).
    * pass kinds: prefix-large (no output head, n_tokens == 2048 == -ub), prefix-small (no output head, n_tokens < 2048),
      suffix (output head kernel + logits copy present).
    * host-side records (cudaLaunchKernel/cudaMemcpyAsync/cudaStreamSynchronize, GPU memcpy records, sync records, CUPTI
      overhead records) belong to the pass whose first H2D input copy API call started last before them.
    * request / trace_total rows reuse the pass columns: n_tokens = sum over passes; gap_n_*, gap_median_us, gap_max_us and
      gap buckets count in-pass gaps only; gap_total_ms = wall - busy including inter-pass idle time
      (gap_intra_pass_ms + gap_inter_pass_ms); n_*, *_ms are sums.
    * MMQ grid note: in this llama.cpp build quantize_mmq_q8_1 uses grid = (n_rows, ceil(K_padded/512), n_channels) and
      quantize_q8_1 (MMVQ) grid = (ceil(K_padded/256), n_rows, n_channels); mul_mat_q grid.x is either the tile count (no
      fix-up) or n_SM=38 (stream-k, followed by mul_mat_q_stream_k_fixup).
    * kernel categories follow the groups of the first analysis of these captures (unary_op_kernel sigmoid/softplus counted with
      gated activations, quantize_q8_1 with quantize to int8, rope_multi/other in "other"); the shares reproduce the table of that analysis.
"""
import argparse
import bisect
import collections
import os
import re
import sqlite3
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import census_lib  # noqa: E402  (data_dir(): the default output folder)

TRACES = ["long", "medium", "short"]
OUT = None        # output folder, set by main()
OLD_DIR = None    # optional folder of an older export, set by main()

# client order of the captures: cases of the size class in bench/cases.json order, cycling i % n_cases
CASES = {
    "long": ["long00-storage", "long01-memory", "long02-network", "long03-none"],
    "medium": ["medium00-storage", "medium01-memory", "medium02-network", "medium03-none"],
    "short": ["triage%02d" % i for i in range(12)],
}

UB = 2048                 # -ub 2048
N_VOCAB = 248320          # output head rows (tied token_embd, Q8_0)
EMBD_BYTES_PER_TOKEN = 2560 * 4
LOGITS_BYTES = N_VOCAB * 4

GGML_TYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
             12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 24: "I8", 26: "I32", 30: "BF16"}
DS_LAYOUT = {0: "D4", 1: "DS4", 2: "D2S6"}
PREC = {0: "default", 10: "F32", 15: "BF16", 20: "F16", 30: "Q8", 40: "Q4"}

CATS = ["MMQ matmul (incl. stream-k fix-up)", "GDN recurrence", "gated activations", "flash attention",
        "quantize to int8", "copies/concat/get_rows", "RMS norm", "conv1d", "element-wise", "mat-vec (MMVQ)", "other"]
CAT_KEY = ["mmq", "gdn", "gated_act", "flash", "quant", "copy", "rms_norm", "conv1d", "elementwise", "mmvq", "other"]
BASE_CAT = {
    "mul_mat_q": 0, "mul_mat_q_stream_k_fixup": 0,
    "gated_delta_net_cuda": 1, "ggml_cuda_gated_delta_net_mma::gdn_single": 1,
    "unary_gated_op_kernel": 2, "unary_op_kernel": 2,   # sigmoid/softplus gate math of the Gated DeltaNet layers counted with
                                                        # the gated activations (reproduces the shares of the first analysis)
    "flash_attn_ext_f16": 3, "flash_attn_mask_to_KV_max": 3, "flash_attn_stream_k_fixup_uniform": 3,
    "flash_attn_stream_k_fixup_general": 3,
    "quantize_mmq_q8_1": 4, "quantize_q8_1": 4,
    "cpy_scalar": 5, "concat_non_cont": 5, "k_get_rows_float": 5, "k_get_rows_float_vec": 5, "k_set_rows": 5,
    "rms_norm_f32": 6,
    "ssm_conv_f32": 7, "ssm_conv_long_token_f32": 7,
    "k_bin_bcast": 8, "scale_f32": 8,
    "mul_mat_vec_q": 9,
}

# ---------------------------------------------------------------------------------------------------------------
# kernel name parsing
# ---------------------------------------------------------------------------------------------------------------

def split_top(s, sep=","):
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


def split_name(demangled):
    """-> (base, [template args]) of a demangled CUDA kernel name"""
    s = demangled
    if s.startswith("void "):
        s = s[5:]
    depth, lt, paren = 0, -1, -1
    for i, ch in enumerate(s):
        if ch == "<":
            if depth == 0 and lt < 0:
                lt = i
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            paren = i
            break
    if lt >= 0 and (paren < 0 or lt < paren):
        base = s[:lt]
        # find the matching '>'
        d = 0
        for j in range(lt, len(s)):
            if s[j] == "<":
                d += 1
            elif s[j] == ">":
                d -= 1
                if d == 0:
                    return base, split_top(s[lt + 1:j])
        return base, []
    return (s[:paren] if paren >= 0 else s), []


def norm_arg(a):
    m = re.fullmatch(r"\(ggml_type\)(\d+)", a)
    if m:
        return GGML_TYPE.get(int(m.group(1)), "type%s" % m.group(1))
    m = re.fullmatch(r"\(mmq_q8_1_ds_layout\)(\d+)", a)
    if m:
        return DS_LAYOUT.get(int(m.group(1)), "ds%s" % m.group(1))
    m = re.fullmatch(r"\(ggml_prec\)(\d+)", a)
    if m:
        return "prec" + PREC.get(int(m.group(1)), m.group(1))
    m = re.fullmatch(r"\(bool\)([01])", a)
    if m:
        return "true" if m.group(1) == "1" else "false"
    m = re.fullmatch(r"\((?:unsigned )?(?:int|long)\)(-?\d+)", a)
    if m:
        return m.group(1)
    m = re.fullmatch(r"\(unsigned long\)(\d+)", a)
    if m:
        return m.group(1)
    a = re.sub(r"&op_", "", a)
    a = re.sub(r"&cpy_1_scalar", "cpy_1_scalar", a)
    return a.replace(", ", ",")


def short_name(base, args):
    """short key, e.g. mul_mat_q<Q8_0,64,false>"""
    b = base.split("::")[-1]
    raw = args
    n = [norm_arg(a) for a in raw]
    if base in ("mul_mat_q",):           # <type, J(mmq_x), need_check(fallback), prec_src1, has_fusion>
        s = "mul_mat_q<%s,%s,%s" % (n[0], n[1], n[2])
        if len(n) > 4 and n[4] == "true":
            s += ",fusion"
        return s + ">"
    if base == "mul_mat_q_stream_k_fixup":   # <type, J, fallback, prec_src1>
        return "mul_mat_q_stream_k_fixup<%s,%s,%s>" % (n[0], n[1], n[2])
    if base == "mul_mat_vec_q":          # <type, ncols_dst, has_fusion, small_k, halve_iters>
        s = "mul_mat_vec_q<%s,%s" % (n[0], n[1])
        for label, v in zip(("fusion", "small_k", "halve_iters"), n[2:5]):
            if v == "true":
                s += "," + label
        return s + ">"
    if base == "quantize_mmq_q8_1":      # <ds_layout, scatter>
        s = "quantize_mmq_q8_1<%s" % n[0]
        if len(n) > 1 and n[1] == "true":
            s += ",scatter"
        return s + ">"
    if base == "flash_attn_ext_f16":     # <DKQ, DV, ncols1, ncols2, softcap, V_is_K_view, sparse>
        s = "flash_attn_ext_f16<%s,%s,%s,%s" % tuple(n[:4])
        for label, v in zip(("softcap", "V_is_K_view", "sparse"), n[4:7]):
            if v == "true":
                s += "," + label
        return s + ">"
    if not n:
        return b
    return "%s<%s>" % (b, ",".join(n))


def category_of(base):
    return BASE_CAT.get(base, len(CATS) - 1)


# ---------------------------------------------------------------------------------------------------------------
# load one trace
# ---------------------------------------------------------------------------------------------------------------

class Trace:
    pass


def load_trace(name, path):
    db = sqlite3.connect(str(path))
    t = Trace()
    t.name = name
    sid = dict(db.execute("select id, value from StringIds"))
    rows = db.execute(
        "select start, end, streamId, correlationId, demangledName, gridX, gridY, gridZ, blockX, blockY, blockZ, "
        "registersPerThread, staticSharedMemory, dynamicSharedMemory, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL "
        "order by start, end").fetchall()
    parsed = {}
    K = []
    for r in rows:
        dn = r[4]
        if dn not in parsed:
            full = sid[dn]
            base, args = split_name(full)
            parsed[dn] = (full, base, short_name(base, args), category_of(base))
        full, base, short, cat = parsed[dn]
        K.append({"start": r[0], "end": r[1], "stream": r[2], "corr": r[3], "full": full, "base": base, "short": short,
                  "cat": cat, "gx": r[5], "gy": r[6], "gz": r[7], "bx": r[8], "by": r[9], "bz": r[10], "regs": r[11],
                  "ssmem": r[12], "dsmem": r[13], "gnode": r[14]})
    t.K = K
    t.api = [dict(zip(("start", "end", "name", "corr", "tid"), (r[0], r[1], sid[r[2]], r[3], r[4]))) for r in db.execute(
        "select start, end, nameId, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME order by start")]
    t.mc = [dict(zip(("start", "end", "stream", "kind", "bytes", "corr"), r)) for r in db.execute(
        "select start, end, streamId, copyKind, bytes, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY order by start")]
    t.sync = [dict(zip(("start", "end", "stream", "corr"), r)) for r in db.execute(
        "select start, end, streamId, correlationId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION order by start")]
    t.ovh = [dict(zip(("start", "end", "type", "corr"), r)) for r in db.execute(
        "select start, end, overheadType, correlationId from CUPTI_ACTIVITY_KIND_OVERHEAD order by start")]
    t.graph_nodes = db.execute("select count(*), min(start), max(end) from CUDA_GRAPH_NODE_EVENTS").fetchone()
    t.graph_node_ids = [r[0] for r in db.execute("select graphNodeId from CUDA_GRAPH_NODE_EVENTS")]
    t.graph_names = db.execute("select s.value, count(*) from CUDA_GRAPH_NODE_EVENTS g join StringIds s on s.id=g.nameId "
                               "group by 1").fetchall()
    t.dev = db.execute("select name, smCount, memoryBandwidth from TARGET_INFO_GPU").fetchone()
    t.session = db.execute("select utcTime from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    t.procs = db.execute("select pid, name from PROCESSES").fetchall()
    t.capture = dict(db.execute("select name, value from META_DATA_CAPTURE"))
    db.close()
    return t


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------

def median(xs):
    return statistics.median(xs) if xs else None


def union_ns(iv):
    """total length of union of (start,end) intervals"""
    tot, cs, ce = 0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        tot += ce - cs
    return tot


def gaps_of(ks):
    """gaps (ns, clamped at 0) between consecutive kernels (sorted by start); returns list of (gap, prev_idx_in_list)"""
    out, run_end = [], None
    for i, k in enumerate(ks):
        if run_end is not None:
            out.append((max(0, k["start"] - run_end), i - 1))
        run_end = k["end"] if run_end is None else max(run_end, k["end"])
    return out


BUCKETS = [("lt1us", 0, 1000), ("1_5us", 1000, 5000), ("5_20us", 5000, 20000), ("20_100us", 20000, 100000),
           ("gt100us", 100000, 1 << 62)]


def bucket_stats(gaps):
    res = {}
    for key, lo, hi in BUCKETS:
        sel = [g for g in gaps if lo <= g < hi]
        res[key] = (len(sel), sum(sel))
    return res


def ms(ns):
    return None if ns is None else ns / 1e6


def us(ns):
    return None if ns is None else ns / 1e3


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:
            return ""
        return ("%.6f" % v).rstrip("0").rstrip(".") if abs(v) < 1e6 else "%.1f" % v
    return str(v)


def write_tsv(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            vals = []
            for h in header:
                v = fmt(r.get(h))
                assert "\t" not in v and "\n" not in v, (h, v)
                vals.append(v)
            f.write("\t".join(vals) + "\n")


# ---------------------------------------------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------------------------------------------

def segment(t, problems):
    K = t.K
    kstarts = [k["start"] for k in K]
    api_by_corr = {a["corr"]: a for a in t.api}
    t.api_by_corr = api_by_corr
    # H2D bursts
    bursts = collections.OrderedDict()
    for m in t.mc:
        if m["kind"] != 1:
            continue
        a = bisect.bisect_left(kstarts, m["end"])
        bursts.setdefault(a, []).append(m)
    anchors = sorted(bursts)
    if not anchors or anchors[0] != 0:
        problems.append("%s: first kernel is not a pass anchor (anchors[0]=%s)" % (t.name, anchors[:1]))
    passes = []
    for pi, a in enumerate(anchors):
        b = anchors[pi + 1] if pi + 1 < len(anchors) else len(K)
        P = {"idx": pi, "k0": a, "k1": b, "ks": K[a:b], "h2d": sorted(bursts[a], key=lambda m: m["start"])}
        passes.append(P)
    # CPU-side window starts: start of the API call that issued the first copy of the burst
    for P in passes:
        first = P["h2d"][0]
        P["win_start"] = api_by_corr[first["corr"]]["start"] if first["corr"] in api_by_corr else first["start"]
    wstarts = [P["win_start"] for P in passes]
    t.wstarts = wstarts
    # assign records to passes
    for P in passes:
        P["api"], P["mc"], P["sync"], P["ovh"] = [], [], [], []

    def widx(ts):
        return max(0, bisect.bisect_right(wstarts, ts) - 1)
    for a in t.api:
        passes[widx(a["start"])]["api"].append(a)
    for m in t.mc:
        a = api_by_corr.get(m["corr"])
        passes[widx(a["start"] if a else m["start"])]["mc"].append(m)
    for s in t.sync:
        a = api_by_corr.get(s["corr"])
        passes[widx(a["start"] if a else s["start"])]["sync"].append(s)
    for o in t.ovh:
        a = api_by_corr.get(o["corr"]) if o["corr"] else None
        passes[widx(a["start"] if a else o["start"])]["ovh"].append(o)
    return passes


def classify(P, problems, tname):
    ks = P["ks"]
    first = ks[0]
    n_tokens = first["gx"]
    h2d = P["h2d"]
    ev = []
    if not (first["base"] == "rms_norm_f32" and first["bx"] == 1024):
        problems.append("%s pass %d: anchor kernel is %s block %d" % (tname, P["idx"], first["short"], first["bx"]))
    if h2d[0]["bytes"] != n_tokens * EMBD_BYTES_PER_TOKEN:
        problems.append("%s pass %d: first H2D copy %d B != n_tokens*10240 (%d)" % (tname, P["idx"], h2d[0]["bytes"], n_tokens))
    P["n_tokens"] = n_tokens
    mask = h2d[8]["bytes"] if len(h2d) > 8 else None
    P["n_kv_padded"] = (mask // (2 * n_tokens)) if mask and mask % (2 * n_tokens) == 0 else None
    if P["n_kv_padded"] is None:
        problems.append("%s pass %d: cannot derive n_kv from mask copy" % (tname, P["idx"]))
    d2h = [m for m in P["mc"] if m["kind"] == 2]
    d2h_b = sum(m["bytes"] for m in d2h)
    head = [k for k in ks if k["base"] == "mul_mat_vec_q" and k["gx"] == N_VOCAB]
    out_ids = h2d[9]["bytes"] // 4 if len(h2d) > 9 else 0
    P["n_head_kernels"] = len(head)
    P["d2h_logits_bytes"] = d2h_b
    P["n_outputs"] = d2h_b // LOGITS_BYTES if d2h_b % LOGITS_BYTES == 0 else None
    has_out = bool(head) or d2h_b > 0
    if has_out:
        kind = "suffix"
        ev.append("head_kernel=%d" % len(head))
        ev.append("d2h_logits=%dB" % d2h_b)
        ev.append("out_ids_h2d=%dB" % (out_ids * 4))
        if not head or d2h_b == 0:
            problems.append("%s pass %d: inconsistent suffix evidence" % (tname, P["idx"]))
    else:
        kind = "prefix-large" if n_tokens == UB else "prefix-small"
        ev.append("no_head_kernel;no_d2h_logits;no_out_ids_h2d")
        ev.append("n_tokens==ub" if n_tokens == UB else "n_tokens<ub")
    P["kind"] = kind
    P["evidence"] = ";".join(ev)
    P["rs5"] = h2d[4]["bytes"] // 4 if len(h2d) > 4 else None


def requests_of(passes):
    req, ridx = [], -1
    prev_suffix = True
    for P in passes:
        starts = P["kind"].startswith("prefix") and prev_suffix
        if starts:
            ridx += 1
        P["request"] = ridx
        prev_suffix = P["kind"] == "suffix"
    n = ridx + 1
    for r in range(n):
        ps = [P for P in passes if P["request"] == r]
        for j, P in enumerate(ps):
            P["pass_in_req"] = j
    return n


# ---------------------------------------------------------------------------------------------------------------
# per-pass metrics
# ---------------------------------------------------------------------------------------------------------------

def pass_metrics(t, P, prev):
    ks = P["ks"]
    n = len(ks)
    api_by_corr = t.api_by_corr
    start, end = ks[0]["start"], max(k["end"] for k in ks)
    P["start"], P["end"] = start, end
    P["wall"] = end - start
    P["busy"] = union_ns([(k["start"], k["end"]) for k in ks])
    P["ksum"] = sum(k["end"] - k["start"] for k in ks)
    g = gaps_of(ks)
    gl = [x[0] for x in g]
    P["gaps"] = gl
    P["gap_total"] = P["wall"] - P["busy"]
    P["gap_n5"] = sum(1 for x in gl if x > 5000)
    P["gap_n20"] = sum(1 for x in gl if x > 20000)
    P["gap_med"] = median(gl)
    mx = max(g, key=lambda x: x[0]) if g else (0, 0)
    P["gap_max"] = mx[0]
    P["gap_max_after"] = ks[mx[1]]["short"] if g else ""
    P["buckets"] = bucket_stats(gl)
    # d2d memcpy time inside the kernel window
    d2d_in = [m for m in P["mc"] if m["kind"] == 8 and m["start"] >= start and m["end"] <= end]
    P["d2d_in_window_ns"] = union_ns([(m["start"], m["end"]) for m in d2d_in])
    # categories
    cnt = [0] * len(CATS)
    tim = [0] * len(CATS)
    for k in ks:
        cnt[k["cat"]] += 1
        tim[k["cat"]] += k["end"] - k["start"]
    P["cat_n"], P["cat_t"] = cnt, tim
    by = collections.Counter(k["short"].split("<")[0] for k in ks)
    P["n_mmq"] = by["mul_mat_q"]
    P["n_fixup"] = by["mul_mat_q_stream_k_fixup"]
    P["n_mmvq"] = by["mul_mat_vec_q"]
    P["n_quant"] = by["quantize_mmq_q8_1"] + by["quantize_q8_1"]
    P["n_fa"] = by["flash_attn_ext_f16"]
    P["n_fa_aux"] = cnt[3] - P["n_fa"]
    P["n_gdn"] = cnt[1]
    P["n_scale"] = by["scale_f32"]
    P["gdn_kernel"] = ";".join(sorted({k["short"] for k in ks if k["cat"] == 1}))
    P["conv_kernel"] = ";".join(sorted({k["short"] for k in ks if k["cat"] == 7}))
    P["n_other"] = n - (P["n_mmq"] + P["n_fixup"] + P["n_mmvq"] + P["n_quant"] + P["n_fa"] + P["n_fa_aux"] + P["n_gdn"])
    # host side
    P["graph_launches"] = sum(1 for a in P["api"] if "GraphLaunch" in a["name"])
    P["gnode_kernels"] = sum(1 for k in ks if k["gnode"] not in (None, 0))
    P["launch_api"] = [a for a in P["api"] if a["name"].startswith("cudaLaunchKernel")]
    P["n_launch_api"] = len(P["launch_api"])
    P["n_mc"] = len(P["mc"])
    P["n_sync"] = len(P["sync"])
    P["cpu_api"] = sum(a["end"] - a["start"] for a in P["api"])
    P["cpu_launch"] = sum(a["end"] - a["start"] for a in P["launch_api"])
    P["cpu_memcpy"] = sum(a["end"] - a["start"] for a in P["api"] if a["name"].startswith("cudaMemcpy"))
    P["cpu_sync"] = sum(a["end"] - a["start"] for a in P["api"] if "Synchronize" in a["name"])
    la = sorted(a["end"] for a in P["launch_api"])
    lstart = [a["start"] for a in P["launch_api"]]
    P["cpu_launch_span"] = (max(a["end"] for a in P["launch_api"]) - min(lstart)) if P["launch_api"] else 0
    P["launch_med"] = median([a["end"] - a["start"] for a in P["launch_api"]])
    # launch-late analysis: kernel launched (API returned) after the previous kernel had finished
    late_n, late_gap, depth = 0, 0, []
    run_end = None
    for i, k in enumerate(ks):
        a = api_by_corr.get(k["corr"])
        if run_end is not None and a is not None:
            gap = max(0, k["start"] - run_end)
            if a["end"] > run_end:
                late_n += 1
                late_gap += gap
        if a is not None and i > 0:
            # launched-but-not-yet-started kernels when kernel i starts (kernels 0..i started)
            depth.append(bisect.bisect_right(la, k["start"]) - (i + 1))
        run_end = k["end"] if run_end is None else max(run_end, k["end"])
    P["late_n"], P["late_gap"] = late_n, late_gap
    P["qdepth_med"] = median(depth)
    P["qdepth_max"] = max(depth) if depth else None
    P["n_cmdbuf"] = sum(1 for o in P["ovh"] if o["type"] == 8)
    P["cmdbuf_ns"] = sum(o["end"] - o["start"] for o in P["ovh"] if o["type"] == 8)
    P["n_modload"] = sum(1 for o in P["ovh"] if o["type"] in (6, 7))
    P["modload_ns"] = sum(o["end"] - o["start"] for o in P["ovh"] if o["type"] in (6, 7))
    P["streams"] = "+".join(str(s) for s in sorted({k["stream"] for k in ks}))
    P["mc_streams"] = "+".join(str(s) for s in sorted({m["stream"] for m in P["mc"]}))
    h2d = P["h2d"]
    P["h2d_n"] = len(h2d)
    P["h2d_bytes"] = sum(m["bytes"] for m in h2d)
    P["h2d_gpu_ns"] = sum(m["end"] - m["start"] for m in h2d)
    P["h2d_start"], P["h2d_end"] = h2d[0]["start"], max(m["end"] for m in h2d)
    P["pre_gap"] = (start - prev["end"]) if prev else None
    P["h2d_lead"] = (P["h2d_start"] - prev["end"]) if prev else None
    P["h2d_tail"] = start - P["h2d_end"]
    P["n_h2d_mc"] = sum(1 for m in P["mc"] if m["kind"] == 1)
    P["n_d2h_mc"] = sum(1 for m in P["mc"] if m["kind"] == 2)
    P["n_d2d_mc"] = sum(1 for m in P["mc"] if m["kind"] == 8)
    P["d2h_bytes"] = sum(m["bytes"] for m in P["mc"] if m["kind"] == 2)
    P["d2d_bytes"] = sum(m["bytes"] for m in P["mc"] if m["kind"] == 8)


PASS_HEADER = (
    ["row_kind", "trace", "request_idx", "pass_idx_in_request", "pass_idx_in_trace", "case_id_inferred", "pass_kind",
     "pass_kind_evidence", "n_tokens", "n_kv_padded", "n_outputs", "n_passes", "start_ns", "start_rel_ms", "wall_ms",
     "gpu_busy_ms", "kernel_time_sum_ms", "gap_total_ms", "gap_total_excl_d2d_ms", "gap_share_of_wall_pct",
     "gap_n_over_5us", "gap_n_over_20us", "gap_median_us", "gap_max_us", "gap_max_after_kernel",
     "gap_intra_pass_ms", "gap_inter_pass_ms", "pre_gap_ms", "idle_before_request_ms", "gpu_busy_pct_of_span_incl_idle_before", "h2d_lead_ms", "h2d_span_ms",
     "h2d_tail_ms"]
    + sum([["gap_n_%s" % k, "gap_t_%s_ms" % k] for k, _, _ in BUCKETS], [])
    + ["n_kernels", "n_mmq", "n_mmq_fixup", "n_mmvq", "n_quantize", "n_flash_attn", "n_flash_aux", "n_gdn", "gdn_kernel",
       "conv_kernel", "n_scale_f32", "n_other"]
    + ["n_cat_%s" % c for c in CAT_KEY] + ["t_cat_%s_ms" % c for c in CAT_KEY]
    + ["n_cuda_graph_launches", "n_kernels_with_graph_node", "n_cudaLaunchKernel_api_calls", "n_memcpy", "n_memcpy_h2d",
       "n_memcpy_d2h", "n_memcpy_d2d", "memcpy_h2d_bytes", "memcpy_d2h_bytes", "memcpy_d2d_bytes", "memcpy_h2d_gpu_ms",
       "d2d_gpu_in_pass_ms", "n_sync_calls", "cpu_api_time_ms", "cpu_launch_api_ms", "cpu_memcpy_api_ms",
       "cpu_sync_api_ms", "cpu_launch_span_ms", "launch_api_median_us", "n_launch_late", "gap_launch_late_ms",
       "launch_queue_depth_median", "launch_queue_depth_max", "n_cmdbuf_full_events", "cmdbuf_full_ms",
       "n_module_load_events", "module_load_ms", "streams_used", "streams_used_memcpy"]
)


def agg_row(row_kind, tname, group, t0, prev_end=None, extra=None):
    """row for a request / trace_total from a list of passes"""
    ks = [k for P in group for k in P["ks"]]
    start, end = ks[0]["start"], max(k["end"] for k in ks)
    busy = union_ns([(k["start"], k["end"]) for k in ks])
    gl = [x for P in group for x in P["gaps"]]
    intra = sum(P["gap_total"] for P in group)
    r = {"row_kind": row_kind, "trace": tname, "n_passes": len(group), "n_tokens": sum(P["n_tokens"] for P in group),
         "start_ns": start, "start_rel_ms": ms(start - t0), "wall_ms": ms(end - start), "gpu_busy_ms": ms(busy),
         "kernel_time_sum_ms": ms(sum(k["end"] - k["start"] for k in ks)), "gap_total_ms": ms(end - start - busy),
         "gap_total_excl_d2d_ms": ms(end - start - busy - sum(P["d2d_in_window_ns"] for P in group)),
         "gap_share_of_wall_pct": 100.0 * (end - start - busy) / (end - start),
         "gap_n_over_5us": sum(P["gap_n5"] for P in group), "gap_n_over_20us": sum(P["gap_n20"] for P in group),
         "gap_median_us": us(median(gl)), "gap_max_us": us(max(gl)),
         "gap_intra_pass_ms": ms(intra), "gap_inter_pass_ms": ms(end - start - busy - intra)}
    mx = max(group, key=lambda P: P["gap_max"])
    r["gap_max_after_kernel"] = mx["gap_max_after"]
    for key, _, _ in BUCKETS:
        r["gap_n_%s" % key] = sum(P["buckets"][key][0] for P in group)
        r["gap_t_%s_ms" % key] = ms(sum(P["buckets"][key][1] for P in group))
    r["n_kernels"] = len(ks)
    for src, dst in (("n_mmq", "n_mmq"), ("n_fixup", "n_mmq_fixup"), ("n_mmvq", "n_mmvq"), ("n_quant", "n_quantize"),
                     ("n_fa", "n_flash_attn"), ("n_fa_aux", "n_flash_aux"), ("n_gdn", "n_gdn"), ("n_other", "n_other")):
        r[dst] = sum(P[src] for P in group)
    for i, c in enumerate(CAT_KEY):
        r["n_cat_%s" % c] = sum(P["cat_n"][i] for P in group)
        r["t_cat_%s_ms" % c] = ms(sum(P["cat_t"][i] for P in group))
    r["n_cuda_graph_launches"] = sum(P["graph_launches"] for P in group)
    r["n_kernels_with_graph_node"] = sum(P["gnode_kernels"] for P in group)
    r["n_cudaLaunchKernel_api_calls"] = sum(P["n_launch_api"] for P in group)
    r["n_memcpy"] = sum(P["n_mc"] for P in group)
    r["n_memcpy_h2d"] = sum(P["n_h2d_mc"] for P in group)
    r["n_memcpy_d2h"] = sum(P["n_d2h_mc"] for P in group)
    r["n_memcpy_d2d"] = sum(P["n_d2d_mc"] for P in group)
    r["memcpy_h2d_bytes"] = sum(P["h2d_bytes"] for P in group)
    r["memcpy_d2h_bytes"] = sum(P["d2h_bytes"] for P in group)
    r["memcpy_d2d_bytes"] = sum(P["d2d_bytes"] for P in group)
    r["memcpy_h2d_gpu_ms"] = ms(sum(P["h2d_gpu_ns"] for P in group))
    r["d2d_gpu_in_pass_ms"] = ms(sum(P["d2d_in_window_ns"] for P in group))
    r["n_sync_calls"] = sum(P["n_sync"] for P in group)
    r["cpu_api_time_ms"] = ms(sum(P["cpu_api"] for P in group))
    r["cpu_launch_api_ms"] = ms(sum(P["cpu_launch"] for P in group))
    r["cpu_memcpy_api_ms"] = ms(sum(P["cpu_memcpy"] for P in group))
    r["cpu_sync_api_ms"] = ms(sum(P["cpu_sync"] for P in group))
    r["cpu_launch_span_ms"] = ms(sum(P["cpu_launch_span"] for P in group))
    r["launch_api_median_us"] = us(median([a["end"] - a["start"] for P in group for a in P["launch_api"]]))
    r["n_launch_late"] = sum(P["late_n"] for P in group)
    r["gap_launch_late_ms"] = ms(sum(P["late_gap"] for P in group))
    r["launch_queue_depth_max"] = max((P["qdepth_max"] or 0) for P in group)
    r["n_cmdbuf_full_events"] = sum(P["n_cmdbuf"] for P in group)
    r["cmdbuf_full_ms"] = ms(sum(P["cmdbuf_ns"] for P in group))
    r["n_module_load_events"] = sum(P["n_modload"] for P in group)
    r["module_load_ms"] = ms(sum(P["modload_ns"] for P in group))
    r["streams_used"] = "+".join(sorted({s for P in group for s in P["streams"].split("+")}, key=int))
    r["streams_used_memcpy"] = "+".join(sorted({s for P in group for s in P["mc_streams"].split("+") if s}, key=int))
    if extra:
        r.update(extra)
    return r


def pass_row(t, P, t0):
    r = {"row_kind": "pass", "trace": t.name, "request_idx": P["request"], "pass_idx_in_request": P["pass_in_req"],
         "pass_idx_in_trace": P["idx"], "case_id_inferred": P["case"], "pass_kind": P["kind"],
         "pass_kind_evidence": P["evidence"], "n_tokens": P["n_tokens"], "n_kv_padded": P["n_kv_padded"],
         "n_outputs": P["n_outputs"], "start_ns": P["start"], "start_rel_ms": ms(P["start"] - t0),
         "wall_ms": ms(P["wall"]), "gpu_busy_ms": ms(P["busy"]), "kernel_time_sum_ms": ms(P["ksum"]),
         "gap_total_ms": ms(P["gap_total"]), "gap_total_excl_d2d_ms": ms(P["gap_total"] - P["d2d_in_window_ns"]),
         "gap_share_of_wall_pct": 100.0 * P["gap_total"] / P["wall"], "gap_n_over_5us": P["gap_n5"],
         "gap_n_over_20us": P["gap_n20"], "gap_median_us": us(P["gap_med"]), "gap_max_us": us(P["gap_max"]),
         "gap_max_after_kernel": P["gap_max_after"], "gap_intra_pass_ms": ms(P["gap_total"]),
         "pre_gap_ms": ms(P["pre_gap"]), "idle_before_request_ms": ms(P.get("idle_before_request")),
         "h2d_lead_ms": ms(P["h2d_lead"]), "h2d_span_ms": ms(P["h2d_end"] - P["h2d_start"]),
         "h2d_tail_ms": ms(P["h2d_tail"]), "n_kernels": len(P["ks"]), "n_mmq": P["n_mmq"], "n_mmq_fixup": P["n_fixup"],
         "n_mmvq": P["n_mmvq"], "n_quantize": P["n_quant"], "n_flash_attn": P["n_fa"], "n_flash_aux": P["n_fa_aux"],
         "n_gdn": P["n_gdn"], "gdn_kernel": P["gdn_kernel"], "conv_kernel": P["conv_kernel"], "n_scale_f32": P["n_scale"],
         "n_other": P["n_other"], "n_cuda_graph_launches": P["graph_launches"],
         "n_kernels_with_graph_node": P["gnode_kernels"], "n_cudaLaunchKernel_api_calls": P["n_launch_api"],
         "n_memcpy": P["n_mc"], "n_memcpy_h2d": P["n_h2d_mc"], "n_memcpy_d2h": P["n_d2h_mc"], "n_memcpy_d2d": P["n_d2d_mc"],
         "memcpy_h2d_bytes": P["h2d_bytes"], "memcpy_d2h_bytes": P["d2h_bytes"], "memcpy_d2d_bytes": P["d2d_bytes"],
         "memcpy_h2d_gpu_ms": ms(P["h2d_gpu_ns"]), "d2d_gpu_in_pass_ms": ms(P["d2d_in_window_ns"]),
         "n_sync_calls": P["n_sync"], "cpu_api_time_ms": ms(P["cpu_api"]), "cpu_launch_api_ms": ms(P["cpu_launch"]),
         "cpu_memcpy_api_ms": ms(P["cpu_memcpy"]), "cpu_sync_api_ms": ms(P["cpu_sync"]),
         "cpu_launch_span_ms": ms(P["cpu_launch_span"]), "launch_api_median_us": us(P["launch_med"]),
         "n_launch_late": P["late_n"], "gap_launch_late_ms": ms(P["late_gap"]),
         "launch_queue_depth_median": P["qdepth_med"], "launch_queue_depth_max": P["qdepth_max"],
         "n_cmdbuf_full_events": P["n_cmdbuf"], "cmdbuf_full_ms": ms(P["cmdbuf_ns"]),
         "n_module_load_events": P["n_modload"], "module_load_ms": ms(P["modload_ns"]), "streams_used": P["streams"],
         "streams_used_memcpy": P["mc_streams"]}
    for key, _, _ in BUCKETS:
        r["gap_n_%s" % key] = P["buckets"][key][0]
        r["gap_t_%s_ms" % key] = ms(P["buckets"][key][1])
    for i, c in enumerate(CAT_KEY):
        r["n_cat_%s" % c] = P["cat_n"][i]
        r["t_cat_%s_ms" % c] = ms(P["cat_t"][i])
    return r


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def main():
    global OUT, OLD_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="in_dir", default=os.environ.get("DMK_LEGACY_TRACES"),
                    help="folder with long.sqlite, medium.sqlite, short.sqlite (default: $DMK_LEGACY_TRACES)")
    ap.add_argument("--out", default=None, help="output folder (default: raw/traces of the census data set)")
    ap.add_argument("--old-export", default=None, help="optional: folder with a second export of the same captures, for a cross-check")
    args = ap.parse_args()
    if not args.in_dir:
        sys.exit("set --in DIR (or DMK_LEGACY_TRACES) to the folder with long.sqlite, medium.sqlite and short.sqlite (see trace_export.sh)")
    OUT = Path(args.out) if args.out else Path(census_lib.data_dir()) / "raw" / "traces"
    OUT.mkdir(parents=True, exist_ok=True)
    OLD_DIR = Path(args.old_export) if args.old_export else None
    problems = []
    out = []

    def say(*a):
        print(*a)

    traces = {}
    for name in TRACES:
        p = Path(args.in_dir) / ("%s.sqlite" % name)
        if not p.exists():
            sys.exit("missing %s - run `bash trace_export.sh` first" % p)
        traces[name] = load_trace(name, p)

    pass_rows, kernel_rows, mmq_rows = [], [], []
    summary = {}
    for name in TRACES:
        t = traces[name]
        K = t.K
        say("=" * 100)
        say("TRACE %s: %d kernels, %d runtime API records, %d memcpy, %d sync, %d overhead records; session start %s; GPU %s (%d SMs)"
            % (name, len(K), len(t.api), len(t.mc), len(t.sync), len(t.ovh), t.session, t.dev[0], t.dev[1]))
        passes = segment(t, problems)
        for P in passes:
            classify(P, problems, name)
        nreq = requests_of(passes)
        t.passes, t.nreq = passes, nreq
        # inferred cases
        for P in passes:
            P["case"] = CASES[name][P["request"] % len(CASES[name])]
        prev = None
        for P in passes:
            pass_metrics(t, P, prev)
            prev = P
        t0 = K[0]["start"]
        t.t0 = t0
        # idle before request
        for r in range(nreq):
            ps = [P for P in passes if P["request"] == r]
            ps[0]["idle_before_request"] = ps[0]["pre_gap"] if r > 0 else None
        # ---- validation ---------------------------------------------------------------------------------------
        val = []
        for P in passes:
            if P["n_gdn"] != 24:
                problems.append("%s pass %d: n_gdn=%d" % (name, P["idx"], P["n_gdn"]))
            if P["n_fa"] != 8:
                problems.append("%s pass %d: n_flash_attn=%d" % (name, P["idx"], P["n_fa"]))
            if P["n_launch_api"] != len(P["ks"]):
                problems.append("%s pass %d: launch API calls %d != kernels %d" % (name, P["idx"], P["n_launch_api"], len(P["ks"])))
            nrms = sum(1 for k in P["ks"] if k["base"] == "rms_norm_f32" and k["bx"] == 1024)
            exp = 65 if P["kind"] == "suffix" else 63   # prefix passes (0 outputs): last layer's post-attn norm gets 0 rows -> no launch
            P["n_rms1024"] = nrms
            if nrms != exp:
                problems.append("%s pass %d (%s): %d rms_norm<1024> launches, expected %d" % (name, P["idx"], P["kind"], nrms, exp))
            if P["streams"] != "15":
                problems.append("%s pass %d: kernel streams %s" % (name, P["idx"], P["streams"]))
            qn = collections.Counter(k["gx"] for k in P["ks"] if k["base"] == "quantize_mmq_q8_1")
            if qn and qn.most_common(1)[0][0] != P["n_tokens"]:
                problems.append("%s pass %d: modal quantize_mmq grid.x %d != n_tokens %d" % (name, P["idx"], qn.most_common(1)[0][0], P["n_tokens"]))
        # sequence identity among passes with equal (kind, n_tokens)
        groups = collections.defaultdict(list)
        for P in passes:
            groups[(P["kind"], P["n_tokens"])].append(P)
        seq_ok, seq_bad, grid_ok = 0, [], 0
        for key, ps in groups.items():
            if len(ps) < 2:
                continue
            s0 = [k["short"] for k in ps[0]["ks"]]
            g0 = [(k["short"], k["gx"], k["gy"], k["gz"]) for k in ps[0]["ks"] if k["cat"] in (0, 4, 9, 1)]
            for P in ps[1:]:
                s1 = [k["short"] for k in P["ks"]]
                g1 = [(k["short"], k["gx"], k["gy"], k["gz"]) for k in P["ks"] if k["cat"] in (0, 4, 9, 1)]
                if s0 == s1:
                    seq_ok += 1
                else:
                    seq_bad.append((key, ps[0]["idx"], P["idx"]))
                if g0 == g1:
                    grid_ok += 1
        say("segmentation: %d passes, %d requests; equal-(kind,n_tokens) pass pairs with identical kernel-name sequence: %d, "
            "different: %d %s; identical MMQ/quantize/MMVQ/GDN grids: %d" % (len(passes), nreq, seq_ok, len(seq_bad), seq_bad[:6], grid_ok))
        if seq_bad:
            problems.append("%s: passes with equal (kind, n_tokens) but different kernel sequences: %s" % (name, seq_bad))
        covered = sum(len(P["ks"]) for P in passes)
        if covered != len(K):
            problems.append("%s: passes cover %d of %d kernels" % (name, covered, len(K)))
        # ---- rows -------------------------------------------------------------------------------------------
        for P in passes:
            pass_rows.append(pass_row(t, P, t0))
        prev_req_end = None
        for r in range(nreq):
            ps = [P for P in passes if P["request"] == r]
            ex = {"request_idx": r, "case_id_inferred": ps[0]["case"], "pass_kind": "request",
                  "pass_kind_evidence": "prefix_passes=%d;suffix_passes=%d" % (sum(1 for P in ps if P["kind"].startswith("prefix")), sum(1 for P in ps if P["kind"] == "suffix")),
                  "idle_before_request_ms": ms(ps[0].get("idle_before_request")), "pre_gap_ms": ms(ps[0]["pre_gap"])}
            ex["n_outputs"] = sum(P["n_outputs"] or 0 for P in ps)
            row = agg_row("request", name, ps, t0, extra=ex)
            rend = max(P["end"] for P in ps)
            span_start = prev_req_end if prev_req_end is not None else ps[0]["start"]
            row["gpu_busy_pct_of_span_incl_idle_before"] = 100.0 * sum(P["busy"] for P in ps) / (rend - span_start)
            prev_req_end = rend
            pass_rows.append(row)
        ex = {"pass_kind": "trace_total", "pass_kind_evidence": "requests=%d" % nreq,
              "n_outputs": sum(P["n_outputs"] or 0 for P in passes)}
        ttot = agg_row("trace_total", name, passes, t0, extra=ex)
        pass_rows.append(ttot)
        # ---- kernel table -----------------------------------------------------------------------------------
        tot_ns = sum(k["end"] - k["start"] for k in K)
        key_of = lambda k: (k["short"], k["gx"], k["gy"], k["gz"], k["bx"], k["by"], k["bz"], k["ssmem"], k["dsmem"], k["regs"])
        agg = collections.OrderedDict()
        per_pass = collections.defaultdict(lambda: collections.defaultdict(int))
        for P in passes:
            for k in P["ks"]:
                kk = key_of(k)
                e = agg.setdefault(kk, {"full": k["full"], "cat": k["cat"], "n": 0, "tot": 0, "min": None, "max": 0})
                d = k["end"] - k["start"]
                e["n"] += 1
                e["tot"] += d
                e["min"] = d if e["min"] is None else min(e["min"], d)
                e["max"] = max(e["max"], d)
                per_pass[kk][P["idx"]] += 1
        for kk, e in sorted(agg.items(), key=lambda x: -x[1]["tot"]):
            r = {"trace": name, "kernel_short": kk[0], "kernel_full": e["full"], "category": CATS[e["cat"]],
                 "grid_x": kk[1], "grid_y": kk[2], "grid_z": kk[3], "block_x": kk[4], "block_y": kk[5], "block_z": kk[6],
                 "smem_static_b": kk[7], "smem_dynamic_b": kk[8], "regs": kk[9], "n_launches": e["n"],
                 "total_us": us(e["tot"]), "mean_us": us(e["tot"] / e["n"]), "min_us": us(e["min"]), "max_us": us(e["max"]),
                 "share_pct_of_trace_kernel_time": 100.0 * e["tot"] / tot_ns}
            pp = per_pass[kk]
            r["n_passes_containing"] = len(pp)
            r["launches_per_pass_median_all"] = median(list(pp.values()))
            for kind in ("prefix-large", "prefix-small", "suffix"):
                vals = [c for pi, c in pp.items() if passes[pi]["kind"] == kind]
                r["lpp_median_%s" % kind.replace("-", "_")] = median(vals)
                r["n_passes_%s" % kind.replace("-", "_")] = len(vals)
            kernel_rows.append(r)
        # ---- mmq launches ---------------------------------------------------------------------------------
        seen = set()
        for P in passes:
            key = (P["kind"], P["n_tokens"])
            if key in seen:
                continue
            seen.add(key)
            ks = P["ks"]
            last_q = {"mmq": None, "mmvq": None}   # (pos, kernel)
            q_users = collections.defaultdict(int)
            order = {"mmq": 0, "mmvq": 0}
            for pos, k in enumerate(ks):
                if k["base"] == "quantize_mmq_q8_1":
                    last_q["mmq"] = (pos, k)
                    q_users[("mmq", pos)] = 0
                elif k["base"] == "quantize_q8_1":
                    last_q["mmvq"] = (pos, k)
                    q_users[("mmvq", pos)] = 0
                elif k["base"] in ("mul_mat_q", "mul_mat_vec_q"):
                    lk = "mmq" if k["base"] == "mul_mat_q" else "mmvq"
                    args = split_name(k["full"])[1]
                    q = last_q[lk]
                    nargs = [norm_arg(a) for a in args]
                    r = {"trace": name, "pass_idx_in_trace": P["idx"], "request_idx": P["request"],
                         "pass_idx_in_request": P["pass_in_req"], "pass_kind": P["kind"], "n_tokens": P["n_tokens"],
                         "n_kv_padded": P["n_kv_padded"], "launch_kind": lk, "order_in_pass": order[lk],
                         "position_in_pass": pos, "kernel_short": k["short"], "quant_type": nargs[0],
                         "grid_x": k["gx"], "grid_y": k["gy"], "grid_z": k["gz"], "block_x": k["bx"], "block_y": k["by"],
                         "block_z": k["bz"], "smem_static_b": k["ssmem"], "smem_dynamic_b": k["dsmem"], "regs": k["regs"],
                         "dur_us": us(k["end"] - k["start"]), "start_in_pass_us": us(k["start"] - P["start"])}
                    if lk == "mmq":      # <type, J(mmq_x), fallback(need_check), prec_src1, has_fusion>
                        r["mmq_x"], r["need_check"], r["has_fusion"] = nargs[1], nargs[2], nargs[4]
                    else:                # <type, ncols_dst, has_fusion, small_k, halve_iters>
                        r["ncols_dst"], r["has_fusion"], r["small_k"], r["halve_iters"] = nargs[1], nargs[2], nargs[3], nargs[4]
                    order[lk] += 1
                    if q:
                        qpos, qk = q
                        q_users[(lk, qpos)] += 1
                        if lk == "mmq":   # quantize_mmq_q8_1 grid = (n_rows, ceil(K_padded/512), 1)
                            n_rows, k_pad = qk["gx"], qk["gy"] * 512
                        else:             # quantize_q8_1 grid = (ceil(K_padded/256), n_rows, 1)
                            n_rows, k_pad = qk["gy"], qk["gx"] * 256
                        r.update({"prev_quant_kernel": qk["short"], "prev_quant_position": qpos,
                                  "prev_quant_distance": pos - qpos, "prev_quant_grid_x": qk["gx"],
                                  "prev_quant_grid_y": qk["gy"], "prev_quant_grid_z": qk["gz"],
                                  "prev_quant_n_rows": n_rows, "prev_quant_k_padded512": k_pad,
                                  "prev_quant_dur_us": us(qk["end"] - qk["start"]),
                                  "prev_quant_reuse_index": q_users[(lk, qpos)] - 1})
                    if lk == "mmq":
                        nxt = ks[pos + 1] if pos + 1 < len(ks) else None
                        fx = nxt is not None and nxt["base"] == "mul_mat_q_stream_k_fixup"
                        r["has_fixup"] = 1 if fx else 0
                        if fx:
                            r["fixup_dur_us"] = us(nxt["end"] - nxt["start"])
                            r["fixup_grid_x"], r["fixup_grid_y"], r["fixup_grid_z"] = nxt["gx"], nxt["gy"], nxt["gz"]
                            r["fixup_short"] = nxt["short"]
                    mmq_rows.append(r)
        summary[name] = t

    # ---- write TSVs ------------------------------------------------------------------------------------------
    kernel_header = ["trace", "kernel_short", "kernel_full", "category", "grid_x", "grid_y", "grid_z", "block_x", "block_y",
                     "block_z", "smem_static_b", "smem_dynamic_b", "regs", "n_launches", "total_us", "mean_us", "min_us",
                     "max_us", "share_pct_of_trace_kernel_time", "n_passes_containing", "launches_per_pass_median_all",
                     "lpp_median_prefix_large", "n_passes_prefix_large", "lpp_median_prefix_small", "n_passes_prefix_small",
                     "lpp_median_suffix", "n_passes_suffix"]
    write_tsv(OUT / "trace-kernels.tsv", kernel_header, kernel_rows)
    pass_header = [h for h in PASS_HEADER]
    write_tsv(OUT / "trace-passes.tsv", pass_header, pass_rows)
    mmq_header = ["trace", "pass_idx_in_trace", "request_idx", "pass_idx_in_request", "pass_kind", "n_tokens", "n_kv_padded",
                  "launch_kind", "order_in_pass", "position_in_pass", "kernel_short", "quant_type", "mmq_x", "need_check",
                  "ncols_dst", "has_fusion", "small_k", "halve_iters", "grid_x", "grid_y", "grid_z", "block_x", "block_y",
                  "block_z", "smem_static_b", "smem_dynamic_b", "regs", "dur_us", "start_in_pass_us", "prev_quant_kernel",
                  "prev_quant_position", "prev_quant_distance", "prev_quant_grid_x", "prev_quant_grid_y", "prev_quant_grid_z",
                  "prev_quant_n_rows", "prev_quant_k_padded512", "prev_quant_dur_us", "prev_quant_reuse_index", "has_fixup",
                  "fixup_dur_us", "fixup_short", "fixup_grid_x", "fixup_grid_y", "fixup_grid_z"]
    write_tsv(OUT / "trace-mmq-launches.tsv", mmq_header, mmq_rows)

    report(summary, pass_rows, kernel_rows, mmq_rows, problems)


# ---------------------------------------------------------------------------------------------------------------
# stdout report (validation results, category shares, pass-class tables)
# ---------------------------------------------------------------------------------------------------------------

def tbl(rows, header):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(header)]
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(header)))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def report(traces, pass_rows, kernel_rows, mmq_rows, problems):
    print("\n" + "=" * 100)
    print("VALIDATION PROBLEMS: %s" % (len(problems) if problems else "none"))
    for p in problems:
        print("  !", p)
    kinds = ["prefix-large", "prefix-small", "suffix"]
    for name in TRACES:
        t = traces[name]
        P = t.passes
        print("\n" + "=" * 100 + "\nTRACE %s" % name)
        print("passes: %d  requests: %d  kernels: %d  cudaLaunchKernel: %d  cudaGraphLaunch: %d  kernels with graphNodeId: %d"
              % (len(P), t.nreq, len(t.K), sum(1 for a in t.api if a["name"].startswith("cudaLaunchKernel")),
                 sum(1 for a in t.api if "Graph" in a["name"]), sum(1 for k in t.K if k["gnode"] not in (None, 0))))
        apinames = collections.Counter(a["name"] for a in t.api)
        print("runtime API calls: %s" % dict(apinames))
        gn = t.graph_nodes
        ids = t.graph_node_ids
        print("CUDA_GRAPH_NODE_EVENTS: %d rows (%s), ts range %.3f..%.3f s relative to capture start; first kernel at %.3f s; "
              "distinct graphId(high 32 bits of graphNodeId): %d"
              % (gn[0], dict(t.graph_names), gn[1] / 1e9, gn[2] / 1e9, t.K[0]["start"] / 1e9,
                 len({i >> 32 for i in ids})))
        def pclass(p):
            if p["kind"] != "suffix":
                return p["kind"]
            return "suffix-mmvq(n<=8)" if p["n_mmvq"] > 3 else "suffix-mmq"
        classes = ["prefix-large", "prefix-small", "suffix-mmq", "suffix-mmvq(n<=8)"]
        # per class table
        rows = []
        for cl in classes:
            ps = [p for p in P if pclass(p) == cl]
            if not ps:
                continue
            rows.append([cl, len(ps), sorted({p["n_tokens"] for p in ps}) if len(ps) < 9 else "%d..%d" % (min(p["n_tokens"] for p in ps), max(p["n_tokens"] for p in ps)),
                         median([len(p["ks"]) for p in ps]), median([p["n_mmq"] for p in ps]), median([p["n_fixup"] for p in ps]),
                         median([p["n_mmvq"] for p in ps]), median([p["n_quant"] for p in ps]), median([p["n_fa"] for p in ps]),
                         median([p["n_gdn"] for p in ps]),
                         round(median([p["wall"] for p in ps]) / 1e6, 3), round(median([p["gap_total"] for p in ps]) / 1e6, 3),
                         round(100 * median([p["gap_total"] / p["wall"] for p in ps]), 2),
                         round(sum(p["gap_total"] for p in ps) / 1e6, 2), round(sum(p["wall"] for p in ps) / 1e6, 2),
                         round(median([p["pre_gap"] for p in ps if p["pre_gap"] is not None]) / 1e6, 3) if any(p["pre_gap"] is not None for p in ps) else "-",
                         round(median([p["gap_total"] for p in ps if p["modload_ns"] < 1e6] or [0]) / 1e6, 3),
                         round(median([p["gap_total"] / p["wall"] for p in ps if p["modload_ns"] < 1e6] or [0]) * 100, 2)])
        tbl(rows, ["class", "n", "n_tokens", "kern", "mmq", "fixup", "mmvq", "quant", "fa", "gdn", "wall_ms_med", "gap_ms_med",
                   "gap%wall_med", "gap_ms_sum", "wall_ms_sum", "pre_gap_ms_med", "gap_ms_med_noModLoad", "gap%wall_noModLoad"])
        # per-class kernels per category (median) + time
        print("kernels per pass by category / kernel time ms (median over passes of the class):")
        rows = []
        for i, c in enumerate(CATS):
            row = [c]
            for cl in classes:
                ps = [p for p in P if pclass(p) == cl]
                row.append("%g / %.3f" % (median([p["cat_n"][i] for p in ps]), median([p["cat_t"][i] for p in ps]) / 1e6) if ps else "-")
            rows.append(row)
        tbl(rows, ["category"] + classes)
        # catalog by (kind, n_tokens)
        print("pass catalog by (kind, n_tokens): n passes, kernels, mmq, fixup, mmvq, quant, wall_ms (median), busy_ms (median), n_kv_padded values")
        cat = collections.OrderedDict()
        for p in sorted(P, key=lambda p: (p["kind"], p["n_tokens"])):
            cat.setdefault((p["kind"], p["n_tokens"]), []).append(p)
        rows = [[k[0], k[1], len(v), len(v[0]["ks"]), v[0]["n_mmq"], v[0]["n_fixup"], v[0]["n_mmvq"], v[0]["n_quant"],
                 round(median([p["wall"] for p in v]) / 1e6, 2), round(median([p["busy"] for p in v]) / 1e6, 2),
                 sorted({p["n_kv_padded"] for p in v}), sorted({p["rs5"] for p in v})] for k, v in cat.items()]
        tbl(rows, ["kind", "n_tokens", "n", "kern", "mmq", "fixup", "mmvq", "quant", "wall_ms", "busy_ms", "n_kv", "h2d5_int32s"])
        # kernel duration histogram
        dur_b = [(0, 2000), (2000, 5000), (5000, 20000), (20000, 100000), (100000, 1 << 62)]
        dc = [[0, 0] for _ in dur_b]
        for k in t.K:
            d = k["end"] - k["start"]
            for j, (lo, hi) in enumerate(dur_b):
                if lo <= d < hi:
                    dc[j][0] += 1
                    dc[j][1] += d
        tot_k = sum(x[1] for x in dc)
        print("kernel duration histogram (<2, 2-5, 5-20, 20-100, >100 us): %s" % "; ".join("%d kernels / %.1f ms (%.1f%% of kernel time)" % (c, d / 1e6, 100.0 * d / tot_k) for c, d in dc))
        # category share of kernel time for the trace
        tot = sum(k["end"] - k["start"] for k in t.K)
        cat_t = [sum(k["end"] - k["start"] for k in t.K if k["cat"] == i) for i in range(len(CATS))]
        print("category share of trace kernel time (total %.1f ms): %s" % (tot / 1e6, "; ".join("%s %.2f%%" % (CAT_KEY[i], 100.0 * cat_t[i] / tot) for i in range(len(CATS)))))
        # requests
        print("requests:")
        rows = []
        for r in range(t.nreq):
            ps = [p for p in P if p["request"] == r]
            row = [r, ps[0]["case"], "+".join("%d" % p["n_tokens"] for p in ps),
                   "+".join(str(p["n_kv_padded"]) for p in ps[:1]),
                   round((max(p["end"] for p in ps) - ps[0]["start"]) / 1e6, 2),
                   round(sum(p["busy"] for p in ps) / 1e6, 2),
                   round(100.0 * sum(p["busy"] for p in ps) / (max(p["end"] for p in ps) - ps[0]["start"]), 1),
                   round((ps[0]["pre_gap"] or 0) / 1e6, 2)]
            rows.append(row)
        tbl(rows, ["req", "case", "pass n_tokens", "n_kv(1st)", "wall_ms", "busy_ms", "busy%", "idle_before_ms"])
        # gaps by transition
        tr = collections.defaultdict(list)
        prevk = None
        for p in P:
            if prevk is not None and p["pre_gap"] is not None:
                tr["%s->%s%s" % (prevk.split("-")[0], p["kind"].split("-")[0], " (new request)" if p["pass_in_req"] == 0 else "")].append(p)
            prevk = p["kind"]
        print("pre-pass idle gap (previous pass last kernel end -> first kernel start), ms; lead = prev kernel end -> first H2D copy start, "
              "h2d = first H2D copy start -> last H2D copy end, tail = last H2D copy end -> first kernel start:")
        rows = []
        for k, v in sorted(tr.items()):
            g = [p["pre_gap"] / 1e6 for p in v]
            rows.append([k, len(v), round(min(g), 3), round(median(g), 3), round(max(g), 3),
                         round(median([p["h2d_lead"] for p in v]) / 1e6, 3),
                         round(median([(p["h2d_end"] - p["h2d_start"]) for p in v]) / 1e6, 3),
                         round(median([p["h2d_tail"] for p in v]) / 1e6, 3)])
        tbl(rows, ["transition", "n", "min", "median", "max", "lead_med", "h2d_med", "tail_med"])
        # in-pass gap histogram per kind
        print("in-pass kernel-to-kernel gap histogram (count / total ms):")
        rows = []
        for kind in kinds + ["all"]:
            ps = [p for p in P if p["kind"] == kind] if kind != "all" else P
            if not ps:
                continue
            row = [kind]
            for key, _, _ in BUCKETS:
                row.append("%d / %.2f" % (sum(p["buckets"][key][0] for p in ps), sum(p["buckets"][key][1] for p in ps) / 1e6))
            row.append("%.2f" % (sum(p["wall"] for p in ps) / 1e6))
            rows.append(row)
        tbl(rows, ["kind"] + [b[0] for b in BUCKETS] + ["wall_ms_total"])
        # kernels sitting before large gaps (> 5 us and > 20 us) in passes
        for thr in (5000, 20000):
            before = collections.defaultdict(lambda: [0, 0])
            after = collections.defaultdict(lambda: [0, 0])
            for p in P:
                for gap, i in gaps_of(p["ks"]):
                    if gap > thr:
                        before[p["ks"][i]["short"]][0] += 1
                        before[p["ks"][i]["short"]][1] += gap
                        after[p["ks"][i + 1]["short"]][0] += 1
                        after[p["ks"][i + 1]["short"]][1] += gap
            print("kernels immediately BEFORE in-pass gaps > %d us (count, total gap ms):" % (thr // 1000))
            for k, v in sorted(before.items(), key=lambda x: -x[1][1])[:8]:
                print("   %-48s %5d  %9.3f" % (k, v[0], v[1] / 1e6))
            print("kernels immediately AFTER in-pass gaps > %d us (count, total gap ms):" % (thr // 1000))
            for k, v in sorted(after.items(), key=lambda x: -x[1][1])[:8]:
                print("   %-48s %5d  %9.3f" % (k, v[0], v[1] / 1e6))
        # launch / CPU analysis
        print("host-side (median over passes of the class); cadence = (last launch API end - first launch API start) / kernels:")
        rows = []
        for cl in classes:
            ps = [p for p in P if pclass(p) == cl]
            if not ps:
                continue
            rows.append([cl, len(ps), round(median([p["launch_med"] for p in ps]) / 1e3, 2),
                         round(median([p["cpu_launch_span"] / len(p["ks"]) for p in ps]) / 1e3, 2),
                         round(median([p["cpu_launch_span"] for p in ps]) / 1e6, 2), round(median([p["busy"] for p in ps]) / 1e6, 2),
                         round(median([p["wall"] / len(p["ks"]) for p in ps]) / 1e3, 2),
                         round(median([p["late_n"] / max(1, len(p["ks"]) - 1) for p in ps]) * 100, 2),
                         round(median([p["late_gap"] for p in ps]) / 1e6, 3), round(median([p["gap_total"] for p in ps]) / 1e6, 3),
                         median([p["qdepth_med"] for p in ps]), median([p["qdepth_max"] for p in ps]),
                         median([p["n_cmdbuf"] for p in ps]), round(median([p["n_sync"] for p in ps]), 1)])
        tbl(rows, ["class", "n", "launch_api_us_med", "cpu_cadence_us", "cpu_launch_span_ms", "gpu_busy_ms", "gpu_wall_us_per_kernel",
                   "launch_late%", "late_gap_ms", "gap_ms", "qdepth_med", "qdepth_max", "cmdbuf_full_ev", "n_sync"])
        la_all = sorted(a["end"] - a["start"] for a in t.api if a["name"].startswith("cudaLaunchKernel"))
        print("cudaLaunchKernel API duration over the whole trace: median %.2f us, mean %.2f us, p95 %.2f us, p99 %.2f us"
              % (la_all[len(la_all) // 2] / 1e3, sum(la_all) / len(la_all) / 1e3, la_all[int(len(la_all) * 0.95)] / 1e3, la_all[int(len(la_all) * 0.99)] / 1e3))
        d2d = [(p["idx"], p["n_tokens"], p["n_d2d_mc"], p["d2d_bytes"]) for p in P if p["n_d2d_mc"]]
        print("passes with device-to-device memcpy (idx, n_tokens, count, bytes): %s" % d2d)
        ovh = collections.Counter()
        ovt = collections.Counter()
        for o in t.ovh:
            ovh[o["type"]] += 1
            ovt[o["type"]] += o["end"] - o["start"]
        print("CUPTI overhead records: %s" % {k: (ovh[k], round(ovt[k] / 1e6, 3)) for k in ovh})
        # passes with module loading overhead
        ml = [(p["idx"], p["n_tokens"], p["n_modload"], round(p["modload_ns"] / 1e6, 2)) for p in P if p["n_modload"]]
        print("passes with module/lazy-loading overhead (idx, n_tokens, events, ms): %s" % ml)

    # comparison with the first analysis of the captures
    print("\n" + "=" * 100 + "\ncategory share of kernel time (this export) vs the table of the first analysis")
    review = {"long": [61.8, 7.5, 5.5, 5.4, 5.2, 4.1, 3.8, 2.5, 2.2, 1.5, None],
              "medium": [63.5, 7.6, 5.1, 2.9, 5.0, 4.3, 3.6, 2.6, 2.3, 2.8, None],
              "short": [66.0, 7.4, 3.9, 2.0, 3.9, 4.6, 3.1, 2.4, 2.2, 4.3, None]}
    rows = []
    for i, c in enumerate(CATS):
        row = [c]
        for name in TRACES:
            t = traces[name]
            tot = sum(k["end"] - k["start"] for k in t.K)
            v = 100.0 * sum(k["end"] - k["start"] for k in t.K if k["cat"] == i) / tot
            rv = review[name][i]
            row.append("%.2f (first analysis %s)" % (v, "-" if rv is None else rv))
        rows.append(row)
    tbl(rows, ["category"] + TRACES)

    # cross-check old exports
    print("\n" + "=" * 100 + "\ncross-check against an older export of the same captures (opened read-only)")
    for name in TRACES:
        if OLD_DIR is None:
            print("  %s: not checked (no --old-export folder given)" % name)
            continue
        p = OLD_DIR / ("%s.sqlite" % name)
        if not p.exists():
            print("  %s: old export not found" % name)
            continue
        db = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
        n, s, e = db.execute("select count(*), sum(end-start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
        mc = db.execute("select count(*) from CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()[0]
        t = traces[name]
        print("  %s: old kernels %d, kernel time %.3f ms, memcpy %d | new kernels %d, kernel time %.3f ms, memcpy %d -> %s"
              % (name, n, s / 1e6, mc, len(t.K), sum(k["end"] - k["start"] for k in t.K) / 1e6, len(t.mc),
                 "IDENTICAL" if (n == len(t.K) and s == sum(k["end"] - k["start"] for k in t.K) and mc == len(t.mc)) else "DIFFERENT"))
        db.close()
    print("\nwrote %s, %s, %s" % (OUT / "trace-kernels.tsv", OUT / "trace-passes.tsv", OUT / "trace-mmq-launches.tsv"))
    print("rows: kernels %d, passes.tsv %d, mmq-launches %d" % (len(kernel_rows), len(pass_rows), len(mmq_rows)))


if __name__ == "__main__":
    main()
