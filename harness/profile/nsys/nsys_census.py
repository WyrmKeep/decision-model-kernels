#!/usr/bin/env python3
"""nsys_census.py - kernel / pass / gap census of nsys captures (python3 stdlib only, CPU only).

Three sub-commands (see NOTES.txt and COLUMNS.tsv in this folder):

  legacy    python3 nsys_census.py legacy --in DIR --out DIR [--old-exports DIR]
            = the old census trace step (harness/profile/census/trace_census.py), unchanged logic: reads DIR/{long,medium,short}.sqlite
            (nsys exports of the three production captures, no client log) and writes trace-passes.tsv, trace-kernels.tsv,
            trace-mmq-launches.tsv into --out.  The output is byte-identical to the census tables of the data set
            data/2026-10-30-shape-census (test: test/run_tests.sh, which needs DMK_LEGACY_TRACES).

  analyze   python3 nsys_census.py analyze --results DIR [--runs a,b,...] [--out DIR] [--assign window|order] [--clock mono|epoch]
            [--no-mmq-launches]
            new mode: whole-server captures (server start, model load, warm-up kernels, requests, idle) = DIR/<run>.sqlite + the
            client log DIR/<run>.client.json written by extra/prof/nsys_client.py.  Passes are found with the legacy anchor
            rule restricted to valid anchors (everything before the first valid pass is dropped); request windows are cut with
            the client stamps mapped onto the nsys timeline (CLOCK_MONOTONIC_ns - systemClockNs; --clock epoch uses
            t_send_epoch_ns - utcEpochNs) and cross-checked with an order-based assignment (expected passes per request kind).
            Writes passes.tsv (trace-passes.tsv columns + new columns), kernels.tsv, classes.tsv, requests.tsv, mmq-launches.tsv
            (not with --no-mmq-launches; the tables of the checks in summary.txt are computed anyway), osrt-request-start.tsv,
            summary.txt; the same tables go to stdout.

  overhead  python3 nsys_census.py overhead --results DIR --ctl CTL_RUN [--nsys R1,R2,...] [--dec DEC.client.json] [--passes passes.tsv]
            client wall time of the profiled runs against the unprofiled control run (paired by Ref) and against the
            decomposition client; with --passes also where the difference can sit (host-side parts of the request).

Input of legacy mode (the original header of trace_census.py follows):
    DIR/{long,medium,short}.sqlite              fresh nsys 2026.3.2 exports of the .nsys-rep files
    optional cross-check only (opened read-only): --old-exports DIR with the earlier exports {long,medium,short}.sqlite
Outputs of legacy mode:
    trace-kernels.tsv        one row per (trace, kernel_short, grid, block, smem, regs)
    trace-passes.tsv         one row per model pass (row_kind=pass), per request (request), per trace (trace_total)
    trace-mmq-launches.tsv   ordered main mul_mat_q / mul_mat_vec_q launches of one representative pass per
                             distinct (trace, pass_kind, n_tokens)
A human-readable summary (validation results, the tables of the census notes) is printed to stdout.

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
    * kernel categories follow the groups of the earlier kernel review (unary_op_kernel sigmoid/softplus counted with gated activations,
      quantize_q8_1 with quantize to int8, rope_multi/other in "other"); the resulting shares reproduce the table of that review.
"""
import argparse
import bisect
import collections
import json
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# legacy mode: input / output directories (set by the `legacy` sub-command)
WORK = HERE
OUT = HERE
OLD_PROF = None   # legacy mode, optional cross-check: folder with the earlier exports {long,medium,short}.sqlite (--old-exports)
TRACES = ["long", "medium", "short"]

# client order of the old captures: cases of the size class in bench/cases.json order, cycling i % n_cases
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
                                                        # the gated activations (reproduces the shares of the earlier review)
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
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    t.tables = tables

    def q(table, sql):
        """rows of `sql`, or [] when the exporter did not create `table` (no such events in the capture)"""
        return db.execute(sql).fetchall() if table in tables else []
    sid = dict(db.execute("select id, value from StringIds")) if "StringIds" in tables else {}
    rows = q("CUPTI_ACTIVITY_KIND_KERNEL",
             "select start, end, streamId, correlationId, demangledName, gridX, gridY, gridZ, blockX, blockY, blockZ, "
             "registersPerThread, staticSharedMemory, dynamicSharedMemory, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL "
             "order by start, end")
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
    t.api = [dict(zip(("start", "end", "name", "corr", "tid"), (r[0], r[1], sid[r[2]], r[3], r[4]))) for r in q(
        "CUPTI_ACTIVITY_KIND_RUNTIME", "select start, end, nameId, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME order by start")]
    t.mc = [dict(zip(("start", "end", "stream", "kind", "bytes", "corr"), r)) for r in q(
        "CUPTI_ACTIVITY_KIND_MEMCPY", "select start, end, streamId, copyKind, bytes, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY order by start")]
    t.sync = [dict(zip(("start", "end", "stream", "corr"), r)) for r in q(
        "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", "select start, end, streamId, correlationId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION order by start")]
    t.ovh = [dict(zip(("start", "end", "type", "corr"), r)) for r in q(
        "CUPTI_ACTIVITY_KIND_OVERHEAD", "select start, end, overheadType, correlationId from CUPTI_ACTIVITY_KIND_OVERHEAD order by start")]
    t.graph_nodes = (q("CUDA_GRAPH_NODE_EVENTS", "select count(*), min(start), max(end) from CUDA_GRAPH_NODE_EVENTS")
                     or [(0, None, None)])[0]
    t.graph_node_ids = [r[0] for r in q("CUDA_GRAPH_NODE_EVENTS", "select graphNodeId from CUDA_GRAPH_NODE_EVENTS")]
    t.graph_names = q("CUDA_GRAPH_NODE_EVENTS", "select s.value, count(*) from CUDA_GRAPH_NODE_EVENTS g join StringIds s on s.id=g.nameId "
                      "group by 1")
    t.dev = (q("TARGET_INFO_GPU", "select name, smCount, memoryBandwidth from TARGET_INFO_GPU") or [(None, None, None)])[0]
    ss = (q("TARGET_INFO_SESSION_START_TIME", "select utcTime, utcEpochNs, systemClockNs from TARGET_INFO_SESSION_START_TIME")
          or [(None, None, None)])[0]
    t.session, t.utc_epoch_ns, t.sys_clock_ns = ss
    t.duration_ns = (q("ANALYSIS_DETAILS", "select duration from ANALYSIS_DETAILS") or [(None,)])[0][0]
    t.procs = q("PROCESSES", "select pid, name from PROCESSES")
    t.capture = dict(q("META_DATA_CAPTURE", "select name, value from META_DATA_CAPTURE"))
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


def build_kernel_rows(name, passes):
    """trace-kernels.tsv rows (one per kernel_short, grid, block, smem, regs) over the kernels of `passes`"""
    pass_of = {P["idx"]: P for P in passes}
    tot_ns = sum(k["end"] - k["start"] for P in passes for k in P["ks"])
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
    rows = []
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
            vals = [c for pi, c in pp.items() if pass_of[pi]["kind"] == kind]
            r["lpp_median_%s" % kind.replace("-", "_")] = median(vals)
            r["n_passes_%s" % kind.replace("-", "_")] = len(vals)
        rows.append(r)
    return rows


def build_mmq_rows(name, passes, extra=None):
    """trace-mmq-launches.tsv rows: ordered main mul_mat_q / mul_mat_vec_q launches of the first pass of every distinct
    (pass kind, n_tokens); `extra(P)` (new mode) returns {position_in_pass: {column: value}} to add columns."""
    rows = []
    seen = set()
    for P in passes:
        key = (P["kind"], P["n_tokens"])
        if key in seen:
            continue
        seen.add(key)
        ks = P["ks"]
        ex = extra(P) if extra else {}
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
                if pos in ex:
                    r.update(ex[pos])
                rows.append(r)
    return rows


KERNEL_HEADER = ["trace", "kernel_short", "kernel_full", "category", "grid_x", "grid_y", "grid_z", "block_x", "block_y",
                 "block_z", "smem_static_b", "smem_dynamic_b", "regs", "n_launches", "total_us", "mean_us", "min_us",
                 "max_us", "share_pct_of_trace_kernel_time", "n_passes_containing", "launches_per_pass_median_all",
                 "lpp_median_prefix_large", "n_passes_prefix_large", "lpp_median_prefix_small", "n_passes_prefix_small",
                 "lpp_median_suffix", "n_passes_suffix"]
MMQ_HEADER = ["trace", "pass_idx_in_trace", "request_idx", "pass_idx_in_request", "pass_kind", "n_tokens", "n_kv_padded",
              "launch_kind", "order_in_pass", "position_in_pass", "kernel_short", "quant_type", "mmq_x", "need_check",
              "ncols_dst", "has_fusion", "small_k", "halve_iters", "grid_x", "grid_y", "grid_z", "block_x", "block_y",
              "block_z", "smem_static_b", "smem_dynamic_b", "regs", "dur_us", "start_in_pass_us", "prev_quant_kernel",
              "prev_quant_position", "prev_quant_distance", "prev_quant_grid_x", "prev_quant_grid_y", "prev_quant_grid_z",
              "prev_quant_n_rows", "prev_quant_k_padded512", "prev_quant_dur_us", "prev_quant_reuse_index", "has_fixup",
              "fixup_dur_us", "fixup_short", "fixup_grid_x", "fixup_grid_y", "fixup_grid_z"]
MMQ_EXTRA = ["op", "layer_idx", "layer_type", "M", "K_quant", "expected_grid_x", "grid_matches"]   # whole-server mode only


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def legacy_main():
    problems = []
    out = []

    def say(*a):
        print(*a)

    traces = {}
    for name in TRACES:
        p = WORK / ("%s.sqlite" % name)
        if not p.exists():
            sys.exit("missing %s (legacy mode needs {long,medium,short}.sqlite in --in)" % p)
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
        kernel_rows.extend(build_kernel_rows(name, passes))
        # ---- mmq launches ---------------------------------------------------------------------------------
        mmq_rows.extend(build_mmq_rows(name, passes))
        summary[name] = t

    # ---- write TSVs ------------------------------------------------------------------------------------------
    write_tsv(OUT / "trace-kernels.tsv", KERNEL_HEADER, kernel_rows)
    write_tsv(OUT / "trace-passes.tsv", PASS_HEADER, pass_rows)
    write_tsv(OUT / "trace-mmq-launches.tsv", MMQ_HEADER, mmq_rows)

    report(summary, pass_rows, kernel_rows, mmq_rows, problems)


# ---------------------------------------------------------------------------------------------------------------
# stdout report (numbers used in the census notes)
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

    # review comparison
    print("\n" + "=" * 100 + "\ncategory share of kernel time (this export) vs the table of the earlier kernel review")
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
            row.append("%.2f (review %s)" % (v, "-" if rv is None else rv))
        rows.append(row)
    tbl(rows, ["category"] + TRACES)

    # cross-check old exports
    print("\n" + "=" * 100 + "\ncross-check against the earlier exports of --old-exports (opened read-only)")
    for name in TRACES:
        p = None if OLD_PROF is None else OLD_PROF / ("%s.sqlite" % name)
        if p is None or not p.exists():
            print("  %s: old export not given (--old-exports DIR) or not found" % name)
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


# ---------------------------------------------------------------------------------------------------------------
# whole-server mode (analyze / overhead)
# ---------------------------------------------------------------------------------------------------------------

N_BINS = [("N<=8", 0, 8), ("9-32", 9, 32), ("33-64", 33, 64), ("65-128", 65, 128), ("129-256", 129, 256),
          ("257-1024", 257, 1024), ("1025-2047", 1025, 2047), ("2048", 2048, 1 << 30)]
CLASS_OF_CAT = ["MMQ", "GDN", "glu", "FA", "quantize", "copy", "rms_norm", "ssm_conv", "elementwise", "MMVQ", "other"]
PASS_TYPES = ["prefix-large", "prefix-small", "suffix", "single"]
EXPECT_PASSES = {"short1q": 1, "short2q": 3, "medium3q": 4, "long3q": 5}   # passes per request (shape census)
N_SUFFIX = {"short1q": 1, "short2q": 2, "medium3q": 3, "long3q": 3}         # questions = suffix passes
KINDS_ORDER = ["short1q", "short2q", "medium3q", "long3q"]
STRAY_GAP_NS = 50_000_000     # a gap this long inside the kernels of a pass = activity that does not belong to it (cut there)
TAIL_CAP_NS = 5_000_000       # API/copy records after the last kernel of the capture's final pass are cut after this
WAIT_PREFIX = ("pthread_cond", "sem_", "poll", "ppoll", "select", "pselect", "epoll", "accept", "recv", "nanosleep",
               "clock_nanosleep", "usleep", "futex", "pthread_join")

NEW_COLS = ["run", "segment", "client_idx", "kind", "case", "phase", "pass_type", "n_class", "boundary", "pass_idx_global",
            "host_gap_drain_ms", "host_gap_cpu_ms", "host_gap_cuda_api_ms", "host_gap_first_launch_ms",
            "host_gap_cpu_os_wait_ms", "host_gap_cpu_busy_ms", "host_gap_n_osrt_calls", "host_gap_osrt_top",
            "client_wall_ms", "send_to_first_cuda_api_ms", "send_to_first_kernel_ms", "last_kernel_to_recv_ms",
            "window_assignment"]


def n_class(n):
    for name, lo, hi in N_BINS:
        if lo <= n <= hi:
            return name
    return "?"


def load_osrt(path, t):
    """decode-thread OSRT calls (t.osrt = [(start, end, name)] or None when the export has no OSRT_API table)"""
    t.osrt, t.osrt_info = None, {}
    tids = collections.Counter(a["tid"] for a in t.api if a["name"].startswith("cudaLaunchKernel"))
    t.decode_tid = tids.most_common(1)[0][0] if tids else None
    db = sqlite3.connect(str(path))
    try:
        if "OSRT_API" not in {r[0] for r in db.execute("select name from sqlite_master where type='table'")}:
            return
        sid = dict(db.execute("select id, value from StringIds"))
        n, nthr = db.execute("select count(*), count(distinct globalTid) from OSRT_API").fetchone()
        top = db.execute("select nameId, count(*), sum(end-start), max(end-start) from OSRT_API group by nameId "
                         "order by 3 desc limit 12").fetchall()
        rows = db.execute("select start, end, nameId from OSRT_API where globalTid=? and coalesce(nestingLevel, 0)=0 "
                          "order by start", (t.decode_tid,)).fetchall()
        t.osrt = [(r[0], r[1], sid.get(r[2], str(r[2]))) for r in rows]
        t.osrt_starts = [r[0] for r in rows]
        t.osrt_info = {"n_all": n, "n_threads": nthr, "n_decode_thread": len(rows),
                       "top": [(sid.get(r[0], str(r[0])), r[1], r[2], r[3]) for r in top]}
    finally:
        db.close()


def load_client(path, need=None):
    """client JSON -> (meta, requests sorted by idx, errors); `need` = required request fields (default: all nsys_client.py fields)"""
    j = json.load(open(path))
    reqs = sorted(j["requests"], key=lambda r: r["idx"])
    need = need or ["idx", "segment", "kind", "case", "phase", "ref", "t_send_mono_ns", "t_header_mono_ns", "t_recv_mono_ns",
                    "t_send_epoch_ns", "wall_ms", "input_tokens"]
    for r in reqs:
        miss = [f for f in need if f not in r]
        if miss:
            sys.exit("%s: request idx %s lacks fields %s" % (path, r.get("idx"), miss))
    return j.get("meta", {}), reqs, j.get("errors", [])


def map_windows(t, reqs, clock):
    """client request -> window [a, b] on the nsys timeline (ns since capture start).
    mono:  nsys_ns = CLOCK_MONOTONIC_ns - TARGET_INFO_SESSION_START_TIME.systemClockNs
    epoch: nsys_ns = epoch_ns - TARGET_INFO_SESSION_START_TIME.utcEpochNs   (end = start + monotonic duration)"""
    for r in reqs:
        dur = r["t_recv_mono_ns"] - r["t_send_mono_ns"]
        am = r["t_send_mono_ns"] - t.sys_clock_ns
        ae = r["t_send_epoch_ns"] - t.utc_epoch_ns
        r["w_mono"], r["w_epoch"] = (am, am + dur), (ae, ae + dur)
        r["w"] = r["w_mono"] if clock == "mono" else r["w_epoch"]


def segment_all(t, diag):
    """Whole-server segmentation: passes = valid anchors only. A valid anchor is a group of host-to-device copies whose next
    kernel is rms_norm_f32 with block 1024 (first op of a prompt pass) and that contains the embedding copy
    (grid.x * 10240 bytes). Everything before the first valid anchor (model load, other copies) is dropped, kernels that
    follow a pass after a gap > 50 ms are not part of it."""
    K = t.K
    kstarts = [k["start"] for k in K]
    t.api_by_corr = {a["corr"]: a for a in t.api}
    bursts = collections.OrderedDict()
    for m in t.mc:
        if m["kind"] == 1:
            bursts.setdefault(bisect.bisect_left(kstarts, m["end"]), []).append(m)
    valid, rejected = [], 0
    for a in sorted(bursts):
        if a >= len(K):
            rejected += 1
            continue
        k = K[a]
        grp = sorted(bursts[a], key=lambda m: m["start"])
        j = next((i for i, m in enumerate(grp) if m["bytes"] == k["gx"] * EMBD_BYTES_PER_TOKEN), None) \
            if (k["base"] == "rms_norm_f32" and k["bx"] == 1024) else None
        if j is None:
            rejected += 1
            continue
        valid.append((a, grp[j:]))
    diag["h2d_groups"], diag["h2d_groups_rejected"] = len(bursts), rejected
    passes, trimmed = [], 0
    for pi, (a, h2d) in enumerate(valid):
        b = valid[pi + 1][0] if pi + 1 < len(valid) else len(K)
        ks = K[a:b]
        run_end = None
        for i, k in enumerate(ks):
            if run_end is not None and k["start"] - run_end > STRAY_GAP_NS:
                trimmed += len(ks) - i
                ks = ks[:i]
                break
            run_end = k["end"] if run_end is None else max(run_end, k["end"])
        passes.append({"idx": pi, "k0": a, "k1": a + len(ks), "ks": ks, "h2d": h2d})
    diag["stray_kernels_cut"] = trimmed
    diag["kernels_before_first_pass"] = valid[0][0] if valid else len(K)
    if not passes:
        return passes
    for P in passes:
        first = P["h2d"][0]
        P["win_start"] = t.api_by_corr[first["corr"]]["start"] if first["corr"] in t.api_by_corr else first["start"]
        P["api"], P["mc"], P["sync"], P["ovh"] = [], [], [], []
    wstarts = [P["win_start"] for P in passes]
    cap = max(k["end"] for k in passes[-1]["ks"]) + TAIL_CAP_NS

    def widx(ts):
        i = bisect.bisect_right(wstarts, ts) - 1
        if i < 0 or (i == len(passes) - 1 and ts > cap):
            return None
        return i
    dropped = 0
    for a in t.api:
        i = widx(a["start"])
        if i is None:
            dropped += 1
        else:
            passes[i]["api"].append(a)
    for m in t.mc:
        a = t.api_by_corr.get(m["corr"])
        i = widx(a["start"] if a else m["start"])
        if i is not None:
            passes[i]["mc"].append(m)
    for s in t.sync:
        a = t.api_by_corr.get(s["corr"])
        i = widx(a["start"] if a else s["start"])
        if i is not None:
            passes[i]["sync"].append(s)
    for o in t.ovh:
        a = t.api_by_corr.get(o["corr"]) if o["corr"] else None
        i = widx(a["start"] if a else o["start"])
        if i is not None:
            passes[i]["ovh"].append(o)
    diag["api_records_outside_passes"] = dropped
    return passes


def assign_windows(passes, reqs):
    """pass idx -> list index of the request whose [send, recv] window contains the pass's first kernel start"""
    wins = sorted(((r["w"][0], r["w"][1], i) for i, r in enumerate(reqs)))
    starts = [w[0] for w in wins]
    out = {}
    for P in passes:
        s = P["ks"][0]["start"]
        j = bisect.bisect_right(starts, s) - 1
        if j >= 0 and s <= wins[j][1]:
            out[P["idx"]] = wins[j][2]
    return out


def assign_order(passes, reqs, note):
    """pass idx -> request list index from the client order and the expected pass structure (shape census), parsed
    backwards from the last pass: short1q = 1 pass with outputs; multi-pass kinds = (E - q) prefix passes then q suffix
    passes (q = number of questions). Stops (and notes it) at the first request that does not fit."""
    out = {}
    cur = len(passes)
    for ri in range(len(reqs) - 1, -1, -1):
        kind = reqs[ri]["kind"]
        if kind not in EXPECT_PASSES:
            note.append("order assignment: unknown request kind %r" % kind)
            break
        e, q = EXPECT_PASSES[kind], N_SUFFIX[kind]
        if cur - e < 0:
            note.append("order assignment: ran out of passes at client request idx %s" % reqs[ri]["idx"])
            break
        blk = passes[cur - e:cur]
        shape = [P["kind"] == "suffix" for P in blk]
        if shape != [False] * (e - q) + [True] * q:
            note.append("order assignment: passes %d..%d do not have the structure of a %s request (client idx %s); stopped"
                        % (blk[0]["idx"], blk[-1]["idx"], kind, reqs[ri]["idx"]))
            break
        for P in blk:
            out[P["idx"]] = ri
        cur -= e
    return out


# ---------------------------------------------------------------------------------------------------------------
# host-side gap between two passes (pre_gap = start of the pass's first kernel - end of the previous pass's last kernel)
# ---------------------------------------------------------------------------------------------------------------

def merge_iv(iv):
    out = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap_ns(a, b):
    """total overlap of two sorted lists of disjoint [s, e] intervals"""
    i = j = tot = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def host_gap(t, Q, P):
    """Splits [Q.end, P.start] (Q = previous pass, P = this pass) into non-overlapping parts that add up exactly:
      drain        CUDA API call that was already running when the GPU finished Q (stream sync / logits copy) until it returns
      cpu          no CUDA API call in flight (host code: sampling, HTTP, tokenisation, batching ...); OSRT split when present
      cuda_api     other CUDA API calls up to the first launch of P (input upload copies, syncs)
      first_launch from the first cudaLaunchKernel of P to the start of its first kernel
    Only calls of the decode thread are used."""
    t0, t1 = Q["end"], P["start"]
    la = [a["start"] for a in P["launch_api"]]
    tl = min(la) if la else t1
    split = min(max(tl, t0), t1)
    api, starts = t.api_dec, t.api_dec_starts
    j = max(bisect.bisect_right(starts, t0) - 1, 0)
    drain = cuda = cpu = 0
    cur = t0
    cpu_iv = []
    while j < len(api) and api[j]["start"] < split:
        c = api[j]
        j += 1
        if c["end"] <= t0:
            continue
        s, e = max(c["start"], cur), min(c["end"], split)
        if s > cur:
            cpu += s - cur
            cpu_iv.append([cur, s])
        if e > s:
            if c["start"] < t0:
                drain += e - s
            else:
                cuda += e - s
        cur = max(cur, e)
    if split > cur:
        cpu += split - cur
        cpu_iv.append([cur, split])
    res = {"drain": drain, "cpu": cpu, "cuda_api": cuda, "first_launch": t1 - split, "total": t1 - t0,
           "wait": None, "n_osrt": None, "osrt_top": ""}
    if t.osrt is not None and cpu_iv:
        iv = merge_iv(cpu_iv)
        i = max(bisect.bisect_right(t.osrt_starts, iv[0][0]) - 1, 0)   # top-level calls of one thread do not overlap
        names = collections.Counter()
        wait_iv, n = [], 0
        while i < len(t.osrt):
            s, e, name = t.osrt[i]
            i += 1
            if s >= iv[-1][1]:
                break
            ov = overlap_ns(iv, [[s, e]])
            if ov > 0:
                n += 1
                names[name] += ov
                if name.startswith(WAIT_PREFIX):
                    wait_iv.append([s, e])
        res["wait"] = overlap_ns(iv, merge_iv(wait_iv))
        res["n_osrt"] = n
        res["osrt_top"] = ";".join("%s:%.3f" % (k, v / 1e6) for k, v in names.most_common(3))
    elif t.osrt is not None:
        res["wait"], res["n_osrt"] = 0, 0
    return res


# ---------------------------------------------------------------------------------------------------------------
# MMQ / MMVQ launch -> weight tensor (derived from the layer structure, not from names: kernels carry no tensor names)
# ---------------------------------------------------------------------------------------------------------------

# (quant type, K) of the weight matrices of the model (census Table 5): token and rows M of the output
TOKENS = {("Q8_0", 2560): "A", ("Q8_0", 4096): "B", ("Q4_0", 2560): "C", ("Q4_0", 9216): "D"}
OP_M = {"attn_qkv": 8192, "ssm_beta|ssm_alpha": 32, "attn_gate(z)": 4096, "ssm_out": 2560, "attn_q": 8192,
        "attn_k|attn_v": 1024, "attn_output": 2560, "ffn_up|ffn_gate": 9216, "ffn_gate+up(fused)": 9216, "ffn_down": 2560}
LAYER_GDN = ["attn_qkv", "ssm_beta|ssm_alpha", "ssm_beta|ssm_alpha", "attn_gate(z)"]     # launch order inside a GDN layer
LAYER_ATTN = ["attn_q", "attn_k|attn_v", "attn_k|attn_v"]


def derive_ops(P):
    """{position_in_pass: {op, layer_idx, layer_type, M, K, ...}} for every main mul_mat_q / mul_mat_vec_q launch of the pass;
    P['ops_stats'] = counters for the validation table. The tensor of a launch is derived from (quantisation type, K of the
    preceding quantize kernel), the position inside the layer (the layer ends at the ffn_down launch) and the layer type
    (4 Q8_0-K2560 launches = Gated-DeltaNet layer, 3 = attention layer)."""
    ks, ops = P["ks"], {}
    toks = []   # (pos, token, extra)
    last_q = {"mmq": None, "mmvq": None}
    for pos, k in enumerate(ks):
        b = k["base"]
        if b == "quantize_mmq_q8_1":
            last_q["mmq"] = k["gy"] * 512
        elif b == "quantize_q8_1":
            last_q["mmvq"] = k["gx"] * 256
        elif b in ("mul_mat_q", "mul_mat_vec_q"):
            lk = "mmq" if b == "mul_mat_q" else "mmvq"
            args = [norm_arg(a) for a in split_name(k["full"])[1]]
            kk = last_q[lk]
            fusion = (args[4] if lk == "mmq" else args[2]) == "true" if len(args) > 4 else False
            nxt = ks[pos + 1] if pos + 1 < len(ks) else None
            info = {"pos": pos, "lk": lk, "type": args[0], "K": kk, "fusion": fusion, "k": k,
                    "J": int(args[1]) if args[1].isdigit() else None,
                    "fix": lk == "mmq" and nxt is not None and nxt["base"] == "mul_mat_q_stream_k_fixup"}
            if lk == "mmvq" and k["gx"] == N_VOCAB:
                tok = "H"
            elif fusion and (args[0], kk) == ("Q4_0", 2560):
                tok = "F"
            else:
                tok = TOKENS.get((args[0], kk), "?")
            info["tok"] = tok
            toks.append(info)
    layers, cur = [], []
    for it in toks:
        if it["tok"] == "H":
            continue
        cur.append(it)
        if it["tok"] == "D":
            layers.append(cur)
            cur = []
    if cur:
        layers.append(cur)
    st = {"n_launches": len(toks), "n_mapped": 0, "n_layers": len(layers), "layer_types": "", "grid_checked": 0, "grid_ok": 0,
          "mmvq_grid_checked": 0, "mmvq_grid_ok": 0}
    types = []
    for li, lay in enumerate(layers):
        s = "".join(i["tok"] for i in lay)
        ltype, names = None, None
        for pat, lt in (("AAAAB", "GDN"), ("AAAB", "ATTN")):
            if s.startswith(pat) and s[len(pat):] in ("CCD", "FD", "", "C", "CC", "F"):
                ltype = lt
                tail = {"CCD": ["ffn_up|ffn_gate", "ffn_up|ffn_gate", "ffn_down"], "FD": ["ffn_gate+up(fused)", "ffn_down"]}.get(s[len(pat):], [])
                names = (LAYER_GDN if lt == "GDN" else LAYER_ATTN) + ["ssm_out" if lt == "GDN" else "attn_output"] + tail
                break
        types.append(ltype or "?")
        for i, it in enumerate(lay):
            op = names[i] if names and i < len(names) else "?"
            it["op"], it["layer_idx"], it["layer_type"] = op, li, ltype or "?"
    # tokens never reached (head) or unmapped keep op '?'
    for it in toks:
        if it["tok"] == "H":
            it.update({"op": "output_head", "layer_idx": "", "layer_type": "head"})
        N = P["n_tokens"]
        m = OP_M.get(it.get("op"))
        exp = None
        if m:
            st["n_mapped"] += 1
            if it["lk"] == "mmq" and not it["fix"] and it["J"]:
                exp = -(-m // 128) * -(-N // it["J"])
                st["grid_checked"] += 1
                st["grid_ok"] += exp == it["k"]["gx"]
            elif it["lk"] == "mmvq":
                nc_ = int(split_name(it["k"]["full"])[1][1]) if split_name(it["k"]["full"])[1][1].isdigit() else None
                if nc_:
                    exp = m if nc_ == 1 else -(-m // 2)
                    st["mmvq_grid_checked"] += 1
                    st["mmvq_grid_ok"] += exp == it["k"]["gx"]
        elif it["tok"] == "H":
            st["n_mapped"] += 1
        ops[it["pos"]] = {"op": it.get("op", "?"), "layer_idx": it.get("layer_idx", ""), "layer_type": it.get("layer_type", "?"),
                          "M": m if m else (N_VOCAB if it["tok"] == "H" else None), "K_quant": it["K"],
                          "expected_grid_x": exp, "grid_matches": ("" if exp is None else int(exp == it["k"]["gx"])),
                          "wtype": it["type"], "fusion": it["fusion"]}
    # pattern: layer l is attention iff l % 4 == 3, 32 layers (24 GDN + 8 attention)
    st["layer_pattern_ok"] = int(len(types) == 32 and all(t_ == ("ATTN" if i % 4 == 3 else "GDN") for i, t_ in enumerate(types)))
    st["layer_types"] = "".join({"GDN": "G", "ATTN": "A", "?": "?"}[x] for x in types)
    P["ops_stats"] = st
    return ops


def kernel_labels(P, ops):
    """[(class, label)] aligned with P['ks']. MMQ stream-k fix-up kernels take the label of the launch they belong to."""
    out = []
    prev_label = None
    for pos, k in enumerate(P["ks"]):
        cls = CLASS_OF_CAT[k["cat"]]
        b = k["base"]
        if b in ("mul_mat_q", "mul_mat_vec_q") and pos in ops:
            o = ops[pos]
            lab = "%s %s %s K=%s" % (cls, o["op"], o["wtype"], o["K_quant"])
            if b == "mul_mat_vec_q" and o["fusion"]:
                lab += " fused"
            prev_label = lab if b == "mul_mat_q" else None
        elif b == "mul_mat_q_stream_k_fixup" and prev_label:
            lab = prev_label
            prev_label = None
        else:
            lab = k["short"]
            prev_label = None
        out.append((cls, lab))
    return out


# ---------------------------------------------------------------------------------------------------------------
# analyze: one run
# ---------------------------------------------------------------------------------------------------------------

def pctl(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def validate_pass(P, name, problems):
    if P["n_gdn"] != 24:
        problems.append("%s pass %d: n_gdn=%d" % (name, P["idx"], P["n_gdn"]))
    if P["n_fa"] != 8:
        problems.append("%s pass %d: n_flash_attn=%d" % (name, P["idx"], P["n_fa"]))
    if P["n_launch_api"] != len(P["ks"]):
        problems.append("%s pass %d: launch API calls %d != kernels %d" % (name, P["idx"], P["n_launch_api"], len(P["ks"])))
    nrms = sum(1 for k in P["ks"] if k["base"] == "rms_norm_f32" and k["bx"] == 1024)
    exp = 65 if P["kind"] == "suffix" else 63
    if nrms != exp:
        problems.append("%s pass %d (%s): %d rms_norm<1024> launches, expected %d" % (name, P["idx"], P["kind"], nrms, exp))
    if len(P["streams"].split("+")) != 1:
        problems.append("%s pass %d: kernels on several streams (%s)" % (name, P["idx"], P["streams"]))
    qn = collections.Counter(k["gx"] for k in P["ks"] if k["base"] == "quantize_mmq_q8_1")
    if qn and qn.most_common(1)[0][0] != P["n_tokens"]:
        problems.append("%s pass %d: modal quantize_mmq grid.x %d != n_tokens %d" % (name, P["idx"], qn.most_common(1)[0][0], P["n_tokens"]))


def analyze_run(run, db_path, client_path, args):
    R = {"run": run, "problems": [], "notes": []}
    problems = R["problems"]
    t = load_trace(run, db_path)
    load_osrt(db_path, t)
    if args.no_osrt:
        t.osrt = None
    meta, reqs, cerrors = load_client(client_path)
    if t.utc_epoch_ns is None or t.sys_clock_ns is None:
        sys.exit("%s: TARGET_INFO_SESSION_START_TIME missing in the export" % run)
    map_windows(t, reqs, args.clock)
    for e in cerrors:
        problems.append("%s: client error at idx %s: %s" % (run, e.get("idx"), e.get("error")))
    diag = {}
    allp = segment_all(t, diag)
    R["info"] = {"run": run, "kernels": len(t.K), "api": len(t.api), "memcpy": len(t.mc), "sync": len(t.sync),
                 "duration_s": (t.duration_ns or 0) / 1e9, "osrt_all": (t.osrt_info or {}).get("n_all"),
                 "osrt_decode": (t.osrt_info or {}).get("n_decode_thread"), "osrt_threads": (t.osrt_info or {}).get("n_threads"),
                 "osrt_top": (t.osrt_info or {}).get("top", []), "passes_valid": len(allp), "requests": len(reqs),
                 "requests_warmup": sum(1 for r in reqs if r["phase"] == "warmup"),
                 "utc_epoch_ns": t.utc_epoch_ns, "sys_clock_ns": t.sys_clock_ns, "decode_tid": t.decode_tid,
                 "gpu": t.dev[0], "has_osrt": t.osrt is not None, "has_cuda": bool(t.K), **diag}
    if not allp:
        problems.append("%s: no valid pass found in the capture" % run)
        return R
    for P in allp:
        classify(P, [], run)
    t.api_dec = [a for a in t.api if a["tid"] == t.decode_tid]
    t.api_dec_starts = [a["start"] for a in t.api_dec]
    # ---- which passes belong to which client request ------------------------------------------------------
    by_win = assign_windows(allp, reqs)
    onotes = []
    by_ord = assign_order(allp, reqs, onotes)
    both = set(by_win) & set(by_ord)
    R["assign"] = {"window": len(by_win), "order": len(by_ord), "same": sum(1 for i in both if by_win[i] == by_ord[i]),
                   "conflict": sum(1 for i in both if by_win[i] != by_ord[i]), "only_window": len(set(by_win) - set(by_ord)),
                   "only_order": len(set(by_ord) - set(by_win)), "notes": onotes, "used": args.assign}
    chosen = by_win if args.assign == "window" else by_ord
    ana = [P for P in allp if P["idx"] in chosen]
    R["info"]["passes_analysed"] = len(ana)
    R["info"]["passes_not_in_a_request"] = len(allp) - len(ana)
    R["info"]["unassigned_first_n"] = [P["n_tokens"] for P in allp if P["idx"] not in chosen][:6]
    if not ana:
        problems.append("%s: no pass could be assigned to a client request (assign=%s)" % (run, args.assign))
        return R
    per_req = collections.defaultdict(list)
    for P in ana:
        ri = chosen[P["idx"]]
        r = reqs[ri]
        P["req_i"], P["request"], P["case"], P["phase"], P["rkind"], P["segment"] = ri, r["idx"], r["case"], r["phase"], r["kind"], r["segment"]
        per_req[ri].append(P)
    for ri, ps in per_req.items():
        for j, P in enumerate(ps):
            P["pass_in_req"] = j
            P["pass_type"] = "single" if len(ps) == 1 else P["kind"]
        if len(ps) == 1 and ps[0]["kind"] != "suffix":
            problems.append("%s: request idx %d has a single pass without outputs" % (run, reqs[ri]["idx"]))
        e = EXPECT_PASSES.get(reqs[ri]["kind"])
        if e and len(ps) != e:
            problems.append("%s: request idx %d (%s) has %d passes, expected %d" % (run, reqs[ri]["idx"], reqs[ri]["kind"], len(ps), e))
    for ri, r in enumerate(reqs):
        if ri not in per_req:
            problems.append("%s: client request idx %d (%s %s, %s) has no pass in the capture" % (run, r["idx"], r["kind"], r["case"], r["phase"]))
    for P in ana:
        classify(P, problems, run)
    prev = None
    for P in ana:
        pass_metrics(t, P, prev)
        prev = P
    t0 = ana[0]["ks"][0]["start"]
    if t.osrt:
        pass
    prev = None
    for P in ana:
        P["hg"] = host_gap(t, prev, P) if prev else None
        if prev is None:
            P["boundary"] = "first"
        elif P["pass_in_req"] == 0:
            P["boundary"] = "request-start"
        else:
            P["boundary"] = "%s->%s" % (prev["pass_type"], P["pass_type"])
        if P["pass_in_req"] == 0:
            P["idle_before_request"] = P["pre_gap"]
        prev = P
        validate_pass(P, run, problems)
    # sequence identity among passes with equal (kind, n_tokens)
    groups = collections.defaultdict(list)
    for P in ana:
        groups[(P["kind"], P["n_tokens"])].append(P)
    seq_bad = []
    for key, ps in groups.items():
        s0 = [k["short"] for k in ps[0]["ks"]]
        for P in ps[1:]:
            if [k["short"] for k in P["ks"]] != s0:
                seq_bad.append((key, ps[0]["idx"], P["idx"]))
    if seq_bad:
        problems.append("%s: passes with equal (kind, n_tokens) but different kernel sequences: %s" % (run, seq_bad[:6]))
    R["info"]["pass_groups_equal_shape"] = sum(len(v) - 1 for v in groups.values())
    R["info"]["seq_identical"] = sum(len(v) - 1 for v in groups.values()) - len(seq_bad)
    # ---- derived: ops, labels -------------------------------------------------------------------------------
    ops_tot = collections.Counter()
    for P in ana:
        P["ops"] = derive_ops(P)
        P["labels"] = kernel_labels(P, P["ops"])
        for k_, v in P["ops_stats"].items():
            if isinstance(v, int):
                ops_tot[k_] += v
        ops_tot["passes"] += 1
        if not P["ops_stats"]["layer_pattern_ok"] and P["n_tokens"] > 0:
            problems.append("%s pass %d: MMQ/MMVQ layer parse failed (%d layers: %s)" % (run, P["idx"], P["ops_stats"]["n_layers"], P["ops_stats"]["layer_types"]))
    R["ops"] = dict(ops_tot)
    # ---- request level numbers and mapping check ---------------------------------------------------------------
    R["map"], R["req"] = [], []
    for ri in sorted(per_req):
        ps, r = per_req[ri], reqs[ri]
        first = ps[0]["ks"][0]["start"]
        last = max(P["end"] for P in ps)
        a, b = r["w"]
        m = {"kind": r["kind"], "phase": r["phase"], "idx": r["idx"], "L0": first - a, "L1": b - last,
             "L0_mono": first - r["w_mono"][0], "L1_mono": r["w_mono"][1] - last,
             "L0_epoch": first - r["w_epoch"][0], "L1_epoch": r["w_epoch"][1] - last,
             "epoch_minus_mono": r["w_epoch"][0] - r["w_mono"][0]}
        i = bisect.bisect_left(t.api_dec_starts, a)
        m["first_api"] = (t.api_dec[i]["start"] - a) if i < len(t.api_dec) and t.api_dec[i]["start"] <= first else None
        r["_first_api_ns"] = m["first_api"]
        R["map"].append(m)
        busy = union_ns([(k["start"], k["end"]) for P in ps for k in P["ks"]])
        intra = sum(P["gap_total"] for P in ps)
        R["req"].append({"kind": r["kind"], "phase": r["phase"], "idx": r["idx"], "wall": r["wall_ms"], "L0": m["L0"] / 1e6,
                         "L1": m["L1"] / 1e6, "first_api": None if m["first_api"] is None else m["first_api"] / 1e6,
                         "busy": busy / 1e6, "intra": intra / 1e6, "n_passes": len(ps),
                         "inter": (last - first - busy - intra) / 1e6, "span": (last - first) / 1e6,
                         "n_api": sum(len(P["api"]) for P in ps), "n_launch": sum(P["n_launch_api"] for P in ps),
                         "n_sync": sum(P["n_sync"] for P in ps), "n_mc": sum(P["n_mc"] for P in ps),
                         "segment": r["segment"], "case": r["case"], "input_tokens": r["input_tokens"],
                         "L0_epoch": m["L0_epoch"] / 1e6, "L1_epoch": m["L1_epoch"] / 1e6, "off": m["epoch_minus_mono"] / 1e6})
    R["osrt_start"] = []
    if t.osrt is not None:
        seen_seg = set()
        for ri in sorted(per_req):
            r = reqs[ri]
            if r["phase"] == "warmup" or r["segment"] in seen_seg:
                continue
            seen_seg.add(r["segment"])
            R["osrt_start"].extend(osrt_window_rows(db_path, t, run, r, r["w"][0] - 1_000_000, per_req[ri][0]["ks"][0]["start"]))
    # ---- rows ----------------------------------------------------------------------------------------------
    def hg_cols(P):
        h = P["hg"]
        if not h:
            return {}
        d = {"host_gap_drain_ms": ms(h["drain"]), "host_gap_cpu_ms": ms(h["cpu"]), "host_gap_cuda_api_ms": ms(h["cuda_api"]),
             "host_gap_first_launch_ms": ms(h["first_launch"])}
        if h["wait"] is not None:
            d.update({"host_gap_cpu_os_wait_ms": ms(h["wait"]), "host_gap_cpu_busy_ms": ms(h["cpu"] - h["wait"]),
                      "host_gap_n_osrt_calls": h["n_osrt"], "host_gap_osrt_top": h["osrt_top"]})
        return d
    pass_rows = []
    for P in ana:
        row = pass_row(t, P, t0)
        row.update({"run": run, "segment": P["segment"], "client_idx": P["request"], "kind": P["rkind"], "case": P["case"],
                    "phase": P["phase"], "pass_type": P["pass_type"], "n_class": n_class(P["n_tokens"]),
                    "boundary": P["boundary"], "pass_idx_global": P["idx"], "window_assignment": "window" if args.assign == "window" else "order"})
        row.update(hg_cols(P))
        pass_rows.append(row)
    prev_req_end = None
    for ri in sorted(per_req):
        ps, r = per_req[ri], reqs[ri]
        ex = {"request_idx": r["idx"], "case_id_inferred": r["case"], "pass_kind": "request",
              "pass_kind_evidence": "prefix_passes=%d;suffix_passes=%d" % (sum(1 for P in ps if P["kind"].startswith("prefix")), sum(1 for P in ps if P["kind"] == "suffix")),
              "idle_before_request_ms": ms(ps[0].get("idle_before_request")), "pre_gap_ms": ms(ps[0]["pre_gap"]),
              "n_outputs": sum(P["n_outputs"] or 0 for P in ps)}
        row = agg_row("request", run, ps, t0, extra=ex)
        rend = max(P["end"] for P in ps)
        span_start = prev_req_end if prev_req_end is not None else ps[0]["start"]
        row["gpu_busy_pct_of_span_incl_idle_before"] = 100.0 * sum(P["busy"] for P in ps) / (rend - span_start)
        prev_req_end = rend
        m = next(x for x in R["map"] if x["idx"] == r["idx"])
        row.update({"run": run, "segment": r["segment"], "client_idx": r["idx"], "kind": r["kind"], "case": r["case"],
                    "phase": r["phase"], "pass_type": "request", "client_wall_ms": r["wall_ms"],
                    "send_to_first_cuda_api_ms": ms(m["first_api"]), "send_to_first_kernel_ms": ms(m["L0"]),
                    "last_kernel_to_recv_ms": ms(m["L1"]), "window_assignment": "window" if args.assign == "window" else "order"})
        pass_rows.append(row)
    ex = {"pass_kind": "trace_total", "pass_kind_evidence": "requests=%d" % len(per_req), "n_outputs": sum(P["n_outputs"] or 0 for P in ana)}
    row = agg_row("trace_total", run, ana, t0, extra=ex)
    row.update({"run": run, "segment": ",".join(sorted({r["segment"] for r in reqs})), "kind": "all", "pass_type": "trace_total"})
    pass_rows.append(row)
    R["pass_rows"] = pass_rows
    use = [P for P in ana if args.include_warmup or P["phase"] != "warmup"]
    R["kernel_rows"] = build_kernel_rows(run, use) if use else []
    R["mmq_rows"] = build_mmq_rows(run, ana, extra=lambda P: P["ops"])
    # compact per-pass records for classes / boundary tables
    R["stat"] = []
    for P in use:
        cls = collections.defaultdict(lambda: [0, 0])
        lab = collections.defaultdict(lambda: [0, 0, 0])
        for k, (c, l) in zip(P["ks"], P["labels"]):
            d = k["end"] - k["start"]
            cls[c][0] += 1
            cls[c][1] += d
            e = lab[(c, l)]
            if k["base"] == "mul_mat_q_stream_k_fixup" and l != k["short"]:
                e[1] += d
                e[2] += 1
            else:
                e[0] += 1
                e[1] += d
        R["stat"].append({"pass_type": P["pass_type"], "n_class": n_class(P["n_tokens"]), "kind": P["rkind"], "n_tokens": P["n_tokens"],
                          "busy": P["busy"], "wall": P["wall"], "ksum": P["ksum"], "cls": dict(cls), "lab": dict(lab),
                          "boundary": P["boundary"], "hg": P["hg"], "pre_gap": P["pre_gap"], "n_launch": P["n_launch_api"],
                          "n_kernels": len(P["ks"])})
    return R


# ---------------------------------------------------------------------------------------------------------------
# analyze: tables and files
# ---------------------------------------------------------------------------------------------------------------

N_ORDER = {name: i for i, (name, _, _) in enumerate(N_BINS)}
CLASS_HEADER = ["scope", "pass_type", "n_class", "request_kinds", "n_passes", "n_tokens_min", "n_tokens_max", "level", "class",
                "label", "launches_per_pass_mean", "launches_per_pass_p50", "kernel_ms_per_pass_mean", "kernel_ms_per_pass_p50",
                "share_of_gpu_busy_pct", "mean_us_per_launch", "fixup_launches_per_pass_mean"]
REQ_HEADER = ["run", "segment", "client_idx", "kind", "case", "phase", "input_tokens", "n_passes", "client_wall_ms",
              "send_to_first_cuda_api_ms", "send_to_first_kernel_ms", "kernel_span_ms", "gpu_busy_ms", "gap_in_pass_ms",
              "gap_between_passes_ms", "last_kernel_to_recv_ms", "gpu_busy_pct_of_client_wall", "n_cuda_api_calls",
              "n_launch_calls", "n_sync_calls", "n_memcpy", "send_to_first_kernel_epoch_clock_ms",
              "last_kernel_to_recv_epoch_clock_ms", "epoch_minus_mono_send_ms"]
OSRT_HEADER = ["run", "client_idx", "case", "thread", "call", "start_rel_send_ms", "dur_ms", "class"]


class Doc:
    """prints every table to stdout and collects the same tables as markdown text (summary.txt)"""

    def __init__(self):
        self.md = []

    @staticmethod
    def cell(v, nd=3):
        if v is None:
            return ""
        if isinstance(v, float):
            return "%.*f" % (nd, v)
        return str(v)

    def h(self, text, level=2):
        print("\n" + text + "\n" + "-" * len(text))
        self.md.append("\n%s %s\n" % ("#" * level, text))

    def p(self, text):
        print(text)
        self.md.append(text + "\n")

    def table(self, title, header, rows, source):
        cells = [[self.cell(v) for v in r] for r in rows]
        w = [max([len(str(h))] + [len(r[i]) for r in cells]) for i, h in enumerate(header)]
        print("\n" + title)
        print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(header)))
        for r in cells:
            print("  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r)))
        print("source: " + source)
        self.md.append("\n**%s**\n\n| %s |\n|%s|" % (title, " | ".join(map(str, header)), "|".join("---" for _ in header)))
        for r in cells:
            self.md.append("| %s |" % " | ".join(c.replace("|", "\\|") for c in r))
        self.md.append("\n_Source: %s_\n" % source)


def class_rows(stats_by_scope):
    rows = []
    for scope, stats in stats_by_scope.items():
        groups = collections.defaultdict(list)
        for s in stats:
            groups[(s["pass_type"], s["n_class"])].append(s)
        for (pt, nc), ss in sorted(groups.items(), key=lambda x: (PASS_TYPES.index(x[0][0]) if x[0][0] in PASS_TYPES else 9, N_ORDER.get(x[0][1], 99))):
            kinds = collections.Counter(s["kind"] for s in ss)
            base = {"scope": scope, "pass_type": pt, "n_class": nc, "n_passes": len(ss),
                    "request_kinds": ";".join("%s:%d" % kv for kv in sorted(kinds.items())),
                    "n_tokens_min": min(s["n_tokens"] for s in ss), "n_tokens_max": max(s["n_tokens"] for s in ss)}
            busy = sum(s["busy"] for s in ss)
            nk = [s["n_kernels"] for s in ss]
            ks = [s["ksum"] / 1e6 for s in ss]
            rows.append({**base, "level": "total", "launches_per_pass_mean": mean(nk), "launches_per_pass_p50": pctl(nk, 50),
                         "kernel_ms_per_pass_mean": mean(ks), "kernel_ms_per_pass_p50": pctl(ks, 50),
                         "share_of_gpu_busy_pct": 100.0 * sum(s["ksum"] for s in ss) / busy,
                         "mean_us_per_launch": sum(s["ksum"] for s in ss) / sum(nk) / 1e3})
            for c in CLASS_OF_CAT:
                n_ = [s["cls"].get(c, (0, 0))[0] for s in ss]
                t_ = [s["cls"].get(c, (0, 0))[1] / 1e6 for s in ss]
                if not any(n_):
                    continue
                rows.append({**base, "level": "class", "class": c, "launches_per_pass_mean": mean(n_),
                             "launches_per_pass_p50": pctl(n_, 50), "kernel_ms_per_pass_mean": mean(t_),
                             "kernel_ms_per_pass_p50": pctl(t_, 50), "share_of_gpu_busy_pct": 100.0 * sum(t_) * 1e6 / busy,
                             "mean_us_per_launch": sum(t_) * 1e3 / sum(n_)})
            labels = collections.OrderedDict()
            for s in ss:
                for key in s["lab"]:
                    labels.setdefault(key, None)
            lrows = []
            for (c, l) in labels:
                n_ = [s["lab"].get((c, l), (0, 0, 0))[0] for s in ss]
                t_ = [s["lab"].get((c, l), (0, 0, 0))[1] / 1e6 for s in ss]
                f_ = [s["lab"].get((c, l), (0, 0, 0))[2] for s in ss]
                lrows.append({**base, "level": "label", "class": c, "label": l, "launches_per_pass_mean": mean(n_),
                              "launches_per_pass_p50": pctl(n_, 50), "kernel_ms_per_pass_mean": mean(t_),
                              "kernel_ms_per_pass_p50": pctl(t_, 50), "share_of_gpu_busy_pct": 100.0 * sum(t_) * 1e6 / busy,
                              "mean_us_per_launch": (sum(t_) * 1e3 / sum(n_)) if sum(n_) else None,
                              "fixup_launches_per_pass_mean": mean(f_) if any(f_) else None})
            rows.extend(sorted(lrows, key=lambda r: -r["kernel_ms_per_pass_mean"]))
    return rows


def req_rows(R):
    rows = []
    for q in R.get("req", []):
        rows.append({"run": R["run"], "segment": q["segment"], "client_idx": q["idx"], "kind": q["kind"], "case": q["case"],
                     "phase": q["phase"], "input_tokens": q["input_tokens"], "n_passes": q["n_passes"],
                     "client_wall_ms": q["wall"], "send_to_first_cuda_api_ms": q["first_api"], "send_to_first_kernel_ms": q["L0"],
                     "kernel_span_ms": q["span"], "gpu_busy_ms": q["busy"], "gap_in_pass_ms": q["intra"],
                     "gap_between_passes_ms": q["inter"], "last_kernel_to_recv_ms": q["L1"],
                     "gpu_busy_pct_of_client_wall": 100.0 * q["busy"] / q["wall"] if q["wall"] else None,
                     "n_cuda_api_calls": q["n_api"], "n_launch_calls": q["n_launch"], "n_sync_calls": q["n_sync"], "n_memcpy": q["n_mc"],
                     "send_to_first_kernel_epoch_clock_ms": q["L0_epoch"], "last_kernel_to_recv_epoch_clock_ms": q["L1_epoch"],
                     "epoch_minus_mono_send_ms": q["off"]})
    return rows


def fmt3(v):
    return None if v is None else float(v)


def summarize(doc, Rs, args, wall_s):
    ok = [R for R in Rs if "pass_rows" in R]
    doc.h("nsys census: whole-server captures", 1)
    doc.p("clock mapping: %s; pass-to-request assignment: %s; warm-up requests %s in kernels/classes/boundary statistics; "
          "all times in ms unless a header says otherwise." % (args.clock, args.assign, "included" if args.include_warmup else "excluded"))
    doc.h("1. Capture inventory")
    doc.table("Table 1: what each capture contains",
              ["run", "client requests (warm-up)", "valid passes", "passes analysed", "passes outside any request", "GPU kernels",
               "CUDA API records", "OSRT calls, decode thread", "OSRT calls, all threads", "capture length (s)", "kernels before first pass",
               "H2D copy groups that are no pass start", "kernels cut off the end of a pass (gap > 50 ms)"],
              [[R["run"], "%d (%d)" % (R["info"]["requests"], R["info"]["requests_warmup"]), R["info"]["passes_valid"],
                R["info"].get("passes_analysed", 0), R["info"].get("passes_not_in_a_request", ""), R["info"]["kernels"],
                R["info"]["api"], R["info"]["osrt_decode"] if R["info"]["has_osrt"] else "no OSRT table",
                R["info"]["osrt_all"] if R["info"]["has_osrt"] else "", float(R["info"]["duration_s"]),
                R["info"].get("kernels_before_first_pass"), R["info"].get("h2d_groups_rejected"), R["info"].get("stray_kernels_cut")] for R in Rs],
              "<run>.sqlite (CUPTI_ACTIVITY_KIND_KERNEL / _RUNTIME, OSRT_API, ANALYSIS_DETAILS) and <run>.client.json")
    doc.h("2. Request windows on the nsys timeline (mapping check)")
    rows = []
    for R in ok:
        by = collections.defaultdict(list)
        for m in R["map"]:
            by[m["kind"]].append(m)
        for kind in KINDS_ORDER + sorted(set(by) - set(KINDS_ORDER)):
            ms_ = by.get(kind)
            if not ms_:
                continue
            l0, l1 = [m["L0"] / 1e6 for m in ms_], [m["L1"] / 1e6 for m in ms_]
            off = [abs(m["epoch_minus_mono"]) / 1e6 for m in ms_]
            rows.append([R["run"], kind, len(ms_), pctl(off, 50), max(off), min(l0), pctl(l0, 50), max(l0), min(l1), pctl(l1, 50),
                         max(l1), sum(1 for m in ms_ if m["L0"] < 0 or m["L1"] < 0)])
    doc.table("Table 2: first kernel after request send and recv after last kernel (both must be >= 0)",
              ["run", "request kind", "requests", "|epoch clock - mono clock| at send, median (ms)", "same, max (ms)",
               "first kernel - send, min (ms)", "same, median (ms)", "same, max (ms)", "recv - last kernel end, min (ms)",
               "same, median (ms)", "same, max (ms)", "requests with a negative value"],
              rows, "client JSON stamps (t_send_mono_ns, t_recv_mono_ns, t_send_epoch_ns) minus TARGET_INFO_SESSION_START_TIME (systemClockNs / utcEpochNs); kernel times from CUPTI_ACTIVITY_KIND_KERNEL")
    doc.table("Table 3: passes per request, window assignment against order-based assignment",
              ["run", "assignment used", "passes assigned by window", "passes assigned by order", "same request", "different request",
               "only by window", "only by order", "notes"],
              [[R["run"], R["assign"]["used"], R["assign"]["window"], R["assign"]["order"], R["assign"]["same"], R["assign"]["conflict"],
                R["assign"]["only_window"], R["assign"]["only_order"], " / ".join(R["assign"]["notes"]) or "-"] for R in ok],
              "segment_all(): valid pass anchors; assign_windows(): first kernel inside [send, recv]; assign_order(): backward parse with 1/3/4/5 passes per request kind")
    doc.h("3. Where the client wall time goes")
    rows = []
    for R in ok:
        by = collections.defaultdict(list)
        for q in R["req"]:
            if args.include_warmup or q["phase"] != "warmup":
                by[q["kind"]].append(q)
        for kind in KINDS_ORDER + sorted(set(by) - set(KINDS_ORDER)):
            qs = by.get(kind)
            if not qs:
                continue
            m = {k: mean([q[k] for q in qs]) for k in ("wall", "L0", "busy", "intra", "inter", "L1", "n_passes", "span")}
            rows.append([R["run"], kind, len(qs), m["n_passes"], m["wall"], m["L0"], m["busy"], m["intra"], m["inter"], m["L1"],
                         m["L0"] + m["span"] + m["L1"] - m["wall"], 100.0 * m["busy"] / m["wall"]])
    doc.table("Table 4: client wall time = send to first kernel + GPU busy + GPU idle inside passes + GPU idle between passes + last kernel to recv (mean per request)",
              ["run", "request kind", "requests", "passes per request", "client wall (ms)", "send to first kernel (ms)", "GPU busy (ms)",
               "idle inside passes (ms)", "idle between passes (ms)", "last kernel to recv (ms)", "sum minus wall (ms)", "GPU busy share of wall (%)"],
              rows, "requests.tsv; GPU busy = union of kernel intervals; idle between passes = kernel span - busy - idle inside passes")
    doc.h("4. Host gap before a pass, by boundary type")
    has_os = any(R["info"]["has_osrt"] for R in ok)
    rows = []
    for R in ok:
        by = collections.defaultdict(list)
        for s in R["stat"]:
            if s["hg"]:
                by[(s["kind"], s["boundary"])].append(s)
        for kind in KINDS_ORDER + sorted({k for k, _ in by} - set(KINDS_ORDER)):
            for bnd in sorted({b for k, b in by if k == kind}):
                ss = by[(kind, bnd)]
                g = lambda key: mean([s["hg"][key] / 1e6 for s in ss])
                gw = (lambda: mean([s["hg"]["wait"] / 1e6 for s in ss])) if has_os and ss[0]["hg"]["wait"] is not None else (lambda: None)
                wait = gw()
                rows.append([R["run"], kind, bnd, len(ss), mean([s["pre_gap"] / 1e6 for s in ss]), pctl([s["pre_gap"] / 1e6 for s in ss], 50),
                             g("drain"), g("cpu"), wait, None if wait is None else g("cpu") - wait, g("cuda_api"), g("first_launch")])
    doc.table("Table 5: GPU idle time before a pass split into host-side parts (means over passes; the four parts add up to the gap)",
              ["run", "request kind", "boundary (previous pass type -> this pass type)", "passes", "gap mean (ms)", "gap p50 (ms)",
               "GPU done to sync return (ms)", "no CUDA call running (ms)", "of which in OS wait calls (ms)", "of which busy (ms)",
               "CUDA calls: upload copies, syncs (ms)", "first launch to first kernel (ms)"],
              rows, "passes.tsv columns host_gap_*; decode-thread CUPTI_ACTIVITY_KIND_RUNTIME records between the previous pass's last kernel end and this pass's first kernel start; OSRT_API for the wait split")
    if has_os:
        rows = []
        for R in ok:
            for nm, n, tot, mx in R["info"]["osrt_top"][:8]:
                rows.append([R["run"], nm, n, tot / 1e6, mx / 1e6])
        doc.table("Table 6: OS runtime calls with the largest total time, all threads", ["run", "call", "calls", "total time (ms)", "longest call (ms)"],
                  rows, "OSRT_API grouped by nameId")
    doc.h("5. Kernel time per pass by class")
    allstats = [s for R in ok for s in R["stat"]]
    groups = collections.defaultdict(list)
    for s in allstats:
        groups[(s["pass_type"], s["n_class"])].append(s)
    order = sorted(groups, key=lambda x: (PASS_TYPES.index(x[0]) if x[0] in PASS_TYPES else 9, N_ORDER.get(x[1], 99)))
    used_cls = [c for c in CLASS_OF_CAT if any(c in s["cls"] for s in allstats)]
    rows, rows_n = [], []
    for key in order:
        ss = groups[key]
        rows.append([key[0], key[1], len(ss), mean([s["ksum"] / 1e6 for s in ss])] + [mean([s["cls"].get(c, (0, 0))[1] / 1e6 for s in ss]) for c in used_cls])
        rows_n.append([key[0], key[1], len(ss), mean([s["n_kernels"] for s in ss])] + [mean([s["cls"].get(c, (0, 0))[0] for s in ss]) for c in used_cls])
    doc.table("Table 7: kernel milliseconds per pass by class (pooled over all runs; mean over passes)",
              ["pass type", "tokens in pass", "passes", "all kernels (ms)"] + ["%s (ms)" % c for c in used_cls], rows,
              "classes.tsv scope ALL; classes: MMQ includes the stream-k fix-up kernels")
    doc.table("Table 8: kernel launches per pass by class (pooled over all runs; mean over passes)",
              ["pass type", "tokens in pass", "passes", "all kernels (launches)"] + ["%s (launches)" % c for c in used_cls], rows_n,
              "classes.tsv scope ALL")
    rows = []
    for key in order:
        ss = groups[key]
        if len(ss) < 2:
            continue
        busy = sum(s["busy"] for s in ss)
        agg = collections.defaultdict(lambda: [0, 0])
        for s in ss:
            for lk, v in s["lab"].items():
                agg[lk][0] += v[0]
                agg[lk][1] += v[1]
        for (c, l), (n_, t_) in sorted(agg.items(), key=lambda x: -x[1][1])[:8]:
            rows.append([key[0], key[1], len(ss), c, l, n_ / len(ss), t_ / len(ss) / 1e6, 100.0 * t_ / busy])
    doc.table("Table 9: the eight labels with the largest kernel time in each pass group (pooled over all runs)",
              ["pass type", "tokens in pass", "passes", "class", "label", "launches per pass", "kernel time per pass (ms)", "share of GPU busy (%)"],
              rows, "classes.tsv scope ALL level label; MMQ/MMVQ labels are <class> <tensor op> <quant type> K=<columns>; fix-up time is inside the MMQ label")
    doc.h("6. Validation of the derived structure")
    rows = []
    for R in ok:
        o = R["ops"]
        rows.append([R["run"], o.get("passes", 0), o.get("layer_pattern_ok", 0), o.get("n_launches", 0), o.get("n_mapped", 0),
                     "%d / %d" % (o.get("grid_ok", 0), o.get("grid_checked", 0)), "%d / %d" % (o.get("mmvq_grid_ok", 0), o.get("mmvq_grid_checked", 0)),
                     "%d / %d" % (R["info"].get("seq_identical", 0), R["info"].get("pass_groups_equal_shape", 0))])
    doc.table("Table 10: tensor assignment of MMQ/MMVQ launches",
              ["run", "passes", "passes with the 32-layer pattern (24 GDN + 8 attention)", "MMQ + MMVQ launches", "launches with a tensor",
               "MMQ grid = ceil(M/128) * ceil(N/J) (ok / checked, launches without fix-up)", "MMVQ grid = M or M/2 (ok / checked)",
               "equal-shape pass pairs with identical kernel sequence"],
              rows, "derive_ops(): launches are assigned by (quant type, K of the preceding quantize kernel) and position in the layer")
    probs = [p for R in Rs for p in R["problems"]]
    doc.h("7. Validation problems")
    doc.p("%s" % ("none" if not probs else "%d problems; first 25:" % len(probs)))
    for p in probs[:25]:
        doc.p("- " + p)
    doc.p("\nwall time of this analysis: %.1f s" % wall_s)
    return probs


def find_runs(res, runs_arg):
    if runs_arg:
        return [r for r in runs_arg.split(",") if r]
    return sorted(p.name[:-len(".client.json")] for p in res.glob("*.client.json") if (res / (p.name[:-len(".client.json")] + ".sqlite")).exists())


def analyze_main(args):
    t_start = time.time()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(res, args.runs)
    if not runs:
        sys.exit("no run with both <run>.sqlite and <run>.client.json in %s (run nsys_export.sh first)" % res)
    Rs = []
    for run in runs:
        db, cj = res / (run + ".sqlite"), res / (run + ".client.json")
        for p in (db, cj):
            if not p.exists():
                sys.exit("missing %s" % p)
        print("analysing %s ..." % run, flush=True)
        Rs.append(analyze_run(run, db, cj, args))
    ok = [R for R in Rs if "pass_rows" in R]
    stats = {R["run"]: R["stat"] for R in ok}
    stats["ALL"] = [s for R in ok for s in R["stat"]]
    doc = Doc()
    probs = summarize(doc, Rs, args, time.time() - t_start)
    write_tsv(out / "passes.tsv", PASS_HEADER + [c for c in NEW_COLS if c not in PASS_HEADER], [r for R in ok for r in R["pass_rows"]])
    write_tsv(out / "kernels.tsv", KERNEL_HEADER, [r for R in ok for r in R["kernel_rows"]])
    if not args.no_mmq_launches:
        write_tsv(out / "mmq-launches.tsv", MMQ_HEADER + MMQ_EXTRA, [r for R in ok for r in R["mmq_rows"]])
    write_tsv(out / "classes.tsv", CLASS_HEADER, class_rows(stats))
    write_tsv(out / "requests.tsv", REQ_HEADER, [r for R in ok for r in req_rows(R)])
    write_tsv(out / "osrt-request-start.tsv", OSRT_HEADER, [r for R in ok for r in R.get("osrt_start", [])])
    (out / "summary.txt").write_text("\n".join(doc.md) + "\n")
    print("\nwrote %s: passes.tsv kernels.tsv %sclasses.tsv requests.tsv osrt-request-start.tsv summary.txt"
          % (out, "" if args.no_mmq_launches else "mmq-launches.tsv "))
    if probs and args.strict:
        sys.exit(1)


def osrt_window_rows(db_path, t, run, r, lo, hi, limit=400):
    """OSRT calls of all threads between `lo` and `hi` (nsys ns): what the server does between request arrival and its first kernel"""
    db = sqlite3.connect(str(db_path))
    try:
        sid = dict(db.execute("select id, value from StringIds"))
        rows = db.execute("select start, end, nameId, globalTid from OSRT_API where end >= ? and start <= ? and "
                          "coalesce(nestingLevel, 0)=0 order by start limit ?", (lo, hi, limit)).fetchall()
    finally:
        db.close()
    threads, out = {}, []
    for s, e, n, g in rows:
        name = sid.get(n, str(n))
        th = "decode" if g == t.decode_tid else "t%d" % threads.setdefault(g, len(threads) + 1)
        out.append({"run": run, "client_idx": r["idx"], "case": r["case"], "thread": th, "call": name,
                    "start_rel_send_ms": (s - r["w"][0]) / 1e6, "dur_ms": (e - s) / 1e6,
                    "class": "wait" if name.startswith(WAIT_PREFIX) else "other"})
    return out


# ---------------------------------------------------------------------------------------------------------------
# overhead: nsys run vs unprofiled control run (and vs the decomposition client)
# ---------------------------------------------------------------------------------------------------------------

OVH_HEADER = ["run", "kind", "n_pairs", "ctl_mean_ms", "nsys_mean_ms", "delta_mean_ms", "delta_pct_of_ctl_mean", "ctl_p50_ms",
              "nsys_p50_ms", "delta_p50_of_pairs_ms", "delta_sd_ms", "delta_ci95_halfwidth_ms"]


def paired(pairs):
    d = [p["nsys"] - p["ctl"] for p in pairs]
    n = len(d)
    sd = statistics.stdev(d) if n > 1 else None
    cm = mean([p["ctl"] for p in pairs])
    return {"n_pairs": n, "ctl_mean_ms": cm, "nsys_mean_ms": mean([p["nsys"] for p in pairs]), "delta_mean_ms": mean(d),
            "delta_pct_of_ctl_mean": 100.0 * mean(d) / cm, "ctl_p50_ms": pctl([p["ctl"] for p in pairs], 50),
            "nsys_p50_ms": pctl([p["nsys"] for p in pairs], 50), "delta_p50_of_pairs_ms": pctl(d, 50), "delta_sd_ms": sd,
            "delta_ci95_halfwidth_ms": (1.96 * sd / n ** 0.5) if sd is not None else None}


def read_tsv(path):
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in f if line.strip()]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def overhead_main(args):
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctl_path = res / (args.ctl + ".client.json")
    if not ctl_path.exists():
        sys.exit("control client JSON %s not found" % ctl_path)
    _, ctl_reqs, ctl_err = load_client(ctl_path)
    by_ref = {r["ref"]: r for r in ctl_reqs}
    if args.nsys:
        runs = [r for r in args.nsys.split(",") if r]
    else:
        runs = sorted(p.name[:-len(".client.json")] for p in res.glob("*.client.json") if p.name[:-len(".client.json")] != args.ctl
                      and (res / (p.name[:-len(".client.json")] + ".sqlite")).exists())
    if not runs:
        sys.exit("no nsys runs given/found")
    doc = Doc()
    doc.h("nsys overhead: profiled runs against the unprofiled control run", 1)
    doc.p("control run %s: %d requests (%d errors). A request of a profiled run is paired with the control request that has the same Ref "
          "(same case, same text); warm-up requests are %s." % (args.ctl, len(ctl_reqs), len(ctl_err), "included" if args.include_warmup else "excluded"))
    pairs, unmatched, mismatch = [], [], []
    for run in runs:
        _, reqs, errs = load_client(res / (run + ".client.json"))
        for r in reqs:
            if r["phase"] == "warmup" and not args.include_warmup:
                continue
            c = by_ref.get(r["ref"])
            if c is None:
                unmatched.append((run, r["idx"], r["case"]))
            elif (c["case"], c["kind"]) != (r["case"], r["kind"]):
                mismatch.append((run, r["idx"], r["case"], c["case"]))
            else:
                pairs.append({"run": run, "kind": r["kind"], "case": r["case"], "phase": r["phase"], "segment": r["segment"],
                              "ref": r["ref"], "nsys": r["wall_ms"], "ctl": c["wall_ms"], "nsys_idx": r["idx"], "ctl_idx": c["idx"]})
    doc.p("pairs: %d; profiled requests without a control request: %d %s; case/kind mismatches: %d %s" %
          (len(pairs), len(unmatched), unmatched[:5], len(mismatch), mismatch[:5]))
    rows = []
    keys = []
    for p in pairs:
        if (p["run"], p["kind"]) not in keys:
            keys.append((p["run"], p["kind"]))
    stats = {}
    for run, kind in keys:
        s = paired([p for p in pairs if p["run"] == run and p["kind"] == kind])
        stats[(run, kind)] = s
        rows.append({"run": run, "kind": kind, **s})
    for kind in KINDS_ORDER:
        ps = [p for p in pairs if p["kind"] == kind]
        if len({p["run"] for p in ps}) > 1:
            s = paired(ps)
            stats[("ALL", kind)] = s
            rows.append({"run": "ALL", "kind": kind, **s})
    write_tsv(out / "overhead.tsv", OVH_HEADER, rows)
    write_tsv(out / "overhead-pairs.tsv", ["run", "kind", "case", "phase", "segment", "ref", "nsys_idx", "ctl_idx", "nsys", "ctl"], pairs)
    doc.table("Table 1: client wall time of profiled requests against the same requests in the control run",
              ["run", "request kind", "pairs", "control mean (ms)", "nsys mean (ms)", "difference of means (ms)", "difference (% of control mean)",
               "control p50 (ms)", "nsys p50 (ms)", "median of paired differences (ms)", "sd of paired differences (ms)",
               "95% half-width of the mean difference (ms)"],
              [[r["run"], r["kind"], r["n_pairs"], r["ctl_mean_ms"], r["nsys_mean_ms"], r["delta_mean_ms"], r["delta_pct_of_ctl_mean"],
                r["ctl_p50_ms"], r["nsys_p50_ms"], r["delta_p50_of_pairs_ms"], r["delta_sd_ms"], r["delta_ci95_halfwidth_ms"]] for r in rows],
              "client JSONs (wall_ms = t_recv - t_send of one request); half-width = 1.96 * sd / sqrt(pairs), normal approximation (pairs are few)")
    # ---- against the decomposition client -------------------------------------------------------------------
    if args.dec:
        dec = collections.defaultdict(list)
        for pth in args.dec.split(","):
            _, dreqs, _ = load_client(pth, need=["idx", "kind", "case", "phase", "wall_ms"])
            for r in dreqs:
                if r["phase"] in ("unique", "cycle"):
                    dec[(r["kind"], r["case"])].append(r["wall_ms"])
        ns = collections.defaultdict(list)
        cl = collections.defaultdict(list)
        for p in pairs:
            if p["phase"] != "warmup":
                ns[(p["kind"], p["case"])].append(p["nsys"])
                cl[(p["kind"], p["case"])].append(p["ctl"])
        rows = []
        for kind in KINDS_ORDER:
            cs = sorted(c for (k, c) in dec if k == kind and (k, c) in ns)
            if not cs:
                continue
            dm = mean([mean(dec[(kind, c)]) for c in cs])
            cm = mean([mean(cl[(kind, c)]) for c in cs])
            nm = mean([mean(ns[(kind, c)]) for c in cs])
            rows.append([kind, len(cs), dm, cm, nm, nm - dm, 100.0 * (nm - dm) / dm])
        doc.table("Table 2: profiled requests against the decomposition client's requests of the same cases",
                  ["request kind", "cases in both", "decomposition client mean (ms)", "control run mean (ms)", "nsys runs mean (ms)",
                   "nsys - decomposition client (ms)", "same (% of decomposition mean)"], rows,
                  "per-case means averaged over the cases present in the decomposition JSON(s) (phases unique, cycle) and in the nsys runs")
        h2 = ["kind", "cases_in_both", "dec_mean_ms", "ctl_mean_ms", "nsys_mean_ms", "nsys_minus_dec_ms", "nsys_minus_dec_pct"]
        write_tsv(out / "overhead-vs-dec.tsv", h2, [dict(zip(h2, r)) for r in rows])
    # ---- where the difference sits ---------------------------------------------------------------------------
    if args.passes:
        reqrows = [r for r in read_tsv(args.passes) if r.get("row_kind") == "request" and r.get("phase") != "warmup"]
        by = collections.defaultdict(list)
        for r in reqrows:
            by[(r["run"], r["kind"])].append(r)
        rows = []
        for (run, kind), rs in sorted(by.items()):
            s = stats.get((run, kind))
            if not s:
                continue
            g = lambda key: mean([num(r.get(key)) for r in rs])
            l0, busy, intra, inter, l1 = g("send_to_first_kernel_ms"), g("gpu_busy_ms"), g("gap_intra_pass_ms"), g("gap_inter_pass_ms"), g("last_kernel_to_recv_ms")
            wall = g("client_wall_ms")
            napi = g("n_cudaLaunchKernel_api_calls") + g("n_sync_calls") + g("n_memcpy")
            nl = g("n_cudaLaunchKernel_api_calls")
            host = l0 + inter + l1
            d = s["delta_mean_ms"]
            rows.append([run, kind, len(rs), s["ctl_mean_ms"], wall, d, l0, busy, intra, inter, l1, 100.0 * host / wall, nl, napi,
                         1000.0 * d / napi, 1000.0 * d / nl, 100.0 * d / host])
        doc.table("Table 3: where a difference of the size in Table 1 can sit (means over the profiled requests of passes.tsv)",
                  ["run", "request kind", "requests", "control wall (ms)", "nsys wall (ms)", "difference (ms)", "send to first kernel (ms)", "GPU busy (ms)",
                   "GPU idle inside passes (ms)", "GPU idle between passes (ms)", "last kernel to recv (ms)", "host-side share of nsys wall (%)",
                   "kernel launches per request", "CUDA API calls per request (launch + sync + memcpy)",
                   "difference per CUDA API call if all of it is per-call cost (us)", "difference per kernel launch if all of it is launch cost (us)",
                   "difference as % of the host-side time (send to first kernel + between passes + last kernel to recv)"],
                  rows, "passes.tsv request rows (client_wall_ms, gpu_busy_ms, gap_*_ms, n_* columns) and Table 1; nsys wall = L0 + GPU busy + idle inside + idle between + L1")
        h3 = ["run", "kind", "requests", "ctl_wall_ms", "nsys_wall_ms", "delta_ms", "send_to_first_kernel_ms", "gpu_busy_ms", "gap_in_pass_ms",
              "gap_between_passes_ms", "last_kernel_to_recv_ms", "host_share_of_nsys_wall_pct", "launches_per_request",
              "cuda_api_calls_per_request", "delta_us_per_cuda_api_call", "delta_us_per_launch", "delta_pct_of_host_time"]
        write_tsv(out / "overhead-where.tsv", h3, [dict(zip(h3, r)) for r in rows])
        doc.p("Reading Table 3: W = L0 + (GPU busy + idle inside passes) + idle between passes + L1 holds exactly for every profiled request. If the profiler "
              "does not change kernel durations or the idle time inside passes (not testable without GPU counters on the control run), the whole difference "
              "is in L0, L1 and the idle time between passes; the two per-call columns are upper bounds on a per-call cost and are only valid under that hypothesis.")
    (out / "overhead.txt").write_text("\n".join(doc.md) + "\n")
    print("\nwrote %s: overhead.tsv overhead-pairs.tsv overhead.txt%s" % (out, " overhead-vs-dec.tsv" if args.dec else "") + (" overhead-where.tsv" if args.passes else ""))


# ---------------------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------------------

def main():
    global WORK, OUT, OLD_PROF
    ap = argparse.ArgumentParser(prog="nsys_census.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("legacy", help="reproduce the census tables trace-{kernels,passes,mmq-launches}.tsv from the three old exports")
    lg.add_argument("--in", dest="inp", required=True, help="directory with long.sqlite, medium.sqlite, short.sqlite")
    lg.add_argument("--out", required=True, help="output directory for trace-*.tsv")
    lg.add_argument("--old-exports", default="", help="optional cross-check: directory with the earlier exports long.sqlite, medium.sqlite, short.sqlite (opened read-only)")
    an = sub.add_parser("analyze", help="whole-server captures + client JSONs -> passes/kernels/classes/requests/mmq tables, summary.txt")
    an.add_argument("--results", required=True, help="directory with <run>.sqlite and <run>.client.json")
    an.add_argument("--runs", default="", help="comma list of run names (default: every <run>.client.json that has a <run>.sqlite)")
    an.add_argument("--out", default="", help="output directory (default <results>/nsys-analysis)")
    an.add_argument("--assign", choices=["window", "order"], default="window", help="pass-to-request assignment (default window)")
    an.add_argument("--clock", choices=["mono", "epoch"], default="mono", help="client stamp -> nsys time mapping (default mono)")
    an.add_argument("--include-warmup", action="store_true", help="warm-up requests count in kernels/classes/boundary statistics")
    an.add_argument("--no-osrt", action="store_true", help="ignore OSRT_API even if present")
    an.add_argument("--strict", action="store_true", help="exit status 1 when validation problems were found")
    an.add_argument("--no-mmq-launches", action="store_true", help="do not write mmq-launches.tsv (the checks in summary.txt are still computed)")
    ov = sub.add_parser("overhead", help="client wall time of nsys runs against the unprofiled control run")
    ov.add_argument("--results", required=True)
    ov.add_argument("--ctl", required=True, help="run name of the control run (<ctl>.client.json)")
    ov.add_argument("--nsys", default="", help="comma list of profiled run names (default: all other runs that have a .sqlite)")
    ov.add_argument("--dec", default="", help="comma list of decomposition client JSON files")
    ov.add_argument("--passes", default="", help="passes.tsv written by `analyze` (adds Table 3)")
    ov.add_argument("--out", default="")
    ov.add_argument("--include-warmup", action="store_true")
    args = ap.parse_args()
    if args.cmd == "legacy":
        WORK, OUT = Path(args.inp), Path(args.out)
        OLD_PROF = Path(args.old_exports) if args.old_exports else None
        OUT.mkdir(parents=True, exist_ok=True)
        legacy_main()
    elif args.cmd == "analyze":
        args.out = args.out or str(Path(args.results) / "nsys-analysis")
        analyze_main(args)
    else:
        args.out = args.out or str(Path(args.results) / "nsys-analysis")
        overhead_main(args)


if __name__ == "__main__":
    main()
