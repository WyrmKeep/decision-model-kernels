#!/usr/bin/env python3
"""Planner of the phase-3 ncu runs: which kernel launches of a fresh server's first request does `ncu -k/-s/-c` select?

Every run starts a fresh llama-server under ncu and sends ONE request (extra/prof/ncu_client.py). ncu counts kernel launches
of the whole process, so a window is (kernel-name filter, launch-skip, launch-count). The skip is the number of launches that match the
filter before the window: all earlier passes of the request plus the earlier launches of the target pass. The pass list comes from
census_lib.simulate_request (fresh server: parent slot 0, children slots 1 and 2, as seen in the server logs of the earlier measurements).
The kernel order of a pass comes from a traced pass of the same kind (data/pass-kernels.tsv.gz, cut from the nsys exports of the three
old census captures by extract_pass_kernels.py, which needs DMK_LEGACY_TRACES); where no pass of the exact size was traced a neighbour
is the model (MODEL_OF, with the reason). The census tables and the model come from the census data set ($DMK_CENSUS_DATA, census_lib).

  plan_ncu.py                    check tables + window table + estimates (markdown)
  plan_ncu.py --run NAME         the expected kernels of one run, one per line
  plan_ncu.py --write            also write plan/<run>.json (read by analyze_ncu.py) and sched/4-ncu-*.txt
                                 (--plan-dir DIR and --sched-dir DIR write to other folders; the ncu lines have no host path, so the
                                 files need no @REMOTE_DIR@)
  plan_ncu.py --shift KEY=N      add N to the skip of every window with filter KEY (correct an offset seen in the smoke run)
  plan_ncu.py --check-metrics    check the metric names of ncu_cfg.py against the section files of the local ncu ($DMK_NCU)
Exit status 1 when an arithmetic cross-check fails.
"""
import argparse
import collections
import csv
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ncu_cfg as cfg  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "census"))
import census_lib as cl  # noqa: E402

DATA = os.path.join(HERE, "data", "pass-kernels.tsv.gz")
PLAN_DIR = os.path.join(HERE, "plan")              # --plan-dir
SCHED_DIR = os.path.join(HERE, "..", "sched")      # --sched-dir

# ----------------------------------------------------------------------------------------------------------------- requests
# first request after server start: parent task on slot 0, children on slots 1 (and 2); the ref variant of the census
# (Ref line = common.tag("seq", 0, case id), what ncu_client.py sends) gives these token counts (census/token-counts.tsv)
FRESH = {
    "long00-storage": dict(kind="long3q", shared=3221, suffix=[43, 47, 73], slots=[0, 1, 2]),
    "medium01-memory": dict(kind="medium3q", shared=1288, suffix=[43, 47, 73], slots=[0, 1, 2]),
    "triage09": dict(kind="short2q", shared=86, suffix=[120, 51], slots=[0, 1]),
    "check03": dict(kind="short1q", shared=0, suffix=[158], slots=[0]),
}
# traced pass that stands in for a pass of the fresh request (kernel ORDER only; grids/shapes of the matmuls come from census_lib)
MODEL_OF = {
    ("long00-storage", 0): ("long-i0-N2048", "traced, same N"),
    ("long00-storage", 1): ("long-i1-N1172", "traced N=1172 for N=1173 (same 10 column tiles)"),
    ("long00-storage", 2): ("long-i12-N129", "traced, same N (3 x 43)"),
    ("long00-storage", 3): ("long-i13-N8", "traced, same N (2 x 4, all-MMVQ pass)"),
    ("long00-storage", 4): ("long-i14-N26", "traced, same N"),
    ("medium01-memory", 0): ("medium-i4-N1288", "traced, same N"),
    ("medium01-memory", 1): ("medium-i5-N129", "traced, same N"),
    ("medium01-memory", 2): ("medium-i6-N8", "traced, same N"),
    ("medium01-memory", 3): ("medium-i7-N26", "traced, same N"),
    ("triage09", 0): ("short-i6-N86", "traced, same N (the census ref variant of triage09)"),
    ("triage09", 1): ("long-i8-N94", "NOT traced (2 x 51 = 102); N=94 is the traced 2-sequence suffix pass"),
    ("triage09", 2): ("long-i4-N73", "NOT traced (1 x 69, gdn_single); N=73 is the traced continuation pass"),
    ("check03", 0): ("short-i1-N120", "NOT traced (1 x 158); N=120 is the traced 1-sequence first suffix pass"),
}
# partial regexes on the ncu "function" name (= nsys shortName: no template arguments, no parameters). `regex:` is a partial match, so
# `mul_mat_q` also selects mul_mat_q_stream_k_fixup, and a namespace in front of the name (gdn_single lives in a namespace) is harmless.
FILTERS = {
    "ALL": None,
    "MM": "mul_mat_q",
    "MV": "mul_mat_vec_q",
    "L0": "mul_mat_q|gdn_single|gated_delta_net_cuda|concat_non_cont|ssm_conv",
    "EW": "rms_norm|quantize_mmq|unary_gated",
    "ATT": "mul_mat_q|flash_attn_ext|cpy_scalar|rope_multi",
    "MVQ": "mul_mat_vec_q|quantize_q8_1",
}
FILTER_NOTE = {
    "ALL": "no filter: every launch counts",
    "MM": "MMQ main kernels and their stream-k fix-ups",
    "MV": "MMVQ kernels",
    "L0": "MMQ + fix-ups, both GDN kernels, concat_non_cont, ssm_conv (the kernels that are not at the DRAM floor)",
    "EW": "rms_norm, quantize_mmq, glu",
    "ATT": "MMQ + fix-ups, flash attention, rope, cpy_scalar",
    "MVQ": "MMVQ + their quantize_q8_1",
}

# ----------------------------------------------------------------------------------------------------------------- the runs
# window: (label, filter key, pass index in the request, first matching kernel of that pass, count or "L0" = the first layer of the pass)
RUNS = [
    # ---- smoke: two short runs on the smallest request; they check the harness, the alignment and the cost per kernel
    dict(name="p3-ncu-smoke-a", group="smoke", case="check03", sections="std",
         why="harness + alignment: the first 6 launches of the first pass, no filter (rms_norm N=158, ..., first mul_mat_q)",
         windows=[("first 6 launches", "ALL", 0, 0, 6)]),
    dict(name="p3-ncu-smoke-b", group="smoke", case="check03", sections="std",
         why="regex with | through the harness + the 3 MMVQ kernels of the last layer and the head (675 MB)",
         windows=[("last-layer MMVQ + quantize_q8_1", "MVQ", 0, 0, 6)]),
    # ---- core, cheapest first
    dict(name="p3-ncu-core-mv", group="core", case="long00-storage", sections="std",
         why="MMVQ: the 3 launches that end every MMQ-path pass (fused FFN gate+up, fused FFN down, output head) + layer 0 of the N=8 pass",
         windows=[("MMVQ end of P2 + layer 0 of P3", "MV", 2, 0, 11)]),
    dict(name="p3-ncu-core-short-p0", group="core", case="triage09", sections="std",
         why="short prefix N=86 (J=96): layer 0 MMQ",
         windows=[("MMQ layer 0, P0 N=86", "MM", 0, 0, "L0")]),
    dict(name="p3-ncu-core-short-p1", group="core", case="triage09", sections="std",
         why="short suffix pass 2 x 51 = N=102 (J=112), model order from N=94: layer 0 MMQ",
         windows=[("MMQ layer 0, P1 N=102", "MM", 1, 0, "L0")]),
    dict(name="p3-ncu-core-long-p2", group="core", case="long00-storage", sections="std",
         why="suffix pass 3 x 43 = N=129 (J=80, 3 sequences): MMQ + gated_delta_net_cuda + concat + ssm_conv of layer 0",
         windows=[("MMQ+GDN layer 0, P2 N=129", "L0", 2, 0, "L0")]),
    dict(name="p3-ncu-core-long-p1", group="core", case="long00-storage", sections="std",
         why="second prefix chunk N=1173 (J=128, 10 column tiles): layer 0 MMQ",
         windows=[("MMQ layer 0, P1 N=1173", "MM", 1, 0, "L0")]),
    dict(name="p3-ncu-core-long-p0", group="core", case="long00-storage", sections="std",
         why="first prefix chunk N=2048 (J=128): MMQ x 8 + gdn_single + concat_non_cont + ssm_conv_long_token of layer 0",
         windows=[("MMQ+GDN layer 0, P0 N=2048", "L0", 0, 0, "L0")]),
    # ---- extra, cheapest first
    dict(name="p3-ncu-extra-long-att", group="extra", case="long00-storage", sections="std",
         why="attention layer 3 of P0 (N=2048): MMQ x 7 (attn_q/k/v/output + FFN), rope_multi x 2, flash_attn_ext_f16, the large cpy_scalar",
         windows=[("attention layer 3, P0", "ATT", 0, ("layer", 3), "L0")]),
    dict(name="p3-ncu-extra-long-ew", group="extra", case="long00-storage", sections="std",
         why="element-wise classes of P0 layer 0 at N=2048: rms_norm x 4 variants, quantize_mmq x 4, glu",
         windows=[("rms/quantize/glu layer 0, P0", "EW", 0, 0, 9)]),
    dict(name="p3-ncu-extra-long-p4", group="extra", case="long00-storage", sections="std",
         why="last suffix pass N=26 (J=32, 1 sequence): layer 0 MMQ",
         windows=[("MMQ layer 0, P4 N=26", "MM", 4, 0, "L0")]),
    dict(name="p3-ncu-extra-med-p0", group="extra", case="medium01-memory", sections="std",
         why="medium prefix N=1288 (11 column tiles, the last one has 8 of 128 columns): layer 0 MMQ",
         windows=[("MMQ layer 0, P0 N=1288", "MM", 0, 0, "L0")]),
    dict(name="p3-ncu-extra-long-full", group="extra", case="long00-storage", sections="full",
         why="--set full (stalls per source line, roofline, memory chart) for the 8 MMQ kernels of layer 0 at N=2048",
         windows=[("MMQ x 8 layer 0, P0 N=2048, --set full", "MM", 0, 0, 8)]),
]
GROUP_FILE = {"smoke": "4-ncu-smoke.txt", "core": "4-ncu-core.txt", "extra": "4-ncu-extra.txt"}

TYPE_NAME = {2: "Q4_0", 8: "Q8_0"}
SIZE_KEYS = ("pos", "gx", "gy", "gz", "bx", "by", "bz", "regs", "smem_static", "smem_dyn", "dur_ns")


# ----------------------------------------------------------------------------------------------------------------- data
def load_models():
    d = collections.defaultdict(list)
    with gzip.open(DATA, "rt", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            for k in SIZE_KEYS:
                r[k] = int(r[k])
            d[r["model"]].append(r)
    return d


def load_trace_passes():
    """trace-passes.tsv rows of the pass kind, keyed by (trace, pass_idx_in_trace)."""
    out = {}
    with open(cl.data_file("trace-passes.tsv"), newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["row_kind"] == "pass":
                out[(r["trace"], int(r["pass_idx_in_trace"]))] = r
    return out


def matches(filt, k):
    pat = FILTERS[filt]
    return True if pat is None else re.search(pat, k["func"]) is not None


def tmpl_ints(dem):
    """Template arguments of a demangled kernel name: ggml_type, mmq_x (J), need_check."""
    m = re.match(r"void mul_mat_q(?:_stream_k_fixup)?<\(ggml_type\)(\d+), \(int\)(\d+), \(bool\)(\d)", dem)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def layer_end(ks, n_layers=1):
    """Position after the n-th layer of a pass: the kernel after the 2n-th residual add (k_bin_bcast with 128 threads) is the
    attn_norm of the next layer."""
    big = 0
    for i, k in enumerate(ks):
        if k["func"] == "k_bin_bcast" and k["bx"] == 128:
            big += 1
            if big == 2 * n_layers:
                if i + 1 < len(ks) and ks[i + 1]["func"] == "rms_norm_f32":
                    return i + 1
                raise ValueError("kernel after the residual add is not rms_norm_f32")
    raise ValueError("layer end not found")


def exec_order(ops):
    """pass_matmuls() lists a recurrent layer as qkv, z, beta, alpha, out; the CUDA stream runs qkv, beta, alpha, z, out (gate z after
    the delta-net), see the traced launch order (trace-mmq-launches.tsv)."""
    out = []
    by_layer = collections.OrderedDict()
    for o in ops:
        by_layer.setdefault(o["layer"], []).append(o)
    for layer, L in by_layer.items():
        names = [o["op"] for o in L]
        if names[:5] == ["attn_qkv", "attn_gate(z)", "ssm_beta", "ssm_alpha", "ssm_out"]:
            L = [L[0], L[2], L[3], L[1]] + L[4:]
        out.extend(L)
    return out


class Fresh:
    """The passes of the first request of one case, each with the kernel list of its model pass and the census ops."""

    def __init__(self, case, models, model):
        self.case = case
        spec = FRESH[case]
        self.kind = spec["kind"]
        self.passes = cl.simulate_request(spec["shared"], spec["suffix"], spec["slots"])
        self.pass_info = []
        for k, p in enumerate(self.passes):
            mid, why = MODEL_OF[(case, k)]
            ks = [dict(x) for x in models[mid]]
            ops = exec_order(cl.pass_matmuls(model, p["n_tokens"], p["n_out"]))
            mmq_ops = [o for o in ops if o["path"] == "MMQ"]
            mmvq_ops = [o for o in ops if o["path"] != "MMQ"]
            self._annotate(ks, p, mmq_ops, mmvq_ops, int(re.search(r"-N(\d+)$", mid).group(1)))
            self.pass_info.append(dict(k=k, p=p, model=mid, why=why, ks=ks, mmq_ops=mmq_ops, mmvq_ops=mmvq_ops,
                                       costs=cl.pass_costs(model, p)))

    @staticmethod
    def _annotate(ks, p, mmq_ops, mmvq_ops, n_proxy):
        """Attach the census op (M, N, K, type, J, grid) to the i-th MMQ / MMVQ launch and to the fix-up behind it."""
        i_mmq = i_mmvq = 0
        last = None
        for k in ks:
            k["n_tokens"] = p["n_tokens"]
            k["n_proxy"] = n_proxy
            k["n_seqs"] = len(p["seqs"])
            k["op"] = ""
            if k["func"] == "mul_mat_q" and i_mmq < len(mmq_ops):
                last = mmq_ops[i_mmq]
                i_mmq += 1
                k["opd"] = last
                k["op"] = last["op"]
            elif k["func"] == "mul_mat_q_stream_k_fixup" and last is not None:
                k["opd"] = last
                k["op"] = last["op"] + " (fix-up)"
            elif k["func"] == "mul_mat_vec_q" and i_mmvq < len(mmvq_ops):
                k["opd"] = mmvq_ops[i_mmvq]
                k["op"] = mmvq_ops[i_mmvq]["op"]
                i_mmvq += 1

    def stream(self, filt):
        """[(pass k, kernel)] of all launches matching the filter, in launch order across the passes of the request."""
        return [(pi["k"], kk) for pi in self.pass_info for kk in pi["ks"] if matches(filt, kk)]


# ----------------------------------------------------------------------------------------------------------------- checks
def count(ks, func):
    return sum(1 for k in ks if k["func"] == func)


def check_passes(fresh_by_case, trace_rows, models):
    """Arithmetic cross-checks. Returns (rows, failures)."""
    rows, fails = [], []
    for case, fr in fresh_by_case.items():
        for pi in fr.pass_info:
            ks, c, p = pi["ks"], pi["costs"], pi["p"]
            n_mmq, n_fix, n_mmvq = count(ks, "mul_mat_q"), count(ks, "mul_mat_q_stream_k_fixup"), count(ks, "mul_mat_vec_q")
            # kernel list of the model pass vs trace-passes.tsv (the extractor already asserts n_kernels)
            tr = re.match(r"(long|medium|short)-i(\d+)-N(\d+)", pi["model"])
            trow = trace_rows[(tr.group(1), int(tr.group(2)))]
            for col, have in (("n_mmq", n_mmq), ("n_mmq_fixup", n_fix), ("n_mmvq", n_mmvq), ("n_quantize", count(ks, "quantize_mmq_q8_1") + count(ks, "quantize_q8_1")),
                              ("n_flash_attn", count(ks, "flash_attn_ext_f16")), ("n_kernels", len(ks))):
                if int(trow[col]) != have:
                    fails.append(f"{case} P{pi['k']}: trace-passes.tsv {col}={trow[col]} but the kernel list of {pi['model']} has {have}")
            # census_lib prediction for the fresh pass vs the model pass
            exact = int(tr.group(3)) == p["n_tokens"]
            grid_bad = fix_bad = 0
            i = 0
            for j, k in enumerate(ks):
                if k["func"] != "mul_mat_q" or "opd" not in k:
                    continue
                o = k["opd"]
                if k["gx"] != o["grid_x"]:
                    grid_bad += 1
                has_fix = j + 1 < len(ks) and ks[j + 1]["func"] == "mul_mat_q_stream_k_fixup"
                if int(has_fix) != int(o["fixup"]):
                    fix_bad += 1
                if o["wtype"] != TYPE_NAME.get(tmpl_ints(k["demangled"])[0]) or (o["J"] != tmpl_ints(k["demangled"])[1]):
                    grid_bad += 1
            census = (c["n_mmq"], c["n_mmq_fixup"], c["n_mmvq"])
            rows.append(dict(case=case, k=pi["k"], n=p["n_tokens"], seqs=len(p["seqs"]), n_out=p["n_out"], model=pi["model"],
                             exact=exact, why=pi["why"], trace=(n_mmq, n_fix, n_mmvq), census=census, grid_bad=grid_bad, fix_bad=fix_bad))
            if census != (n_mmq, n_fix, n_mmvq):
                msg = f"{case} P{pi['k']} N={p['n_tokens']}: census_lib (mmq, fix-up, mmvq) = {census}, model pass {pi['model']} = {(n_mmq, n_fix, n_mmvq)}"
                if exact:
                    fails.append(msg)
                else:
                    rows[-1].setdefault("notes", []).append(msg)
            if exact and (grid_bad or fix_bad):
                fails.append(f"{case} P{pi['k']}: census_lib grid/fix-up prediction differs from the exact traced pass in {grid_bad}+{fix_bad} launches")
    return rows, fails


# ----------------------------------------------------------------------------------------------------------------- windows
def plan_window(fr, filt, pass_k, m, cnt, shifts):
    stream = fr.stream(filt)
    before = sum(1 for k, _ in stream if k < pass_k)
    pi = fr.pass_info[pass_k]
    in_pass = [kk for kk in pi["ks"] if matches(filt, kk)]
    if isinstance(m, tuple):                       # ("layer", l): start at the first matching launch of layer l
        lo = m[1]
        l0 = 0 if lo == 0 else layer_end(pi["ks"], lo)
        m = sum(1 for kk in in_pass if kk["pos"] < l0)
        if cnt == "L0":
            cnt = ("layers", 1)
        if isinstance(cnt, tuple):                 # ("layers", n): through the end of layer lo + n - 1
            hi = layer_end(pi["ks"], lo + cnt[1])
            cnt = sum(1 for kk in in_pass if l0 <= kk["pos"] < hi)
    elif cnt == "L0":                              # the first layer of the pass
        cnt = sum(1 for kk in in_pass if kk["pos"] < layer_end(pi["ks"], 1))
        assert m == 0
    skip = before + m + shifts.get(filt, 0)
    sel = stream[before + m: before + m + cnt]
    if len(sel) != cnt:
        raise ValueError(f"window runs past the end of the request: skip {skip} count {cnt} of {len(stream)}")
    return dict(filter=filt, regex=FILTERS[filt], skip=skip, count=cnt, shift=shifts.get(filt, 0), pass_k=pass_k, kernels=sel,
                passes=sorted({k for k, _ in sel}), n_stream=len(stream), before=before)


def plan_all(fresh_by_case, shifts):
    out = []
    for run in RUNS:
        fr = fresh_by_case[run["case"]]
        wins = []
        for label, filt, pk, m, cnt in run["windows"]:
            w = plan_window(fr, filt, pk, m, cnt, shifts)
            w["label"] = label
            wins.append(w)
        out.append(dict(run, plan=wins))
    return out


def fix_grid(kk, n):
    """Grid of a launch of the fresh pass: where the model pass is a neighbour (N differs) the token-count dimension of the launches whose
    grid is known to follow N is replaced. Returns (gx, gy, gz, exact): exact = the grid is certain."""
    gx, gy, gz, npx = kk["gx"], kk["gy"], kk["gz"], kk["n_proxy"]
    if npx == n:
        return gx, gy, gz, True
    f = kk["func"]
    if f in ("rms_norm_f32",):
        return (n if gx == npx else gx), (n if gy == npx else gy), gz, True
    if f in ("quantize_mmq_q8_1", "quantize_q8_1") and gx == npx:
        return n, gy, gz, True
    if f == "mul_mat_q" and "opd" in kk:
        return kk["opd"]["grid_x"], gy, gz, True
    return gx, gy, gz, False


def floor_inputs(kk, pi, model):
    """Inputs of the speed-of-light floor of one launch (analyze_ncu.py divides them by the limits of the card): bytes that must cross DRAM
    (all tensors from/to DRAM), int8 tensor ops, fp16-tensor flops, fp32 flops. Empty = no model (the analysis then uses the measured
    DRAM bytes). Source of the shapes: census_lib (matmul, gdn_cost, attn_cost) and the grid of the launch."""
    p, n, S = pi["p"], kk["n_tokens"], kk["n_seqs"]
    o, f = kk.get("opd"), kk["func"]
    ne, cc = model.n_embd, model.conv_channels
    d = dict(bytes_model="", ops_int8="", flops_fp16="", flops_fp32="", model_note="")
    if f == "mul_mat_q" and o:
        d.update(bytes_model=o["weight_bytes"] + o["act_q8_bytes"] + o["act_out_f32_bytes"], ops_int8=o["flops"],
                 model_note="weights + q8_1 input + f32 output; useful int8 ops 2*M*N*K")
    elif f == "mul_mat_vec_q" and o:
        d.update(bytes_model=o["weight_bytes"] + o["act_q8_bytes"] + o["act_out_f32_bytes"], model_note="weights + q8_1 input + f32 output")
    elif f == "quantize_mmq_q8_1":
        k_pad = kk["gy"] * 512
        d.update(bytes_model=n * k_pad * 4 + n * k_pad * 9 // 8, model_note="f32 input N*K*4 + q8_1 output N*K*9/8, K = grid.y * 512")
    elif f == "concat_non_cont":
        d.update(bytes_model=2 * n * cc * 4, model_note="read + write N*8192 f32 (conv state rows ignored)")
    elif f.startswith("ssm_conv"):
        d.update(bytes_model=2 * n * cc * 4, model_note="read + write N*8192 f32")
    elif f == "unary_gated_op_kernel":
        per_tok = round(kk["gx"] * 256 / kk["n_proxy"])
        d.update(bytes_model=3 * n * per_tok * 4, model_note=f"gate + up read, out write: 3*N*{per_tok}*4")
    elif f == "rms_norm_f32" and kk["gy"] == 1 and kk["bx"] == 1024:
        d.update(bytes_model=2 * n * ne * 4, model_note="read + write N*2560 f32")
    elif f in ("gdn_single", "gated_delta_net_cuda"):
        c = cl.gdn_cost(model, n, S)
        d.update(bytes_model=c["bytes"], model_note="state read+write + token streams (census_lib.gdn_cost)")
        d["flops_fp16" if c["mma"] else "flops_fp32"] = c["flops"]
    elif f == "flash_attn_ext_f16":
        a = cl.attn_cost(model, p["seqs"])
        d.update(bytes_model=a["kv_bytes"] + a["kv_write_bytes"] + 2 * n * model.n_head * model.head_dim * 4, flops_fp16=a["flops"],
                 model_note="useful causal flops 4*hd*heads*keys; K+V read once, Q read, O write (census_lib.attn_cost)")
    elif f == "cpy_scalar" and kk["gx"] * kk["bx"] >= 100000:
        el = kk["gx"] * kk["bx"] * n // kk["n_proxy"]
        d.update(bytes_model=2 * el * 4, model_note="f32 -> f32 copy of N*4096 elements: read + write [INFERENCE: contiguous copy of the attention output]")
    return d


def launches_per_pass(kk, pi):
    f, o = kk["func"], kk.get("opd")
    if f == "mul_mat_q" and o:
        return sum(1 for x in pi["mmq_ops"] if x["op"] == o["op"])
    if f == "mul_mat_q_stream_k_fixup" and o:
        return sum(1 for x in pi["mmq_ops"] if x["op"] == o["op"] and x["fixup"])
    if f == "mul_mat_vec_q" and o:
        return sum(1 for x in pi["mmvq_ops"] if x["op"] == o["op"])
    return sum(1 for x in pi["ks"] if (x["func"], x["gx"], x["gy"], x["gz"], x["bx"], x["by"]) == (f, kk["gx"], kk["gy"], kk["gz"], kk["bx"], kk["by"]))


def class_of(kk, gx, gy, gz):
    f, o, n = kk["func"], kk.get("opd"), kk["n_tokens"]
    if f == "mul_mat_q" and o:
        return f"mmq {o['op']} {o['wtype']} J={o['J']} N={o['N']}"
    if f == "mul_mat_q_stream_k_fixup" and o:
        return f"mmq-fixup {o['op']} {o['wtype']} J={o['J']} N={o['N']}"
    if f == "mul_mat_vec_q" and o:
        return f"mmvq {o['op']} {o['wtype']} n={o['N']}"
    return f"{f} grid=({gx},{gy},{gz}) N={n}"


def kernel_record(i, pk, kk, label, fr, model):
    pi = fr.pass_info[pk]
    o = kk.get("opd")
    gx, gy, gz, exact = fix_grid(kk, kk["n_tokens"])
    r = dict(i=i, window=label, pass_k=pk, pos=kk["pos"], func=kk["func"], demangled=kk["demangled"], gx=gx, gy=gy, gz=gz, grid_exact=exact,
             bx=kk["bx"], by=kk["by"], bz=kk["bz"], regs=kk["regs"], smem_dyn=kk["smem_dyn"], dur_ns_nsys_proxy=kk["dur_ns"],
             n_tokens=kk["n_tokens"], n_proxy=kk["n_proxy"], n_seqs=kk["n_seqs"], op=kk["op"], cls=class_of(kk, gx, gy, gz),
             per_pass=launches_per_pass(kk, pi), pass_nsys_ms=sum(x["dur_ns"] for x in pi["ks"]) / 1e6)
    if o:
        for key in ("wtype", "M", "N", "K", "path", "J", "tiles", "grid_x", "fixup", "flops", "weight_bytes", "act_q8_bytes", "act_out_f32_bytes", "padded_ops"):
            r[key] = o.get(key, "")
    r.update(floor_inputs(kk, pi, model))
    return r


def estimate(run):
    n = sum(w["count"] for w in run["plan"])
    c = cfg.COST
    passes = c["passes"][run["sections"]]
    out = {"kernels": n}
    for mode in ("opt", "pess"):
        out[mode] = c["run_fixed"][mode] + n * (c["t_save"][mode] + passes * c["t_pass"][mode])
    return out


# ----------------------------------------------------------------------------------------------------------------- schedule
ENV_COMMON = ('DMK_SERVER_ARGS="-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots --no-warmup" '
              "DMK_TOOL_CAPS=SYS_ADMIN DMK_STOP_TIMEOUT=120")


def wrap_of(run):
    w = run["plan"][0]
    a = [cfg.NCU_REMOTE, f"-o /runs/{run['name']}", "--target-processes application-only", "--clock-control none", "--cache-control all",
         "--replay-mode kernel", "--pipeline-boost-state dynamic", "--apply-rules no", f"--section-folder {cfg.SECTIONS_REMOTE}"]
    if run["sections"] == "full":
        a.append("--set full")
    else:
        a += [f"--section {s}" for s in cfg.SECTIONS_STD]
    a.append("--metrics " + ",".join(cfg.EXTRA_METRICS))
    a.append("--kernel-name-base function")
    if w["regex"]:
        a.append(f"-k regex:{w['regex']}")
    a += [f"-s {w['skip']}", f"-c {w['count']}", "--kill yes", "--forward-signals"]
    return " ".join(a)


def line_of(run):
    fr = FRESH[run["case"]]
    return (f'{ENV_COMMON} DMK_SERVER_WRAP="{wrap_of(run)}" client {run["name"]} build:b12004-sm86 /extra/prof/ncu_client.py '
            f'--kind {fr["kind"]} --case {run["case"]} --out /runs/{{name}}.client.json')


CUTS = """\
# What was cut (budget: 25 GPU-minutes of ncu in total, pessimistic estimate; every cut kernel is covered by a measured sibling or by nsys):
#   - attn_k / attn_v MMQ (M=1024: 2 x 38 CTAs + fix-up, about 1.3 ms of a 420 ms N=2048 pass) and the other 3 attention-layer MMQ:
#     they have the shapes of the profiled ones (J, type, grid); only the attention layer 3 window of the extra run sees them.
#   - the tiny kernels (scale_f32, k_get_rows_float(_vec), cpy_scalar with 384 blocks, k_bin_bcast 1-256 blocks, unary_op_kernel; 2-12 us
#     each, 0.3 % of an N=2048 pass), k_bin_bcast residual adds, quantize_q8_1 (N<=8), flash_attn aux kernels, k_set_rows, mask_to_KV_max.
#   - quantize_mmq / rms_norm / glu only once per variant at N=2048 (extra-long-ew); by arithmetic from the nsys durations they run at
#     370-413 GB/s there, i.e. at the DRAM floor, so more instances add nothing.
#   - the J=48 (N=43) and J=64 (N=51) classes: a fresh server's first request never produces them (the first request of every case
#     batches the questions: 3 x 43 = 129, 2 x 4, 26; 2 x 51, 69); they need a warm server with other slot placement.
#   - N=69 (triage09 P2, J=80, 1 tile column) and the N=102/N=69 shapes of untraced passes beyond layer 0 MMQ.
#   - second and later layers: a layer repeats; layer 0 of a pass is profiled, the census gives the per-layer launch counts.
#   - application replay (needs a deterministic app and a client that waits for the replays), --set full outside extra-long-full.
"""


def fmt_min(s):
    return f"{s / 60:.1f}"


def sched_text(group, runs):
    rs = [r for r in runs if r["group"] == group]
    tot = {m: sum(estimate(r)[m] for r in rs) for m in ("opt", "pess")}
    nk = sum(estimate(r)["kernels"] for r in rs)
    head = [f"# phase 3, package 4 (ncu): {group}. Baseline D (build:b12004-sm86), RTX 3060 Ti, ncu 2026.3.1 from /tools.",
            f"# {len(rs)} runs, {nk} profiled kernels, estimate {fmt_min(tot['opt'])} GPU-min optimistic / {fmt_min(tot['pess'])} pessimistic "
            f"(run overhead {cfg.COST['run_fixed']['opt']:.0f}-{cfg.COST['run_fixed']['pess']:.0f} s each, 7-section set "
            f"{cfg.COST['t_save']['opt'] + cfg.COST['passes']['std'] * cfg.COST['t_pass']['opt']:.1f}-"
            f"{cfg.COST['t_save']['pess'] + cfg.COST['passes']['std'] * cfg.COST['t_pass']['pess']:.1f} s per kernel; see harness/profile/ncu/NOTES.txt).",
            "# One request per run on a fresh server (-np 3 --kv-unified --no-warmup); the window is (kernel regex, skip, count) from harness/profile/ncu/plan_ncu.py.",
            "# --clock-control none always; --kill yes ends the server after the last kernel of the window, so the client sees a reset."]
    body = []
    for r in rs:
        e = estimate(r)
        w = r["plan"][0]
        body.append(f"# {r['name']}: {r['why']}; {w['count']} kernels ({w['label']}); {fmt_min(e['opt'])}-{fmt_min(e['pess'])} min")
        body.append(line_of(r))
    return "\n".join(head) + "\n" + CUTS + "\n".join(body) + "\n", tot, nk


CTX = 40     # launches of the filter stream before/after the window stored for the alignment search of analyze_ncu.py


def write_files(runs, fresh, model, plan_dir=PLAN_DIR, sched_dir=SCHED_DIR):
    os.makedirs(plan_dir, exist_ok=True)
    os.makedirs(sched_dir, exist_ok=True)
    for r in runs:
        fr = fresh[r["case"]]
        exp, i = [], 0
        for w in r["plan"]:
            for pk, kk in w["kernels"]:
                exp.append(kernel_record(i, pk, kk, w["label"], fr, model))
                i += 1
        w0 = r["plan"][0]
        stream = fr.stream(w0["filter"])
        lo = max(0, w0["skip"] - CTX)
        ctx = []
        for j in range(lo, min(len(stream), w0["skip"] + w0["count"] + CTX)):
            pk, kk = stream[j]
            gx, gy, gz, exact = fix_grid(kk, kk["n_tokens"])
            ctx.append([j - w0["skip"], kk["func"], gx, gy, gz, kk["bx"], kk["by"], int(exact), kk["opd"]["J"] if kk.get("opd") else "",
                        kk["opd"]["wtype"] if kk.get("opd") else ""])
        passes = [dict(k=pi["k"], n_tokens=pi["p"]["n_tokens"], n_seqs=len(pi["p"]["seqs"]), n_out=pi["p"]["n_out"], model=pi["model"],
                       nsys_ms=sum(x["dur_ns"] for x in pi["ks"]) / 1e6) for pi in fr.pass_info]
        doc = dict(run=r["name"], group=r["group"], case=r["case"], kind=FRESH[r["case"]]["kind"], sections=r["sections"], why=r["why"],
                   windows=[{k: v for k, v in w.items() if k != "kernels"} for w in r["plan"]], expected=exp,
                   context_columns=["rel", "func", "gx", "gy", "gz", "bx", "by", "grid_exact", "J", "wtype"], context=ctx, passes=passes,
                   estimate=estimate(r))
        with open(os.path.join(plan_dir, r["name"] + ".json"), "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
    parts = []
    for g in ("smoke", "core", "extra"):
        txt, _, _ = sched_text(g, runs)
        with open(os.path.join(sched_dir, GROUP_FILE[g]), "w") as f:
            f.write(txt)
        parts.append((g, txt))
    allr = runs
    tot = {m: sum(estimate(r)[m] for r in allr) for m in ("opt", "pess")}
    nk = sum(estimate(r)["kernels"] for r in allr)
    with open(os.path.join(sched_dir, "4-ncu.txt"), "w") as f:
        f.write(f"# phase 3, package 4 (ncu), all runs in order: smoke, core, extra (= 4-ncu-smoke.txt + 4-ncu-core.txt + 4-ncu-extra.txt).\n"
                f"# {len(allr)} runs, {nk} profiled kernels, {fmt_min(tot['opt'])} GPU-min optimistic / {fmt_min(tot['pess'])} pessimistic "
                f"(budget 25). Headers of the three parts follow.\n")
        for g, txt in parts:
            f.write(txt)
    return tot, nk


# ----------------------------------------------------------------------------------------------------------------- output
def md_checks(rows):
    out = ["| case | pass | N (tokens) | sequences | outputs | model pass | model N = N | mul_mat_q: model / census | fix-ups: model / census | MMVQ: model / census | grid or type differs | fix-up flag differs |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t, c = r["trace"], r["census"]
        out.append(f"| {r['case']} | P{r['k']} | {r['n']} | {r['seqs']} | {r['n_out']} | {r['model']} | {'yes' if r['exact'] else 'no'} | {t[0]} / {c[0]} | "
                   f"{t[1]} / {c[1]} | {t[2]} / {c[2]} | {r['grid_bad']} | {r['fix_bad']} |")
    out.append("")
    out.append("Source: harness/profile/census/census_lib.py (pass_matmuls, mmq_launch) vs the kernel lists of data/pass-kernels.tsv.gz (nsys traces of the three old census captures, counts also "
               "checked against trace-passes.tsv); `grid or type differs` compares the census grid, J and weight type of every mul_mat_q launch with the traced pass "
               "(0 expected when the model pass has the same N; for the other model passes the census value is used).")
    return "\n".join(out)


def md_windows(runs):
    out = ["| run | request | pass(es) of the window | kernel filter (partial regex on the function name) | skip | count | kernels (first ... last) | nsys time of the window, ms |",
           "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        for w in r["plan"]:
            ks = [kk for _, kk in w["kernels"]]
            names = " ".join(f"{kk['func']}" + (f"[{kk['op']}]" if kk["op"] and not kk["op"].endswith("(fix-up)") else "") for kk in ks[:3])
            tail = f" ... {ks[-1]['func']}" + (f"[{ks[-1]['op']}]" if ks[-1]["op"] else "") if len(ks) > 3 else ""
            ms = sum(kk["dur_ns"] for kk in ks) / 1e6
            ps = ",".join(f"P{p}" for p in w["passes"])
            rx = (w["regex"] or "(none)").replace("|", "\\|")
            out.append(f"| {r['name']} | {r['case']} | {ps} | `{rx}` | {w['skip']} | {w['count']} | {names}{tail} | {ms:.2f} |")
    out.append("")
    out.append("Source: plan_ncu.py from census_lib.simulate_request and data/pass-kernels.tsv.gz (nsys traces); the last column is the sum of the nsys kernel durations of "
               "the model pass (not of ncu).")
    return "\n".join(out)


def md_estimates(runs):
    c = cfg.COST
    out = ["| run | request | windows | profiled kernels | sections | passes per kernel | optimistic, min | pessimistic, min | start/stop overhead included, s (opt / pess) |",
           "|---|---|---|---|---|---|---|---|---|"]
    tot = collections.defaultdict(lambda: [0, 0.0, 0.0, 0, 0])
    for r in runs:
        e = estimate(r)
        out.append(f"| {r['name']} | {r['case']} | {len(r['plan'])} | {e['kernels']} | {r['sections']} | {c['passes'][r['sections']]} | {fmt_min(e['opt'])} | "
                   f"{fmt_min(e['pess'])} | {c['run_fixed']['opt']:.0f} / {c['run_fixed']['pess']:.0f} |")
        t = tot[r["group"]]
        t[0] += 1
        t[1] += e["opt"]
        t[2] += e["pess"]
        t[3] += e["kernels"]
    out.append("")
    out.append("| file | runs | profiled kernels | optimistic, min | pessimistic, min |")
    out.append("|---|---|---|---|---|")
    to = [0, 0.0, 0.0, 0]
    for g in ("smoke", "core", "extra"):
        t = tot[g]
        out.append(f"| {GROUP_FILE[g]} | {t[0]} | {t[3]} | {fmt_min(t[1])} | {fmt_min(t[2])} |")
        to = [to[0] + t[0], to[1] + t[1], to[2] + t[2], to[3] + t[3]]
    out.append(f"| 4-ncu.txt (all) | {to[0]} | {to[3]} | {fmt_min(to[1])} | {fmt_min(to[2])} |")
    out.append("")
    out.append("Source: ncu_cfg.COST. seconds per run = run_fixed (opt 22 / pess 55); seconds per kernel = t_save + passes x t_pass "
               "(opt 0.3 + passes x 0.016, pess 1.5 + passes x 0.10); passes = 25 (7 sections, [INFERENCE]) or 42 (--set full, measured on the sample reports). "
               "None of these was measured on the RTX 3060 Ti; the smoke run measures them.")
    return "\n".join(out)


def print_run(r):
    print(f"== {r['name']} ({r['case']}, {r['sections']}): {r['why']}")
    i = 0
    for w in r["plan"]:
        print(f"-- window '{w['label']}': filter {w['filter']} regex={w['regex']!r} skip={w['skip']} count={w['count']} passes={w['passes']} "
              f"(stream length {w['n_stream']}, {w['before']} matching launches in earlier passes)")
        for pk, kk in w["kernels"]:
            print(f"  {i:3d} P{pk} pos {kk['pos']:4d} {kk['func']:28s} grid=({kk['gx']},{kk['gy']},{kk['gz']}) block=({kk['bx']},{kk['by']},{kk['bz']}) "
                  f"nsys={kk['dur_ns'] / 1000:8.1f}us {kk['op']}")
            i += 1


def check_metrics():
    """Every metric of ncu_cfg.METRICS must be collected by the section set or by EXTRA_METRICS (parsed from the section files)."""
    if not cfg.LOCAL_NCU_DIR or not os.path.isdir(os.path.join(cfg.LOCAL_NCU_DIR, "sections")):
        print("SKIP: no ncu installation with section files (set DMK_NCU to the ncu command line tool of Nsight Compute 2026.3.1)")
        return 0
    sec_dir = os.path.join(cfg.LOCAL_NCU_DIR, "sections")
    have = set(cfg.EXTRA_METRICS)
    by_id = {}                      # `--section` takes the Identifier inside the file, which differs from the file name (LaunchStats = LaunchStatistics.section)
    for fn in sorted(os.listdir(sec_dir)):
        if fn.endswith(".section"):
            with open(os.path.join(sec_dir, fn)) as f:
                text = f.read()
            m = re.search(r'^Identifier:\s*"([^"]+)"', text, re.M)
            if m:
                by_id[m.group(1)] = (fn, text)
    for s in cfg.SECTIONS_STD:
        if s not in by_id:
            print(f"MISSING section identifier {s} in {sec_dir}")
            return 1
        have |= set(re.findall(r'Name:\s*"([A-Za-z0-9_.]+)"', by_id[s][1]))
        print(f"section {s:24s} = {by_id[s][0]}")
    want = dict(cfg.METRICS)
    for s in cfg.STALLS:
        want["stall_" + s] = cfg.STALL_METRIC.format(s)
    bad = {k: v for k, v in want.items() if v not in have and v not in cfg.NOT_IN_SECTIONS_OK}
    for k, v in sorted(bad.items()):
        print(f"MISSING in the 7 sections + extras: {k} = {v}")
    print(f"{len(want) - len(bad)} of {len(want)} analysis metrics are collected by {len(cfg.SECTIONS_STD)} sections + {len(cfg.EXTRA_METRICS)} extra metrics")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="print the expected kernels of this run")
    ap.add_argument("--write", action="store_true", help="write plan/*.json and sched/4-ncu-*.txt")
    ap.add_argument("--plan-dir", default=PLAN_DIR, help="folder for plan/<run>.json (--write; default: plan/ of this folder)")
    ap.add_argument("--sched-dir", default=SCHED_DIR, help="folder for the 4-ncu-*.txt schedules (--write; default: sched/ of harness/profile)")
    ap.add_argument("--shift", action="append", default=[], metavar="KEY=N", help="add N to the skip of the windows with filter KEY")
    ap.add_argument("--check-metrics", action="store_true")
    ap.add_argument("--md", action="store_true", help="markdown tables only")
    a = ap.parse_args()
    if a.check_metrics:
        sys.exit(check_metrics())
    shifts = {}
    for s in a.shift:
        k, v = s.split("=")
        if k not in FILTERS:
            sys.exit(f"unknown filter key {k}: {sorted(FILTERS)}")
        shifts[k] = int(v)
    models = load_models()
    model = cl.Model()
    fresh = {c: Fresh(c, models, model) for c in FRESH}
    trace_rows = load_trace_passes()
    rows, fails = check_passes(fresh, trace_rows, models)
    runs = plan_all(fresh, shifts)
    if a.run:
        for r in runs:
            if r["name"] == a.run:
                print_run(r)
                return
        sys.exit(f"no run {a.run}: {[r['name'] for r in runs]}")
    print("### Pass structure of the first request on a fresh server and cross-checks\n")
    print(md_checks(rows))
    for r in rows:
        for n in r.get("notes", []):
            print(f"\nnote: {n}  (model pass is a neighbour; the census value is used for the matmul shapes)")
    print("\n### Windows\n")
    print(md_windows(runs))
    print("\n### Estimates\n")
    print(md_estimates(runs))
    if fails:
        print("\nFAILED CROSS-CHECKS:")
        for f in fails:
            print("  " + f)
    if a.write:
        tot, nk = write_files(runs, fresh, model, a.plan_dir, a.sched_dir)
        print(f"\nwrote {a.plan_dir}/*.json and {a.sched_dir}/4-ncu-*.txt: {len(runs)} runs, {nk} kernels, {fmt_min(tot['opt'])} / {fmt_min(tot['pess'])} min")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
