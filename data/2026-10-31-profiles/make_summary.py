#!/usr/bin/env python3
"""make_summary.py - the small tables that the notebook entry of phase 3 shows.

Reads the analysis tables in <tables>/ (made by make_tables.sh) and writes <tables>/summary/*.tsv.
Every table that the notebook shows is one file here. One tab between the fields. Units are in the column names.

Usage: make_summary.py [--tables DIR] [--out DIR]
"""
import argparse
import csv
import os
import re
import sys
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ["short1q", "short2q", "medium3q", "long3q"]


def read_tsv(path):
    """Rows as dicts. A header like `duration_us [measured duration, us]` becomes `duration_us`."""
    with open(path, newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        header = [re.sub(r" \[.*$", "", h) for h in next(rd)]
        return [dict(zip(header, row)) for row in rd]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def fmt(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return ("%." + str(nd) + "f") % x


class Writer:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.names = []

    def write(self, name, header, rows):
        path = os.path.join(self.out, name)
        with open(path, "w") as f:
            f.write("\t".join(header) + "\n")
            for r in rows:
                if len(r) != len(header):
                    sys.exit("make_summary: row of %s has %d fields, header has %d" % (name, len(r), len(header)))
                f.write("\t".join(str(v) for v in r) + "\n")
        self.names.append(name)


def where(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


# ---------------------------------------------------------------- decomposition (part 1)
def decomposition(T, W):
    stack = {r["kind"]: r for r in where(read_tsv(T("decomp/stack.tsv")), phase="unique")}
    hb = {r["kind"]: r for r in where(read_tsv(T("decomp/hostbound.tsv")), phase="unique")}
    rows = []
    for k in KINDS:
        s, h = stack[k], hb[k]
        c = {n: num(s[n]) for n in s if n.endswith("_ms")}
        before = c["http_in_ms"] + c["parse_ms"] + c["build_ms"] + c["sched_ms"] + c["first_setup_ms"] + c["prepare_ms"]
        after = c["between_ms"] + c["post_submit_ms"] + c["tail_server_ms"] + c["format_ms"] + c["http_out_ms"]
        span = num(h["gpu_span_ms"])
        rows.append([k, s["n"], fmt(c["wall_ms"], 1), fmt(before, 1), fmt(c["submit_ms"], 1), fmt(c["gpu_wait_ms"], 1), fmt(after, 1),
                     fmt(span, 1), fmt(num(h["floor_max_ms"]), 1), fmt(num(h["floor_roofline_ms"]), 1), fmt(num(h["span_over_floor"]), 2)])
    W.write("request-time-decomposition.tsv",
            ["request_type", "requests", "wall_ms", "before_submit_ms", "submit_ms", "gpu_wait_ms", "other_steps_ms", "gpu_span_ms",
             "floor_pass_max_ms", "floor_roofline_ms", "span_over_floor_pass_max"], rows)

    labels = ["Weight-read floor of one pass", "HTTP and client time", "Prompt build and use of the tokenizer", "Start of the task and start of decode",
              "Graph build and submit", "End of the request after the last wait", "GPU wait above the floor", "Wall time"]
    wf = read_tsv(T("decomp/waterfall-short1q.tsv"))[:8]
    rows = [[lab, fmt(num(r["mean_ms"]), 2), fmt(num(r["cumulative_mean_ms"]), 2), fmt(num(r["share_of_wall_pct"]), 1)] for lab, r in zip(labels, wf)]
    W.write("short1q-waterfall.tsv", ["step", "mean_ms", "cumulative_ms", "share_of_wall_pct"], rows)

    ref = read_tsv(T("decomp/waterfall-short1q-ref.tsv"))
    names = {0: ("weight-read floor of a pass with the output head", "ms"), 1: ("int8 floor of the pass", "ms"),
             3: ("roofline floor of the pass", "ms"), 4: ("measurement of one short pass before phase 3", "ms"),
             5: ("wall time, mean", "ms"), 7: ("GPU span, mean (maximum of the GPU time)", "ms"),
             8: ("GPU wait, mean (minimum of the GPU time)", "ms"), 13: ("GPU span divided by the weight-read floor", "ratio"),
             14: ("wall time divided by the weight-read floor", "ratio")}
    rows = []
    for i, (nm, unit) in names.items():
        rows.append([nm, unit, fmt(num(list(ref[i].values())[1]), 2)])
    W.write("short1q-reference.tsv", ["quantity", "unit", "value"], rows)

    rows = []
    for r in where(read_tsv(T("decomp/instrumentation.tsv")), phase="all timed"):
        rows.append([r["kind"], r["n_timing"], fmt(num(r["mean_timing_ms"]), 2), fmt(num(r["mean_ref_ms"]), 2), fmt(num(r["d_mean_pct"]), 2)])
    W.write("instrumentation-cost.tsv", ["request_type", "requests_each_arm", "timing_build_mean_ms", "reference_build_mean_ms", "difference_pct"], rows)

    rows = []
    for r in read_tsv(T("decomp/warm-path.tsv")):
        rows.append([r["kind"], r["case"], r["n_pairs"], fmt(num(r["wall1_mean_ms"]), 1), fmt(num(r["wall2_mean_ms"]), 1), fmt(num(r["d_wall_ms"]), 2),
                     fmt(num(r["d_submit_ms"]), 2), fmt(num(r["d_gpu_wait_ms"]), 2)])
    W.write("warm-path.tsv", ["request_type", "case", "pairs", "first_wall_ms", "second_wall_ms", "wall_change_ms", "submit_change_ms", "gpu_wait_change_ms"], rows)


# ---------------------------------------------------------------- nsys (part 2)
def nsys(T, W):
    run = "p3-nsys-all48"
    req = [r for r in read_tsv(T("nsys/requests.tsv")) if r["run"] == run and r["phase"] != "warmup"]
    rows = []
    for k in KINDS:
        rs = [r for r in req if r["kind"] == k]
        m = lambda c: mean([num(r[c]) for r in rs])
        wall, busy = m("client_wall_ms"), m("gpu_busy_ms")
        rows.append([k, len(rs), fmt(m("n_passes"), 0), fmt(wall, 1), fmt(m("send_to_first_kernel_ms"), 1), fmt(busy, 1), fmt(m("gap_in_pass_ms"), 1),
                     fmt(m("gap_between_passes_ms"), 1), fmt(m("last_kernel_to_recv_ms"), 1), fmt(100 * busy / wall, 1)])
    W.write("nsys-request-time.tsv",
            ["request_type", "requests", "passes_per_request", "client_wall_ms", "send_to_first_kernel_ms", "gpu_busy_ms", "idle_inside_passes_ms",
             "idle_between_passes_ms", "last_kernel_to_recv_ms", "gpu_busy_share_pct"], rows)

    passes = read_tsv(T("nsys/passes.tsv"))
    ps = [r for r in passes if r["row_kind"] == "pass" and r["trace"] == run and r["phase"] != "warmup" and r["boundary"] != "first"]
    order = ["request-start", "prefix-large->prefix-small", "prefix-small->suffix", "suffix->suffix"]
    rows = []
    for k in KINDS:
        for b in order:
            rs = [r for r in ps if r["kind"] == k and r["boundary"] == b]
            if not rs:
                continue
            m = lambda c: mean([num(r[c]) for r in rs])
            parts = [m("host_gap_drain_ms"), m("host_gap_cpu_ms"), m("host_gap_cuda_api_ms"), m("host_gap_first_launch_ms")]
            rows.append([k, b, len(rs), fmt(sum(parts), 2), fmt(parts[0], 2), fmt(parts[1], 2), fmt(m("host_gap_cpu_os_wait_ms"), 2),
                         fmt(m("host_gap_cpu_busy_ms"), 2), fmt(parts[2], 2), fmt(parts[3], 3)])
    W.write("nsys-gap-before-pass.tsv",
            ["request_type", "boundary", "passes", "gap_mean_ms", "gpu_done_to_sync_return_ms", "no_cuda_call_ms", "no_cuda_call_in_os_wait_ms",
             "no_cuda_call_busy_ms", "cuda_calls_ms", "first_launch_to_first_kernel_ms"], rows)

    rq = [r for r in passes if r["row_kind"] == "request" and r["trace"] == run and r["phase"] != "warmup"]
    rows_api, rows_up = [], []
    for k in KINDS:
        rs = [r for r in rq if r["kind"] == k]
        m = lambda c: mean([num(r[c]) for r in rs])
        rows_api.append([k, len(rs), fmt(m("n_cudaLaunchKernel_api_calls"), 0), fmt(m("cpu_launch_api_ms"), 1), fmt(m("n_sync_calls"), 1),
                         fmt(m("cpu_sync_api_ms"), 1), fmt(m("cpu_api_time_ms"), 1), fmt(m("n_cmdbuf_full_events"), 1), fmt(m("cmdbuf_full_ms"), 1),
                         fmt(m("launch_queue_depth_max"), 0)])
        mb, ms = m("memcpy_h2d_bytes") / 1e6, m("memcpy_h2d_gpu_ms")
        rows_up.append([k, len(rs), fmt(m("n_memcpy_h2d"), 1), fmt(mb, 1), fmt(ms, 2), fmt(mb / ms, 2)])
    W.write("nsys-host-cuda-calls.tsv",
            ["request_type", "requests", "launch_calls", "launch_calls_ms", "sync_calls", "sync_calls_ms", "all_cuda_calls_ms", "command_buffer_full_events",
             "command_buffer_full_ms", "launch_queue_depth_max"], rows_api)
    W.write("nsys-uploads.tsv", ["request_type", "requests", "upload_copies", "uploaded_mb", "upload_gpu_ms", "upload_rate_gb_s"], rows_up)

    # kernel time and launches per pass type, pooled over the captures (scope ALL)
    cl = [r for r in read_tsv(T("nsys/classes.tsv")) if r["scope"] == "ALL"]
    groups = OrderedDict()
    for r in cl:
        if r["level"] == "total":
            groups[(r["pass_type"], r["n_class"])] = {"passes": r["n_passes"], "all": num(r["kernel_ms_per_pass_mean"]), "launches": num(r["launches_per_pass_mean"]), "classes": {}}
    for r in cl:
        if r["level"] == "class":
            groups[(r["pass_type"], r["n_class"])]["classes"][r["class"]] = num(r["kernel_ms_per_pass_mean"])
    cls = ["MMQ", "GDN", "glu", "FA", "quantize", "copy", "rms_norm", "ssm_conv", "elementwise", "MMVQ", "other"]
    rows = []
    for (pt, n), g in groups.items():
        rows.append([pt, n, g["passes"], fmt(g["launches"], 0), fmt(g["all"], 1)] + [fmt(g["classes"].get(c, 0.0), 1) for c in cls])
    W.write("nsys-kernel-ms-per-pass.tsv",
            ["pass_type", "tokens_in_pass", "passes", "launches_per_pass", "all_kernels_ms"] + [c + "_ms" for c in cls], rows)

    rows = []
    for r in read_tsv(T("nsys/overhead.tsv")):
        if r["run"] == "ALL":
            rows.append([r["kind"], r["n_pairs"], fmt(num(r["ctl_mean_ms"]), 1), fmt(num(r["nsys_mean_ms"]), 1), fmt(num(r["delta_mean_ms"]), 2),
                         fmt(num(r["delta_pct_of_ctl_mean"]), 2), fmt(num(r["delta_ci95_halfwidth_ms"]), 2)])
    W.write("nsys-overhead.tsv", ["request_type", "pairs", "without_nsys_mean_ms", "with_nsys_mean_ms", "difference_ms", "difference_pct", "ci95_halfwidth_ms"], rows)


# ---------------------------------------------------------------- ncu (part 4)
MMQ_OPS = ["attn_qkv", "attn_gate(z)", "ssm_out", "ffn_up", "ffn_gate", "ffn_down"]


def ncu(T, W):
    sol = read_tsv(T("ncu/sol.tsv"))
    kern = read_tsv(T("ncu/kernels.tsv"))
    align = {r["run"]: r["status"].split(" ")[0] for r in read_tsv(T("ncu/alignment.tsv"))}

    def parse_mmq(cls):
        m = re.match(r"^mmq (\S+) (Q\d_0) J=(\d+) N=(\d+)$", cls)
        return m.groups() if m else None

    big = ["p3-ncu-core-long-p1", "p3-ncu-extra-med-p0", "p3-ncu-extra-long-full"]
    rows = []
    for run in big:
        for r in where(sol, run=run):
            p = parse_mmq(r["class"])
            if p and p[0] in MMQ_OPS:
                rows.append([p[3], p[0], p[1], r["launches_per_pass"], fmt(num(r["duration_us"]), 1), fmt(num(r["share_of_pass_pct"]), 1),
                             fmt(num(r["useful_tops_pct"]), 1), fmt(num(r["imma_pipe_pct"]), 1), fmt(num(r["dram_pct_of_422"]), 1), r["stall_top3"]])
    W.write("ncu-mmq-large-n.tsv",
            ["n_tokens", "op", "weight_type", "launches_per_pass", "duration_us", "share_of_pass_pct", "int8_of_peak_pct", "imma_pipe_active_pct",
             "dram_of_422_gb_s_pct", "top_stall_reasons"], rows)

    rows = []
    for r in where(kern, run="p3-ncu-extra-long-full"):
        if r["function"] == "mul_mat_q":
            p = parse_mmq(r["class"])
            rows.append([p[1] + " " + p[0], r["grid"], r["block"], r["regs"], r["smem_dyn_b"], r["occ_limit_regs"], r["occ_limit_smem"], r["occ_limit_warps"],
                         r["occ_limit_blocks"], fmt(num(r["warps_active_pct"]), 1), fmt(num(r["issue_active_pct"]), 1), fmt(num(r["imma_pipe_pct"]), 1),
                         fmt(num(r["sm_clock_mhz"]), 0)])
    W.write("ncu-mmq-resources.tsv",
            ["kernel", "grid", "block", "registers_per_thread", "dynamic_smem_b", "blocks_per_sm_limit_registers", "blocks_per_sm_limit_smem",
             "blocks_per_sm_limit_warps", "blocks_per_sm_limit_blocks", "warps_active_pct", "issue_slots_active_pct", "imma_pipe_active_pct",
             "sm_clock_mhz"], rows)

    small = ["p3-ncu-extra-long-p4", "p3-ncu-core-short-p0", "p3-ncu-core-short-p1"]
    rows = []
    for run in small:
        for r in where(sol, run=run):
            p = parse_mmq(r["class"])
            if p and p[0] in ("attn_qkv", "ssm_out", "ffn_up", "ffn_down"):
                rows.append([p[3], p[2], p[0], p[1], fmt(num(r["duration_us"]), 1), r["bound"], fmt(num(r["pct_of_limit"]), 1), fmt(num(r["imma_pipe_pct"]), 1),
                             fmt(num(r["useful_tops_pct"]), 1)])
    W.write("ncu-mmq-small-n.tsv",
            ["n_tokens", "tile_columns_j", "op", "weight_type", "duration_us", "limit", "of_limit_pct", "imma_pipe_active_pct", "int8_of_peak_pct"], rows)

    keep = [("p3-ncu-core-mv", None), ("p3-ncu-core-long-p2", "gdn_single"), ("p3-ncu-core-long-p0", "concat_non_cont"), ("p3-ncu-core-long-p0", "ssm_conv_f32"),
            ("p3-ncu-extra-long-att", "flash_attn_ext_f16"), ("p3-ncu-extra-long-ew", None)]
    rows = []
    for run, prefix in keep:
        for r in where(sol, run=run):
            c = r["class"]
            if prefix and not c.startswith(prefix):
                continue
            if run == "p3-ncu-core-long-p2" and not c.startswith("gdn_single"):
                continue
            rows.append([run.replace("p3-ncu-", ""), align[run], c, fmt(num(r["duration_us"]), 1), r["bound"], fmt(num(r["pct_of_limit"]), 1),
                         fmt(num(r["dram_gb_s"]), 1), fmt(num(r["dram_pct_of_422"]), 1), r["stall_top3"]])
    W.write("ncu-other-kernels.tsv",
            ["run", "run_alignment", "kernel", "duration_us", "limit", "of_limit_pct", "dram_gb_s", "dram_of_422_gb_s_pct", "top_stall_reasons"], rows)

    rows = []
    for r in read_tsv(T("ncu/alignment.tsv")):
        rows.append([r["run"].replace("p3-ncu-", ""), r["planned_launches"], r["matching"], r["shift"], r["status"].split(" ")[0]])
    W.write("ncu-run-alignment.tsv", ["run", "planned_launches", "matching_launches", "shift_launches", "status"], rows)

    ops = read_tsv(T("ncu/ffn_up_q4_0_N2048-opcodes.tsv"))
    total = [r for r in ops if r["opcode"] == "TOTAL"][0]
    imma = [r for r in ops if r["opcode"] == "IMMA"][0]
    rows = []
    for r in [r for r in ops if r["opcode"] != "TOTAL"][:10]:
        rows.append([r["opcode"], r["sass_lines"], fmt(num(r["stall_samples_share_pct"]), 1), fmt(num(r["instructions_executed"]) / 1e6, 2),
                     fmt(num(r["instructions_executed"]) / num(imma["instructions_executed"]), 2)])
    rows.append(["all opcodes", total["sass_lines"], "100.0", fmt(num(total["instructions_executed"]) / 1e6, 2),
                 fmt(num(total["instructions_executed"]) / num(imma["instructions_executed"]), 2)])
    W.write("ncu-instruction-mix.tsv", ["opcode", "sass_lines", "stall_samples_pct", "instructions_executed_millions", "instructions_per_imma"], rows)


# ---------------------------------------------------------------- leads (part 3)
def lead_28495(T, W):
    rows = []
    for run, label in (("kvwin-nsys-passes.tsv", "kv-unified"), ("kvwin-nsys-nokvu-passes.tsv", "no-kv-unified")):
        ps = [r for r in read_tsv(T("leads-kvwin/" + run)) if r["phase"] not in ("warmup", "other")]
        t1 = {(r["j"], r["N"]): r for r in ps if r["phase"] == "T1"}
        for ph in sorted({r["phase"] for r in ps} - {"T1"}):
            same = [r for r in ps if r["phase"] == ph and (r["j"], r["N"]) in t1]
            fa, fa1 = sum(num(r["fa"]) for r in same), sum(num(t1[(r["j"], r["N"])]["fa"]) for r in same)
            bz, bz1 = sum(num(r["busy"]) for r in same), sum(num(t1[(r["j"], r["N"])]["busy"]) for r in same)
            n_all = len([r for r in ps if r["phase"] == ph])
            rows.append([label, ph, "%d of %d" % (len(same), n_all), fmt(fa, 2), fmt(fa1, 2), fmt(fa / fa1, 3), fmt(bz, 1), fmt(bz1, 1), fmt(bz / bz1, 3)])
    W.write("lead-28495-nsys-phases.tsv",
            ["cache_mode", "phase", "passes_compared", "fa_ms", "fa_same_passes_of_t1_ms", "fa_ratio", "gpu_busy_ms", "gpu_busy_same_passes_of_t1_ms", "gpu_busy_ratio"],
            rows)


def lead_30779(T, W):
    reqs = read_tsv(T("leads-childpass/childpass-requests.tsv"))
    rows = []
    for k in KINDS:
        rs = [r for r in reqs if r["kind"] == k]
        below = [r for r in rs if r["below"] == "True"]
        more_dec = [r for r in rs if num(r["dec_vs_fixed"]) and num(r["dec_vs_fixed"]) > 0]
        diff_sizes = [r for r in rs if r["sizes_vs_fixed"] == "True"]
        rows.append([k, len(rs), sum(1 for r in rs if r["unified_ok"] == "True"), sum(1 for r in rs if r["more_ub"] == "True"), len(below), len(more_dec), len(diff_sizes)])
    W.write("lead-30779-passes.tsv",
            ["request_type", "requests", "passes_equal_census", "requests_with_more_passes_than_census", "requests_child_below_parent", "requests_with_more_decode_calls_than_fixed_order",
             "requests_with_other_pass_sizes_than_fixed_order"], rows)

    rows = []
    for k in ("medium3q", "long3q"):
        for below in ("True", "False"):
            rs = [r for r in reqs if r["kind"] == k and r["below"] == below]
            rows.append([k, "yes" if below == "True" else "no", len(rs), fmt(mean([num(r["total_us"]) / 1000 for r in rs]), 2), fmt(mean([num(r["n_dec"]) for r in rs]), 2)])
    W.write("lead-30779-cost.tsv", ["request_type", "child_below_parent", "requests", "request_total_mean_ms", "decode_calls_mean"], rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", default=os.path.join(HERE, "tables"))
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(a.tables, "summary")
    T = lambda rel: os.path.join(a.tables, rel)
    W = Writer(out)
    parts = [
        ("decomp", decomposition, ["decomp/stack.tsv", "decomp/hostbound.tsv", "decomp/waterfall-short1q.tsv", "decomp/waterfall-short1q-ref.tsv",
                                   "decomp/instrumentation.tsv", "decomp/warm-path.tsv"]),
        ("nsys", nsys, ["nsys/requests.tsv", "nsys/passes.tsv", "nsys/classes.tsv", "nsys/overhead.tsv"]),
        ("ncu", ncu, ["ncu/sol.tsv", "ncu/kernels.tsv", "ncu/alignment.tsv", "ncu/ffn_up_q4_0_N2048-opcodes.tsv"]),
        ("lead 28495", lead_28495, ["leads-kvwin/kvwin-nsys-passes.tsv", "leads-kvwin/kvwin-nsys-nokvu-passes.tsv"]),
        ("lead 30779", lead_30779, ["leads-childpass/childpass-requests.tsv"]),
    ]
    for name, fn, needs in parts:
        missing = [r for r in needs if not os.path.isfile(T(r))]
        if missing:
            print("make_summary: skipped %s, missing %s" % (name, ", ".join(missing)))
        else:
            fn(T, W)
    print("make_summary: wrote %d tables to %s" % (len(W.names), out))


if __name__ == "__main__":
    main()
