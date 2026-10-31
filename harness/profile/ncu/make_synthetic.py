#!/usr/bin/env python3
"""Synthetic ncu raw-page CSVs with llama kernel names and KNOWN values, to test analyze_ncu.py without a GPU.

The kernels are the 13 launches of the plan of run p3-ncu-core-long-p0 (plan/p3-ncu-core-long-p0.json: demangled names, grids and blocks of
the traced pass). Durations, bytes, pipe utilisation and stall ratios are invented but fixed; for five kernels the speed-of-light result is also
computed here BY HAND from the matmul shapes (explicit arithmetic below, not from the plan or from analyze_ncu.py) and written to
synthetic-expected.json, which selftest_ncu.py compares with the output of analyze_ncu.py.

Writes into <tmp>/ncu-synthetic/ (tmp = $TMPDIR or /tmp; override with --out):
  aligned/p3-ncu-core-long-p0.csv      all 13 launches, ncu default (auto-scaled) units, 6 decimals
  base/p3-ncu-core-long-p0.csv         the same data as `--print-units base` (ns, byte, hz, thousands separators, 2 decimals)
  shifted/p3-ncu-core-long-p0.csv      the window starts 3 launches late (launches 3..15 of the stream)
  misaligned/p3-ncu-core-long-p0.csv   13 launches of another part of the stream (the window of p3-ncu-extra-long-ew)
  missing/p3-ncu-core-long-p0.csv      aligned data without DRAM / L2 / pipe / stall / SM clock columns
  empty/p3-ncu-core-long-p0.csv        what ncu prints for a report with 0 kernels (a warning, no CSV)
  synthetic-expected.json
usage: make_synthetic.py [--out DIR]
"""
import argparse
import csv
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ncu_cfg as cfg  # noqa: E402

RUN = "p3-ncu-core-long-p0"
FIXED = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC"]
SM_HZ = 1.8e9
DRAM_HZ = 7.0e9 / 2          # invented
STALL_RATIOS = {"long_scoreboard": 1.0, "short_scoreboard": 0.5, "barrier": 0.25, "mio_throttle": 0.25, "selected": 0.4, "not_selected": 0.1}

# limits used by the hand computation (the same numbers as ncu_cfg, written out so the test does not import the code under test)
BW, INT8, FP16, FP32, FLOOR_US = 422e9, 149.3e12, 38e12, 18.1e12, 1.0

# measured values of the hand-checked kernels (plan index -> dict)
OVERRIDE = {
    0: dict(dur_us=1800.0, rd=28_000_000, wr=67_108_864, imma=31.5, ops=85_899_345_920),    # attn_qkv Q8_0 M=8192 N=2048 K=2560
    1: dict(dur_us=640.0, rd=67_108_864, wr=67_108_864, imma=0.0, ops=0),                    # concat_non_cont
    3: dict(dur_us=42.0, rd=6_000_000, wr=262_144, imma=12.0, ops=335_544_320),               # ssm_beta Q8_0 M=32 N=2048 K=2560
    7: dict(dur_us=1500.0, rd=50_000_000, wr=50_000_000, imma=18.0, ops=0),                  # gdn_single
    12: dict(dur_us=1950.0, rd=34_500_000, wr=20_971_520, imma=33.0, ops=96_636_764_160),     # ffn_down Q4_0 M=2560 N=2048 K=9216
}


def hand_expected():
    """Speed of light of the hand-checked kernels from the shapes (all numbers spelled out)."""
    out = {}

    def sol(parts):
        b = max(parts, key=parts.get)
        return b, parts[b]

    # attn_qkv: Q8_0 (34 B per 32), M=8192, N=2048, K=2560
    w = 8192 * 2560 // 32 * 34              # 22,282,240
    q = 2048 * 2560 * 9 // 8                # 5,898,240
    o = 2048 * 8192 * 4                     # 67,108,864
    ops = 2 * 8192 * 2048 * 2560            # 85,899,345,920
    parts = {"DRAM": (w + q + o) / BW * 1e6, "INT8": ops / INT8 * 1e6, "launch": FLOOR_US}
    b, f = sol(parts)
    out["mmq attn_qkv Q8_0 J=128 N=2048"] = dict(bound=b, floor_us=f, duration_us=1800.0, pct_of_limit=100 * f / 1800.0, gap_us=1800.0 - f, launches_per_pass=24,
                                                  gap_ms_per_pass=24 * (1800.0 - f) / 1e3, ms_per_pass=24 * 1800.0 / 1e3, useful_tops=ops / 1800e-6 / 1e12,
                                                  dram_gb_s=(28_000_000 + 67_108_864) / 1800e-6 / 1e9, parts=parts, padded_over_useful=1.0)
    # concat_non_cont: read + write 2048 x 8192 f32
    by = 2 * 2048 * 8192 * 4                # 134,217,728
    parts = {"DRAM": by / BW * 1e6, "launch": FLOOR_US}
    b, f = sol(parts)
    out["concat_non_cont grid=(8192,1,1) N=2048"] = dict(bound=b, floor_us=f, duration_us=640.0, pct_of_limit=100 * f / 640.0, gap_us=640.0 - f, launches_per_pass=24,
                                                          dram_gb_s=(67_108_864 * 2) / 640e-6 / 1e9, parts=parts)
    # ssm_beta: Q8_0, M=32, N=2048, K=2560
    w = 32 * 2560 // 32 * 34
    q = 2048 * 2560 * 9 // 8
    o = 2048 * 32 * 4
    ops = 2 * 32 * 2048 * 2560
    parts = {"DRAM": (w + q + o) / BW * 1e6, "INT8": ops / INT8 * 1e6, "launch": FLOOR_US}
    b, f = sol(parts)
    out["mmq ssm_beta Q8_0 J=128 N=2048"] = dict(bound=b, floor_us=f, duration_us=42.0, pct_of_limit=100 * f / 42.0, gap_us=42.0 - f, launches_per_pass=24, parts=parts)
    # ffn_down: Q4_0 (18 B per 32), M=2560, N=2048, K=9216
    w = 2560 * 9216 // 32 * 18
    q = 2048 * 9216 * 9 // 8
    o = 2048 * 2560 * 4
    ops = 2 * 2560 * 2048 * 9216
    parts = {"DRAM": (w + q + o) / BW * 1e6, "INT8": ops / INT8 * 1e6, "launch": FLOOR_US}
    b, f = sol(parts)
    out["mmq ffn_down Q4_0 J=128 N=2048"] = dict(bound=b, floor_us=f, duration_us=1950.0, pct_of_limit=100 * f / 1950.0, gap_us=1950.0 - f, launches_per_pass=31,
                                                  useful_tops=ops / 1950e-6 / 1e12, useful_tops_pct=100 * ops / 1950e-6 / 1e12 / 149.3, parts=parts)
    # gdn_single (mma path): flops 3*2*128*128*32*2048, bytes = state r+w + token streams (plan value, checked against census_lib.gdn_cost by plan_ncu)
    fl = 3 * 2 * 128 * 128 * 32 * 2048
    state_rw = 2 * 128 * 128 * 32 * 4                       # recurrent state of one sequence, read + write
    streams = 2048 * (2 * 16 * 128 + 4096) * 4 + 2048 * 4096 * 4 + 2 * 2048 * 32 * 4   # q, k, v in; output; gates
    parts = {"DRAM": (state_rw + streams) / BW * 1e6, "FP16": fl / FP16 * 1e6, "launch": FLOOR_US}
    b, f = sol(parts)
    out["gdn_single grid=(32,1,1) N=2048"] = dict(bound=b, floor_us=f, duration_us=1500.0, pct_of_limit=100 * f / 1500.0, gap_us=1500.0 - f, launches_per_pass=24, parts=parts)
    return out


STALL_SHARES = {"long_scoreboard": 50.0, "short_scoreboard": 25.0, "barrier": 12.5, "mio_throttle": 12.5}   # of the 4 stall reasons, selected/not_selected excluded


def metric_columns(missing=False):
    cols = []
    for key, name in cfg.METRICS.items():
        if missing and key in ("dram_rd", "dram_wr", "dram_bps", "dram_pct", "dram_hz", "l2_bytes", "l2_hit", "l1_hit", "imma_pct", "hmma_pct", "imma_inst_pct",
                               "hmma_inst_pct", "imma_inst", "hmma_inst", "int8_ops", "sm_hz", "sm_hz_alt", "latency", "warps_pct"):
            continue
        cols.append((key, name))
    return cols


def unit_and_scale(key, base):
    """(unit, divisor) of a metric in the CSV: ncu default = auto-scaled, --print-units base = ns / byte / hz."""
    if key == "duration_ns":
        return ("ns", 1.0) if base else ("us", 1e3)
    if key in ("dram_rd", "dram_wr", "l2_bytes"):
        return ("byte", 1.0) if base else ("Mbyte", 1e6)
    if key == "dram_bps":
        return ("byte/s", 1.0) if base else ("Gbyte/s", 1e9)
    if key in ("sm_hz", "sm_hz_alt", "dram_hz"):
        return ("hz", 1.0) if base else ("Ghz", 1e9)
    if key.endswith("_pct") or key in ("l2_hit", "l1_hit"):
        return "%", 1.0
    if key in ("inst", "imma_inst", "hmma_inst"):
        return "inst", 1.0
    return "", 1.0


def kernel_values(e, i):
    """Invented measured values of plan kernel e (index i)."""
    ov = OVERRIDE.get(i, {})
    dur = ov.get("dur_us", round(e["dur_ns_nsys_proxy"] / 1e3 * 1.10, 3))
    bm = e["bytes_model"] if e["bytes_model"] != "" else 1_000_000
    rd = ov.get("rd", int(bm * 0.6))
    wr = ov.get("wr", int(bm * 0.4))
    ops = ov.get("ops", e["ops_int8"] if e["ops_int8"] != "" else 0)
    v = {
        "duration_ns": dur * 1e3, "sm_hz": SM_HZ, "sm_hz_alt": SM_HZ, "dram_hz": DRAM_HZ, "sm_cycles_active": dur * 1e3 * 1.8 * 0.9, "dram_rd": rd, "dram_wr": wr,
        "dram_bps": (rd + wr) / (dur * 1e-6), "dram_pct": 100 * (rd + wr) / (dur * 1e-6) / 448e9, "l2_bytes": (rd + wr) * 1.5, "l2_hit": 55.25, "l1_hit": 10.5,
        "imma_pct": ov.get("imma", 20.0), "hmma_pct": 0.0, "imma_inst_pct": ov.get("imma", 20.0) / 2, "hmma_inst_pct": 0.0, "imma_inst": ops / 8192,
        "hmma_inst": 0, "int8_ops": ops, "inst": 5_000_000, "issue_pct": 40.0, "sm_thr_pct": 60.0, "warps_pct": 45.5, "warps_max_pct": 50.0, "latency": 25.0,
        "regs": e["regs"], "smem_static": 0, "smem_dyn": e["smem_dyn"], "smem_driver": 1024, "grid": e["gx"] * e["gy"] * e["gz"], "block": e["bx"] * e["by"] * e["bz"],
        "waves": 3.5, "occ_lim_regs": 1, "occ_lim_smem": 1, "occ_lim_warps": 6, "occ_lim_blocks": 16, "passes": 25,
    }
    return v


def fmt_cell(x, base):
    if x is None:
        return ""
    if base:
        return f"{int(x):,}" if float(x).is_integer() else f"{x:,.2f}"
    return f"{x:.6f}"


def demangled(func, c=None):
    if func == "mul_mat_q" and c:
        return f"void mul_mat_q<(ggml_type){ {'Q4_0': 2, 'Q8_0': 8}[c[9]] }, (int){c[8]}, (bool)0, (ggml_prec)30, (bool)0>(const char *, const char *)"
    return f"void {func}<(int)1>(const float *, float *, int)"


def build_csv(rows, base=False, missing=False):
    """rows: list of (kernel name, grid, block, values dict). Returns the CSV text of the raw page."""
    mcols = metric_columns(missing)
    stall_names = list(STALL_RATIOS)
    header = FIXED + [n for _, n in mcols] + [cfg.STALL_METRIC.format(s) for s in stall_names if not missing]
    units = [""] * len(FIXED)
    for key, name in mcols:
        units.append(unit_and_scale(key, base)[0])
    if not missing:
        units += ["inst"] * len(stall_names)
    buf = []
    for idx, (name, grid, block, v) in enumerate(rows):
        r = [str(idx), "1234", "llama-server", "host", name, "1", "7", f"({block[0]}, {block[1]}, {block[2]})", f"({grid[0]}, {grid[1]}, {grid[2]})", "0", "86"]
        for key, _ in mcols:
            u, div = unit_and_scale(key, base)
            r.append(fmt_cell(v[key] / div, base) if v.get(key) is not None else "")
        if not missing:
            for s in stall_names:
                r.append(f"{STALL_RATIOS[s]:.6f}" if not base else f"{STALL_RATIOS[s]:.2f}")
        buf.append(r)
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n", quoting=csv.QUOTE_ALL)
    w.writerow(header)
    w.writerow(units)
    for r in buf:
        w.writerow(r)
    return "==PROF== Disconnected from process 1234\n" + out.getvalue()



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "ncu-synthetic"))
    a = ap.parse_args()
    plan = json.load(open(os.path.join(HERE, "plan", RUN + ".json")))
    ex = plan["expected"]
    ctx = {c[0]: c for c in plan["context"]}
    ew = json.load(open(os.path.join(HERE, "plan", "p3-ncu-extra-long-ew.json")))

    def row_of_plan(e, i):
        return (e["demangled"], (e["gx"], e["gy"], e["gz"]), (e["bx"], e["by"], e["bz"]), kernel_values(e, i))

    aligned = [row_of_plan(e, i) for i, e in enumerate(ex)]
    # shifted: the window starts 3 launches late: stream launches 3..15 (plan kernels 3..12 and context entries 13, 14, 15)
    shifted = aligned[3:]
    for rel in (13, 14, 15):
        c = ctx[rel]
        fake = dict(ex[0], dur_ns_nsys_proxy=100_000, func=c[1], gx=c[2], gy=c[3], gz=c[4], bx=c[5], by=c[6], bytes_model="", ops_int8="")
        shifted.append((demangled(c[1], c), (c[2], c[3], c[4]), (c[5], c[6], 1), kernel_values(fake, 99)))
    # misaligned: 13 launches that are not those of the window (the element-wise window + context of the other run)
    mis = []
    for i, e in enumerate(ew["expected"]):
        mis.append(row_of_plan(e, 50 + i))
    mis = (mis * 2)[:13]
    files = {"aligned": build_csv(aligned), "base": build_csv(aligned, base=True), "shifted": build_csv(shifted), "misaligned": build_csv(mis),
             "missing": build_csv(aligned, missing=True), "empty": "==WARNING== No kernels were profiled.\n==PROF== Disconnected from process 1234\n"}
    for name, text in files.items():
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, RUN + ".csv"), "w") as f:
            f.write(text)
    exp = hand_expected()
    with open(os.path.join(a.out, "synthetic-expected.json"), "w") as f:
        json.dump(exp, f, indent=1)
        f.write("\n")
    print(f"wrote {len(files)} synthetic reports and synthetic-expected.json to {a.out}")


if __name__ == "__main__":
    main()
