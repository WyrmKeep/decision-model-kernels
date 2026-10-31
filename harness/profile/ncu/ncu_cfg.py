"""Constants shared by plan_ncu.py (schedule/plan), analyze_ncu.py (report analysis) and make_synthetic.py (tests).

Metric names were verified against the ncu 2026.3.1 binary (the ncu command line tool of the Nsight Compute 2026.3.1 installation) with
`ncu --section-folder <sections> --chips ga104 --query-metrics [--query-metrics-mode suffix --metrics <base>]`
(`plan_ncu.py --check-metrics` repeats the check against the section files), and against the 840-column raw page of the
six sample reports shipped with ncu (RTX A4500, cc 8.6, --set full).
"""
import os
import shutil

# The local ncu command line tool: $DMK_NCU (path of the `ncu` command), else `ncu` on PATH; "" when there is none. It turns a
# .ncu-rep / .ncu-repz into CSV (analyze_ncu.py) and holds the section files (plan_ncu.py --check-metrics) and the sample reports
# (the self-test). LOCAL_NCU_DIR is the installation folder (the one with `sections/`), found from the real path of the tool.
LOCAL_NCU = os.environ.get("DMK_NCU") or shutil.which("ncu") or ""


def _install_dir(tool):
    if not tool:
        return ""
    d = os.path.dirname(os.path.realpath(tool))
    for _ in range(3):
        if os.path.isdir(os.path.join(d, "sections")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.realpath(tool))


LOCAL_NCU_DIR = _install_dir(LOCAL_NCU)

# On the GPU host: $DMK_REMOTE_DIR/tools is mounted read-only at /tools in the server container. /tools/nsight-compute-2026.3.1/ncu is a
# /bin/sh wrapper that runs the real binary without `exec`; as the container entrypoint it would be PID 1 and would swallow the
# SIGTERM of `docker stop` (see NOTES.txt, risk 1), so the schedules call the real binary.
NCU_REMOTE = "/tools/nsight-compute-2026.3.1/target/linux-desktop-glibc_2_11_3-x64/ncu"
SECTIONS_REMOTE = "/tools/nsight-compute-2026.3.1/sections"

# core / smoke section set: SoL, launch, occupancy, pipes (tensor), memory tables, stall reasons, scheduler
SECTIONS_STD = ["SpeedOfLight", "LaunchStats", "Occupancy", "ComputeWorkloadAnalysis", "MemoryWorkloadAnalysis", "WarpStateStats",
                "SchedulerStats"]
# not in the sections above (checked with the section files): raw DRAM bytes, L2 bytes, tensor instruction / op counts
EXTRA_METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "smsp__inst_executed.sum",
    "sm__inst_executed_pipe_tensor_op_imma.sum",
    "sm__inst_executed_pipe_tensor_op_hmma.sum",
    "sm__ops_path_tensor_src_int8.sum",
]

# --- limits of the card (measured, task description): the denominators of the speed-of-light table
LIM_DRAM_GBPS = 422.0         # sustained DRAM read, GB/s (nominal 448)
LIM_INT8_TOPS = 149.3         # dense int8 mma.sync, TOP/s
LIM_FP16_TFLOPS = 38.0        # fp16 mma with fp32 accumulate, TFLOP/s
LIM_FP32_TFLOPS = 18.1        # CUDA-core FP32 FMA at 1,861 MHz (census_lib.FP32_TFLOPS)
LAUNCH_FLOOR_US = 1.0         # shortest kernel in the 13 nsys-traced passes: 0.992 us (unary_op_kernel; data/pass-kernels.tsv.gz)

# --- metrics the analysis reads (name -> short column); a missing one gives an empty cell, never an error
STALLS = ["barrier", "branch_resolving", "dispatch_stall", "drain", "imc_miss", "lg_throttle", "long_scoreboard", "math_pipe_throttle",
          "membar", "mio_throttle", "misc", "no_instruction", "short_scoreboard", "sleeping", "tex_throttle", "wait", "gmma"]
METRICS = {
    "duration_ns": "gpu__time_duration.sum",
    "sm_hz": "gpc__cycles_elapsed.avg.per_second",
    "sm_hz_alt": "sm__cycles_elapsed.avg.per_second",
    "dram_hz": "dram__cycles_elapsed.avg.per_second",
    "sm_cycles_active": "sm__cycles_active.avg",
    "dram_rd": "dram__bytes_read.sum",
    "dram_wr": "dram__bytes_write.sum",
    "dram_bps": "dram__bytes.sum.per_second",
    "dram_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_bytes": "lts__t_bytes.sum",
    "l2_hit": "lts__t_sector_hit_rate.pct",
    "l1_hit": "l1tex__t_sector_hit_rate.pct",
    "imma_pct": "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active",
    "hmma_pct": "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "imma_inst_pct": "sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active",
    "hmma_inst_pct": "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
    "imma_inst": "sm__inst_executed_pipe_tensor_op_imma.sum",
    "hmma_inst": "sm__inst_executed_pipe_tensor_op_hmma.sum",
    "int8_ops": "sm__ops_path_tensor_src_int8.sum",
    "inst": "smsp__inst_executed.sum",
    "issue_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm_thr_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "warps_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "warps_max_pct": "sm__maximum_warps_per_active_cycle_pct",
    "latency": "smsp__average_warp_latency_per_inst_issued.ratio",
    "regs": "launch__registers_per_thread",
    "smem_static": "launch__shared_mem_per_block_static",
    "smem_dyn": "launch__shared_mem_per_block_dynamic",
    "smem_driver": "launch__shared_mem_per_block_driver",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
    "waves": "launch__waves_per_multiprocessor",
    "occ_lim_regs": "launch__occupancy_limit_registers",
    "occ_lim_smem": "launch__occupancy_limit_shared_mem",
    "occ_lim_warps": "launch__occupancy_limit_warps",
    "occ_lim_blocks": "launch__occupancy_limit_blocks",
    "passes": "profiler__replayer_passes",
}
STALL_METRIC = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"
# not part of any section: collected through EXTRA_METRICS / always present
NOT_IN_SECTIONS_OK = {"profiler__replayer_passes", "sm__cycles_elapsed.avg.per_second"}
# estimate model (plan_ncu.py): seconds per profiled kernel = t_save + passes * t_pass; seconds per run = fixed overhead
COST = {
    "passes": {"std": 25, "full": 42},            # 42 measured on the A4500 sample reports (--set full, cc 8.6); 25 for the 7 sections [INFERENCE]
    "t_save": {"opt": 0.3, "pess": 1.5},          # s, one-off save of the device memory the kernel can access [INFERENCE]
    "t_pass": {"opt": 0.016, "pess": 0.10},       # s per replay pass: restore of the written subset, counters, read-out [INFERENCE]
    "run_fixed": {"opt": 22.0, "pess": 55.0},     # s per run: server start + ncu init + first-kernel config + report write + teardown
}
