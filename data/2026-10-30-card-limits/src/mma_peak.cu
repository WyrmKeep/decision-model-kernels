// mb-mma: tensor-core mma.sync throughput from registers, no memory traffic in the inner loop.
//
// Modes (instruction shapes are the ones llama.cpp b12004 issues on sm_86, see ggml/src/ggml-cuda/mma.cuh):
//   s8k32   mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32      MMQ Q4_0 / Q8_0 (mmq-vec-dot.cuh, tile<16,8,int> x tile<8,8,int>)
//   s8k16   mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32      MMQ K-quant paths (tile<16,4,int> x tile<8,4,int>)
//   f16f32  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32    flash attention KQ (fattn-mma-f16.cuh, T_C_KQ = float)
//   f16f16  mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16    flash attention VKQ (T_C_VKQ = half2)
//   bf16f32 mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  gated-delta-net-mma.cu
//
// Every warp runs ILP independent accumulator chains (independent mma instructions) so the tensor pipe is never
// latency-bound. The operand registers never change between iterations except for a sign flip on the float modes
// (keeps the f16 accumulators bounded). Operands are random bits (--data rand) so the datapath toggles like real data;
// --data zero uses all-zero operands to show how much of the power budget is data dependent.
// One launch runs ~100 ms; launches are queued back to back for --seconds and timed with cudaEvents.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "common.cuh"

namespace {

__host__ __device__ __forceinline__ uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
__device__ __forceinline__ float rand_pm1(uint32_t& s) {
    return (float) (xorshift32(s) >> 8) * (1.0f / 8388608.0f) - 1.0f;  // [-1, 1)
}

// ------------------------------------------------------------------------------------------ per-mode traits
struct TS8K32 {
    static constexpr int NA = 4, NB = 2, NC = 4;
    static constexpr uint32_t FLIP = 0;
    static __device__ __forceinline__ uint32_t gen(uint32_t& s) { return xorshift32(s); }
    static __device__ __forceinline__ void mma(uint32_t* c, const uint32_t* a, const uint32_t* b) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct TS8K16 {
    static constexpr int NA = 2, NB = 1, NC = 4;
    static constexpr uint32_t FLIP = 0;
    static __device__ __forceinline__ uint32_t gen(uint32_t& s) { return xorshift32(s); }
    static __device__ __forceinline__ void mma(uint32_t* c, const uint32_t* a, const uint32_t* b) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    }
};
struct TF16F32 {
    static constexpr int NA = 4, NB = 2, NC = 4;
    static constexpr uint32_t FLIP = 0x80008000u;
    static __device__ __forceinline__ uint32_t gen(uint32_t& s) {
        const __half2 h = __floats2half2_rn(rand_pm1(s), rand_pm1(s));
        return *reinterpret_cast<const uint32_t*>(&h);
    }
    static __device__ __forceinline__ void mma(uint32_t* c, const uint32_t* a, const uint32_t* b) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct TF16F16 {
    static constexpr int NA = 4, NB = 2, NC = 2;
    static constexpr uint32_t FLIP = 0x80008000u;
    static __device__ __forceinline__ uint32_t gen(uint32_t& s) { return TF16F32::gen(s); }
    static __device__ __forceinline__ void mma(uint32_t* c, const uint32_t* a, const uint32_t* b) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                     : "+r"(c[0]), "+r"(c[1])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct TBF16F32 {
    static constexpr int NA = 4, NB = 2, NC = 4;
    static constexpr uint32_t FLIP = 0x80008000u;
    static __device__ __forceinline__ uint32_t gen(uint32_t& s) {
        const __nv_bfloat162 h = __floats2bfloat162_rn(rand_pm1(s), rand_pm1(s));
        return *reinterpret_cast<const uint32_t*>(&h);
    }
    static __device__ __forceinline__ void mma(uint32_t* c, const uint32_t* a, const uint32_t* b) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};

constexpr int kThreads = 128;  // 4 warps: one per SM sub-partition

template <class M, int ILP>
__global__ void __launch_bounds__(kThreads)
    k_mma(long long iters, uint32_t seed, int zero, uint32_t sentinel, uint32_t* sink) {
    uint32_t a[ILP][M::NA], b[ILP][M::NB], c[ILP][M::NC];
    uint32_t s = (seed ^ (threadIdx.x * 0x9E3779B1u) ^ (blockIdx.x * 0x85EBCA6Bu)) | 1u;
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
#pragma unroll
        for (int k = 0; k < M::NA; ++k) a[j][k] = zero ? 0u : M::gen(s);
#pragma unroll
        for (int k = 0; k < M::NB; ++k) b[j][k] = zero ? 0u : M::gen(s);
#pragma unroll
        for (int k = 0; k < M::NC; ++k) c[j][k] = 0u;
    }
    for (long long i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < ILP; ++j) M::mma(c[j], a[j], b[j]);
        if (M::FLIP != 0) {
#pragma unroll
            for (int j = 0; j < ILP; ++j)
#pragma unroll
                for (int k = 0; k < M::NA; ++k) a[j][k] ^= M::FLIP;
        }
    }
    uint32_t x = 0;
#pragma unroll
    for (int j = 0; j < ILP; ++j)
#pragma unroll
        for (int k = 0; k < M::NC; ++k) x += c[j][k];
    if (x == sentinel) sink[0] = x;  // sentinel is a run-time value: keeps the loop alive, never true in practice
}

using KernelFn = void (*)(long long, uint32_t, int, uint32_t, uint32_t*);

enum Mode { MODE_S8K32 = 0, MODE_S8K16, MODE_F16F32, MODE_F16F16, MODE_BF16F32, NMODES };

struct ModeInfo {
    const char* name;
    double ops_per_mma;         // multiply and add counted as 2 ops: 2*M*N*K of the instruction shape
    double nominal_ops_clk_sm;  // GA10x GeForce dense rate, public-spec arithmetic (see README), not measured
};
const ModeInfo kModes[NMODES] = {
    {"s8k32", 2.0 * 16 * 8 * 32, 2048}, {"s8k16", 2.0 * 16 * 8 * 16, 2048},  {"f16f32", 2.0 * 16 * 8 * 16, 512},
    {"f16f16", 2.0 * 16 * 8 * 16, 1024}, {"bf16f32", 2.0 * 16 * 8 * 16, 512},
};

template <class M>
KernelFn pick_ilp(int ilp) {
    switch (ilp) {
        case 1: return k_mma<M, 1>;
        case 2: return k_mma<M, 2>;
        case 4: return k_mma<M, 4>;
        case 8: return k_mma<M, 8>;
        default: return nullptr;
    }
}
KernelFn get_kernel(int mode, int ilp) {
    switch (mode) {
        case MODE_S8K32: return pick_ilp<TS8K32>(ilp);
        case MODE_S8K16: return pick_ilp<TS8K16>(ilp);
        case MODE_F16F32: return pick_ilp<TF16F32>(ilp);
        case MODE_F16F16: return pick_ilp<TF16F16>(ilp);
        case MODE_BF16F32: return pick_ilp<TBF16F32>(ilp);
        default: return nullptr;
    }
}

// ------------------------------------------------------------------------------------------ host
struct Config {
    std::string mode = "all";
    double seconds = 2.0;
    double warmup = 1.0;
    int ilp = 4;
    int bps = 4;
    std::string data = "rand";
    int device = 0;
    int sample_ms = 20;
    bool header = true;
    std::string series;
    double launch_ms = 100.0;
    std::vector<int> sweep_ilp = {2, 4, 8};
    std::vector<int> sweep_bps = {1, 2, 4, 8};
    double sweep_seconds = 1.0;
};

void usage() {
    std::fputs(
        "usage: mb-mma [options]\n"
        "  --mode M           all | sweep | comma list of s8k32,s8k16,f16f32,f16f16,bf16f32   (default all)\n"
        "                       all   = every instruction shape once at --ilp/--blocks-per-sm\n"
        "                       sweep = s8k32 over --sweep-ilp x --sweep-bps (how many warps saturate the SM)\n"
        "  --seconds S        measured time per configuration, >= 1 recommended   (default 2)\n"
        "  --warmup S         warm-up before each configuration, clocks ramp up    (default 1)\n"
        "  --ilp N            independent mma chains per warp: 1, 2, 4 or 8       (default 4)\n"
        "  --blocks-per-sm B  128-thread blocks per SM (4 warps each), clamped to occupancy (default 4 = 16 warps/SM)\n"
        "  --data rand|zero   operand bits (default rand)\n"
        "  --launch-ms MS     target duration of one kernel launch                 (default 100)\n"
        "  --sweep-ilp L --sweep-bps L --sweep-seconds S   sweep grid (comma lists; defaults 2,4,8 / 1,2,4,8 / 1)\n"
        "  --device N         CUDA device (default 0)\n"
        "  --sample-ms N      NVML sampling interval (default 20)\n"
        "  --series FILE      write the raw NVML time series (TSV)\n"
        "  --no-header        do not print the TSV header line\n"
        "  --print-header     print the TSV header line and exit\n"
        "TSV rows go to stdout, human-readable progress to stderr.\n",
        stderr);
}

struct Row {
    std::string mode, data;
    int ilp = 0, bps = 0, warps = 0, launches = 0;
    double seconds = 0, mma_total = 0, tops = 0, tops_last_half = 0, tops_min = 0, tops_max = 0;
    double ops_clk_sm = NAN, nominal = 0, eff = NAN;
    mb::HwStats hw;
};

void print_header() {
    std::printf(
        "bench\tmode\tdata\tilp\tblocks_per_sm\twarps_per_sm\tlaunches\tseconds\tmma_total\ttops\ttops_last_half\t"
        "tops_launch_min\ttops_launch_max\tops_per_clk_sm\tnominal_ops_per_clk_sm\teff_vs_nominal_pct\t%s\n",
        mb::hw_header().c_str());
}
void print_row(const Row& r) {
    std::printf("mma\t%s\t%s\t%d\t%d\t%d\t%d\t%.3f\t%.4g\t%.2f\t%.2f\t%.2f\t%.2f\t%s\t%.0f\t%s\t%s\n", r.mode.c_str(),
                r.data.c_str(), r.ilp, r.bps, r.warps, r.launches, r.seconds, r.mma_total, r.tops, r.tops_last_half,
                r.tops_min, r.tops_max, mb::num(r.ops_clk_sm, "%.0f").c_str(), r.nominal, mb::num(r.eff).c_str(),
                mb::hw_row(r.hw).c_str());
    std::fflush(stdout);
}

struct Runner {
    mb::Session& ses;
    const Config& cfg;
    cudaStream_t st;
    uint32_t* sink;

    bool run(int mode, int ilp, int bps_req, bool zero, double seconds, Row& out) {
        KernelFn k = get_kernel(mode, ilp);
        if (!k) {
            std::fprintf(stderr, "unsupported --ilp %d (use 1, 2, 4 or 8)\n", ilp);
            std::exit(2);
        }
        int occ = 0;
        MB_CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*) k, kThreads, 0));
        const int bps = std::min(bps_req, occ);
        if (bps < 1) {
            std::fprintf(stderr, "kernel %s ilp %d cannot be launched (occupancy 0)\n", kModes[mode].name, ilp);
            return false;
        }
        if (bps != bps_req)
            std::fprintf(stderr, "# note: %s ilp=%d: blocks/SM %d clamped to occupancy limit %d\n", kModes[mode].name, ilp, bps_req, bps);
        const int grid = ses.dev.sm_count * bps;
        const uint32_t seed = 0x1234567u;
        const uint32_t sentinel = 0x9E3779B1u;
        const int z = zero ? 1 : 0;

        auto launch = [&](long long iters) {
            long long it = iters;
            uint32_t sd = seed, sn = sentinel;
            int zz = z;
            uint32_t* sk = sink;
            void* args[] = {&it, &sd, &zz, &sn, &sk};
            MB_CK(cudaLaunchKernel((const void*) k, dim3(grid), dim3(kThreads), args, 0, st));
        };
        auto time_launch = [&](long long iters) {
            cudaEvent_t a, b;
            MB_CK(cudaEventCreate(&a));
            MB_CK(cudaEventCreate(&b));
            MB_CK(cudaEventRecord(a, st));
            launch(iters);
            MB_CK(cudaEventRecord(b, st));
            MB_CK(cudaEventSynchronize(b));
            float ms = 0;
            MB_CK(cudaEventElapsedTime(&ms, a, b));
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return (double) ms;
        };
        auto scale_iters = [&](long long iters, double ms) {
            const double want = cfg.launch_ms / std::max(ms, 1e-3) * (double) iters;
            return std::max(1LL, std::min((long long) want, 4000000000LL));  // a bogus first timing must not make a launch run for hours
        };

        ses.hw.set_phase(fmt_phase(mode, ilp, bps, zero, "warmup"));
        launch(1 << 10);  // module load, first clock boost
        MB_CK(cudaStreamSynchronize(st));
        long long iters = scale_iters(1 << 16, time_launch(1 << 16));
        mb::LaunchTimes w = mb::run_timed([&] { launch(iters); }, cfg.warmup, st);
        iters = scale_iters(iters, mb::median(std::vector<double>(w.ms.end() - std::min<size_t>(w.ms.size(), 4), w.ms.end())));

        ses.hw.set_phase(fmt_phase(mode, ilp, bps, zero, "measure"));
        mb::LaunchTimes t = mb::run_timed([&] { launch(iters); }, seconds, st);
        ses.hw.set_phase("idle");

        const double mma_per_launch = (double) grid * (kThreads / 32) * ilp * (double) iters;
        const double ops_per_launch = mma_per_launch * kModes[mode].ops_per_mma;
        Row r;
        r.mode = kModes[mode].name;
        r.data = zero ? "zero" : "rand";
        r.ilp = ilp;
        r.bps = bps;
        r.warps = bps * (kThreads / 32);
        r.launches = (int) t.ms.size();
        r.seconds = t.total_ms / 1e3;
        r.mma_total = mma_per_launch * t.ms.size();
        r.tops = ops_per_launch * t.ms.size() / (t.total_ms * 1e-3) / 1e12;
        const size_t half = t.ms.size() / 2;
        double ms_h = 0;
        for (size_t i = half; i < t.ms.size(); ++i) ms_h += t.ms[i];
        r.tops_last_half = ops_per_launch * (t.ms.size() - half) / (ms_h * 1e-3) / 1e12;
        r.tops_min = 1e30;
        r.tops_max = 0;
        for (double ms : t.ms) {
            const double x = ops_per_launch / (ms * 1e-3) / 1e12;
            r.tops_min = std::min(r.tops_min, x);
            r.tops_max = std::max(r.tops_max, x);
        }
        r.nominal = kModes[mode].nominal_ops_clk_sm;
        r.hw = ses.hw.stats(t.t_begin, t.t_end);
        if (std::isfinite(r.hw.sm_mean) && r.hw.sm_mean > 0) {
            r.ops_clk_sm = r.tops * 1e12 / (ses.dev.sm_count * r.hw.sm_mean * 1e6);
            r.eff = 100.0 * r.ops_clk_sm / r.nominal;
        }
        std::fprintf(stderr,
                     "%-8s %-4s ilp=%d warps/SM=%-2d  %7.2f TOPS (last half %7.2f, launch min/max %.2f/%.2f)  SM %.0f MHz  "
                     "%.1f W  %.0f ops/clk/SM (%.1f%% of nominal %.0f)  swcap %.0f%%\n",
                     r.mode.c_str(), r.data.c_str(), ilp, r.warps, r.tops, r.tops_last_half, r.tops_min, r.tops_max,
                     r.hw.sm_mean, r.hw.pw_mean, r.ops_clk_sm, r.eff, r.nominal, r.hw.swcap_pct);
        out = r;
        return true;
    }

    static std::string fmt_phase(int mode, int ilp, int bps, bool zero, const char* what) {
        return mb::fmt("%s ilp%d bps%d %s %s", kModes[mode].name, ilp, bps, zero ? "zero" : "rand", what);
    }
};

int mode_from_name(const std::string& n) {
    for (int i = 0; i < NMODES; ++i)
        if (n == kModes[i].name) return i;
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string v;
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (mb::opt_value(argc, argv, i, "--mode", v)) cfg.mode = v;
        else if (mb::opt_value(argc, argv, i, "--seconds", v)) cfg.seconds = mb::parse_double(v, "--seconds");
        else if (mb::opt_value(argc, argv, i, "--warmup", v)) cfg.warmup = mb::parse_double(v, "--warmup");
        else if (mb::opt_value(argc, argv, i, "--ilp", v)) cfg.ilp = (int) mb::parse_ll(v, "--ilp");
        else if (mb::opt_value(argc, argv, i, "--blocks-per-sm", v)) cfg.bps = (int) mb::parse_ll(v, "--blocks-per-sm");
        else if (mb::opt_value(argc, argv, i, "--data", v)) cfg.data = v;
        else if (mb::opt_value(argc, argv, i, "--launch-ms", v)) cfg.launch_ms = mb::parse_double(v, "--launch-ms");
        else if (mb::opt_value(argc, argv, i, "--sweep-ilp", v)) cfg.sweep_ilp = mb::parse_int_list(v, "--sweep-ilp");
        else if (mb::opt_value(argc, argv, i, "--sweep-bps", v)) cfg.sweep_bps = mb::parse_int_list(v, "--sweep-bps");
        else if (mb::opt_value(argc, argv, i, "--sweep-seconds", v)) cfg.sweep_seconds = mb::parse_double(v, "--sweep-seconds");
        else if (mb::opt_value(argc, argv, i, "--device", v)) cfg.device = (int) mb::parse_ll(v, "--device");
        else if (mb::opt_value(argc, argv, i, "--sample-ms", v)) cfg.sample_ms = (int) mb::parse_ll(v, "--sample-ms");
        else if (mb::opt_value(argc, argv, i, "--series", v)) cfg.series = v;
        else if (a == "--no-header") cfg.header = false;
        else if (a == "--print-header") { print_header(); return 0; }
        else {
            std::fprintf(stderr, "unknown option '%s'\n\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (cfg.data != "rand" && cfg.data != "zero") { std::fprintf(stderr, "--data must be rand or zero\n"); return 2; }
    if (cfg.seconds <= 0 || cfg.warmup < 0 || cfg.bps < 1 || cfg.launch_ms <= 0 || cfg.sample_ms < 1) {
        std::fprintf(stderr, "bad numeric option\n");
        return 2;
    }
    std::vector<int> modes;
    bool sweep = false;
    if (cfg.mode == "all") {
        for (int i = 0; i < NMODES; ++i) modes.push_back(i);
    } else if (cfg.mode == "sweep") {
        sweep = true;
        modes.push_back(MODE_S8K32);
    } else {
        for (const auto& n : mb::split(cfg.mode, ',')) {
            const int m = mode_from_name(n);
            if (m < 0) { std::fprintf(stderr, "unknown mode '%s'\n", n.c_str()); return 2; }
            modes.push_back(m);
        }
    }

    mb::Session ses;
    ses.series_path = cfg.series;
    ses.begin("mb-mma", cfg.device, cfg.sample_ms);

    cudaStream_t st;
    MB_CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    uint32_t* sink = nullptr;
    MB_CK(cudaMalloc(&sink, 256));
    MB_CK(cudaMemset(sink, 0, 256));

    if (cfg.header) print_header();
    Runner R{ses, cfg, st, sink};
    Row row;
    int rc = 0;
    if (sweep) {
        std::vector<std::pair<int, int>> seen;
        for (int ilp : cfg.sweep_ilp) {
            KernelFn k = get_kernel(MODE_S8K32, ilp);
            if (!k) { std::fprintf(stderr, "unsupported ilp %d in --sweep-ilp\n", ilp); return 2; }
            int occ = 0;
            MB_CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*) k, kThreads, 0));
            for (int bps : cfg.sweep_bps) {
                const int eff = std::min(bps, occ);
                if (std::find(seen.begin(), seen.end(), std::make_pair(ilp, eff)) != seen.end()) continue;
                seen.push_back({ilp, eff});
                if (R.run(MODE_S8K32, ilp, eff, cfg.data == "zero", cfg.sweep_seconds, row)) print_row(row);
                else rc = 1;
            }
        }
    } else {
        for (int m : modes) {
            if (R.run(m, cfg.ilp, cfg.bps, cfg.data == "zero", cfg.seconds, row)) print_row(row);
            else rc = 1;
        }
    }
    ses.end("mma");
    cudaFree(sink);
    cudaStreamDestroy(st);
    return rc;
}
