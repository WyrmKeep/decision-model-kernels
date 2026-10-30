// mb-membw: DRAM and L2 bandwidth with vectorized 16-byte accesses.
//
//   dram read   grid-stride ld.global.v4.u32 over a 1 GiB buffer, 32-bit sum reduce (kept alive through a run-time sentinel)
//               variants: plain (no cache operator), cg (L2 only), cs (streaming, evict-first), nc (read-only path)
//   dram write  grid-stride st.global.v4.u32 over a 1 GiB buffer (variants wb = default, cs = streaming)
//   dram copy   ld.v4 + st.v4 between two buffers (variants: ldcs_stcs, ldplain_stwb) and cudaMemcpyAsync device-to-device;
//               GB/s counts read bytes + written bytes
//   l2 read     the same read kernel over a buffer that fits in L2 (default 0.25 .. 3 MiB), many passes per launch; a few
//               larger sizes (6 MiB .. 128 MiB) show where the data stops fitting in the 4 MiB L2
//   sweep       dram read (cs) over unroll {1,2,4,8} x blocks/SM {1,2,3,4,6}
//
// GB/s is decimal (1e9 bytes/s). The theoretical DRAM peak is derived from the NVML memory clock sampled during the run:
// clock x 2 (double data rate) x bus width / 8.
// One launch runs ~40 ms (many passes); launches are queued back to back for --seconds and timed with cudaEvents.
#include "common.cuh"

namespace {

enum { LD_PLAIN = 0, LD_CG = 1, LD_CS = 2, LD_NC = 3 };
enum { ST_WB = 0, ST_CG = 1, ST_CS = 2 };

template <int L>
__device__ __forceinline__ uint4 ldv(const uint4* p) {
    uint4 v;
    if (L == LD_PLAIN)
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else if (L == LD_CG)
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else if (L == LD_CS)
        asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else
        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
template <int S>
__device__ __forceinline__ void stv(uint4* p, const uint4& v) {
    if (S == ST_WB)
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    else if (S == ST_CG)
        asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    else
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

constexpr int kThreads = 256;

template <int L, int U>
__global__ void __launch_bounds__(kThreads)
    k_read(const uint4* __restrict__ p, size_t n, int passes, uint32_t sentinel, uint32_t* sink) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (int ps = 0; ps < passes; ++ps) {
        size_t i = tid;
        for (; i + (U - 1) * stride < n; i += U * stride) {
            uint4 v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) v[u] = ldv<L>(p + i + u * stride);
#pragma unroll
            for (int u = 0; u < U; ++u) acc += v[u].x + v[u].y + v[u].z + v[u].w;
        }
        for (; i < n; i += stride) {
            const uint4 v = ldv<L>(p + i);
            acc += v.x + v.y + v.z + v.w;
        }
    }
    if (acc == sentinel) sink[0] = acc;
}

template <int S, int U>
__global__ void __launch_bounds__(kThreads) k_write(uint4* __restrict__ p, size_t n, int passes, uint32_t val) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (int ps = 0; ps < passes; ++ps) {
        const uint4 v = make_uint4(val + ps, val ^ (uint32_t) tid, val + 2u, val + 3u);
        size_t i = tid;
        for (; i + (U - 1) * stride < n; i += U * stride) {
#pragma unroll
            for (int u = 0; u < U; ++u) stv<S>(p + i + u * stride, v);
        }
        for (; i < n; i += stride) stv<S>(p + i, v);
    }
}

template <int L, int S, int U>
__global__ void __launch_bounds__(kThreads) k_copy(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n, int passes) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (int ps = 0; ps < passes; ++ps) {
        size_t i = tid;
        for (; i + (U - 1) * stride < n; i += U * stride) {
            uint4 v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) v[u] = ldv<L>(src + i + u * stride);
#pragma unroll
            for (int u = 0; u < U; ++u) stv<S>(dst + i + u * stride, v[u]);
        }
        for (; i < n; i += stride) stv<S>(dst + i, ldv<L>(src + i));
    }
}

using ReadFn = void (*)(const uint4*, size_t, int, uint32_t, uint32_t*);
using WriteFn = void (*)(uint4*, size_t, int, uint32_t);
using CopyFn = void (*)(const uint4*, uint4*, size_t, int);

template <int L>
ReadFn read_unroll(int u) {
    switch (u) {
        case 1: return k_read<L, 1>;
        case 2: return k_read<L, 2>;
        case 4: return k_read<L, 4>;
        case 8: return k_read<L, 8>;
        default: return nullptr;
    }
}
ReadFn get_read(int ld, int u) {
    switch (ld) {
        case LD_PLAIN: return read_unroll<LD_PLAIN>(u);
        case LD_CG: return read_unroll<LD_CG>(u);
        case LD_CS: return read_unroll<LD_CS>(u);
        case LD_NC: return read_unroll<LD_NC>(u);
        default: return nullptr;
    }
}
template <int S>
WriteFn write_unroll(int u) {
    switch (u) {
        case 1: return k_write<S, 1>;
        case 2: return k_write<S, 2>;
        case 4: return k_write<S, 4>;
        case 8: return k_write<S, 8>;
        default: return nullptr;
    }
}
WriteFn get_write(int st, int u) {
    switch (st) {
        case ST_WB: return write_unroll<ST_WB>(u);
        case ST_CS: return write_unroll<ST_CS>(u);
        default: return nullptr;
    }
}
template <int L, int S>
CopyFn copy_unroll(int u) {
    switch (u) {
        case 1: return k_copy<L, S, 1>;
        case 2: return k_copy<L, S, 2>;
        case 4: return k_copy<L, S, 4>;
        case 8: return k_copy<L, S, 8>;
        default: return nullptr;
    }
}
CopyFn get_copy(int ld, int st, int u) {
    if (ld == LD_CS && st == ST_CS) return copy_unroll<LD_CS, ST_CS>(u);
    if (ld == LD_PLAIN && st == ST_WB) return copy_unroll<LD_PLAIN, ST_WB>(u);
    return nullptr;
}

const char* kLdName[] = {"plain", "cg", "cs", "nc"};
const char* kStName[] = {"wb", "cg", "cs"};

// ------------------------------------------------------------------------------------------ host
struct Config {
    std::string mode = "all";
    double seconds = 1.5;
    double l2_seconds = 1.0;
    double warmup = 0.5;
    double buf_mib = 1024;       // read / write buffer
    double copy_mib = 768;       // each of the two copy buffers
    int unroll = 4;
    int bps = 0;                 // 0 = occupancy maximum
    std::vector<double> l2_mib = {0.25, 0.5, 1, 2, 3, 3.5, 6, 12, 32, 128};
    int device = 0;
    int sample_ms = 20;
    bool header = true;
    std::string series;
    double launch_ms = 40.0;
};

void usage() {
    std::fputs(
        "usage: mb-membw [options]\n"
        "  --mode M           all | dram | l2 | sweep                               (default all = dram + l2)\n"
        "  --seconds S        measured time per DRAM test                           (default 1.5)\n"
        "  --l2-seconds S     measured time per L2 size                             (default 1.0)\n"
        "  --warmup S         warm-up per test                                      (default 0.5)\n"
        "  --buf-mib X        DRAM read/write buffer in MiB                         (default 1024)\n"
        "  --copy-mib X       each of the two copy buffers in MiB                   (default 768)\n"
        "  --l2-mib LIST      L2 test buffer sizes in MiB, comma list               (default 0.25,0.5,1,2,3,3.5,6,12,32,128)\n"
        "  --unroll N         loads in flight per thread: 1, 2, 4 or 8              (default 4)\n"
        "  --blocks-per-sm B  256-thread blocks per SM, 0 = occupancy maximum       (default 0)\n"
        "  --launch-ms MS     target duration of one kernel launch                  (default 40)\n"
        "  --device N         CUDA device (default 0)\n"
        "  --sample-ms N      NVML sampling interval (default 20)\n"
        "  --series FILE      write the raw NVML time series (TSV)\n"
        "  --no-header        do not print the TSV header line\n"
        "  --print-header     print the TSV header line and exit\n"
        "TSV rows go to stdout, human-readable progress to stderr.\n"
        "VRAM: the largest allocation is max(--buf-mib, 2 x --copy-mib) (1.5 GiB by default) plus the CUDA context.\n",
        stderr);
}

struct Row {
    std::string test, variant;
    double buf_mib = 0;
    int bps = 0, unroll = 0, passes = 0, launches = 0;
    double seconds = 0, gbps = 0, gbps_med = 0, gbps_max = 0, nominal = NAN, eff = NAN;
    double bytes_per_pass = 0;
    mb::HwStats hw;
};

void print_header() {
    std::printf(
        "bench\ttest\tvariant\tbuf_mib\tblocks_per_sm\tunroll\tpasses_per_launch\tlaunches\tseconds\tgbps\tgbps_median_launch\t"
        "gbps_max_launch\tnominal_gbps\teff_vs_nominal_pct\tbytes_per_pass\t%s\n",
        mb::hw_header().c_str());
}
void print_row(const Row& r) {
    std::printf("membw\t%s\t%s\t%.3f\t%d\t%d\t%d\t%d\t%.3f\t%.1f\t%.1f\t%.1f\t%s\t%s\t%.0f\t%s\n", r.test.c_str(), r.variant.c_str(),
                r.buf_mib, r.bps, r.unroll, r.passes, r.launches, r.seconds, r.gbps, r.gbps_med, r.gbps_max,
                mb::num(r.nominal).c_str(), mb::num(r.eff).c_str(), r.bytes_per_pass, mb::hw_row(r.hw).c_str());
    std::fflush(stdout);
}

struct Runner {
    mb::Session& ses;
    const Config& cfg;
    cudaStream_t st;
    uint32_t* sink;

    int grid_for(const void* fn, int bps_req, int* bps_out) {
        int occ = 0;
        MB_CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, kThreads, 0));
        int b = bps_req <= 0 ? occ : std::min(bps_req, occ);
        if (b < 1) b = 1;
        *bps_out = b;
        return ses.dev.sm_count * b;
    }

    // launch(passes) enqueues one batch of `passes` passes; bytes_pass = bytes moved per pass (read + written).
    template <class F>
    Row measure(const std::string& test, const std::string& variant, double buf_bytes, double bytes_pass, int bps, int unroll,
                double seconds, bool dram, F&& launch) {
        auto time_launch = [&](int passes) {
            cudaEvent_t a, b;
            MB_CK(cudaEventCreate(&a));
            MB_CK(cudaEventCreate(&b));
            MB_CK(cudaEventRecord(a, st));
            launch(passes);
            MB_CK(cudaEventRecord(b, st));
            MB_CK(cudaEventSynchronize(b));
            float ms = 0;
            MB_CK(cudaEventElapsedTime(&ms, a, b));
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return (double) ms;
        };
        auto scale = [&](int passes, double ms) {
            return (int) std::max(1.0, std::min(2.0e6, std::floor(cfg.launch_ms / std::max(ms, 1e-4) * passes)));
        };
        const std::string tag = test + " " + variant + mb::fmt(" %.3fMiB", buf_bytes / 1048576.0);
        ses.hw.set_phase(tag + " warmup");
        launch(1);
        MB_CK(cudaStreamSynchronize(st));
        int passes = scale(1, time_launch(1));
        passes = scale(passes, time_launch(passes));
        mb::LaunchTimes w = mb::run_timed([&] { launch(passes); }, cfg.warmup, st);
        passes = scale(passes, mb::median(std::vector<double>(w.ms.end() - std::min<size_t>(w.ms.size(), 4), w.ms.end())));

        ses.hw.set_phase(tag + " measure");
        mb::LaunchTimes t = mb::run_timed([&] { launch(passes); }, seconds, st);
        ses.hw.set_phase("idle");

        Row r;
        r.test = test;
        r.variant = variant;
        r.buf_mib = buf_bytes / 1048576.0;
        r.bps = bps;
        r.unroll = unroll;
        r.passes = passes;
        r.launches = (int) t.ms.size();
        r.seconds = t.total_ms / 1e3;
        r.bytes_per_pass = bytes_pass;
        const double bytes_launch = bytes_pass * passes;
        r.gbps = bytes_launch * t.ms.size() / (t.total_ms * 1e-3) / 1e9;
        std::vector<double> per;
        for (double ms : t.ms) per.push_back(bytes_launch / (ms * 1e-3) / 1e9);
        r.gbps_med = mb::median(per);
        r.gbps_max = *std::max_element(per.begin(), per.end());
        r.hw = ses.hw.stats(t.t_begin, t.t_end);
        if (dram && std::isfinite(r.hw.mem_mean)) {
            r.nominal = r.hw.mem_mean * 2.0 * ses.dev.mem_bus_bits / 8.0 / 1000.0;
            r.eff = 100.0 * r.gbps / r.nominal;
        }
        std::fprintf(stderr, "%-10s %-9s %9.3f MiB  bps=%d u=%d  %8.1f GB/s (median launch %.1f, max %.1f)  nominal %s  eff %s%%  mem %.0f MHz  SM %.0f MHz  %.1f W\n",
                     test.c_str(), variant.c_str(), r.buf_mib, bps, unroll, r.gbps, r.gbps_med, r.gbps_max,
                     mb::num(r.nominal).c_str(), mb::num(r.eff).c_str(), r.hw.mem_mean, r.hw.sm_mean, r.hw.pw_mean);
        return r;
    }

    Row run_read(int ld, double mib, double seconds, bool dram, int unroll, int bps_req, void* buf) {
        ReadFn k = get_read(ld, unroll);
        if (!k) { std::fprintf(stderr, "unsupported --unroll %d (use 1, 2, 4 or 8)\n", unroll); std::exit(2); }
        int bps = 0;
        const int grid = grid_for((const void*) k, bps_req, &bps);
        const size_t bytes = (size_t) (mib * 1048576.0);
        const size_t n = bytes / 16;
        auto launch = [&](int passes) {
            const uint4* p = (const uint4*) buf;
            size_t nn = n;
            int ps = passes;
            uint32_t sn = 0x9E3779B1u;
            uint32_t* sk = sink;
            void* args[] = {&p, &nn, &ps, &sn, &sk};
            MB_CK(cudaLaunchKernel((const void*) k, dim3(grid), dim3(kThreads), args, 0, st));
        };
        return measure(dram ? "dram_read" : "l2_read", kLdName[ld], (double) bytes, (double) n * 16, bps, unroll, seconds, dram, launch);
    }
    Row run_write(int sv, double mib, double seconds, int unroll, int bps_req, void* buf) {
        WriteFn k = get_write(sv, unroll);
        if (!k) { std::fprintf(stderr, "unsupported --unroll %d (use 1, 2, 4 or 8)\n", unroll); std::exit(2); }
        int bps = 0;
        const int grid = grid_for((const void*) k, bps_req, &bps);
        const size_t bytes = (size_t) (mib * 1048576.0);
        const size_t n = bytes / 16;
        auto launch = [&](int passes) {
            uint4* p = (uint4*) buf;
            size_t nn = n;
            int ps = passes;
            uint32_t val = 0xA5A5A5A5u;
            void* args[] = {&p, &nn, &ps, &val};
            MB_CK(cudaLaunchKernel((const void*) k, dim3(grid), dim3(kThreads), args, 0, st));
        };
        return measure("dram_write", kStName[sv], (double) bytes, (double) n * 16, bps, unroll, seconds, true, launch);
    }
    Row run_copy(int ld, int sv, double mib, double seconds, int unroll, int bps_req, void* src, void* dst) {
        CopyFn k = get_copy(ld, sv, unroll);
        if (!k) { std::fprintf(stderr, "unsupported copy variant / --unroll\n"); std::exit(2); }
        int bps = 0;
        const int grid = grid_for((const void*) k, bps_req, &bps);
        const size_t bytes = (size_t) (mib * 1048576.0);
        const size_t n = bytes / 16;
        auto launch = [&](int passes) {
            const uint4* s = (const uint4*) src;
            uint4* d = (uint4*) dst;
            size_t nn = n;
            int ps = passes;
            void* args[] = {&s, &d, &nn, &ps};
            MB_CK(cudaLaunchKernel((const void*) k, dim3(grid), dim3(kThreads), args, 0, st));
        };
        return measure("dram_copy", std::string("ld") + kLdName[ld] + "_st" + kStName[sv], (double) bytes, 2.0 * n * 16, bps, unroll, seconds, true, launch);
    }
    Row run_memcpy(double mib, double seconds, void* src, void* dst) {
        const size_t bytes = (size_t) (mib * 1048576.0);
        auto launch = [&](int passes) {
            for (int i = 0; i < passes; ++i) MB_CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, st));
        };
        return measure("dram_copy", "memcpy_d2d", (double) bytes, 2.0 * bytes, 0, 0, seconds, true, launch);
    }
};

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string v;
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (mb::opt_value(argc, argv, i, "--mode", v)) cfg.mode = v;
        else if (mb::opt_value(argc, argv, i, "--seconds", v)) cfg.seconds = mb::parse_double(v, "--seconds");
        else if (mb::opt_value(argc, argv, i, "--l2-seconds", v)) cfg.l2_seconds = mb::parse_double(v, "--l2-seconds");
        else if (mb::opt_value(argc, argv, i, "--warmup", v)) cfg.warmup = mb::parse_double(v, "--warmup");
        else if (mb::opt_value(argc, argv, i, "--buf-mib", v)) cfg.buf_mib = mb::parse_double(v, "--buf-mib");
        else if (mb::opt_value(argc, argv, i, "--copy-mib", v)) cfg.copy_mib = mb::parse_double(v, "--copy-mib");
        else if (mb::opt_value(argc, argv, i, "--l2-mib", v)) cfg.l2_mib = mb::parse_double_list(v, "--l2-mib");
        else if (mb::opt_value(argc, argv, i, "--unroll", v)) cfg.unroll = (int) mb::parse_ll(v, "--unroll");
        else if (mb::opt_value(argc, argv, i, "--blocks-per-sm", v)) cfg.bps = (int) mb::parse_ll(v, "--blocks-per-sm");
        else if (mb::opt_value(argc, argv, i, "--launch-ms", v)) cfg.launch_ms = mb::parse_double(v, "--launch-ms");
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
    if (cfg.mode != "all" && cfg.mode != "dram" && cfg.mode != "l2" && cfg.mode != "sweep") {
        std::fprintf(stderr, "--mode must be all, dram, l2 or sweep\n");
        return 2;
    }
    if (cfg.seconds <= 0 || cfg.l2_seconds <= 0 || cfg.warmup < 0 || cfg.buf_mib < 1 || cfg.copy_mib < 1 || cfg.launch_ms <= 0 ||
        cfg.sample_ms < 1 || cfg.bps < 0) {
        std::fprintf(stderr, "bad numeric option\n");
        return 2;
    }
    for (double m : cfg.l2_mib)
        if (m <= 0) { std::fprintf(stderr, "bad --l2-mib entry\n"); return 2; }

    mb::Session ses;
    ses.series_path = cfg.series;
    ses.begin("mb-membw", cfg.device, cfg.sample_ms);

    const bool do_dram = cfg.mode == "all" || cfg.mode == "dram" || cfg.mode == "sweep";
    const bool do_l2 = cfg.mode == "all" || cfg.mode == "l2";
    double need_mib = 0;
    if (do_dram) need_mib = std::max(cfg.buf_mib, cfg.mode == "sweep" ? 0.0 : 2.0 * cfg.copy_mib);
    if (do_l2) need_mib = std::max(need_mib, *std::max_element(cfg.l2_mib.begin(), cfg.l2_mib.end()));
    {
        size_t fr = 0, tot = 0;
        MB_CK(cudaMemGetInfo(&fr, &tot));
        if (need_mib + 64 > fr / 1048576.0) {
            std::fprintf(stderr, "not enough free VRAM: need %.0f MiB + 64 MiB margin, free %.0f MiB\n", need_mib, fr / 1048576.0);
            return 4;
        }
    }

    cudaStream_t st;
    MB_CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    uint32_t* sink = nullptr;
    MB_CK(cudaMalloc(&sink, 256));
    MB_CK(cudaMemset(sink, 0, 256));

    if (cfg.header) print_header();
    Runner R{ses, cfg, st, sink};

    if (do_dram) {
        void* buf = nullptr;
        MB_CK(cudaMalloc(&buf, (size_t) (cfg.buf_mib * 1048576.0)));
        MB_CK(cudaMemset(buf, 0x5A, (size_t) (cfg.buf_mib * 1048576.0)));
        if (cfg.mode == "sweep") {
            for (int bps : {1, 2, 3, 4, 6})
                for (int u : {1, 2, 4, 8}) print_row(R.run_read(LD_CS, cfg.buf_mib, 0.8, true, u, bps, buf));
        } else {
            for (int ld : {LD_CS, LD_CG, LD_NC, LD_PLAIN}) print_row(R.run_read(ld, cfg.buf_mib, cfg.seconds, true, cfg.unroll, cfg.bps, buf));
            for (int sv : {ST_CS, ST_WB}) print_row(R.run_write(sv, cfg.buf_mib, cfg.seconds, cfg.unroll, cfg.bps, buf));
        }
        MB_CK(cudaFree(buf));
        if (cfg.mode != "sweep") {
            void *src = nullptr, *dst = nullptr;
            const size_t cb = (size_t) (cfg.copy_mib * 1048576.0);
            MB_CK(cudaMalloc(&src, cb));
            MB_CK(cudaMalloc(&dst, cb));
            MB_CK(cudaMemset(src, 0x5A, cb));
            MB_CK(cudaMemset(dst, 0, cb));
            print_row(R.run_copy(LD_CS, ST_CS, cfg.copy_mib, cfg.seconds, cfg.unroll, cfg.bps, src, dst));
            print_row(R.run_copy(LD_PLAIN, ST_WB, cfg.copy_mib, cfg.seconds, cfg.unroll, cfg.bps, src, dst));
            print_row(R.run_memcpy(cfg.copy_mib, cfg.seconds, src, dst));
            MB_CK(cudaFree(src));
            MB_CK(cudaFree(dst));
        }
    }
    if (do_l2) {
        const double max_mib = *std::max_element(cfg.l2_mib.begin(), cfg.l2_mib.end());
        void* buf = nullptr;
        MB_CK(cudaMalloc(&buf, (size_t) (max_mib * 1048576.0)));
        MB_CK(cudaMemset(buf, 0x5A, (size_t) (max_mib * 1048576.0)));
        for (double mib : cfg.l2_mib) print_row(R.run_read(LD_CG, mib, cfg.l2_seconds, false, cfg.unroll, cfg.bps, buf));
        MB_CK(cudaFree(buf));
    }
    ses.end("membw");
    cudaFree(sink);
    cudaStreamDestroy(st);
    return 0;
}
