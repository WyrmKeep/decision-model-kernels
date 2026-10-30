// mb-cublaslt-i8: cuBLASLt int8 x int8 -> int32 GEMM (the IMMA tensor-core path) for a list of shapes.
//
// This is a PURE int8 GEMM, an upper reference for what int8 tensor cores can do on this card with a tuned library.
// llama.cpp's MMQ does more per output: it reads Q4_0 / Q8_0 blocks (0.5625 / 1.0625 bytes per weight instead of
// 1 byte), unpacks 4-bit values, and applies a per-block scale to every 32-wide K block. Nothing of that is in here.
//
// Shape convention (GEMM of a weight matrix with a batch of token activations):
//   M = rows of the weight matrix (output features),  N = tokens,  K = reduction (input features)
//   D[M x N] = W[M x K] * X[N x K]^T,  2*M*N*K operations.
//
// Layout variants ("--variants"), because int8 IMMA kernels in cuBLASLt are picky about layouts:
//   col     default column-major, transa=T transb=N ("TN"): A is the row-major weight W (K contiguous, lda=K), B is the
//           row-major activation X (K contiguous, ldb=K), D is column-major MxN (M contiguous). This is also the natural
//           layout of ggml tensors, so it needs no data conversion.
//   col32   A and D in CUBLASLT_ORDER_COL32, B in CUBLASLT_ORDER_COL32_2R_4R4 (the Ampere IMMA layout), transa=N transb=T
//   col32t  as col32 but B in CUBLASLT_ORDER_COL4_4R2_8C (the Turing IMMA layout)
// A variant that cuBLASLt has no algorithm for is reported with status NO_ALGO. Timing the COL32 variants uses random data in
// the target layout (no conversion in the timed loop); --check converts a small problem and compares with a CPU reference.
//
// Algorithm search: cublasLtMatmulAlgoGetHeuristic (--max-algos candidates, optionally + --exhaustive tile/stage enumeration),
// every candidate is timed briefly, the best --retime are timed for --seconds, the fastest is reported.
// Timing: matmuls are captured in a CUDA graph (removes host launch cost) and the graph is launched back to back; the weight
// matrix is replicated --rotate-mib worth of copies and the copies are used in turn, so the weights come from DRAM, as in a
// real forward pass, instead of staying in L2. The activations and the output are reused.
#include <cublasLt.h>

#include "common.cuh"

namespace {

struct LtError {
    std::string msg;
};
// cublasGetStatusName lives in libcublas, which this binary does not link: name the statuses here.
const char* status_name(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "LICENSE_ERROR";
        default: return "UNKNOWN_STATUS";
    }
}
inline void lt_check(cublasStatus_t s, const char* what, int line) {
    if (s != CUBLAS_STATUS_SUCCESS) throw LtError{mb::fmt("%s (line %d): %s", status_name(s), line, what)};
}
inline void cu_check(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) throw LtError{mb::fmt("CUDA %s (line %d): %s", cudaGetErrorName(e), line, what)};
}
#define LT(x) lt_check((x), #x, __LINE__)
#define CU(x) cu_check((x), #x, __LINE__)

constexpr double kNominalOpsPerClkSm = 2048;  // int8 dense, GA10x GeForce (README: public-spec arithmetic)

struct Shape {
    long long M = 0, N = 0, K = 0;
    std::string label = "-", count = "1", wtype = "-";
};

struct Config {
    std::string shapes_file;
    std::vector<Shape> cli_shapes;
    std::string variants = "col,col32,col32t";
    std::vector<int> n_only;
    double seconds = 0.25;         // final timing per retimed candidate
    double search_seconds = 0.012; // quick timing per candidate
    int max_algos = 16;
    int retime = 3;
    bool exhaustive = false;
    int workspace_mib = 32;
    double rotate_mib = 32;        // weight copies total, 0 = a single copy
    double max_mem_mib = 1800;
    bool graph = true;
    bool check = true;
    bool dry_run = false;
    double peak_tops = NAN;
    int device = 0;
    int sample_ms = 20;
    bool header = true;
    std::string series;
};

void usage() {
    std::fputs(
        "usage: mb-cublaslt-i8 [options]\n"
        "  --shapes FILE        TSV: M N K [label [count [wtype]]], '#' lines are comments  (see shapes/*.tsv)\n"
        "  --shape M,N,K[,label]  add one shape (repeatable); with --shapes both are used\n"
        "  --n-only LIST        keep only shapes with N in this comma list\n"
        "  --variants LIST      col,col32,col32t                                         (default all three)\n"
        "  --seconds S          timing of each of the best candidates                      (default 0.25)\n"
        "  --search-seconds S   quick timing of every candidate                            (default 0.012)\n"
        "  --max-algos N        heuristic candidates per shape and variant                 (default 16)\n"
        "  --retime K           candidates re-timed with --seconds                         (default 3)\n"
        "  --exhaustive         also enumerate algo ids x tile ids x stage ids (many more candidates, slow)\n"
        "  --workspace-mib N    cuBLASLt workspace                                         (default 32)\n"
        "  --rotate-mib X       weight copies cycled through per timing batch, 0 = one copy (default 32)\n"
        "  --max-mem-mib X      skip shapes that need more device memory than this         (default 1800)\n"
        "  --no-graph           time plain launches instead of CUDA graphs\n"
        "  --no-check           skip the small CPU-reference correctness check\n"
        "  --peak-tops X        measured int8 peak (from mb-mma, s8k32) for the efficiency column\n"
        "  --dry-run            parse the shapes and print the memory plan, no CUDA needed\n"
        "  --device N  --sample-ms N  --series FILE  --no-header  --print-header (print the TSV header and exit)\n"
        "TSV rows go to stdout, human-readable progress to stderr.\n",
        stderr);
}

// ------------------------------------------------------------------------------------------ shapes
bool load_shapes(const std::string& path, std::vector<Shape>& out) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) {
        std::fprintf(stderr, "cannot open shapes file %s\n", path.c_str());
        return false;
    }
    char line[4096];
    int ln = 0;
    while (std::fgets(line, sizeof line, f)) {
        ++ln;
        std::string s = line;
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (s.empty() || s[0] == '#') continue;
        std::vector<std::string> t = mb::split(s, '\t');
        if (t.size() < 3) t = mb::split(s, ' ');
        if (t.size() < 3) {
            std::fprintf(stderr, "%s:%d: need at least M N K\n", path.c_str(), ln);
            std::fclose(f);
            return false;
        }
        char* e1 = nullptr;
        char* e2 = nullptr;
        char* e3 = nullptr;
        const long long M = std::strtoll(t[0].c_str(), &e1, 10);
        const long long N = std::strtoll(t[1].c_str(), &e2, 10);
        const long long K = std::strtoll(t[2].c_str(), &e3, 10);
        if (*e1 || *e2 || *e3 || t[0].empty() || t[1].empty() || t[2].empty()) {
            if (out.empty() && (t[0] == "M" || t[0] == "m")) continue;  // header line without '#'
            std::fprintf(stderr, "%s:%d: M N K must be integers\n", path.c_str(), ln);
            std::fclose(f);
            return false;
        }
        if (M < 1 || N < 1 || K < 1) {
            std::fprintf(stderr, "%s:%d: M N K must be positive\n", path.c_str(), ln);
            std::fclose(f);
            return false;
        }
        Shape sh;
        sh.M = M;
        sh.N = N;
        sh.K = K;
        if (t.size() > 3 && !t[3].empty()) sh.label = t[3];
        if (t.size() > 4 && !t[4].empty()) sh.count = t[4];
        if (t.size() > 5 && !t[5].empty()) sh.wtype = t[5];
        out.push_back(sh);
    }
    std::fclose(f);
    return true;
}

// ------------------------------------------------------------------------------------------ layouts
struct Variant {
    const char* name;
    bool col32;
    cublasLtOrder_t border;
};
const Variant kVariants[] = {
    {"col", false, CUBLASLT_ORDER_COL},
    {"col32", true, CUBLASLT_ORDER_COL32_2R_4R4},
    {"col32t", true, CUBLASLT_ORDER_COL4_4R2_8C},
};

inline long long roundup(long long x, long long m) { return (x + m - 1) / m * m; }

struct Geometry {
    cublasOperation_t ta, tb;
    long long rowsA, colsA, ldA, rowsB, colsB, ldB, rowsC, colsC, ldC;
    cublasLtOrder_t oa, ob, oc;
    size_t bytesA, bytesB, bytesC;
};

Geometry geometry(const Variant& v, const Shape& s) {
    Geometry g{};
    if (!v.col32) {
        g.ta = CUBLAS_OP_T;
        g.tb = CUBLAS_OP_N;
        g.rowsA = s.K; g.colsA = s.M; g.ldA = s.K;
        g.rowsB = s.K; g.colsB = s.N; g.ldB = s.K;
        g.rowsC = s.M; g.colsC = s.N; g.ldC = s.M;
        g.oa = g.ob = g.oc = CUBLASLT_ORDER_COL;
        g.bytesA = (size_t) s.K * s.M;
        g.bytesB = (size_t) s.K * s.N;
        g.bytesC = (size_t) 4 * s.M * s.N;
    } else {
        g.ta = CUBLAS_OP_N;
        g.tb = CUBLAS_OP_T;
        g.rowsA = s.M; g.colsA = s.K; g.ldA = 32 * s.M;
        g.rowsB = s.N; g.colsB = s.K; g.ldB = 32 * roundup(s.N, 8);
        g.rowsC = s.M; g.colsC = s.N; g.ldC = 32 * s.M;
        g.oa = CUBLASLT_ORDER_COL32;
        g.ob = v.border;
        g.oc = CUBLASLT_ORDER_COL32;
        g.bytesA = (size_t) g.ldA * (roundup(s.K, 32) / 32);
        g.bytesB = (size_t) g.ldB * (roundup(s.K, 32) / 32);
        g.bytesC = (size_t) 4 * g.ldC * (roundup(s.N, 32) / 32);
    }
    return g;
}

struct MemPlan {
    std::string status = "OK";
    int ncopies = 1;
    size_t strideA = 0;
    double total_mib = 0;
};

MemPlan plan_mem(const Variant& v, const Shape& s, const Config& cfg) {
    MemPlan p;
    const bool dim_ok = !v.col32 ? (s.M % 4 == 0 && s.K % 4 == 0 && s.N % 4 == 0) : (s.M % 4 == 0 && s.K % 32 == 0 && s.N % 8 == 0);
    if (!dim_ok) {
        p.status = "SKIP_DIM";
        return p;
    }
    const Geometry g = geometry(v, s);
    p.strideA = (size_t) roundup((long long) g.bytesA, 256);
    const double others = (double) roundup((long long) g.bytesB, 256) + (double) roundup((long long) g.bytesC, 256) + cfg.workspace_mib * 1048576.0;
    const double cap = cfg.max_mem_mib * 1048576.0;
    int n = 1;
    if (cfg.rotate_mib > 0) n = (int) std::max(1.0, std::min(1024.0, std::ceil(cfg.rotate_mib * 1048576.0 / (double) p.strideA)));
    while (n > 1 && others + (double) n * p.strideA > cap) --n;
    p.ncopies = n;
    p.total_mib = (others + (double) n * p.strideA) / 1048576.0;
    if (p.total_mib > cfg.max_mem_mib) p.status = "SKIP_MEM";
    return p;
}

// ------------------------------------------------------------------------------------------ device helpers
__global__ void k_fill(uint32_t* p, size_t n32, uint32_t seed) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n32; i += stride) {
        uint32_t x = (uint32_t) (i + 1) * 2654435761u ^ seed;
        x ^= x >> 15; x *= 0x2c1b3c6du; x ^= x >> 12; x *= 0x297a2d39u; x ^= x >> 15;
        p[i] = x;
    }
}

struct DevBuf {
    void* p = nullptr;
    ~DevBuf() {
        if (p) cudaFree(p);
    }
    void alloc(size_t bytes) { CU(cudaMalloc(&p, bytes)); }
};

template <class T, cublasStatus_t (*D)(T)>
struct LtRes {
    T v = nullptr;
    ~LtRes() {
        if (v) D(v);
    }
};
using MatmulDesc = LtRes<cublasLtMatmulDesc_t, cublasLtMatmulDescDestroy>;
using Layout = LtRes<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy>;
using Pref = LtRes<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceDestroy>;

void make_layout(Layout& l, cudaDataType t, long long rows, long long cols, long long ld, cublasLtOrder_t order) {
    LT(cublasLtMatrixLayoutCreate(&l.v, t, rows, cols, ld));
    LT(cublasLtMatrixLayoutSetAttribute(l.v, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof order));
}

const int32_t kAlpha = 1, kBeta = 0;

// One problem (shape x variant) with its descriptors and buffers.
struct Case {
    cublasLtHandle_t lt;
    cudaStream_t st;
    Variant var;
    Shape sh;
    Geometry g;
    MemPlan mp;
    size_t ws_bytes;

    MatmulDesc op;
    Layout Ad, Bd, Cd;
    DevBuf bufA, bufB, bufC, ws;

    Case(cublasLtHandle_t lt_, cudaStream_t st_, const Variant& v, const Shape& s, const MemPlan& m, size_t ws)
        : lt(lt_), st(st_), var(v), sh(s), g(geometry(v, s)), mp(m), ws_bytes(ws) {}

    void create(bool fill) {
        LT(cublasLtMatmulDescCreate(&op.v, CUBLAS_COMPUTE_32I, CUDA_R_32I));
        LT(cublasLtMatmulDescSetAttribute(op.v, CUBLASLT_MATMUL_DESC_TRANSA, &g.ta, sizeof g.ta));
        LT(cublasLtMatmulDescSetAttribute(op.v, CUBLASLT_MATMUL_DESC_TRANSB, &g.tb, sizeof g.tb));
        make_layout(Ad, CUDA_R_8I, g.rowsA, g.colsA, g.ldA, g.oa);
        make_layout(Bd, CUDA_R_8I, g.rowsB, g.colsB, g.ldB, g.ob);
        make_layout(Cd, CUDA_R_32I, g.rowsC, g.colsC, g.ldC, g.oc);
        bufA.alloc((size_t) mp.ncopies * mp.strideA);
        bufB.alloc((size_t) roundup((long long) g.bytesB, 256));
        bufC.alloc((size_t) roundup((long long) g.bytesC, 256));
        ws.alloc(ws_bytes);
        if (fill) {
            fill_dev(bufA.p, (size_t) mp.ncopies * mp.strideA, 0x1111u);
            fill_dev(bufB.p, (size_t) roundup((long long) g.bytesB, 256), 0x2222u);
            CU(cudaMemsetAsync(bufC.p, 0, (size_t) roundup((long long) g.bytesC, 256), st));
            ws_init();
            CU(cudaStreamSynchronize(st));
        }
    }
    void ws_init() { CU(cudaMemsetAsync(ws.p, 0, ws_bytes, st)); }
    void fill_dev(void* p, size_t bytes, uint32_t seed) {
        k_fill<<<256, 256, 0, st>>>((uint32_t*) p, bytes / 4, seed);
        CU(cudaGetLastError());
    }

    void launch(const cublasLtMatmulAlgo_t* algo, int copy) {
        const char* a = (const char*) bufA.p + (size_t) (copy % mp.ncopies) * mp.strideA;
        LT(cublasLtMatmul(lt, op.v, &kAlpha, a, Ad.v, bufB.p, Bd.v, &kBeta, bufC.p, Cd.v, bufC.p, Cd.v, algo, ws.p, ws_bytes, st));
    }
};

std::string algo_desc(const cublasLtMatmulAlgo_t& a) {
    uint32_t id = 0, tile = 0, stages = 0, splitk = 0, red = 0, swz = 0, custom = 0;
    size_t w = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof swz, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof custom, &w);
    return mb::fmt("id%u/tile%u/stages%u/splitk%u/red%u/swz%u/opt%u", id, tile, stages, splitk, red, swz, custom);
}

struct Cand {
    cublasLtMatmulAlgo_t algo;
    size_t ws = 0;
    float waves = 0;
    double quick_us = INFINITY;
};

struct Timing {
    double us = NAN;
    int reps = 0;
    int launches = 0;
    bool graph = false;
    double t_begin = 0, t_end = 0;
};

// Time per matmul: `batch_ms` of work per graph (clamped to max_reps matmuls), repeated for `seconds`.
Timing time_algo(Case& c, const cublasLtMatmulAlgo_t& algo, double seconds, double batch_ms, int max_reps, bool use_graph) {
    Timing r;
    c.launch(&algo, 0);  // warm-up and validation
    CU(cudaStreamSynchronize(c.st));
    // calibration with plain launches
    const int cal = 6;
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    CU(cudaEventRecord(e0, c.st));
    for (int i = 0; i < cal; ++i) c.launch(&algo, i);
    CU(cudaEventRecord(e1, c.st));
    CU(cudaEventSynchronize(e1));
    float ms = 0;
    CU(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    const double per_call_ms = std::max(1e-4, (double) ms / cal);
    int reps = (int) std::ceil(batch_ms / per_call_ms);
    reps = std::max(reps, std::min(c.mp.ncopies, max_reps));
    reps = std::max(1, std::min(reps, max_reps));
    r.reps = reps;

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool ok = false;
    if (use_graph) {
        if (cudaStreamBeginCapture(c.st, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
            bool cap_ok = true;
            try {
                for (int i = 0; i < reps; ++i) c.launch(&algo, i);
            } catch (const LtError&) {
                cap_ok = false;
            }
            cudaError_t ee = cudaStreamEndCapture(c.st, &graph);
            if (cap_ok && ee == cudaSuccess && graph && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess) ok = true;
        }
        if (!ok) {
            cudaGetLastError();  // clear a failed capture
            if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
            if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
        }
    }
    r.graph = ok;
    mb::LaunchTimes t;
    if (ok) {
        t = mb::run_timed([&] { CU(cudaGraphLaunch(exec, c.st)); }, seconds, c.st);
    } else {
        t = mb::run_timed([&] { for (int i = 0; i < reps; ++i) c.launch(&algo, i); }, seconds, c.st);
    }
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
    r.launches = (int) t.ms.size();
    r.us = t.total_ms * 1e3 / ((double) t.ms.size() * reps);
    r.t_begin = t.t_begin;
    r.t_end = t.t_end;
    return r;
}

// ------------------------------------------------------------------------------------------ correctness check
std::string check_variant(cublasLtHandle_t lt, cudaStream_t st, const Variant& v, size_t ws_bytes) {
    Shape s;
    s.M = 64; s.N = 48; s.K = 128;
    MemPlan mp;
    mp.ncopies = 1;
    mp.strideA = (size_t) roundup((long long) geometry(v, s).bytesA, 256);
    try {
        Case c(lt, st, v, s, mp, ws_bytes);
        c.create(false);
        // logical data: W[m][k], X[n][k], values in [-16, 15]
        std::vector<int8_t> W(s.M * s.K), X(s.N * s.K);
        uint32_t r = 12345;
        auto rnd = [&]() { r = r * 1664525u + 1013904223u; return (int8_t) ((int) ((r >> 16) & 31) - 16); };
        for (auto& x : W) x = rnd();
        for (auto& x : X) x = rnd();
        std::vector<int32_t> ref(s.M * s.N);
        for (long long m = 0; m < s.M; ++m)
            for (long long n = 0; n < s.N; ++n) {
                int32_t acc = 0;
                for (long long k = 0; k < s.K; ++k) acc += (int32_t) W[m * s.K + k] * (int32_t) X[n * s.K + k];
                ref[m + n * s.M] = acc;  // column-major MxN
            }

        std::vector<int32_t> got(s.M * s.N, 0x7fffffff);
        DevBuf tmpA, tmpB, tmpC;
        // candidate algorithm: first heuristic result
        Pref pref;
        LT(cublasLtMatmulPreferenceCreate(&pref.v));
        const uint64_t wsb = ws_bytes;
        LT(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb));
        cublasLtMatmulHeuristicResult_t h[4];
        int n = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, c.op.v, c.Ad.v, c.Bd.v, c.Cd.v, c.Cd.v, pref.v, 4, h, &n);
        if (hs != CUBLAS_STATUS_SUCCESS || n < 1) return "NO_ALGO";

        if (!v.col32) {
            CU(cudaMemcpy(c.bufA.p, W.data(), W.size(), cudaMemcpyHostToDevice));   // K x M col-major = row-major W
            CU(cudaMemcpy(c.bufB.p, X.data(), X.size(), cudaMemcpyHostToDevice));   // K x N col-major = row-major X
            CU(cudaMemset(c.bufC.p, 0, (size_t) 4 * s.M * s.N));
            c.launch(&h[0].algo, 0);
            CU(cudaStreamSynchronize(st));
            CU(cudaMemcpy(got.data(), c.bufC.p, (size_t) 4 * s.M * s.N, cudaMemcpyDeviceToHost));
        } else {
            // plain column-major logical matrices, converted to the cuBLASLt layouts with cublasLtMatrixTransform
            std::vector<int8_t> Acm(s.M * s.K), Bcm(s.N * s.K);
            for (long long m = 0; m < s.M; ++m)
                for (long long k = 0; k < s.K; ++k) Acm[m + k * s.M] = W[m * s.K + k];
            for (long long n2 = 0; n2 < s.N; ++n2)
                for (long long k = 0; k < s.K; ++k) Bcm[n2 + k * s.N] = X[n2 * s.K + k];
            tmpA.alloc(Acm.size());
            tmpB.alloc(Bcm.size());
            tmpC.alloc((size_t) 4 * s.M * s.N);
            CU(cudaMemcpy(tmpA.p, Acm.data(), Acm.size(), cudaMemcpyHostToDevice));
            CU(cudaMemcpy(tmpB.p, Bcm.data(), Bcm.size(), cudaMemcpyHostToDevice));
            LtRes<cublasLtMatrixTransformDesc_t, cublasLtMatrixTransformDescDestroy> td;
            LT(cublasLtMatrixTransformDescCreate(&td.v, CUDA_R_32F));
            const float one = 1.0f, zero = 0.0f;
            Layout sa, sb, dc;
            make_layout(sa, CUDA_R_8I, s.M, s.K, s.M, CUBLASLT_ORDER_COL);
            make_layout(sb, CUDA_R_8I, s.N, s.K, s.N, CUBLASLT_ORDER_COL);
            make_layout(dc, CUDA_R_32I, s.M, s.N, s.M, CUBLASLT_ORDER_COL);
            LT(cublasLtMatrixTransform(lt, td.v, &one, tmpA.p, sa.v, &zero, nullptr, nullptr, c.bufA.p, c.Ad.v, st));
            LT(cublasLtMatrixTransform(lt, td.v, &one, tmpB.p, sb.v, &zero, nullptr, nullptr, c.bufB.p, c.Bd.v, st));
            c.launch(&h[0].algo, 0);
            LT(cublasLtMatrixTransform(lt, td.v, &one, c.bufC.p, c.Cd.v, &zero, nullptr, nullptr, tmpC.p, dc.v, st));
            CU(cudaStreamSynchronize(st));
            CU(cudaMemcpy(got.data(), tmpC.p, (size_t) 4 * s.M * s.N, cudaMemcpyDeviceToHost));
        }
        long long bad = 0;
        for (size_t i = 0; i < ref.size(); ++i) bad += (ref[i] != got[i]);
        return bad == 0 ? "OK" : mb::fmt("MISMATCH_%lld_of_%zu", bad, ref.size());
    } catch (const LtError& e) {
        std::string m = "ERR:" + e.msg;
        for (auto& ch : m) if (ch == '\t' || ch == '\n') ch = ' ';
        cudaGetLastError();
        return m;
    }
}

// ------------------------------------------------------------------------------------------ one row
struct Row {
    std::string variant, status = "OK", algo = "NA", check = "NA";
    Shape sh;
    size_t ws = 0;
    int ncopies = 0, ncand = 0, reps = 0;
    bool graph = false;
    double time_us = NAN, tops = NAN, a_gbps = NAN, eff_peak = NAN, eff_nom = NAN, quick_best_us = NAN, waves = NAN, mem_mib = NAN;
    mb::HwStats hw;
};

void print_header() {
    std::printf(
        "bench\tvariant\tlabel\tcount\twtype\tM\tN\tK\tstatus\talgo\tworkspace_bytes\twaves\ta_copies\tgraph\treps_per_graph\t"
        "candidates\tquick_best_us\ttime_us\ttops\ta_read_gbps\teff_vs_peak_pct\teff_vs_nominal_pct\tcheck\tmem_mib\t%s\n",
        mb::hw_header().c_str());
}
void print_row(const Row& r) {
    std::printf("cublaslt_i8\t%s\t%s\t%s\t%s\t%lld\t%lld\t%lld\t%s\t%s\t%zu\t%s\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                r.variant.c_str(), r.sh.label.c_str(), r.sh.count.c_str(), r.sh.wtype.c_str(), r.sh.M, r.sh.N, r.sh.K,
                r.status.c_str(), r.algo.c_str(), r.ws, mb::num(r.waves, "%.2f").c_str(), r.ncopies, r.graph ? 1 : 0, r.reps,
                r.ncand, mb::num(r.quick_best_us, "%.2f").c_str(), mb::num(r.time_us, "%.2f").c_str(),
                mb::num(r.tops, "%.2f").c_str(), mb::num(r.a_gbps, "%.1f").c_str(), mb::num(r.eff_peak).c_str(),
                mb::num(r.eff_nom).c_str(), r.check.c_str(), mb::num(r.mem_mib, "%.0f").c_str(), mb::hw_row(r.hw).c_str());
    std::fflush(stdout);
}

struct Ctx {
    mb::Session& ses;
    const Config& cfg;
    cublasLtHandle_t lt;
    cudaStream_t st;
};

std::vector<Cand> candidates(Ctx& cx, Case& c, size_t ws_bytes) {
    std::vector<Cand> out;
    Pref pref;
    LT(cublasLtMatmulPreferenceCreate(&pref.v));
    const uint64_t wsb = ws_bytes;
    LT(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb));
    const uint32_t al = 16;
    LT(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &al, sizeof al));
    LT(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &al, sizeof al));
    LT(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &al, sizeof al));
    LT(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &al, sizeof al));
    std::vector<cublasLtMatmulHeuristicResult_t> h(cx.cfg.max_algos);
    int n = 0;
    const cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(cx.lt, c.op.v, c.Ad.v, c.Bd.v, c.Cd.v, c.Cd.v, pref.v, cx.cfg.max_algos, h.data(), &n);
    if (s == CUBLAS_STATUS_SUCCESS)
        for (int i = 0; i < n; ++i)
            if (h[i].state == CUBLAS_STATUS_SUCCESS) {
                Cand k;
                k.algo = h[i].algo;
                k.ws = h[i].workspaceSize;
                k.waves = h[i].wavesCount;
                out.push_back(k);
            }
    if (cx.cfg.exhaustive) {
        int ids[256];
        int nid = 0;
        if (cublasLtMatmulAlgoGetIds(cx.lt, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, 256, ids, &nid) == CUBLAS_STATUS_SUCCESS) {
            for (int a = 0; a < nid; ++a) {
                cublasLtMatmulAlgo_t base;
                if (cublasLtMatmulAlgoInit(cx.lt, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, ids[a], &base) != CUBLAS_STATUS_SUCCESS) continue;
                std::vector<uint32_t> tiles, stages;
                size_t sz = 0;
                if (cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz) == CUBLAS_STATUS_SUCCESS && sz) {
                    tiles.resize(sz / sizeof(uint32_t));
                    if (cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), sz, &sz) != CUBLAS_STATUS_SUCCESS) tiles.clear();
                }
                sz = 0;
                if (cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &sz) == CUBLAS_STATUS_SUCCESS && sz) {
                    stages.resize(sz / sizeof(uint32_t));
                    if (cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_STAGES_IDS, stages.data(), sz, &sz) != CUBLAS_STATUS_SUCCESS) stages.clear();
                }
                if (tiles.empty()) tiles.push_back(0);
                if (stages.empty()) stages.push_back(0);
                for (uint32_t t : tiles) {
                    if (t == 0 && tiles.size() > 1) continue;  // CUBLASLT_MATMUL_TILE_UNDEFINED
                    for (uint32_t sg : stages) {
                        if (sg == 0 && stages.size() > 1) continue;  // CUBLASLT_MATMUL_STAGES_UNDEFINED
                        cublasLtMatmulAlgo_t al2 = base;
                        if (t) cublasLtMatmulAlgoConfigSetAttribute(&al2, CUBLASLT_ALGO_CONFIG_TILE_ID, &t, sizeof t);
                        if (sg) cublasLtMatmulAlgoConfigSetAttribute(&al2, CUBLASLT_ALGO_CONFIG_STAGES_ID, &sg, sizeof sg);
                        cublasLtMatmulHeuristicResult_t chk;
                        std::memset(&chk, 0, sizeof chk);
                        if (cublasLtMatmulAlgoCheck(cx.lt, c.op.v, c.Ad.v, c.Bd.v, c.Cd.v, c.Cd.v, &al2, &chk) == CUBLAS_STATUS_SUCCESS &&
                            chk.workspaceSize <= ws_bytes) {
                            Cand k;
                            k.algo = al2;
                            k.ws = chk.workspaceSize;
                            k.waves = chk.wavesCount;
                            out.push_back(k);
                        }
                    }
                }
            }
        }
    }
    return out;
}

Row bench_one(Ctx& cx, const Shape& sh, const Variant& var, const std::string& check_status, const MemPlan& mp) {
    Row r;
    r.variant = var.name;
    r.sh = sh;
    r.check = check_status;
    r.ncopies = mp.ncopies;
    r.mem_mib = mp.total_mib;
    r.status = mp.status;
    if (mp.status != "OK") return r;
    const size_t ws_bytes = (size_t) cx.cfg.workspace_mib * 1048576ull;
    try {
        Case c(cx.lt, cx.st, var, sh, mp, ws_bytes);
        c.create(true);
        std::vector<Cand> cands = candidates(cx, c, ws_bytes);
        r.ncand = (int) cands.size();
        if (cands.empty()) {
            r.status = "NO_ALGO";
            return r;
        }
        cx.ses.hw.set_phase(mb::fmt("%s %s %lldx%lldx%lld search", var.name, sh.label.c_str(), sh.M, sh.N, sh.K));
        for (auto& k : cands) {
            try {
                const Timing t = time_algo(c, k.algo, cx.cfg.search_seconds, 1.5, 64, cx.cfg.graph);
                k.quick_us = t.us;
            } catch (const LtError&) {
                cudaGetLastError();
                k.quick_us = INFINITY;
            }
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.quick_us < b.quick_us; });
        r.quick_best_us = cands[0].quick_us;
        if (!std::isfinite(cands[0].quick_us)) {
            r.status = "ERR:no candidate ran";
            return r;
        }
        cx.ses.hw.set_phase(mb::fmt("%s %s %lldx%lldx%lld final", var.name, sh.label.c_str(), sh.M, sh.N, sh.K));
        Timing best;
        best.us = INFINITY;
        const Cand* best_c = nullptr;
        const int top = std::min<int>(cx.cfg.retime, (int) cands.size());
        for (int i = 0; i < top; ++i) {
            if (!std::isfinite(cands[i].quick_us)) break;
            const Timing t = time_algo(c, cands[i].algo, cx.cfg.seconds, 10.0, 4096, cx.cfg.graph);
            if (t.us < best.us) {
                best = t;
                best_c = &cands[i];
            }
        }
        cx.ses.hw.set_phase("idle");
        if (!best_c) {
            r.status = "ERR:no candidate ran";
            return r;
        }
        r.algo = algo_desc(best_c->algo);
        r.ws = best_c->ws;
        r.waves = best_c->waves;
        r.graph = best.graph;
        r.reps = best.reps;
        r.time_us = best.us;
        const double ops = 2.0 * (double) sh.M * (double) sh.N * (double) sh.K;
        r.tops = ops / (best.us * 1e-6) / 1e12;
        r.a_gbps = (double) sh.M * (double) sh.K / (best.us * 1e-6) / 1e9;
        r.hw = cx.ses.hw.stats(best.t_begin, best.t_end);
        if (std::isfinite(cx.cfg.peak_tops) && cx.cfg.peak_tops > 0) r.eff_peak = 100.0 * r.tops / cx.cfg.peak_tops;
        if (std::isfinite(r.hw.sm_mean) && r.hw.sm_mean > 0)
            r.eff_nom = 100.0 * r.tops * 1e12 / ((double) cx.ses.dev.sm_count * r.hw.sm_mean * 1e6 * kNominalOpsPerClkSm);
    } catch (const LtError& e) {
        r.status = "ERR:" + e.msg;
        for (auto& ch : r.status) if (ch == '\t' || ch == '\n') ch = ' ';
        cudaGetLastError();
    }
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string v;
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (mb::opt_value(argc, argv, i, "--shapes", v)) cfg.shapes_file = v;
        else if (mb::opt_value(argc, argv, i, "--shape", v)) {
            const auto t = mb::split(v, ',');
            if (t.size() < 3) { std::fprintf(stderr, "--shape needs M,N,K[,label]\n"); return 2; }
            Shape s;
            s.M = mb::parse_ll(t[0], "--shape M");
            s.N = mb::parse_ll(t[1], "--shape N");
            s.K = mb::parse_ll(t[2], "--shape K");
            if (t.size() > 3) s.label = t[3];
            if (s.M < 1 || s.N < 1 || s.K < 1) { std::fprintf(stderr, "--shape values must be positive\n"); return 2; }
            cfg.cli_shapes.push_back(s);
        }
        else if (mb::opt_value(argc, argv, i, "--n-only", v)) cfg.n_only = mb::parse_int_list(v, "--n-only");
        else if (mb::opt_value(argc, argv, i, "--variants", v)) cfg.variants = v;
        else if (mb::opt_value(argc, argv, i, "--seconds", v)) cfg.seconds = mb::parse_double(v, "--seconds");
        else if (mb::opt_value(argc, argv, i, "--search-seconds", v)) cfg.search_seconds = mb::parse_double(v, "--search-seconds");
        else if (mb::opt_value(argc, argv, i, "--max-algos", v)) cfg.max_algos = (int) mb::parse_ll(v, "--max-algos");
        else if (mb::opt_value(argc, argv, i, "--retime", v)) cfg.retime = (int) mb::parse_ll(v, "--retime");
        else if (a == "--exhaustive") cfg.exhaustive = true;
        else if (mb::opt_value(argc, argv, i, "--workspace-mib", v)) cfg.workspace_mib = (int) mb::parse_ll(v, "--workspace-mib");
        else if (mb::opt_value(argc, argv, i, "--rotate-mib", v)) cfg.rotate_mib = mb::parse_double(v, "--rotate-mib");
        else if (mb::opt_value(argc, argv, i, "--max-mem-mib", v)) cfg.max_mem_mib = mb::parse_double(v, "--max-mem-mib");
        else if (a == "--no-graph") cfg.graph = false;
        else if (a == "--no-check") cfg.check = false;
        else if (mb::opt_value(argc, argv, i, "--peak-tops", v)) cfg.peak_tops = mb::parse_double(v, "--peak-tops");
        else if (a == "--dry-run") cfg.dry_run = true;
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
    if (cfg.seconds <= 0 || cfg.search_seconds <= 0 || cfg.max_algos < 1 || cfg.max_algos > 256 || cfg.retime < 1 || cfg.workspace_mib < 0 ||
        cfg.rotate_mib < 0 || cfg.max_mem_mib < 64 || cfg.sample_ms < 1) {
        std::fprintf(stderr, "bad numeric option\n");
        return 2;
    }
    std::vector<const Variant*> vars;
    for (const auto& n : mb::split(cfg.variants, ',')) {
        const Variant* found = nullptr;
        for (const auto& v : kVariants)
            if (n == v.name) found = &v;
        if (!found) { std::fprintf(stderr, "unknown variant '%s' (col, col32, col32t)\n", n.c_str()); return 2; }
        vars.push_back(found);
    }
    std::vector<Shape> shapes;
    if (!cfg.shapes_file.empty() && !load_shapes(cfg.shapes_file, shapes)) return 2;
    for (const auto& s : cfg.cli_shapes) shapes.push_back(s);
    if (shapes.empty()) {
        std::fprintf(stderr, "no shapes: use --shapes FILE or --shape M,N,K\n");
        return 2;
    }
    if (!cfg.n_only.empty()) {
        std::vector<Shape> keep;
        for (const auto& s : shapes)
            if (std::find(cfg.n_only.begin(), cfg.n_only.end(), (int) s.N) != cfg.n_only.end()) keep.push_back(s);
        shapes.swap(keep);
        if (shapes.empty()) { std::fprintf(stderr, "--n-only removed every shape\n"); return 2; }
    }

    if (cfg.dry_run) {
        std::printf("label\tcount\twtype\tM\tN\tK\tvariant\tstatus\ta_copies\tmem_mib\tgflop\n");
        double max_mem = 0;
        for (const auto& s : shapes)
            for (const Variant* v : vars) {
                const MemPlan mp = plan_mem(*v, s, cfg);
                std::printf("%s\t%s\t%s\t%lld\t%lld\t%lld\t%s\t%s\t%d\t%.1f\t%.3f\n", s.label.c_str(), s.count.c_str(), s.wtype.c_str(), s.M, s.N,
                            s.K, v->name, mp.status.c_str(), mp.ncopies, mp.total_mib, 2.0 * s.M * s.N * s.K / 1e9);
                if (mp.status == "OK") max_mem = std::max(max_mem, mp.total_mib);
            }
        std::fprintf(stderr, "# %zu shapes x %zu variants, largest planned allocation %.0f MiB (cap --max-mem-mib %.0f), plus the CUDA context\n",
                     shapes.size(), vars.size(), max_mem, cfg.max_mem_mib);
        return 0;
    }

    mb::Session ses;
    ses.series_path = cfg.series;
    ses.begin("mb-cublaslt-i8", cfg.device, cfg.sample_ms);
    cudaStream_t st;
    MB_CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cublasLtHandle_t lt = nullptr;
    {
        cublasStatus_t s = cublasLtCreate(&lt);
        if (s != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "cublasLtCreate failed: %s\n", status_name(s));
            return 3;
        }
    }
    std::fprintf(stderr, "# cuBLASLt version %zu, %zu shapes x %zu variants, workspace %d MiB, weights rotate %.0f MiB, graph %s%s\n",
                 cublasLtGetVersion(), shapes.size(), vars.size(), cfg.workspace_mib, cfg.rotate_mib, cfg.graph ? "on" : "off",
                 cfg.exhaustive ? ", exhaustive search" : "");
    if (std::isfinite(cfg.peak_tops)) std::fprintf(stderr, "# measured int8 peak for the efficiency column: %.2f TOPS\n", cfg.peak_tops);

    Ctx cx{ses, cfg, lt, st};
    std::vector<std::string> checks;
    for (const Variant* v : vars) {
        ses.hw.set_phase(std::string("check ") + v->name);
        const std::string c = cfg.check ? check_variant(lt, st, *v, (size_t) cfg.workspace_mib * 1048576ull) : "NOT_RUN";
        checks.push_back(c);
        std::fprintf(stderr, "# self-check %-6s (64x48x128 against a CPU reference): %s\n", v->name, c.c_str());
    }

    if (cfg.header) print_header();
    double min_free_mib = 1e30;
    for (const auto& s : shapes) {
        for (size_t vi = 0; vi < vars.size(); ++vi) {
            const Variant& v = *vars[vi];
            const MemPlan mp = plan_mem(v, s, cfg);
            Row r = bench_one(cx, s, v, checks[vi], mp);
            size_t fr = 0, tot = 0;
            if (cudaMemGetInfo(&fr, &tot) == cudaSuccess) min_free_mib = std::min(min_free_mib, fr / 1048576.0);
            print_row(r);
            std::fprintf(stderr, "%-6s %-18s M=%-6lld N=%-5lld K=%-5lld %-9s %9.2f us %8.2f TOPS  A-read %7.1f GB/s  eff(peak) %s%%  %s  cand=%d\n",
                         r.variant.c_str(), s.label.c_str(), s.M, s.N, s.K, r.status.substr(0, 9).c_str(), r.time_us, r.tops, r.a_gbps,
                         mb::num(r.eff_peak).c_str(), r.algo.c_str(), r.ncand);
        }
    }
    cublasLtDestroy(lt);
    ses.end("cublaslt_i8");
    cudaStreamDestroy(st);
    return 0;
}
