// common.cuh - helpers shared by the dm-kernels microbenchmarks (CUDA 12.8, sm_86).
//
//  * MB_CK            CUDA error check that prints file:line and exits with code 3
//  * HwSampler        NVML sampler (SM clock, memory clock, power, temperature, P-state, throttle reasons,
//                     utilization, VRAM used) in a background thread; NVML is loaded with dlopen at run time,
//                     so the binaries start and run without libnvidia-ml (the hw_* columns are then NaN / 0)
//  * run_timed        back-to-back launches timed with cudaEvents for at least N seconds
//  * small CLI helpers (opt_value / parse_*)
//
// Nothing in here changes clocks, power limits or any other GPU setting. NVML is used read-only.
#pragma once

#include <cuda_runtime.h>
#include <nvml.h>

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define MB_CK(call)                                                                                              \
    do {                                                                                                         \
        cudaError_t err_ = (call);                                                                               \
        if (err_ != cudaSuccess) {                                                                               \
            std::fprintf(stderr, "CUDA error %s (%s) at %s:%d: %s\n", cudaGetErrorName(err_),                    \
                         cudaGetErrorString(err_), __FILE__, __LINE__, #call);                                   \
            std::exit(3);                                                                                        \
        }                                                                                                        \
    } while (0)

namespace mb {

inline double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
inline double epoch_s() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------------------------- CLI helpers
// `--name value` and `--name=value`. Returns true when argv[i] is that option; i is advanced past the value.
inline bool opt_value(int argc, char** argv, int& i, const char* name, std::string& val) {
    const std::string a = argv[i];
    const std::string n = name;
    if (a == n) {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "option %s needs a value\n", name);
            std::exit(2);
        }
        val = argv[++i];
        return true;
    }
    if (a.size() > n.size() && a.compare(0, n.size(), n) == 0 && a[n.size()] == '=') {
        val = a.substr(n.size() + 1);
        return true;
    }
    return false;
}
inline double parse_double(const std::string& s, const char* what) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || *end != '\0' || !std::isfinite(v)) {
        std::fprintf(stderr, "bad number for %s: '%s'\n", what, s.c_str());
        std::exit(2);
    }
    return v;
}
inline long long parse_ll(const std::string& s, const char* what) {
    char* end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0') {
        std::fprintf(stderr, "bad integer for %s: '%s'\n", what, s.c_str());
        std::exit(2);
    }
    return v;
}
inline std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t p = 0;
    while (true) {
        const size_t q = s.find(sep, p);
        out.push_back(s.substr(p, q == std::string::npos ? std::string::npos : q - p));
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out;
}
inline std::vector<double> parse_double_list(const std::string& s, const char* what) {
    std::vector<double> v;
    for (const auto& t : split(s, ',')) v.push_back(parse_double(t, what));
    return v;
}
inline std::vector<int> parse_int_list(const std::string& s, const char* what) {
    std::vector<int> v;
    for (const auto& t : split(s, ',')) v.push_back((int) parse_ll(t, what));
    return v;
}

inline std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
inline std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

// ---------------------------------------------------------------------------------------------- NVML sampler
struct HwSample {
    double t;                 // now_s()
    double epoch;             // epoch_s()
    unsigned sm_mhz, mem_mhz;
    double power_w;           // nvmlDeviceGetPowerUsage (the driver reports a short average)
    double power_inst_w;      // NVML_FI_DEV_POWER_INSTANT, NaN when unavailable
    unsigned temp_c;
    int pstate;               // 0..15, -1 unknown
    unsigned long long reasons;
    unsigned util_gpu;
    double vram_used_mib;
    int phase;
};

struct HwStats {
    int n = 0;
    double sm_mean = NAN, sm_min = NAN, sm_max = NAN;
    double mem_mean = NAN;
    double pw_mean = NAN, pw_max = NAN, pwi_mean = NAN, pwi_max = NAN;
    double temp_max = NAN;
    int pstate_last = -1;
    unsigned long long reasons_or = 0;
    double swcap_pct = NAN;   // share of samples with the SW power cap throttle reason set
    double vram_max_mib = NAN;
};

inline std::string hw_header() {
    return "sm_mhz_mean\tsm_mhz_min\tsm_mhz_max\tmem_mhz_mean\tpower_w_mean\tpower_w_max\tpower_inst_w_mean\t"
           "power_inst_w_max\ttemp_c_max\tpstate\tthrottle_or\tswcap_pct\tvram_used_mib_max\thw_samples";
}
inline std::string num(double v, const char* f = "%.1f") {
    if (!std::isfinite(v)) return "NA";
    return fmt(f, v);
}
inline std::string hw_row(const HwStats& s) {
    return num(s.sm_mean) + "\t" + num(s.sm_min, "%.0f") + "\t" + num(s.sm_max, "%.0f") + "\t" + num(s.mem_mean) + "\t" +
           num(s.pw_mean) + "\t" + num(s.pw_max) + "\t" + num(s.pwi_mean) + "\t" + num(s.pwi_max) + "\t" +
           num(s.temp_max, "%.0f") + "\t" + (s.pstate_last < 0 ? std::string("NA") : fmt("P%d", s.pstate_last)) + "\t" +
           fmt("0x%llx", s.reasons_or) + "\t" + num(s.swcap_pct) + "\t" + num(s.vram_max_mib, "%.0f") + "\t" +
           fmt("%d", s.n);
}

class HwSampler {
  public:
    ~HwSampler() { stop(); close(); }

    // pci_bus_id: from cudaDeviceGetPCIBusId; fallback_index is used when the lookup by bus id fails.
    bool open(const char* pci_bus_id, int fallback_index, std::string* msg) {
        lib_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!lib_) {
            if (msg) *msg = "libnvidia-ml.so.1 not found (no NVML sampling)";
            return false;
        }
        auto sym = [&](const char* n) { return dlsym(lib_, n); };
        init_      = (fn_void) sym("nvmlInit_v2");
        shutdown_  = (fn_void) sym("nvmlShutdown");
        by_pci_    = (fn_by_pci) sym("nvmlDeviceGetHandleByPciBusId_v2");
        by_idx_    = (fn_by_idx) sym("nvmlDeviceGetHandleByIndex_v2");
        clock_     = (fn_clock) sym("nvmlDeviceGetClockInfo");
        maxclock_  = (fn_clock) sym("nvmlDeviceGetMaxClockInfo");
        power_     = (fn_u32) sym("nvmlDeviceGetPowerUsage");
        limit_     = (fn_u32) sym("nvmlDeviceGetEnforcedPowerLimit");
        temp_      = (fn_temp) sym("nvmlDeviceGetTemperature");
        pstate_    = (fn_pstate) sym("nvmlDeviceGetPerformanceState");
        reasons_   = (fn_reasons) sym("nvmlDeviceGetCurrentClocksEventReasons");
        if (!reasons_) reasons_ = (fn_reasons) sym("nvmlDeviceGetCurrentClocksThrottleReasons");
        util_      = (fn_util) sym("nvmlDeviceGetUtilizationRates");
        meminfo_   = (fn_meminfo) sym("nvmlDeviceGetMemoryInfo");
        fields_    = (fn_fields) sym("nvmlDeviceGetFieldValues");
        name_      = (fn_name) sym("nvmlDeviceGetName");
        if (!init_ || !by_idx_ || !clock_) {
            if (msg) *msg = "NVML symbols missing";
            return false;
        }
        if (init_() != NVML_SUCCESS) {
            if (msg) *msg = "nvmlInit_v2 failed";
            return false;
        }
        inited_ = true;
        nvmlReturn_t r = NVML_ERROR_UNKNOWN;
        if (by_pci_ && pci_bus_id && *pci_bus_id) r = by_pci_(pci_bus_id, &dev_);
        if (r != NVML_SUCCESS) r = by_idx_((unsigned) fallback_index, &dev_);
        if (r != NVML_SUCCESS) {
            if (msg) *msg = "cannot get NVML device handle";
            return false;
        }
        ok_ = true;
        return true;
    }

    bool ok() const { return ok_; }

    std::string device_summary() {
        if (!ok_) return "NVML unavailable";
        char nm[96] = "?";
        if (name_) name_(dev_, nm, sizeof nm);
        unsigned lim = 0, smax = 0, mmax = 0;
        if (limit_) limit_(dev_, &lim);
        if (maxclock_) {
            maxclock_(dev_, NVML_CLOCK_SM, &smax);
            maxclock_(dev_, NVML_CLOCK_MEM, &mmax);
        }
        return fmt("NVML: %s, enforced power limit %.0f W, max SM clock %u MHz, max mem clock %u MHz", nm, lim / 1000.0,
                   smax, mmax);
    }

    int set_phase(const std::string& name) {
        std::lock_guard<std::mutex> g(mu_);
        phases_.push_back(name);
        phase_ = (int) phases_.size() - 1;
        return phase_;
    }

    void start(int interval_ms) {
        if (!ok_ || running_) return;
        interval_ms_ = interval_ms < 1 ? 1 : interval_ms;
        running_ = true;
        th_ = std::thread([this] { loop(); });
    }
    void stop() {
        if (!running_) return;
        running_ = false;
        if (th_.joinable()) th_.join();
    }

    // Statistics over samples with t0 <= t <= t1 (now_s() time base).
    HwStats stats(double t0, double t1) {
        HwStats s;
        std::lock_guard<std::mutex> g(mu_);
        double sm_sum = 0, mem_sum = 0, pw_sum = 0, pwi_sum = 0;
        int pwi_n = 0, pw_n = 0, cap = 0;
        for (const auto& x : samples_) {
            if (x.t < t0 || x.t > t1) continue;
            ++s.n;
            sm_sum += x.sm_mhz;
            mem_sum += x.mem_mhz;
            s.sm_min = std::isnan(s.sm_min) ? x.sm_mhz : std::min(s.sm_min, (double) x.sm_mhz);
            s.sm_max = std::isnan(s.sm_max) ? x.sm_mhz : std::max(s.sm_max, (double) x.sm_mhz);
            if (std::isfinite(x.power_w)) { pw_sum += x.power_w; ++pw_n; s.pw_max = std::isnan(s.pw_max) ? x.power_w : std::max(s.pw_max, x.power_w); }
            if (std::isfinite(x.power_inst_w)) { pwi_sum += x.power_inst_w; ++pwi_n; s.pwi_max = std::isnan(s.pwi_max) ? x.power_inst_w : std::max(s.pwi_max, x.power_inst_w); }
            s.temp_max = std::isnan(s.temp_max) ? x.temp_c : std::max(s.temp_max, (double) x.temp_c);
            s.pstate_last = x.pstate;
            s.reasons_or |= x.reasons;
            if (x.reasons & 0x4ULL) ++cap;
            s.vram_max_mib = std::isnan(s.vram_max_mib) ? x.vram_used_mib : std::max(s.vram_max_mib, x.vram_used_mib);
        }
        if (s.n) {
            s.sm_mean = sm_sum / s.n;
            s.mem_mean = mem_sum / s.n;
            s.swcap_pct = 100.0 * cap / s.n;
            if (pw_n) s.pw_mean = pw_sum / pw_n;
            if (pwi_n) s.pwi_mean = pwi_sum / pwi_n;
        }
        return s;
    }

    double vram_used_mib_now() {
        if (!ok_ || !meminfo_) return NAN;
        nvmlMemory_t m;
        if (meminfo_(dev_, &m) != NVML_SUCCESS) return NAN;
        return m.used / 1048576.0;
    }

    // Raw time series, one row per sample. header=true writes the column line first.
    void write_series(FILE* f, const char* bench, bool header) {
        std::lock_guard<std::mutex> g(mu_);
        if (header)
            std::fprintf(f, "bench\tphase\tt_s\tepoch_s\tsm_mhz\tmem_mhz\tpower_w\tpower_inst_w\ttemp_c\tpstate\tthrottle\tutil_pct\tvram_used_mib\n");
        const double t_base = samples_.empty() ? 0.0 : samples_.front().t;
        for (const auto& x : samples_) {
            std::fprintf(f, "%s\t%s\t%.3f\t%.3f\t%u\t%u\t%.1f\t%s\t%u\t%d\t0x%llx\t%u\t%.0f\n", bench,
                         x.phase >= 0 && x.phase < (int) phases_.size() ? phases_[x.phase].c_str() : "-", x.t - t_base, x.epoch,
                         x.sm_mhz, x.mem_mhz, x.power_w, std::isfinite(x.power_inst_w) ? fmt("%.1f", x.power_inst_w).c_str() : "NA",
                         x.temp_c, x.pstate, x.reasons, x.util_gpu, x.vram_used_mib);
        }
    }

  private:
    using fn_void = nvmlReturn_t (*)();
    using fn_by_pci = nvmlReturn_t (*)(const char*, nvmlDevice_t*);
    using fn_by_idx = nvmlReturn_t (*)(unsigned, nvmlDevice_t*);
    using fn_clock = nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned*);
    using fn_u32 = nvmlReturn_t (*)(nvmlDevice_t, unsigned*);
    using fn_temp = nvmlReturn_t (*)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned*);
    using fn_pstate = nvmlReturn_t (*)(nvmlDevice_t, nvmlPstates_t*);
    using fn_reasons = nvmlReturn_t (*)(nvmlDevice_t, unsigned long long*);
    using fn_util = nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t*);
    using fn_meminfo = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*);
    using fn_fields = nvmlReturn_t (*)(nvmlDevice_t, int, nvmlFieldValue_t*);
    using fn_name = nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned);

    void loop() {
        while (running_) {
            const double t_begin = now_s();
            HwSample s{};
            s.t = t_begin;
            s.epoch = epoch_s();
            unsigned u = 0;
            if (clock_(dev_, NVML_CLOCK_SM, &u) == NVML_SUCCESS) s.sm_mhz = u;
            if (clock_(dev_, NVML_CLOCK_MEM, &u) == NVML_SUCCESS) s.mem_mhz = u;
            s.power_w = (power_ && power_(dev_, &u) == NVML_SUCCESS) ? u / 1000.0 : NAN;
            s.power_inst_w = NAN;
            if (fields_) {
                nvmlFieldValue_t fv;
                std::memset(&fv, 0, sizeof fv);
                fv.fieldId = NVML_FI_DEV_POWER_INSTANT;
                if (fields_(dev_, 1, &fv) == NVML_SUCCESS && fv.nvmlReturn == NVML_SUCCESS) {
                    if (fv.valueType == NVML_VALUE_TYPE_UNSIGNED_INT) s.power_inst_w = fv.value.uiVal / 1000.0;
                    else if (fv.valueType == NVML_VALUE_TYPE_UNSIGNED_LONG) s.power_inst_w = fv.value.ulVal / 1000.0;
                    else if (fv.valueType == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG) s.power_inst_w = fv.value.ullVal / 1000.0;
                }
            }
            if (temp_ && temp_(dev_, NVML_TEMPERATURE_GPU, &u) == NVML_SUCCESS) s.temp_c = u;
            nvmlPstates_t ps;
            s.pstate = (pstate_ && pstate_(dev_, &ps) == NVML_SUCCESS) ? (int) ps : -1;
            unsigned long long rs = 0;
            if (reasons_ && reasons_(dev_, &rs) == NVML_SUCCESS) s.reasons = rs;
            nvmlUtilization_t ut;
            if (util_ && util_(dev_, &ut) == NVML_SUCCESS) s.util_gpu = ut.gpu;
            nvmlMemory_t mem;
            s.vram_used_mib = (meminfo_ && meminfo_(dev_, &mem) == NVML_SUCCESS) ? mem.used / 1048576.0 : NAN;
            {
                std::lock_guard<std::mutex> g(mu_);
                s.phase = phase_;
                samples_.push_back(s);
            }
            const double wait = interval_ms_ / 1000.0 - (now_s() - t_begin);
            if (wait > 0) std::this_thread::sleep_for(std::chrono::duration<double>(wait));
        }
    }

    void close() {
        if (inited_ && shutdown_) shutdown_();
        inited_ = false;
        ok_ = false;
        if (lib_) dlclose(lib_);
        lib_ = nullptr;
    }

    void* lib_ = nullptr;
    nvmlDevice_t dev_ = nullptr;
    bool ok_ = false, inited_ = false;
    std::atomic<bool> running_{false};
    std::thread th_;
    std::mutex mu_;
    std::vector<HwSample> samples_;
    std::vector<std::string> phases_;
    int phase_ = -1;
    int interval_ms_ = 20;
    fn_void init_ = nullptr, shutdown_ = nullptr;
    fn_by_pci by_pci_ = nullptr;
    fn_by_idx by_idx_ = nullptr;
    fn_clock clock_ = nullptr, maxclock_ = nullptr;
    fn_u32 power_ = nullptr, limit_ = nullptr;
    fn_temp temp_ = nullptr;
    fn_pstate pstate_ = nullptr;
    fn_reasons reasons_ = nullptr;
    fn_util util_ = nullptr;
    fn_meminfo meminfo_ = nullptr;
    fn_fields fields_ = nullptr;
    fn_name name_ = nullptr;
};

// ---------------------------------------------------------------------------------------------- device info
struct DevInfo {
    int dev = 0;
    std::string name;
    int cc_major = 0, cc_minor = 0;
    int sm_count = 0;
    int l2_bytes = 0;
    int mem_bus_bits = 0;
    int mem_clock_khz = 0;     // cudaDevAttrMemoryClockRate (the base value the runtime reports, not the live clock)
    int max_threads_per_sm = 0;
    int max_regs_per_sm = 0;
    int max_smem_per_sm = 0;
    std::string pci_bus_id;
};

// Selects the device. Prints a clear message and exits with code 3 when there is none.
inline DevInfo select_device(int dev) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n <= 0) {
        std::fprintf(stderr, "no CUDA device available (%s)\n", e == cudaSuccess ? "device count 0" : cudaGetErrorString(e));
        std::exit(3);
    }
    if (dev < 0 || dev >= n) {
        std::fprintf(stderr, "--device %d out of range (0..%d)\n", dev, n - 1);
        std::exit(2);
    }
    MB_CK(cudaSetDevice(dev));
    cudaDeviceProp p;
    MB_CK(cudaGetDeviceProperties(&p, dev));
    DevInfo d;
    d.dev = dev;
    d.name = p.name;
    d.cc_major = p.major;
    d.cc_minor = p.minor;
    MB_CK(cudaDeviceGetAttribute(&d.sm_count, cudaDevAttrMultiProcessorCount, dev));
    MB_CK(cudaDeviceGetAttribute(&d.l2_bytes, cudaDevAttrL2CacheSize, dev));
    MB_CK(cudaDeviceGetAttribute(&d.mem_bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    MB_CK(cudaDeviceGetAttribute(&d.mem_clock_khz, cudaDevAttrMemoryClockRate, dev));
    MB_CK(cudaDeviceGetAttribute(&d.max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
    MB_CK(cudaDeviceGetAttribute(&d.max_regs_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, dev));
    MB_CK(cudaDeviceGetAttribute(&d.max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    char bus[32] = {0};
    MB_CK(cudaDeviceGetPCIBusId(bus, sizeof bus, dev));
    d.pci_bus_id = bus;
    return d;
}

inline void print_banner(const char* tool, const DevInfo& d, HwSampler& hw, const std::string& nvml_msg) {
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    std::fprintf(stderr, "# %s  start epoch %.3f\n", tool, epoch_s());
    std::fprintf(stderr, "# GPU %d: %s, sm_%d%d, %d SMs, L2 %.1f MiB, mem bus %d bit, runtime-reported mem clock %.0f MHz, PCI %s\n",
                 d.dev, d.name.c_str(), d.cc_major, d.cc_minor, d.sm_count, d.l2_bytes / 1048576.0, d.mem_bus_bits,
                 d.mem_clock_khz / 1000.0, d.pci_bus_id.c_str());
    std::fprintf(stderr, "# CUDA runtime %d.%d, driver API %d.%d\n", rt / 1000, (rt % 1000) / 10, drv / 1000, (drv % 1000) / 10);
    if (hw.ok()) std::fprintf(stderr, "# %s\n", hw.device_summary().c_str());
    else std::fprintf(stderr, "# NVML: %s\n", nvml_msg.c_str());
    size_t fr = 0, tot = 0;
    if (cudaMemGetInfo(&fr, &tot) == cudaSuccess)
        std::fprintf(stderr, "# VRAM total %.0f MiB, free %.0f MiB (before the benchmark allocates)\n", tot / 1048576.0, fr / 1048576.0);
}

// ---------------------------------------------------------------------------------------------- timing
struct LaunchTimes {
    std::vector<double> ms;   // GPU time of every enqueued batch
    double total_ms = 0;
    double t_begin = 0, t_end = 0;   // now_s() bounds (host clock), for matching NVML samples
};

// Enqueues `launch_once()` repeatedly (always keeping the next batch queued) until `seconds` of wall time have
// passed, and returns the GPU time of every batch measured with cudaEvents.
template <class F>
inline LaunchTimes run_timed(F&& launch_once, double seconds, cudaStream_t st) {
    std::vector<cudaEvent_t> ev;
    cudaEvent_t e0;
    MB_CK(cudaEventCreate(&e0));
    MB_CK(cudaStreamSynchronize(st));
    LaunchTimes r;
    r.t_begin = now_s();
    MB_CK(cudaEventRecord(e0, st));
    auto enqueue = [&]() {
        cudaEvent_t e;
        MB_CK(cudaEventCreate(&e));
        launch_once();
        MB_CK(cudaEventRecord(e, st));
        ev.push_back(e);
    };
    enqueue();
    enqueue();
    while (true) {
        MB_CK(cudaEventSynchronize(ev[ev.size() - 2]));
        if (now_s() - r.t_begin >= seconds) break;
        enqueue();
    }
    MB_CK(cudaEventSynchronize(ev.back()));
    r.t_end = now_s();
    cudaEvent_t last = e0;
    for (cudaEvent_t e : ev) {
        float ms = 0;
        MB_CK(cudaEventElapsedTime(&ms, last, e));
        r.ms.push_back(ms);
        last = e;
    }
    for (double m : r.ms) r.total_ms += m;
    for (cudaEvent_t e : ev) cudaEventDestroy(e);
    cudaEventDestroy(e0);
    return r;
}

inline double median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Opens the series file, if any, and creates the sampler. Returns the sampler (possibly not ok()).
struct Session {
    DevInfo dev;
    HwSampler hw;
    std::string nvml_msg;
    std::string series_path;
    int sample_ms = 20;

    void begin(const char* tool, int device, int sample_interval_ms) {
        dev = select_device(device);
        sample_ms = sample_interval_ms;
        hw.open(dev.pci_bus_id.c_str(), device, &nvml_msg);
        print_banner(tool, dev, hw, nvml_msg);
        hw.start(sample_ms);
        // let the sampler collect an idle baseline sample before any load starts
        std::this_thread::sleep_for(std::chrono::milliseconds(3 * sample_ms));
    }
    void end(const char* bench) {
        hw.stop();
        if (!series_path.empty()) {
            FILE* f = std::fopen(series_path.c_str(), "w");
            if (!f) {
                std::fprintf(stderr, "cannot write %s\n", series_path.c_str());
                return;
            }
            hw.write_series(f, bench, true);
            std::fclose(f);
            std::fprintf(stderr, "# NVML series written to %s\n", series_path.c_str());
        }
    }
};

}  // namespace mb
