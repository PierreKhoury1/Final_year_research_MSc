// clockcal: long-run CPU CLOCK_MONOTONIC_RAW <-> GPU %globaltimer calibration (DESIGN.md section 6).
// Every --interval-s it collects --samples brackets (pingpong or launch), fits them with sb::fit_clock
// and appends one flushed row to clock_windows.csv, so a 6-hour run survives being killed. At the end it
// fits all kept brackets globally and writes meta.json (also written at start with status "running").
// GPU temperature / SM clock come from NVML loaded with dlopen (no link dependency, never nvidia-smi).
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/utsname.h>

#include "clock_fit.h"
#include "cuda_check.h"
#include "host_time.h"
#include "json_writer.h"

namespace {

// ---------------- config ----------------
struct Config {
    std::string out;
    double duration_s = 21600;
    double interval_s = 60;
    int samples = 10000;
    std::string method = "pingpong";
    int core = -1;
    int fifo = 0;
    int gpu = 0;
    int raw = 0;
    std::string load = "none";
    bool print_config = false;
};

void usage(FILE *f) {
    fprintf(f,
            "usage: clockcal --out DIR [--duration-s 21600] [--interval-s 60] [--samples 10000]\n"
            "                [--method pingpong|launch] [--core -1] [--fifo 0] [--gpu 0] [--raw 0|1]\n"
            "                [--load none|spin] [--print-config]\n");
}

bool parse_double(const char *s, double &v) {
    char *e = nullptr;
    errno = 0;
    v = strtod(s, &e);
    return errno == 0 && e != s && *e == 0 && std::isfinite(v);
}
bool parse_int(const char *s, int &v) {
    char *e = nullptr;
    errno = 0;
    long x = strtol(s, &e, 10);
    if (errno != 0 || e == s || *e != 0 || x < -2147483647L || x > 2147483647L) return false;
    v = (int)x;
    return true;
}

// Returns 0 ok, 2 on bad usage (message already printed).
int parse_args(int argc, char **argv, Config &c) {
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        if (k == "--print-config") { c.print_config = true; continue; }
        if (k == "-h" || k == "--help") { usage(stdout); exit(0); }
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); usage(stderr); return 2; }
        const char *v = argv[++i];
        bool ok = true;
        if (k == "--out") c.out = v;
        else if (k == "--duration-s") ok = parse_double(v, c.duration_s) && c.duration_s > 0;
        else if (k == "--interval-s") ok = parse_double(v, c.interval_s) && c.interval_s > 0;
        else if (k == "--samples") ok = parse_int(v, c.samples) && c.samples >= 100;
        else if (k == "--method") { c.method = v; ok = c.method == "pingpong" || c.method == "launch"; }
        else if (k == "--core") ok = parse_int(v, c.core);
        else if (k == "--fifo") ok = parse_int(v, c.fifo) && c.fifo >= 0 && c.fifo <= 99;
        else if (k == "--gpu") ok = parse_int(v, c.gpu) && c.gpu >= 0;
        else if (k == "--raw") ok = parse_int(v, c.raw) && (c.raw == 0 || c.raw == 1);
        else if (k == "--load") { c.load = v; ok = c.load == "none" || c.load == "spin"; }
        else { fprintf(stderr, "unknown flag %s\n", k.c_str()); usage(stderr); return 2; }
        if (!ok) { fprintf(stderr, "bad value for %s: %s\n", k.c_str(), v); usage(stderr); return 2; }
    }
    if (c.out.empty() && !c.print_config) { fprintf(stderr, "--out is required\n"); usage(stderr); return 2; }
    return 0;
}

std::string config_json(const Config &c) {
    sb::Json j;
    j.add("out", c.out).add("duration_s", c.duration_s).add("interval_s", c.interval_s).add("samples", c.samples)
        .add("method", c.method).add("core", c.core).add("fifo", c.fifo).add("gpu", c.gpu).add("raw", c.raw)
        .add("load", c.load);
    return j.str();
}

// ---------------- signals ----------------
volatile sig_atomic_t g_stop = 0;
int g_signal = 0;
void on_signal(int s) { g_stop = 1; g_signal = s; }

// clock_nanosleep in <= 100 ms slices so a signal ends the wait promptly. Returns false if stopped.
bool sleep_until_raw_or_stop(int64_t t_raw) {
    while (!g_stop) {
        int64_t now = sb::now_ns();
        if (now >= t_raw) return true;
        sb::sleep_until_raw(std::min<int64_t>(t_raw, now + 100000000LL));
    }
    return false;
}

// ---------------- NVML via dlopen ----------------
typedef struct nvmlDevice_st *nvmlDevice_t;
typedef int (*nvmlInit_t)(void);
typedef int (*nvmlGetByIndex_t)(unsigned, nvmlDevice_t *);
typedef int (*nvmlGetByPci_t)(const char *, nvmlDevice_t *);
typedef int (*nvmlGetTemp_t)(nvmlDevice_t, int, unsigned *);
typedef int (*nvmlGetClock_t)(nvmlDevice_t, int, unsigned *);
typedef int (*nvmlShutdown_t)(void);
const int NVML_TEMPERATURE_GPU = 0;
const int NVML_CLOCK_SM = 1;

struct Nvml {
    void *lib = nullptr;
    nvmlDevice_t dev = nullptr;
    nvmlGetTemp_t get_temp = nullptr;
    nvmlGetClock_t get_clock = nullptr;
    nvmlShutdown_t shutdown = nullptr;
    bool ok = false;
    std::string match = "none";  // pci / index / none
    std::string error;

    // pci: CUDA's bus id string, e.g. "0000:01:00.0"; cuda_index used only if the PCI lookup fails.
    void open(const std::string &pci, int cuda_index) {
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) { error = "dlopen libnvidia-ml.so.1 failed"; return; }
        auto init = (nvmlInit_t)dlsym(lib, "nvmlInit_v2");
        auto by_idx = (nvmlGetByIndex_t)dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
        auto by_pci = (nvmlGetByPci_t)dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2");
        get_temp = (nvmlGetTemp_t)dlsym(lib, "nvmlDeviceGetTemperature");
        get_clock = (nvmlGetClock_t)dlsym(lib, "nvmlDeviceGetClockInfo");
        shutdown = (nvmlShutdown_t)dlsym(lib, "nvmlShutdown");
        if (!init || !get_temp || !get_clock) { error = "missing NVML symbols"; return; }
        int rc = init();
        if (rc != 0) { error = "nvmlInit_v2 returned " + std::to_string(rc); return; }
        if (by_pci && !pci.empty()) {
            // NVML prints 8-digit domains; try CUDA's form first, then the padded one.
            if (by_pci(pci.c_str(), &dev) == 0) match = "pci";
            else if (by_pci(("0000" + pci).c_str(), &dev) == 0) match = "pci";
        }
        if (match == "none" && by_idx && by_idx((unsigned)cuda_index, &dev) == 0) {
            match = "index";  // may be the wrong board on multi-GPU hosts; recorded in meta.json
        }
        if (match == "none") { error = "no NVML handle for the CUDA device"; return; }
        ok = true;
    }
    // NAN when unavailable.
    double temp_c() const {
        unsigned v = 0;
        return ok && get_temp(dev, NVML_TEMPERATURE_GPU, &v) == 0 ? (double)v : NAN;
    }
    double sm_mhz() const {
        unsigned v = 0;
        return ok && get_clock(dev, NVML_CLOCK_SM, &v) == 0 ? (double)v : NAN;
    }
    void close() {
        if (ok && shutdown) shutdown();
        ok = false;
    }
};

// ---------------- kernels ----------------
const unsigned STOP = 0xFFFFFFFEu;

__device__ __forceinline__ uint64_t gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Resident ping-pong: answer host sequence number base+i+1 with a %globaltimer reading (lockstep.cu).
__global__ void pp_kernel(volatile unsigned *ctl, unsigned long long *gt, int n, unsigned base) {
    for (int i = 0; i < n; i++) {
        unsigned want = base + (unsigned)i + 1, c;
        while ((c = ctl[0]) != want) {
            if (c == STOP) return;
        }
        gt[i] = gtimer();
        __threadfence_system();
        ctl[1] = want;
        __threadfence_system();
    }
}

// Launch method: one stamp, then the sequence number after it is globally visible.
struct Stamp {
    unsigned long long g;
    unsigned seq;
    unsigned pad;
};
__global__ void stamp_kernel(volatile Stamp *s, unsigned seq) {
    s->g = gtimer();
    __threadfence_system();
    s->seq = seq;
    __threadfence_system();
}

// %globaltimer update granularity: n successive distinct steps.
__global__ void gres_kernel(unsigned long long *out, int n) {
    uint64_t last = gtimer();
    int k = 0;
    while (k < n) {
        uint64_t t = gtimer();
        if (t != last) { out[k++] = t - last; last = t; }
    }
}

// Short-block FMA load (each block ~10 us) so the calibration kernel can still get an SM slot.
__global__ void fma_load(float *a, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = a[i];
    for (int k = 0; k < iters; k++) v = v * 1.0000001f + 0.5f;
    a[i] = v;
}

// ---------------- collection ----------------
struct Collector {
    cudaStream_t s = nullptr;
    volatile unsigned *ctl_h = nullptr;
    unsigned *ctl_d = nullptr;
    unsigned long long *gt_h = nullptr, *gt_d = nullptr;
    volatile Stamp *st_h = nullptr;
    Stamp *st_d = nullptr;
    unsigned seq = 0;
    static const int BATCH = 400;
    static constexpr int64_t TIMEOUT_NS = 1000000000LL;
    long timeouts = 0;

    void init(cudaStream_t stream) {
        s = stream;
        CK(cudaHostAlloc((void **)&ctl_h, 2 * sizeof(unsigned), cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void **)&ctl_d, (void *)ctl_h, 0));
        CK(cudaHostAlloc((void **)&gt_h, BATCH * sizeof(unsigned long long), cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void **)&gt_d, gt_h, 0));
        CK(cudaHostAlloc((void **)&st_h, sizeof(Stamp), cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void **)&st_d, (void *)st_h, 0));
        st_h->g = 0;
        st_h->seq = 0;
    }

    // Wait for the stream with a 10 s guard: a hung GPU must not hang a 6-hour run silently.
    void sync_guarded() {
        int64_t lim = sb::now_ns() + 10000000000LL;
        for (;;) {
            cudaError_t e = cudaStreamQuery(s);
            if (e == cudaSuccess) return;
            if (e != cudaErrorNotReady) CK(e);
            if (sb::now_ns() > lim) { fprintf(stderr, "clockcal: GPU stream stuck for 10 s, aborting\n"); exit(3); }
            sb::cpu_relax();
        }
    }

    void pingpong(int n, std::vector<sb::Bracket> &out) {
        std::vector<std::pair<int64_t, int64_t>> tt;
        tt.reserve(BATCH);
        while ((int)out.size() < n) {
            int m = std::min(BATCH, n - (int)out.size());
            unsigned base = seq;
            ctl_h[0] = base;
            ctl_h[1] = base;
            __sync_synchronize();
            pp_kernel<<<1, 1, 0, s>>>((volatile unsigned *)ctl_d, gt_d, m, base);
            CK(cudaGetLastError());
            tt.clear();
            for (int i = 0; i < m; i++) {
                unsigned w = base + (unsigned)i + 1;
                int64_t t0 = sb::now_ns();
                ctl_h[0] = w;
                __sync_synchronize();
                int64_t lim = t0 + TIMEOUT_NS;
                bool ok = true;
                while (ctl_h[1] != w) {
                    if (sb::now_ns() > lim) { ok = false; break; }
                }
                int64_t t1 = sb::now_ns();
                if (!ok) { timeouts++; break; }
                tt.push_back({t0, t1});
            }
            ctl_h[0] = STOP;
            __sync_synchronize();
            sync_guarded();
            for (size_t i = 0; i < tt.size(); i++) out.push_back({tt[i].first, tt[i].second, gt_h[i]});
            seq = base + (unsigned)m + 1;
            if (seq >= STOP - 2 * BATCH) seq = 0;  // never let a sequence number collide with STOP
            if (tt.empty()) return;  // GPU not answering at all; the fit reports what it got
        }
    }

    void launch(int n, std::vector<sb::Bracket> &out) {
        for (int i = 0; i < n; i++) {
            unsigned w = ++seq;
            if (w == 0 || w >= STOP) { seq = 1; w = 1; st_h->seq = 0; }
            int64_t t0 = sb::now_ns();
            stamp_kernel<<<1, 1, 0, s>>>((volatile Stamp *)st_d, w);
            int64_t lim = t0 + TIMEOUT_NS;
            bool ok = true;
            while (st_h->seq != w) {
                if (sb::now_ns() > lim) { ok = false; break; }
            }
            int64_t t1 = sb::now_ns();
            if (!ok) { timeouts++; CK(cudaGetLastError()); sync_guarded(); continue; }
            out.push_back({t0, t1, st_h->g});
        }
        CK(cudaGetLastError());
        sync_guarded();
    }
};

// ---------------- helpers ----------------
// Same selection as sb::fit_clock: the tightest max(ceil(1%), 50) valid brackets (ties included).
std::vector<sb::Bracket> kept_brackets(const std::vector<sb::Bracket> &br, double keep_frac = 0.01, int min_keep = 50) {
    std::vector<int64_t> ws;
    for (auto &b : br)
        if (b.t1 >= b.t0 && b.g != 0) ws.push_back(b.t1 - b.t0);
    std::vector<sb::Bracket> out;
    if (ws.size() < 2) return out;
    std::sort(ws.begin(), ws.end());
    size_t keep = (size_t)std::ceil(keep_frac * (double)ws.size());
    keep = std::max(keep, (size_t)std::max(min_keep, 2));
    keep = std::min(keep, ws.size());
    int64_t thr = ws[keep - 1];
    for (auto &b : br)
        if (b.t1 >= b.t0 && b.g != 0 && b.t1 - b.t0 <= thr) out.push_back(b);
    return out;
}

// The fit's local GPU->host offset at GPU reading g, split into an exact integer part and a small
// double part: offset(g) = host_of(g) - g. %globaltimer is ~1.7e18 ns, beyond double's integer range.
struct Offset {
    int64_t whole;
    double frac;
};
Offset local_offset(const sb::ClockFit &f, uint64_t g) {
    double x = (double)(int64_t)(g - f.g_ref);
    return {f.t_ref - (int64_t)f.g_ref, f.b + (f.a - 1.0) * x};
}

std::string csv_num(double v, const char *fmt = "%.3f") {
    if (!std::isfinite(v)) return "";
    char b[64];
    snprintf(b, sizeof b, fmt, v);
    return b;
}

bool mkdir_p(const std::string &p) {
    std::string cur;
    for (size_t i = 0; i <= p.size(); i++) {
        if (i == p.size() || p[i] == '/') {
            if (!cur.empty() && mkdir(cur.c_str(), 0775) != 0 && errno != EEXIST) return false;
        }
        if (i < p.size()) cur += p[i];
    }
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string uuid_str(const cudaUUID_t &u) {
    char b[64];
    const unsigned char *x = (const unsigned char *)u.bytes;
    snprintf(b, sizeof b, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", x[0], x[1], x[2],
             x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]);
    return b;
}

// ---------------- load thread ----------------
std::atomic<bool> g_load_stop{false};

void load_thread(int gpu, cudaStream_t ls, int sms) {
    CK(cudaSetDevice(gpu));
    const int threads = 256, blocks = sms * 16;
    float *a = nullptr;
    CK(cudaMalloc(&a, (size_t)threads * blocks * sizeof(float)));
    CK(cudaMemsetAsync(a, 0, (size_t)threads * blocks * sizeof(float), ls));
    while (!g_load_stop.load(std::memory_order_relaxed)) {
        for (int i = 0; i < 4; i++) fma_load<<<blocks, threads, 0, ls>>>(a, 4096);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(ls));  // keep the queue shallow
    }
    CK(cudaFree(a));
}

}  // namespace

int main(int argc, char **argv) {
    Config cfg;
    if (int rc = parse_args(argc, argv, cfg)) return rc;
    if (cfg.print_config) {
        printf("%s\n", config_json(cfg).c_str());
        return 0;
    }
    if (!mkdir_p(cfg.out)) { fprintf(stderr, "cannot create %s: %s\n", cfg.out.c_str(), strerror(errno)); return 1; }

    std::string cmdline;
    for (int i = 0; i < argc; i++) cmdline += (i ? " " : "") + std::string(argv[i]);

    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost));
    CK(cudaSetDevice(cfg.gpu));
    CK(cudaFree(0));

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, cfg.gpu));
    int clk_khz = 0, mclk_khz = 0, drv = 0, rtv = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, cfg.gpu);
    cudaDeviceGetAttribute(&mclk_khz, cudaDevAttrMemoryClockRate, cfg.gpu);
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rtv);
    char pci[32] = {0};
    CK(cudaDeviceGetPCIBusId(pci, sizeof pci, cfg.gpu));

    Nvml nvml;
    nvml.open(pci, cfg.gpu);
    if (!nvml.ok) fprintf(stderr, "clockcal: NVML unavailable (%s); temperature/clock columns stay empty\n", nvml.error.c_str());

    int prio_lo = 0, prio_hi = 0;
    CK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
    cudaStream_t cs, ls = nullptr;
    CK(cudaStreamCreateWithPriority(&cs, cudaStreamNonBlocking, prio_hi));

    // %globaltimer granularity (bounds how sharp a single bracket can be).
    const int NRES = 64;
    unsigned long long *res_d = nullptr, res_h[NRES];
    CK(cudaMalloc(&res_d, sizeof res_h));
    gres_kernel<<<1, 1, 0, cs>>>(res_d, NRES);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(res_h, res_d, sizeof res_h, cudaMemcpyDeviceToHost, cs));
    CK(cudaStreamSynchronize(cs));
    CK(cudaFree(res_d));
    std::vector<unsigned long long> steps(res_h, res_h + NRES);
    std::sort(steps.begin(), steps.end());

    Collector col;
    col.init(cs);

    // Start the load thread before pinning/FIFO so it does not inherit the calibration core or priority.
    std::thread loader;
    if (cfg.load == "spin") {
        CK(cudaStreamCreateWithPriority(&ls, cudaStreamNonBlocking, prio_lo));
        loader = std::thread(load_thread, cfg.gpu, ls, prop.multiProcessorCount);
    }

    bool pin_ok = sb::pin_thread(cfg.core);
    bool fifo_ok = sb::set_fifo(cfg.fifo);
    if (!pin_ok) fprintf(stderr, "clockcal: pinning to core %d failed\n", cfg.core);
    if (!fifo_ok) fprintf(stderr, "clockcal: SCHED_FIFO %d failed\n", cfg.fifo);

    struct utsname un{};
    uname(&un);
    const std::string start_iso = sb::iso_utc_now();
    const int64_t t_start = sb::now_ns();
    const int64_t interval_ns = (int64_t)std::llround(cfg.interval_s * 1e9);
    const int64_t duration_ns = (int64_t)std::llround(cfg.duration_s * 1e9);
    const long n_windows_planned = (long)((duration_ns + interval_ns - 1) / interval_ns);

    int n_windows = 0, n_ok = 0, n_skipped = 0;
    bool have_ref = false;
    Offset ref{0, 0.0};
    sb::ClockFit ref_fit, global_fit;
    std::vector<sb::Bracket> all_kept;
    std::string exit_reason = "running";

    auto write_meta = [&](const std::string &status) {
        sb::Json gpu;
        gpu.add("name", std::string(prop.name)).add("uuid", uuid_str(prop.uuid)).add("pci_bus_id", std::string(pci))
            .add("sm_count", prop.multiProcessorCount).add("cc", std::to_string(prop.major) + "." + std::to_string(prop.minor))
            .add("clock_khz", clk_khz).add("mem_clock_khz", mclk_khz);
        sb::Json nv;
        nv.add("available", nvml.ok).add("match", nvml.match).add("error", nvml.error);
        sb::Json gstep;
        gstep.add("min", steps.front()).add("median", steps[NRES / 2]).add("max", steps.back());
        sb::Json j;
        j.add("tool", "clockcal").add("status", status).add("exit_reason", exit_reason).add_raw("config", config_json(cfg)).add("cmdline", cmdline).add("gpu", gpu)
            .add("driver_version", drv).add("runtime_version", rtv).add("host", sb::hostname())
            .add("kernel_release", std::string(un.release)).add("method", cfg.method).add("load", cfg.load)
            .add("stream_priority_range", std::vector<double>{(double)prio_lo, (double)prio_hi})
            .add("stream_priority", prio_hi).add("pin_ok", pin_ok).add("fifo_ok", fifo_ok).add("nvml", nv)
            .add("globaltimer_step_ns", gstep).add("start_utc", start_iso).add("start_monoraw_ns", (long long)t_start)
            .add("end_utc", status == "running" ? std::string("") : sb::iso_utc_now())
            .add("n_windows_planned", n_windows_planned).add("n_windows", n_windows).add("n_windows_ok", n_ok)
            .add("n_windows_skipped", n_skipped).add("timeouts", col.timeouts)
            .add("offset_definition",
                 "offset_ns = [host_k(g_c) - g_c] - [host_ref(g_ref_c) - g_ref_c]: window k's fitted GPU->host offset "
                 "at its centre GPU reading g_c minus the reference window's at its own centre")
            .add_raw("reference_window_fit", have_ref ? ref_fit.json() : "null")
            .add_raw("global_fit", global_fit.ok ? global_fit.json() : "null");
        std::string path = cfg.out + "/meta.json";
        std::string tmp = path + ".tmp";
        FILE *f = fopen(tmp.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", tmp.c_str()); return; }
        fputs(j.str().c_str(), f);
        fputc('\n', f);
        fclose(f);
        rename(tmp.c_str(), path.c_str());
    };
    write_meta("running");

    std::string csv_path = cfg.out + "/clock_windows.csv";
    FILE *csv = fopen(csv_path.c_str(), "w");
    if (!csv) { fprintf(stderr, "cannot write %s\n", csv_path.c_str()); return 1; }
    fprintf(csv,
            "# clockcal %s method=%s samples=%d interval_s=%g load=%s gpu=\"%s\"\n"
            "# t_host_ns: host CLOCK_MONOTONIC_RAW at the window centre = host_k(g_c), g_c = (min g + max g)/2 of the window\n"
            "# offset_ns: [host_k(g_c) - g_c] - [host_ref(g_ref_c) - g_ref_c], i.e. how the GPU->host offset wandered since\n"
            "#   the reference (first successfully fitted) window; each window is evaluated with its own fit at its own\n"
            "#   centre, so no fitted rate is extrapolated across windows. Slope of offset_ns vs time = relative rate.\n"
            "# rate_ppm: (a - 1) * 1e6 of the window fit (host ns per GPU ns); positive = host clock runs faster\n"
            "# residual_*/eps_ns/widths: from sb::fit_clock over the window (common/clock_fit.h); temp/clock from NVML, empty if absent\n"
            "t_host_ns,n,n_kept,offset_ns,rate_ppm,residual_rms_ns,residual_max_ns,min_width_ns,median_width_ns,eps_ns,"
            "gpu_temp_c,sm_clock_mhz\n",
            start_iso.c_str(), cfg.method.c_str(), cfg.samples, cfg.interval_s, cfg.load.c_str(), prop.name);
    fflush(csv);

    std::vector<sb::Bracket> br;
    br.reserve(cfg.samples);
    for (long k = 0; k < n_windows_planned && !g_stop; k++) {
        int64_t t_win = t_start + k * interval_ns;
        if (sb::now_ns() > t_win + interval_ns / 2 && k > 0) { n_skipped++; continue; }  // overran: skip boundary
        if (!sleep_until_raw_or_stop(t_win)) break;

        br.clear();
        if (cfg.method == "pingpong") col.pingpong(cfg.samples, br);
        else col.launch(cfg.samples, br);
        sb::ClockFit f = sb::fit_clock(br);
        double temp = nvml.temp_c(), smclk = nvml.sm_mhz();
        n_windows++;

        uint64_t gmin = UINT64_MAX, gmax = 0;
        for (auto &b : br)
            if (b.g != 0) { gmin = std::min(gmin, b.g); gmax = std::max(gmax, b.g); }

        std::string t_host, off;
        if (f.ok) {
            uint64_t gc = gmin + (gmax - gmin) / 2;
            Offset o = local_offset(f, gc);
            if (!have_ref) { have_ref = true; ref = o; ref_fit = f; }
            double offset = (double)(o.whole - ref.whole) + (o.frac - ref.frac);
            t_host = std::to_string((long long)std::llround(f.host_of(gc)));
            off = csv_num(offset);
            n_ok++;
            auto kb = kept_brackets(br);
            all_kept.insert(all_kept.end(), kb.begin(), kb.end());
        } else {
            t_host = std::to_string((long long)(br.empty() ? sb::now_ns() : br[br.size() / 2].t0));
        }
        fprintf(csv, "%s,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", t_host.c_str(), f.n, f.n_kept, off.c_str(),
                f.ok ? csv_num(f.rate_ppm(), "%.6f").c_str() : "", csv_num(f.resid_rms_ns).c_str(),
                csv_num(f.resid_max_ns).c_str(), csv_num(f.min_width_ns, "%.0f").c_str(),
                csv_num(f.median_width_ns, "%.0f").c_str(), csv_num(f.eps_ns).c_str(), csv_num(temp, "%.0f").c_str(),
                csv_num(smclk, "%.0f").c_str());
        fflush(csv);

        if (cfg.raw) {
            char name[64];
            snprintf(name, sizeof name, "/brackets_%05ld.bin", k);
            FILE *rf = fopen((cfg.out + name).c_str(), "wb");
            if (rf) {  // int64 t0, int64 t1, uint64 g per bracket, little endian, no header
                fwrite(br.data(), sizeof(sb::Bracket), br.size(), rf);
                fclose(rf);
            }
        }
        fprintf(stderr, "window %ld/%ld n=%d kept=%d eps=%.0f ns rate=%.3f ppm offset=%s ns temp=%s\n", k + 1,
                n_windows_planned, f.n, f.n_kept, f.eps_ns, f.rate_ppm(), off.c_str(), csv_num(temp, "%.0f").c_str());
        if (k > 0 && k % 10 == 0) write_meta("running");
    }
    fclose(csv);

    exit_reason = g_stop ? std::string("signal ") + std::to_string(g_signal) : "completed";
    if (loader.joinable()) {
        g_load_stop = true;
        loader.join();
    }
    // Global fit: every kept bracket of every window, all of them used (keep_frac 1).
    if (all_kept.size() >= 2) global_fit = sb::fit_clock(all_kept, 1.0, 2);
    write_meta(n_ok > 0 ? "ok" : "no_fit");
    nvml.close();
    return n_ok > 0 ? 0 : 1;
}
