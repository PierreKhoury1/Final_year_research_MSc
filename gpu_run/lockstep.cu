// Lockstep test: how precisely can the CPU make an NVIDIA GPU start work at a chosen instant T?
// Methods: normal launch at T, CUDA Graph launch at T, persistent block released by a host flag at T,
// and GPU self-timed (resident block waits on %globaltimer for T converted to GPU time).
// Start error = GPU-side start time (mapped to host time) - T. Host clock: CLOCK_MONOTONIC_RAW.
// Build: nvcc -O2 -std=c++17 -o lockstep lockstep.cu -lpthread
// Run:   ./lockstep <condition-label> [load=none|stream] [targets=500] [gap_us=2000] [host_core=1] > out.json
// Output: one JSON object per run. A statistic that could not be measured is null, never NaN, so the
// line always parses. If LOCKSTEP_RAW names a directory, every start error (us) is also written there,
// one file per condition and method, for histograms later.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <string>
#include <pthread.h>
#include <sched.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static int N = 500;                       // targets per method (argv[3])
static int64_t GAP_NS = 2000000;          // between targets (argv[4], given in us)
static const unsigned STOP = 0xFFFFFFFEu;

__device__ __forceinline__ uint64_t gtimer() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// ---- kernels ----
// work: block 0 thread 0 stamps start; all threads do a small fixed amount of FMA work; stamps end.
__global__ void work(unsigned long long *st, unsigned long long *en, unsigned *ctr, float *a) {
    __shared__ unsigned idx;
    if (blockIdx.x == 0 && threadIdx.x == 0) { idx = atomicAdd(ctr, 1); st[idx] = gtimer(); }
    float v = a[blockIdx.x * blockDim.x + threadIdx.x];
    for (int k = 0; k < 400; k++) v = v * 1.0000001f + 0.5f;
    a[blockIdx.x * blockDim.x + threadIdx.x] = v;
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) en[idx] = gtimer();
}

// persistent: one resident block. Thread 0 waits until the host flag has reached i+1. If the block arrived
// late and the flag is already past i+1 it stamps at once, so lateness is recorded instead of the kernel
// waiting forever for a value that will never come back (the failure seen on the laptop). STOP ends it.
__global__ void flagged(volatile unsigned *flag, unsigned long long *st, unsigned long long *en, float *a, int n) {
    __shared__ int quit;
    for (int i = 0; i < n; i++) {
        if (threadIdx.x == 0) {
            unsigned c;
            while ((c = flag[0]) < (unsigned)(i + 1)) {}
            quit = (c == STOP);
            st[i] = quit ? 0 : gtimer();
        }
        __syncthreads();
        if (quit) return;
        float v = a[threadIdx.x];
        for (int k = 0; k < 400; k++) v = v * 1.0000001f + 0.5f;
        a[threadIdx.x] = v;
        __syncthreads();
        if (threadIdx.x == 0) en[i] = gtimer();
    }
}

// self-timed: one resident block; thread 0 waits on the GPU clock for each target tick. Every 4096 spins it
// also glances at the host stop flag, so a badly mapped target cannot hang the run.
__global__ void selftimed(const unsigned long long *tgt, unsigned long long *st, unsigned long long *en, float *a, int n, volatile unsigned *stopflag) {
    __shared__ int quit;
    for (int i = 0; i < n; i++) {
        if (threadIdx.x == 0) {
            uint64_t t; unsigned k = 0; int q = 0;
            while ((t = gtimer()) < tgt[i]) { if ((++k & 4095u) == 0 && stopflag[0] == STOP) { q = 1; break; } }
            quit = q;
            st[i] = q ? 0 : t;
        }
        __syncthreads();
        if (quit) return;
        float v = a[threadIdx.x];
        for (int k = 0; k < 400; k++) v = v * 1.0000001f + 0.5f;
        a[threadIdx.x] = v;
        __syncthreads();
        if (threadIdx.x == 0) en[i] = gtimer();
    }
}

// calibration ping-pong over mapped pinned memory
__global__ void pp(volatile unsigned *ctl, unsigned long long *gt, int n, unsigned base) {
    for (int i = 0; i < n; i++) {
        unsigned want = base + i + 1, c;
        while ((c = ctl[0]) != want) { if (c == STOP) return; }
        gt[i] = gtimer(); __threadfence_system(); ctl[1] = want; __threadfence_system();
    }
}

// "AI job": big FMA kernels keeping every SM busy
__global__ void ai_job(float *a, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; float v = a[i];
    for (int k = 0; k < iters; k++) v = v * 1.0000001f + 0.5f;
    a[i] = v;
}

// globaltimer update granularity
__global__ void gres(unsigned long long *out, int n) {
    uint64_t last = gtimer(); int k = 0;
    while (k < n) { uint64_t t = gtimer(); if (t != last) { out[k++] = t - last; last = t; } }
}

// ---- host helpers ----
static inline int64_t now_ns() { timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts); return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec; }
static inline void spin_until(int64_t t) { while (now_ns() < t) {} }
static bool pin_to(int core) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(core, &s); return pthread_setaffinity_np(pthread_self(), sizeof s, &s) == 0; }
static void pin_away_from(int core) {
    int ncpu = (int)std::thread::hardware_concurrency(); if (ncpu < 2) return;
    cpu_set_t s; CPU_ZERO(&s); for (int c = 0; c < ncpu && c < CPU_SETSIZE; c++) if (c != core) CPU_SET(c, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}
static bool set_fifo(int prio) { sched_param p{}; p.sched_priority = prio; return pthread_setschedparam(pthread_self(), SCHED_FIFO, &p) == 0; }
// wait for a stream, but give up after ms: a hung kernel must not take the whole run with it
static bool sync_within(cudaStream_t s, int64_t ms) {
    int64_t end = now_ns() + ms * 1000000LL;
    for (;;) {
        cudaError_t e = cudaStreamQuery(s);
        if (e == cudaSuccess) return true;
        if (e != cudaErrorNotReady) CK(e);
        if (now_ns() > end) return false;
    }
}
static std::string jnum(double v, const char *fmt = "%.3f") {
    if (std::isnan(v) || std::isinf(v)) return "null";
    char b[64]; snprintf(b, sizeof b, fmt, v); return b;
}
static std::string jstr(const std::string &s) {   // minimal JSON string escaping
    std::string o = "\"";
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; if (c == '\n') { o += "\\n"; continue; } o += c; }
    return o + "\"";
}

struct Br { int64_t t0, t1; uint64_t g; };
struct Map { double a, b; uint64_t g0; double eps_ns; int used; };

static volatile unsigned *ctl_h; static unsigned *ctl_d;
static unsigned long long *gt_h, *gt_d;
static unsigned seq = 0;

static std::vector<Br> calibrate(cudaStream_t s, int batches = 8, int n = 400) {
    std::vector<Br> out;
    for (int b = 0; b < batches; b++) {
        unsigned base = seq; ctl_h[0] = base; ctl_h[1] = base; __sync_synchronize();
        pp<<<1, 1, 0, s>>>((volatile unsigned *)ctl_d, gt_d, n, base);
        int got = 0; std::vector<std::pair<int64_t, int64_t>> tt;
        for (int i = 0; i < n; i++) {
            unsigned w = base + i + 1;
            int64_t t0 = now_ns(); ctl_h[0] = w; __sync_synchronize();
            int64_t lim = t0 + 1000000000LL; bool ok = true;
            while (ctl_h[1] != w) { if (now_ns() > lim) { ok = false; break; } }
            int64_t t1 = now_ns();
            if (!ok) break;
            tt.push_back({t0, t1}); got++;
        }
        ctl_h[0] = STOP; __sync_synchronize();
        if (!sync_within(s, 5000)) { fprintf(stderr, "calibration kernel did not finish\n"); exit(2); }
        for (int i = 0; i < got; i++) out.push_back({tt[i].first, tt[i].second, gt_h[i]});
        seq = base + n + 1;
    }
    return out;
}

// min-filter fit: host_ns = a*(g-g0) + b on the tightest brackets; eps = half median tight width + worst violation
static Map fit(const std::vector<Br> &br) {
    if (br.size() < 4) return {1.0, 0.0, 0, NAN, 0};
    std::vector<double> w; for (auto &x : br) w.push_back((double)(x.t1 - x.t0));
    std::vector<double> ws = w; std::sort(ws.begin(), ws.end());
    double thr = std::max(ws[(size_t)(ws.size() * 0.05)], 1500.0);
    uint64_t g0 = UINT64_MAX; for (auto &x : br) g0 = std::min(g0, x.g);
    double sx = 0, sy = 0, sxx = 0, sxy = 0; int n = 0;
    for (size_t i = 0; i < br.size(); i++) if (w[i] <= thr) {
        double X = (double)(br[i].g - g0), Y = 0.5 * (double)(br[i].t0 + br[i].t1);
        sx += X; sy += Y; sxx += X * X; sxy += X * Y; n++;
    }
    if (n < 2) return {1.0, 0.0, g0, NAN, n};
    double a = (n * sxy - sx * sy) / (n * sxx - sx * sx), b = (sy - a * sx) / n;
    std::vector<double> tw; double viol = 0;
    for (size_t i = 0; i < br.size(); i++) if (w[i] <= thr) {
        double p = a * (double)(br[i].g - g0) + b;
        viol = std::max(viol, std::max((double)br[i].t0 - p, p - (double)br[i].t1));
        tw.push_back(w[i]);
    }
    std::sort(tw.begin(), tw.end());
    return {a, b, g0, tw[tw.size() / 2] / 2 + std::max(viol, 0.0), n};
}

int main(int argc, char **argv) {
    std::string cond = argc > 1 ? argv[1] : "idle";
    std::string load = argc > 2 ? argv[2] : "none";
    if (argc > 3) N = std::max(1, atoi(argv[3]));
    if (argc > 4) GAP_NS = (int64_t)atoll(argv[4]) * 1000;
    int hcore = argc > 5 ? atoi(argv[5]) : 1;
    const char *rawdir = getenv("LOCKSTEP_RAW");

    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int drv = 0, rt = 0; CK(cudaDriverGetVersion(&drv)); CK(cudaRuntimeGetVersion(&rt));

    CK(cudaHostAlloc((void **)&ctl_h, 64, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void **)&ctl_d, (void *)ctl_h, 0));
    CK(cudaHostAlloc((void **)&gt_h, 400 * 8, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void **)&gt_d, gt_h, 0));
    unsigned *flag_h, *flag_d; CK(cudaHostAlloc((void **)&flag_h, 64, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void **)&flag_d, flag_h, 0));
    unsigned *stop_h, *stop_d; CK(cudaHostAlloc((void **)&stop_h, 64, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void **)&stop_d, stop_h, 0));
    flag_h[0] = 0; stop_h[0] = 0;

    int lo, hi; CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    cudaStream_t s, s_load; CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, hi)); CK(cudaStreamCreateWithPriority(&s_load, cudaStreamNonBlocking, lo));

    unsigned long long *st, *en, *tgt; unsigned *ctr; float *a, *big;
    CK(cudaMalloc(&st, (size_t)N * 8)); CK(cudaMalloc(&en, (size_t)N * 8)); CK(cudaMalloc(&tgt, (size_t)N * 8)); CK(cudaMalloc(&ctr, 4));
    CK(cudaMalloc(&a, 1 << 20)); CK(cudaMemset(a, 0, 1 << 20));
    size_t bigN = (size_t)prop.multiProcessorCount * 2048 * 8; CK(cudaMalloc(&big, bigN * 4)); CK(cudaMemset(big, 0, bigN * 4));

    // globaltimer granularity
    unsigned long long *rd; CK(cudaMallocManaged(&rd, 1000 * 8)); gres<<<1, 1>>>(rd, 1000); CK(cudaDeviceSynchronize());
    std::vector<unsigned long long> inc(rd, rd + 1000); std::sort(inc.begin(), inc.end());

    // background AI load (same process, low-priority stream), kept off the timing core
    std::atomic<bool> stop{false}; std::thread th;
    if (load == "stream") th = std::thread([&] { pin_away_from(hcore); while (!stop) { ai_job<<<(unsigned)(bigN / 256), 256, 0, s_load>>>(big, 20000); cudaStreamSynchronize(s_load); } });
    if (load == "stream") { int64_t t = now_ns() + 300000000LL; spin_until(t); }

    // timing thread: one core, real-time priority when the container allows it
    bool pinned = pin_to(hcore);
    bool fifo = set_fifo(80);
    if (!pinned) fprintf(stderr, "warning: could not pin the host thread to core %d\n", hcore);
    if (!fifo) fprintf(stderr, "note: SCHED_FIFO refused (needs CAP_SYS_NICE); running at normal priority\n");

    std::vector<Br> br = calibrate(s);
    Map m = fit(br);
    if (std::isnan(m.eps_ns)) { fprintf(stderr, "calibration produced too few brackets (%zu); is mapped pinned memory working?\n", br.size()); }

    // CUDA Graph of one work launch. Thread-local capture: the load thread's own stream calls must not
    // be counted as unsafe activity during capture.
    cudaGraph_t graph; cudaGraphExec_t gexec;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    work<<<8, 256, 0, s>>>(st, en, ctr, a);
    CK(cudaStreamEndCapture(s, &graph)); CK(cudaGraphInstantiate(&gexec, graph, 0));

    const char *names[] = {"normal launch", "CUDA Graph launch", "persistent block + flag", "GPU self-timed"};
    std::vector<std::vector<double>> errs(4), durs(4);
    bool timed_out[4] = {false, false, false, false};
    std::vector<unsigned long long> hs(N), he(N);
    for (int meth = 0; meth < 4; meth++) {
        CK(cudaMemsetAsync(st, 0, (size_t)N * 8, s)); CK(cudaMemsetAsync(en, 0, (size_t)N * 8, s)); CK(cudaMemsetAsync(ctr, 0, 4, s)); CK(cudaStreamSynchronize(s));
        std::vector<int64_t> T(N); int64_t t0 = now_ns() + 50000000LL;
        for (int i = 0; i < N; i++) T[i] = t0 + i * GAP_NS;
        if (meth == 0) {
            for (int i = 0; i < N; i++) { spin_until(T[i]); work<<<8, 256, 0, s>>>(st, en, ctr, a); }
        } else if (meth == 1) {
            for (int i = 0; i < N; i++) { spin_until(T[i]); CK(cudaGraphLaunch(gexec, s)); }
        } else if (meth == 2) {
            flag_h[0] = 0; __sync_synchronize();
            flagged<<<1, 256, 0, s>>>((volatile unsigned *)flag_d, st, en, a, N);
            for (int i = 0; i < N; i++) { spin_until(T[i]); ((volatile unsigned *)flag_h)[0] = i + 1; __sync_synchronize(); }
            spin_until(T[N - 1] + GAP_NS);
        } else {
            std::vector<unsigned long long> tg(N);
            for (int i = 0; i < N; i++) tg[i] = (unsigned long long)llround(((double)T[i] - m.b) / m.a) + m.g0;
            CK(cudaMemcpy(tgt, tg.data(), (size_t)N * 8, cudaMemcpyHostToDevice));
            stop_h[0] = 0; __sync_synchronize();
            selftimed<<<1, 256, 0, s>>>(tgt, st, en, a, N, (volatile unsigned *)stop_d);
            spin_until(T[N - 1] + GAP_NS);
        }
        // give the method 5 s past its last target; then tell resident kernels to quit and wait once more
        if (!sync_within(s, 5000)) {
            timed_out[meth] = true;
            fprintf(stderr, "method '%s' did not finish 5 s after its last target; stopping it\n", names[meth]);
            ((volatile unsigned *)flag_h)[0] = STOP; ((volatile unsigned *)stop_h)[0] = STOP; __sync_synchronize();
            if (!sync_within(s, 5000)) { fprintf(stderr, "kernel still resident; aborting run\n"); exit(2); }
        }
        CK(cudaMemcpy(hs.data(), st, (size_t)N * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(he.data(), en, (size_t)N * 8, cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++) if (hs[i]) {
            double start_host = m.a * (double)(hs[i] - m.g0) + m.b;
            errs[meth].push_back((start_host - (double)T[i]) / 1000.0);
            if (he[i] > hs[i]) durs[meth].push_back((double)(he[i] - hs[i]) / 1000.0);
        }
        if (rawdir) {
            std::string fn = std::string(rawdir) + "/" + cond + "__" + names[meth] + ".txt";
            for (char &c : fn) if (c == ' ' || c == ',' || c == '+') c = '_';
            if (FILE *f = fopen(fn.c_str(), "w")) { for (double e : errs[meth]) fprintf(f, "%.3f\n", e); fclose(f); }
        }
    }
    // re-check the clock mapping after the run
    std::vector<Br> br2 = calibrate(s, 4); br.insert(br.end(), br2.begin(), br2.end()); Map m2 = fit(br);
    stop = true; if (th.joinable()) th.join();

    auto pct = [](std::vector<double> v, double p) { if (v.empty()) return (double)NAN; std::sort(v.begin(), v.end()); return v[(size_t)std::min(v.size() - 1.0, p / 100.0 * (v.size() - 1))]; };
    auto over = [](const std::vector<double> &v, double us) { if (v.empty()) return (double)NAN; size_t k = 0; for (double e : v) if (e > us) k++; return 100.0 * k / v.size(); };

    printf("{\"gpu\":%s,\"sm\":%d,\"driver\":%d,\"runtime\":%d,\"condition\":%s,\"load\":%s,\"n_targets\":%d,\"gap_us\":%lld,",
           jstr(prop.name).c_str(), prop.multiProcessorCount, drv, rt, jstr(cond).c_str(), jstr(load).c_str(), N, (long long)(GAP_NS / 1000));
    printf("\"host_core\":%d,\"pinned\":%s,\"sched_fifo\":%s,\"globaltimer_step_ns\":{\"min\":%llu,\"median\":%llu,\"max\":%llu},",
           hcore, pinned ? "true" : "false", fifo ? "true" : "false", inc[0], inc[500], inc[999]);
    printf("\"self_timed_floor_us\":%s,\"clock_bound_us\":%s,\"clock_bound_after_us\":%s,\"rate_change_ppm\":%s,\"calib_brackets_used\":%d,\"methods\":{",
           jnum(inc[500] / 1000.0).c_str(), jnum(m.eps_ns / 1000).c_str(), jnum(m2.eps_ns / 1000).c_str(), jnum((m2.a / m.a - 1) * 1e6, "%.4f").c_str(), m.used);
    for (int k = 0; k < 4; k++) {
        auto &e = errs[k];
        double mx = e.empty() ? NAN : *std::max_element(e.begin(), e.end());
        printf("%s%s:{\"n\":%zu,\"timeout\":%s,\"p1_us\":%s,\"median_us\":%s,\"p99_us\":%s,\"p999_us\":%s,\"max_us\":%s,\"over_100us_pct\":%s,\"work_median_us\":%s}",
               k ? "," : "", jstr(names[k]).c_str(), e.size(), timed_out[k] ? "true" : "false",
               jnum(pct(e, 1)).c_str(), jnum(pct(e, 50)).c_str(), jnum(pct(e, 99)).c_str(), jnum(pct(e, 99.9)).c_str(), jnum(mx).c_str(),
               jnum(over(e, 100.0), "%.2f").c_str(), jnum(pct(durs[k], 50)).c_str());
    }
    printf("}}\n");
    return 0;
}
