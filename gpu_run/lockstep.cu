// Lockstep test: how precisely can the CPU make an NVIDIA GPU start work at a chosen instant T?
// Methods: normal launch at T, CUDA Graph launch at T, persistent block released by a host flag at T,
// and GPU self-timed (resident block waits on %globaltimer for T converted to GPU time).
// Start error = GPU-side start time (mapped to host time) - T. Host clock: CLOCK_MONOTONIC_RAW.
// Build: nvcc -O2 -std=c++17 -o lockstep lockstep.cu -lpthread
// Run:   ./lockstep <condition-label> [load=none|stream] > out.json
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

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static const int N = 500;                 // targets per method
static const int64_t GAP_NS = 2000000;    // 2 ms between targets
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

// persistent: one resident block; thread 0 waits for the host flag, stamps, then the block does the same work.
__global__ void flagged(volatile unsigned *flag, unsigned long long *st, unsigned long long *en, float *a, int n) {
    for (int i = 0; i < n; i++) {
        if (threadIdx.x == 0) {
            unsigned c;
            while ((c = flag[0]) != (unsigned)(i + 1) && c != STOP) {}
            st[i] = (c == STOP) ? 0 : gtimer();
        }
        __syncthreads();
        if (flag[0] == STOP) return;
        float v = a[threadIdx.x];
        for (int k = 0; k < 400; k++) v = v * 1.0000001f + 0.5f;
        a[threadIdx.x] = v;
        __syncthreads();
        if (threadIdx.x == 0) en[i] = gtimer();
    }
}

// self-timed: one resident block; thread 0 waits on the GPU clock for each target tick.
__global__ void selftimed(const unsigned long long *tgt, unsigned long long *st, unsigned long long *en, float *a, int n) {
    for (int i = 0; i < n; i++) {
        if (threadIdx.x == 0) { uint64_t t; while ((t = gtimer()) < tgt[i]) ; st[i] = t; }
        __syncthreads();
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

struct Br { int64_t t0, t1; uint64_t g; };
struct Map { double a, b; uint64_t g0; double eps_ns; int used; };

static volatile unsigned *ctl_h; static unsigned *ctl_d;
static unsigned long long *gt_h, *gt_d;
static unsigned seq = 0;

static std::vector<Br> calibrate(cudaStream_t s, int batches = 8, int n = 400) {
    std::vector<Br> out;
    for (int b = 0; b < batches; b++) {
        unsigned base = seq; ctl_h[0] = base; ctl_h[1] = base;
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
        ctl_h[0] = STOP; CK(cudaStreamSynchronize(s));
        for (int i = 0; i < got; i++) out.push_back({tt[i].first, tt[i].second, gt_h[i]});
        seq = base + n + 1;
    }
    return out;
}

// min-filter fit: host_ns = a*(g-g0) + b on the tightest brackets; eps = half median tight width + worst violation
static Map fit(const std::vector<Br> &br) {
    std::vector<double> w; for (auto &x : br) w.push_back((double)(x.t1 - x.t0));
    std::vector<double> ws = w; std::sort(ws.begin(), ws.end());
    double thr = std::max(ws[(size_t)(ws.size() * 0.05)], 1500.0);
    uint64_t g0 = UINT64_MAX; for (auto &x : br) g0 = std::min(g0, x.g);
    double sx = 0, sy = 0, sxx = 0, sxy = 0; int n = 0;
    for (size_t i = 0; i < br.size(); i++) if (w[i] <= thr) {
        double X = (double)(br[i].g - g0), Y = 0.5 * (br[i].t0 + br[i].t1);
        sx += X; sy += Y; sxx += X * X; sxy += X * Y; n++;
    }
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
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));

    CK(cudaHostAlloc((void **)&ctl_h, 64, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void **)&ctl_d, (void *)ctl_h, 0));
    CK(cudaHostAlloc((void **)&gt_h, 400 * 8, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void **)&gt_d, gt_h, 0));
    unsigned *flag_h, *flag_d; CK(cudaHostAlloc((void **)&flag_h, 64, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void **)&flag_d, flag_h, 0));

    int lo, hi; CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    cudaStream_t s, s_load; CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, hi)); CK(cudaStreamCreateWithPriority(&s_load, cudaStreamNonBlocking, lo));

    unsigned long long *st, *en, *tgt; unsigned *ctr; float *a, *big;
    CK(cudaMalloc(&st, N * 8)); CK(cudaMalloc(&en, N * 8)); CK(cudaMalloc(&tgt, N * 8)); CK(cudaMalloc(&ctr, 4));
    CK(cudaMalloc(&a, 1 << 20)); CK(cudaMemset(a, 0, 1 << 20));
    size_t bigN = (size_t)prop.multiProcessorCount * 2048 * 8; CK(cudaMalloc(&big, bigN * 4)); CK(cudaMemset(big, 0, bigN * 4));

    // globaltimer granularity
    unsigned long long *rd; CK(cudaMallocManaged(&rd, 1000 * 8)); gres<<<1, 1>>>(rd, 1000); CK(cudaDeviceSynchronize());
    std::vector<unsigned long long> inc(rd, rd + 1000); std::sort(inc.begin(), inc.end());

    // background AI load (same process, low-priority stream)
    std::atomic<bool> stop{false}; std::thread th;
    if (load == "stream") th = std::thread([&] { while (!stop) { ai_job<<<(unsigned)(bigN / 256), 256, 0, s_load>>>(big, 20000); cudaStreamSynchronize(s_load); } });
    if (load == "stream") { int64_t t = now_ns() + 300000000LL; spin_until(t); }

    std::vector<Br> br = calibrate(s);
    Map m = fit(br);

    // CUDA Graph of one work launch
    cudaGraph_t graph; cudaGraphExec_t gexec;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    work<<<8, 256, 0, s>>>(st, en, ctr, a);
    CK(cudaStreamEndCapture(s, &graph)); CK(cudaGraphInstantiate(&gexec, graph, 0));

    const char *names[] = {"normal launch", "CUDA Graph launch", "persistent block + flag", "GPU self-timed"};
    std::vector<std::vector<double>> errs(4), durs(4);
    unsigned long long hs[N], he[N];
    for (int meth = 0; meth < 4; meth++) {
        CK(cudaMemset(st, 0, N * 8)); CK(cudaMemset(en, 0, N * 8)); CK(cudaMemset(ctr, 0, 4)); CK(cudaStreamSynchronize(s));
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
        } else {
            std::vector<unsigned long long> tg(N);
            for (int i = 0; i < N; i++) tg[i] = (unsigned long long)llround(((double)T[i] - m.b) / m.a) + m.g0;
            CK(cudaMemcpy(tgt, tg.data(), N * 8, cudaMemcpyHostToDevice));
            selftimed<<<1, 256, 0, s>>>(tgt, st, en, a, N);
        }
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(hs, st, N * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(he, en, N * 8, cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++) if (hs[i]) {
            double start_host = m.a * (double)(hs[i] - m.g0) + m.b;
            errs[meth].push_back((start_host - (double)T[i]) / 1000.0);
            if (he[i] > hs[i]) durs[meth].push_back((double)(he[i] - hs[i]) / 1000.0);
        }
    }
    // re-check the clock mapping after the run
    std::vector<Br> br2 = calibrate(s, 4); br.insert(br.end(), br2.begin(), br2.end()); Map m2 = fit(br);
    stop = true; if (th.joinable()) th.join();

    auto pct = [](std::vector<double> v, double p) { if (v.empty()) return NAN; std::sort(v.begin(), v.end()); return v[(size_t)std::min(v.size() - 1.0, p / 100.0 * (v.size() - 1))]; };
    printf("{\"gpu\":\"%s\",\"sm\":%d,\"condition\":\"%s\",\"load\":\"%s\",\"globaltimer_step_ns\":{\"min\":%llu,\"median\":%llu,\"max\":%llu},",
           prop.name, prop.multiProcessorCount, cond.c_str(), load.c_str(), inc[0], inc[500], inc[999]);
    printf("\"clock_bound_us\":%.3f,\"clock_bound_after_us\":%.3f,\"rate_change_ppm\":%.4f,\"methods\":{", m.eps_ns / 1000, m2.eps_ns / 1000, (m2.a / m.a - 1) * 1e6);
    for (int k = 0; k < 4; k++) {
        auto &e = errs[k];
        printf("%s\"%s\":{\"n\":%zu,\"p1_us\":%.2f,\"median_us\":%.2f,\"p99_us\":%.2f,\"max_us\":%.2f,\"work_median_us\":%.2f}",
               k ? "," : "", names[k], e.size(), pct(e, 1), pct(e, 50), pct(e, 99), e.empty() ? NAN : *std::max_element(e.begin(), e.end()), pct(durs[k], 50));
    }
    printf("}}\n");
    return 0;
}
