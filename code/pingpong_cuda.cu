// CUDA port of pingpong.c (Linux). CPU-GPU clock ping-pong over mapped pinned host memory.
// Host: tsc0 -> write seq -> spin until GPU acks -> tsc1. GPU: see seq -> read %globaltimer -> ack.
// Output record layout matches pingpong.c so ana2.py / improve.py work unchanged.
// Build: nvcc -O2 -o pingpong_cuda pingpong_cuda.cu -lpthread
// Run:   ./pingpong_cuda [secs=60] [host_core=0] [fifo=0] [out=pp_cuda.bin]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <x86intrin.h>

#define STOP 0xFFFFFFFEu
#define BATCH 400
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

struct rec_t { uint32_t phase, ok; uint64_t tsc0, tsc1, gpu; };

__device__ __forceinline__ uint64_t gtimer() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__global__ void pp(volatile unsigned *ctl, unsigned long long *gt, unsigned n, unsigned base) {
    for (unsigned i = 0; i < n; i++) {
        unsigned want = base + i + 1, c;
        unsigned long long spins = 0;
        while ((c = ctl[0]) != want) { if (c == STOP || ++spins > 4000000000ull) return; }
        gt[i] = gtimer();
        __threadfence_system();
        ctl[1] = want;
        __threadfence_system();
    }
}

__global__ void burn(float *a) {
    float v = a[blockIdx.x * blockDim.x + threadIdx.x];
    for (int i = 0; i < 4000; i++) v = v * 1.0000001f + 0.5f;
    a[blockIdx.x * blockDim.x + threadIdx.x] = v;
}

// globaltimer update granularity: distinct increments seen by one thread spinning on it
__global__ void gres(unsigned long long *out, int n) {
    uint64_t last = gtimer(); int k = 0;
    while (k < n) { uint64_t t = gtimer(); if (t != last) { out[k++] = t - last; last = t; } }
}

static inline uint64_t tsc() { _mm_lfence(); uint64_t t = __rdtsc(); _mm_lfence(); return t; }
static double now_s() { timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static void pin(int core) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(core, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); }

static volatile unsigned *ctl_h; static unsigned *ctl_d;
static unsigned long long *gt_h, *gt_d;
static cudaStream_t sp, sb;
static uint32_t seq;

static size_t run_phase(uint32_t ph, double secs, int sleep_us, std::vector<rec_t> &out) {
    size_t n0 = out.size(); double end = now_s() + secs;
    while (now_s() < end) {
        uint32_t base = seq, nb = BATCH;
        ctl_h[0] = base; ctl_h[1] = base;
        pp<<<1, 1, 0, sp>>>((volatile unsigned *)ctl_d, gt_d, nb, base);
        size_t first = out.size(); bool aborted = false;
        for (uint32_t i = 0; i < nb; i++) {
            uint32_t want = base + i + 1;
            if (sleep_us > 0) usleep(sleep_us);
            uint64_t t0 = tsc(); ctl_h[0] = want;
            uint64_t lim = t0 + 5000000000ull; int ok = 1;  // ~1-2 s timeout
            while (ctl_h[1] != want) { if (__rdtsc() > lim) { ok = 0; break; } }
            uint64_t t1 = tsc();
            out.push_back({ph, (uint32_t)ok, t0, t1, 0});
            if (!ok) { aborted = true; break; }
        }
        ctl_h[0] = STOP; CK(cudaStreamSynchronize(sp));
        for (size_t k = first; k < out.size(); k++) if (out[k].ok) out[k].gpu = gt_h[k - first];
        seq = base + nb + 1;
        if (aborted) fprintf(stderr, "phase %u: timeout, batch aborted\n", ph);
    }
    return out.size() - n0;
}

int main(int argc, char **argv) {
    double secs = argc > 1 ? atof(argv[1]) : 60;
    int hcore = argc > 2 ? atoi(argv[2]) : 0;
    int fifo = argc > 3 ? atoi(argv[3]) : 0;
    const char *outf = argc > 4 ? argv[4] : "pp_cuda.bin";

    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d), %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    CK(cudaHostAlloc((void **)&ctl_h, 256, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void **)&ctl_d, (void *)ctl_h, 0));
    CK(cudaHostAlloc((void **)&gt_h, BATCH * 8, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void **)&gt_d, gt_h, 0));
    CK(cudaStreamCreateWithFlags(&sp, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));
    float *big; CK(cudaMalloc(&big, (1 << 22) * sizeof(float)));

    // globaltimer granularity
    unsigned long long *rd; CK(cudaMallocManaged(&rd, 1000 * 8));
    gres<<<1, 1>>>(rd, 1000); CK(cudaDeviceSynchronize());
    std::vector<unsigned long long> inc(rd, rd + 1000); std::sort(inc.begin(), inc.end());
    printf("globaltimer increments ns: min %llu median %llu max %llu\n", inc[0], inc[500], inc[999]);

    pin(hcore);
    if (fifo) { sched_param sp_{}; sp_.sched_priority = 90; if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp_)) puts("SCHED_FIFO failed (need root)"); }

    std::vector<rec_t> r; r.reserve(20000000);
    double qa = now_s(); uint64_t ta = tsc();
    const char *names[] = {"tight", "sleepy", "cpu_load", "gpu_load", "tight2"};
    int ncpu = (int)std::thread::hardware_concurrency();
    for (uint32_t ph = 0; ph < 5; ph++) {
        std::atomic<bool> stop{false}; std::vector<std::thread> th;
        if (ph == 2) for (int c = 0; c < ncpu; c++) if (c != hcore) th.emplace_back([&stop, c] { pin(c); volatile uint64_t x = 0; while (!stop) x++; });
        if (ph == 3) th.emplace_back([&stop, big] { while (!stop) { burn<<<(1 << 22) / 256, 256, 0, sb>>>(big); cudaStreamSynchronize(sb); } });
        size_t m = run_phase(ph, ph == 1 ? secs / 2 : secs, ph == 1 ? 1000 : 0, r);
        stop = true; for (auto &t : th) t.join();
        size_t ok = 0; for (size_t k = r.size() - m; k < r.size(); k++) ok += r[k].ok;
        printf("%-9s %zu samples, %zu ok\n", names[ph], m, ok); fflush(stdout);
    }
    double qb = now_s(); uint64_t tb = tsc();
    double tsc_hz = (double)(tb - ta) / (qb - qa);
    printf("TSC %.3f MHz (vs CLOCK_MONOTONIC_RAW)\n", tsc_hz / 1e6);
    FILE *fo = fopen(outf, "wb"); fwrite(&tsc_hz, 8, 1, fo); fwrite(r.data(), sizeof(rec_t), r.size(), fo); fclose(fo);
    return 0;
}
