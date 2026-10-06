// gputrace: map the latencies of a GPU's scheduling path with hardware timers, from every angle one process
// can reach, on one time axis with a hard host<->GPU bound. One run = one strategy:
//
//   launch       host launch call -> first instruction on the GPU (by idle gap, queue depth, graph vs stream)
//   notify       kernel end -> host sees it (mapped flag poll, cudaEventQuery poll, cudaStreamSynchronize)
//   dispatch     one kernel's blocks: which SM, in what order, how fast, how many waves, tail
//   concurrency  two streams (optionally priorities): time for the second kernel to get SMs, co-residency,
//                whether running blocks were preempted (gaps in their own timer reads)
//   clocks       SM clock vs %globaltimer over time while load is applied: boost/ramp/throttle
//   copy         H2D/D2H memcpy latency seen by the host; PCIe read latency seen by the GPU (dependent loads)
//   timeslice    another process (role hog) shares the GPU: when did our resident thread not run (quanta),
//                which SMs did the hog get (MPS partition)
//
// Every strategy runs the clock sync (clocksync.cuh, the pcieclock samplers) before and after, so the
// analysis can put %globaltimer on CLOCK_MONOTONIC_RAW with a bound that needs no symmetry assumption.
// Output: PREFIX.gpu.bin, PREFIX.host.bin, PREFIX.json, PREFIX.{pre,post}.{classic,up,down}.bin.
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "host_time.h"
#include "json_writer.h"
#include "gputrace.cuh"
#include "clocksync.cuh"

using namespace sb;
using namespace sb::gt;

#define LAUNCH_CK() do { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { \
    fprintf(stderr, "gputrace: kernel launch failed: %s\n", cudaGetErrorString(e_)); exit(3); } } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "gputrace: %s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

// ------------------------------------------------------------------ kernels

// Every block: trace_begin, spin dur_ns of GPU time (all threads), trace_end. flag (optional, mapped host
// memory): thread 0 of the last block to finish... no: of every block, writes its end time (notify uses 1 block).
__constant__ uint32_t c_timer_every = 32;
__global__ void k_spin(TraceDev td, uint32_t kid, uint32_t tag, uint64_t dur_ns, volatile uint64_t *flag) {
    __shared__ BlockTrace bt;
    if (threadIdx.x == 0) trace_begin(td, bt);
    __syncthreads();
    uint64_t gap; uint32_t it;
    spin_ns(bt.g0, dur_ns, gap, it, c_timer_every);
    __syncthreads();
    if (threadIdx.x == 0) {
        trace_end(td, bt, kid, tag, gap, it, 0);
        if (flag) { st_sys(flag, gtimer()); __threadfence_system(); }
    }
}

// One thread samples (%globaltimer, clock64) every sample_ns for dur_ns: SM clock frequency over time.
__global__ void k_sampler(TraceDev td, uint32_t kid, uint64_t dur_ns, uint64_t sample_ns) {
    if (threadIdx.x != 0) return;
    uint64_t start = gtimer(), next = start;
    for (;;) {
        uint64_t g = gtimer();
        if (g >= next) {
            uint64_t c = clock64();
            unsigned s = atomicAdd(td.count, 1u);
            if (s < td.capacity) {
                GpuRec r{};
                r.g_begin = r.g_end = g; r.clk_begin = r.clk_end = c; r.smid = smid(); r.kernel_id = kid; r.tag = 1;
                td.recs[s] = r;
            }
            next += sample_ns;
            if (g >= start + dur_ns) break;
        }
    }
}

// One resident thread for dur_ns: records every interval in which it did not observe the timer for more
// than gap_ns (its context was switched out or it was preempted). Final record (tag 3) carries the window.
__global__ void k_resident(TraceDev td, uint32_t kid, uint64_t dur_ns, uint64_t gap_ns) {
    if (threadIdx.x != 0) return;
    uint64_t start = gtimer(), last = start, maxgap = 0;
    uint32_t n = 0, it = 0;
    for (;;) {
        uint64_t t = gtimer();
        it++;
        if (t - last > gap_ns) {
            n++;
            if (t - last > maxgap) maxgap = t - last;
            unsigned s = atomicAdd(td.count, 1u);
            if (s < td.capacity) {
                GpuRec r{};
                r.g_begin = last; r.g_end = t; r.smid = smid(); r.kernel_id = kid; r.tag = 2; r.block = n;
                td.recs[s] = r;
            }
        }
        last = t;
        if (t >= start + dur_ns) break;
    }
    unsigned s = atomicAdd(td.count, 1u);
    if (s < td.capacity) {
        GpuRec r{};
        r.g_begin = start; r.g_end = last; r.max_gap_ns = maxgap; r.smid = smid(); r.kernel_id = kid; r.tag = 3;
        r.n_iters = it; r.block = n;
        td.recs[s] = r;
    }
}

// One thread: n dependent 64-bit loads from mapped host memory (each address comes from the previous value),
// one record per load with clock64 and %globaltimer around it. buf[j*8] == next index, filled by the host.
__global__ void k_pcie_read(TraceDev td, uint32_t kid, const volatile uint64_t *buf, uint32_t n) {
    if (threadIdx.x != 0) return;
    uint64_t idx = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t g0 = gtimer();
        uint64_t c0 = clock64();
        uint64_t v = ld_sys(buf + (idx & 1023) * 8);
        if (v == ~0ull) asm volatile("trap;");   // consume the value: the next reads wait for the load
        uint64_t c1 = clock64();
        uint64_t g1 = gtimer();
        idx = v;
        unsigned s = atomicAdd(td.count, 1u);
        if (s < td.capacity) {
            GpuRec r{};
            r.g_begin = g0; r.g_end = g1; r.clk_begin = c0; r.clk_end = c1; r.smid = smid(); r.kernel_id = kid;
            r.tag = 4; r.block = i;
            td.recs[s] = r;
        }
    }
}

// ------------------------------------------------------------------ host side

struct Args {
    std::string strategy = "launch", out = "gputrace", role = "main", blocks = "1,sm,2sm,8sm,32sm";
    int gpu = 0, core = -1, clock_core = -1, iters = 2000, depth = 1, graph = 0, threads = 256, reps = 5;
    int priority = 0, sync_rounds = 3, sync_per_phase = 1000, waves = 8, nosync = 0, idle_spin = 0, timer_every = 32;
    double idle_us = 0, dur_us = 200, offset_us = 500, seconds = 10, sample_us = 100, gap_us = 20, dur_b_us = 200;
    double launch_dur_us = 10, hog_dur_us = 5000;
    size_t smem = 0, capacity = 1u << 20;
};

static Args g_args;
static std::vector<HostEvent> g_ev;
static uint32_t g_kid = 0;

static inline void ev(uint32_t type, uint32_t kid, uint64_t a = 0, uint64_t b = 0) {
    g_ev.push_back(HostEvent{now_ns(), type, kid, a, b});
}

struct Dev {
    TraceDev td{};
    cudaDeviceProp prop{};
    int sms = 0;
    void init(size_t capacity) {
        CK(cudaGetDeviceProperties(&prop, g_args.gpu));
        sms = prop.multiProcessorCount;
        CK(cudaMalloc(&td.recs, sizeof(GpuRec) * capacity));
        CK(cudaMalloc(&td.count, sizeof(unsigned)));
        CK(cudaMemset(td.count, 0, sizeof(unsigned)));
        td.capacity = (unsigned)capacity;
    }
    unsigned count() const { unsigned c; CK(cudaMemcpy(&c, td.count, sizeof c, cudaMemcpyDeviceToHost)); return c; }
};
static Dev g_dev;

static std::vector<int> parse_blocks(const std::string &s, int sms) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        std::string tok = s.substr(i, j - i);
        int mult = 1;
        if (tok.size() >= 2 && tok.compare(tok.size() - 2, 2, "sm") == 0) { mult = sms; tok = tok.substr(0, tok.size() - 2); if (tok.empty()) tok = "1"; }
        out.push_back(std::max(1, atoi(tok.c_str()) * mult));
        i = j + 1;
    }
    return out;
}

// idle_spin keeps the CPU busy-waiting (no C-state / frequency drop on the host side) so that the launch
// cost after a GPU-idle gap can be separated from the cost of a sleeping CPU.
static void idle(double us) {
    if (us <= 0) return;
    if (g_args.idle_spin) spin_until(now_ns() + (int64_t)(us * 1000));
    else sleep_until_raw(now_ns() + (int64_t)(us * 1000));
    ev(EV_IDLE_END, 0, (uint64_t)(us * 1000));
}

static cudaStream_t make_stream(int prio_class) {   // 0 default, -1 lowest, +1 highest
    int least, greatest;
    CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    cudaStream_t s;
    int p = prio_class == 0 ? 0 : (prio_class < 0 ? least : greatest);
    CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, p));
    return s;
}

// ---- strategies

static void s_launch() {
    cudaStream_t st = make_stream(0);
    const uint64_t dur = (uint64_t)(g_args.launch_dur_us * 1000);
    cudaGraphExec_t gexec = nullptr;
    if (g_args.graph) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        k_spin<<<1, 32, 0, st>>>(g_dev.td, 0xffffffffu, 0, dur, nullptr);
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&gexec, g, 0));
        CK(cudaGraphLaunch(gexec, st));   // warm (upload)
        CK(cudaStreamSynchronize(st));
        CK(cudaMemset(g_dev.td.count, 0, sizeof(unsigned)));   // drop the warm-up record
    }
    for (int i = 0; i < g_args.iters; i++) {
        idle(g_args.idle_us);
        for (int d = 0; d < g_args.depth; d++) {
            uint32_t kid = ++g_kid;
            if (gexec) {
                ev(EV_GRAPH_ENTER, kid, 1, 32);
                CK(cudaGraphLaunch(gexec, st));
                ev(EV_GRAPH_RETURN, kid);
            } else {
                ev(EV_LAUNCH_ENTER, kid, 1, 32);
                k_spin<<<1, 32, 0, st>>>(g_dev.td, kid, 0, dur, nullptr);
                ev(EV_LAUNCH_RETURN, kid);
                LAUNCH_CK();
            }
        }
        ev(EV_SYNC_ENTER, g_kid);
        CK(cudaStreamSynchronize(st));
        ev(EV_SYNC_RETURN, g_kid);
    }
    if (gexec) cudaGraphExecDestroy(gexec);
}

static void s_notify() {
    cudaStream_t st = make_stream(0);
    volatile uint64_t *flag; uint64_t *flag_d;
    CK(cudaHostAlloc((void **)&flag, 64, cudaHostAllocMapped));
    *flag = 0;
    CK(cudaHostGetDevicePointer((void **)&flag_d, (void *)flag, 0));
    cudaEvent_t e;
    CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    const uint64_t dur = (uint64_t)(g_args.dur_us * 1000);
    uint64_t seen = 0;
    for (int i = 0; i < g_args.iters; i++) {
        int mode = i % 3;   // 0 flag, 1 event, 2 sync
        uint32_t kid = ++g_kid;
        idle(g_args.idle_us);
        ev(EV_LAUNCH_ENTER, kid, 1, 32);
        k_spin<<<1, 32, 0, st>>>(g_dev.td, kid, (uint32_t)mode, dur, mode == 0 ? flag_d : nullptr);
        ev(EV_LAUNCH_RETURN, kid);
        LAUNCH_CK();
        if (mode == 1) CK(cudaEventRecord(e, st));
        if (mode == 0) {
            uint64_t v, polls = 0;
            while ((v = *flag) == seen) polls++;
            ev(EV_FLAG_SEEN, kid, v, polls);
            seen = v;
        } else if (mode == 1) {
            uint64_t polls = 0;
            while (cudaEventQuery(e) != cudaSuccess) polls++;
            ev(EV_EVENT_SEEN, kid, polls);
        } else {
            ev(EV_SYNC_ENTER, kid);
            CK(cudaStreamSynchronize(st));
            ev(EV_SYNC_RETURN, kid);
        }
        CK(cudaStreamSynchronize(st));
    }
    cudaFreeHost((void *)flag);
}

static void s_dispatch() {
    cudaStream_t st = make_stream(0);
    const uint64_t dur = (uint64_t)(g_args.dur_us * 1000);
    for (int B : parse_blocks(g_args.blocks, g_dev.sms)) {
        for (int r = 0; r < g_args.reps; r++) {
            uint32_t kid = ++g_kid;
            idle(g_args.idle_us);
            ev(EV_LAUNCH_ENTER, kid, (uint64_t)B, (uint64_t)g_args.threads);
            k_spin<<<B, g_args.threads, g_args.smem, st>>>(g_dev.td, kid, (uint32_t)r, dur, nullptr);
            ev(EV_LAUNCH_RETURN, kid);
            LAUNCH_CK();
            ev(EV_SYNC_ENTER, kid);
            CK(cudaStreamSynchronize(st));
            ev(EV_SYNC_RETURN, kid);
        }
    }
}

static void s_concurrency() {
    cudaStream_t s0 = make_stream(g_args.priority ? -1 : 0), s1 = make_stream(g_args.priority ? +1 : 0);
    const uint64_t dur_a = (uint64_t)(g_args.dur_us * 1000), dur_b = (uint64_t)(g_args.dur_b_us * 1000);
    const int B_a = g_dev.sms * g_args.waves, B_b = g_dev.sms;
    for (int r = 0; r < g_args.reps; r++) {
        uint32_t ka = ++g_kid;
        idle(g_args.idle_us);
        ev(EV_LAUNCH_ENTER, ka, (uint64_t)B_a, (uint64_t)g_args.threads);
        k_spin<<<B_a, g_args.threads, g_args.smem, s0>>>(g_dev.td, ka, 0, dur_a, nullptr);
        ev(EV_LAUNCH_RETURN, ka);
        LAUNCH_CK();
        spin_until(now_ns() + (int64_t)(g_args.offset_us * 1000));
        uint32_t kb = ++g_kid;
        ev(EV_LAUNCH_ENTER, kb, (uint64_t)B_b, (uint64_t)g_args.threads);
        k_spin<<<B_b, g_args.threads, g_args.smem, s1>>>(g_dev.td, kb, 1, dur_b, nullptr);
        ev(EV_LAUNCH_RETURN, kb);
        LAUNCH_CK();
        ev(EV_SYNC_ENTER, kb);
        CK(cudaStreamSynchronize(s1));
        ev(EV_SYNC_RETURN, kb);
        ev(EV_SYNC_ENTER, ka);
        CK(cudaStreamSynchronize(s0));
        ev(EV_SYNC_RETURN, ka);
    }
}

static void s_clocks() {
    cudaStream_t s0 = make_stream(0), s1 = make_stream(0);
    const uint64_t dur = (uint64_t)(g_args.seconds * 1e9), sample = (uint64_t)(g_args.sample_us * 1000);
    uint32_t ks = ++g_kid;
    ev(EV_LAUNCH_ENTER, ks, 1, 32);
    k_sampler<<<1, 32, 0, s0>>>(g_dev.td, ks, dur, sample);
    ev(EV_LAUNCH_RETURN, ks);
    LAUNCH_CK();
    // idle third, loaded third, idle third
    spin_until(now_ns() + (int64_t)(dur / 3));
    uint32_t kl = ++g_kid;
    ev(EV_MARK, kl, 1, 0);   // load start
    ev(EV_LAUNCH_ENTER, kl, (uint64_t)g_dev.sms * 4, (uint64_t)g_args.threads);
    k_spin<<<g_dev.sms * 4, g_args.threads, 0, s1>>>(g_dev.td, kl, 5, dur / 3, nullptr);
    ev(EV_LAUNCH_RETURN, kl);
    CK(cudaStreamSynchronize(s1));
    ev(EV_MARK, kl, 2, 0);   // load end
    CK(cudaStreamSynchronize(s0));
    ev(EV_SYNC_RETURN, ks);
}

// After an idle gap, one block samples (%globaltimer, clock64) every sample_us for dur_us: the SM clock
// ramp seen by the first kernel after idle. Repeated `reps` times (tag 1 samples, kernel_id per rep).
static void s_ramp() {
    cudaStream_t st = make_stream(0);
    const uint64_t dur = (uint64_t)(g_args.dur_us * 1000), sample = (uint64_t)(g_args.sample_us * 1000);
    for (int r = 0; r < g_args.reps; r++) {
        uint32_t kid = ++g_kid;
        idle(g_args.idle_us);
        ev(EV_LAUNCH_ENTER, kid, 1, 32);
        k_sampler<<<1, 32, 0, st>>>(g_dev.td, kid, dur, sample);
        ev(EV_LAUNCH_RETURN, kid);
        LAUNCH_CK();
        ev(EV_SYNC_ENTER, kid);
        CK(cudaStreamSynchronize(st));
        ev(EV_SYNC_RETURN, kid);
    }
}

static void s_copy() {
    cudaStream_t st = make_stream(0);
    const size_t sizes[] = {8, 4096, 65536, 1u << 20};
    uint8_t *h, *d;
    CK(cudaHostAlloc((void **)&h, 1u << 20, cudaHostAllocDefault));
    CK(cudaMalloc(&d, 1u << 20));
    memset(h, 1, 1u << 20);
    cudaEvent_t e;
    CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    const int reps = std::max(10, g_args.iters / 10);
    for (size_t n : sizes) {
        for (int dir = 0; dir < 2; dir++) {
            for (int r = 0; r < reps; r++) {
                uint32_t kid = ++g_kid;
                idle(g_args.idle_us);
                ev(EV_COPY_ENTER, kid, n, dir);
                CK(dir == 0 ? cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, st)
                            : cudaMemcpyAsync(h, d, n, cudaMemcpyDeviceToHost, st));
                ev(EV_COPY_RETURN, kid, n, dir);
                CK(cudaEventRecord(e, st));
                while (cudaEventQuery(e) != cudaSuccess) {}
                ev(EV_COPY_DONE, kid, n, dir);
            }
        }
    }
    // GPU-side PCIe read latency: dependent loads from mapped host memory, 1024 lines of 64 bytes
    uint64_t *buf, *buf_d;
    CK(cudaHostAlloc((void **)&buf, 1024 * 64, cudaHostAllocMapped));
    for (uint64_t j = 0; j < 1024; j++) buf[j * 8] = (j * 613 + 1) & 1023;   // a permutation step
    CK(cudaHostGetDevicePointer((void **)&buf_d, buf, 0));
    uint32_t kid = ++g_kid;
    ev(EV_LAUNCH_ENTER, kid, 1, 32);
    k_pcie_read<<<1, 32, 0, st>>>(g_dev.td, kid, buf_d, (uint32_t)std::max(100, g_args.iters));
    ev(EV_LAUNCH_RETURN, kid);
    LAUNCH_CK();
    CK(cudaStreamSynchronize(st));
    ev(EV_SYNC_RETURN, kid);
    cudaFreeHost(buf); cudaFreeHost(h); cudaFree(d);
}

static pid_t spawn_hog(double seconds) {
    pid_t pid = fork();
    if (pid != 0) return pid;
    std::string sec = std::to_string(seconds), gpu = std::to_string(g_args.gpu), out = g_args.out + ".hog";
    std::string sr = std::to_string(g_args.sync_rounds), sp = std::to_string(g_args.sync_per_phase);
    execl("/proc/self/exe", "gputrace", "--role", "hog", "--seconds", sec.c_str(), "--gpu", gpu.c_str(),
          "--out", out.c_str(), "--sync-rounds", sr.c_str(), "--sync-per-phase", sp.c_str(), "--threads",
          std::to_string(g_args.threads).c_str(), "--hog-dur-us", std::to_string(g_args.hog_dur_us).c_str(), (char *)nullptr);
    _exit(127);
}

static void s_timeslice() {
    cudaStream_t st = make_stream(0);
    const uint64_t dur = (uint64_t)(g_args.seconds * 1e9), gap = (uint64_t)(g_args.gap_us * 1000);
    uint32_t kid = ++g_kid;
    ev(EV_LAUNCH_ENTER, kid, 1, 32);
    k_resident<<<1, 32, 0, st>>>(g_dev.td, kid, dur, gap);
    ev(EV_LAUNCH_RETURN, kid);
    LAUNCH_CK();
    spin_until(now_ns() + 500000000LL);   // 0.5 s alone first
    ev(EV_MARK, kid, 1, 0);               // hog start
    pid_t hog = spawn_hog(std::max(1.0, g_args.seconds - 2.0));
    CK(cudaStreamSynchronize(st));
    ev(EV_SYNC_RETURN, kid);
    int status = 0;
    waitpid(hog, &status, 0);
    ev(EV_MARK, kid, 2, (uint64_t)status);
}

// role hog: keep the GPU busy with wide spinning kernels for `seconds`, tracing its own blocks.
static void s_hog() {
    cudaStream_t st = make_stream(0);
    const uint64_t dur = (uint64_t)(g_args.hog_dur_us * 1000);   // blocks longer than the time-slice quantum get interrupted
    int64_t until = now_ns() + (int64_t)(g_args.seconds * 1e9);
    while (now_ns() < until) {
        uint32_t kid = ++g_kid;
        ev(EV_LAUNCH_ENTER, kid, (uint64_t)g_dev.sms * 2, (uint64_t)g_args.threads);
        k_spin<<<g_dev.sms * 2, g_args.threads, 0, st>>>(g_dev.td, kid, 9, dur, nullptr);
        ev(EV_LAUNCH_RETURN, kid);
        LAUNCH_CK();
        CK(cudaStreamSynchronize(st));
        ev(EV_SYNC_RETURN, kid);
    }
}

// ---- main

static void usage() {
    fprintf(stderr,
            "usage: gputrace --strategy launch|notify|dispatch|concurrency|clocks|ramp|copy|timeslice --out PREFIX\n"
            "  [--gpu N] [--core C] [--clock-core C] [--iters N] [--idle-us X] [--idle-spin 0|1] [--depth D] [--graph 0|1]\n"
            "  [--blocks LIST e.g. 1,sm,2sm,8sm] [--threads T] [--dur-us D] [--dur-b-us D] [--smem BYTES] [--reps R]\n"
            "  [--waves W] [--offset-us X] [--priority 0|1] [--seconds S] [--sample-us X] [--gap-us X]\n"
            "  [--timer-every N (spin reads %%globaltimer every N iterations, clock64 otherwise)] [--launch-dur-us D] [--hog-dur-us D] [--sync-rounds R] [--sync-per-phase N] [--no-sync 1] [--capacity N]\n");
    exit(2);
}

int main(int argc, char **argv) {
    Args &a = g_args;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string f = argv[i], v = argv[i + 1];
        if (f == "--strategy") a.strategy = v;
        else if (f == "--out") a.out = v;
        else if (f == "--role") a.role = v;
        else if (f == "--gpu") a.gpu = atoi(v.c_str());
        else if (f == "--core") a.core = atoi(v.c_str());
        else if (f == "--clock-core") a.clock_core = atoi(v.c_str());
        else if (f == "--iters") a.iters = atoi(v.c_str());
        else if (f == "--idle-us") a.idle_us = atof(v.c_str());
        else if (f == "--idle-spin") a.idle_spin = atoi(v.c_str());
        else if (f == "--timer-every") a.timer_every = std::max(1, atoi(v.c_str()));
        else if (f == "--depth") a.depth = std::max(1, atoi(v.c_str()));
        else if (f == "--graph") a.graph = atoi(v.c_str());
        else if (f == "--blocks") a.blocks = v;
        else if (f == "--threads") a.threads = atoi(v.c_str());
        else if (f == "--dur-us") a.dur_us = atof(v.c_str());
        else if (f == "--dur-b-us") a.dur_b_us = atof(v.c_str());
        else if (f == "--launch-dur-us") a.launch_dur_us = atof(v.c_str());
        else if (f == "--hog-dur-us") a.hog_dur_us = atof(v.c_str());
        else if (f == "--smem") a.smem = (size_t)atol(v.c_str());
        else if (f == "--reps") a.reps = atoi(v.c_str());
        else if (f == "--waves") a.waves = atoi(v.c_str());
        else if (f == "--offset-us") a.offset_us = atof(v.c_str());
        else if (f == "--priority") a.priority = atoi(v.c_str());
        else if (f == "--seconds") a.seconds = atof(v.c_str());
        else if (f == "--sample-us") a.sample_us = atof(v.c_str());
        else if (f == "--gap-us") a.gap_us = atof(v.c_str());
        else if (f == "--sync-rounds") a.sync_rounds = atoi(v.c_str());
        else if (f == "--sync-per-phase") a.sync_per_phase = atoi(v.c_str());
        else if (f == "--no-sync") a.nosync = atoi(v.c_str());
        else if (f == "--capacity") a.capacity = (size_t)atol(v.c_str());
        else usage();
    }
    if (a.role == "hog") a.strategy = "hog";
    CK(cudaSetDevice(a.gpu));
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    if (a.core >= 0 && !pin_thread(a.core)) fprintf(stderr, "gputrace: could not pin to core %d\n", a.core);
    g_dev.init(a.capacity);
    { uint32_t te = (uint32_t)a.timer_every; CK(cudaMemcpyToSymbol(c_timer_every, &te, sizeof te)); }
    if (a.smem > 0) CK(cudaFuncSetAttribute(k_spin, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)a.smem));
    g_ev.reserve((size_t)a.iters * (size_t)std::max(a.depth, 1) * 4 + 1024);
    // warm the context and the kernels so the first traced launch is not a module load
    k_spin<<<1, 32>>>(g_dev.td, 0, 0, 1000, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(g_dev.td.count, 0, sizeof(unsigned)));

    ClockSamples pre, post;
    std::string err;
    bool pre_ok = true, post_ok = true;
    const int64_t t_start = now_ns();
    if (!a.nosync) pre_ok = clocksync(a.sync_rounds, a.sync_per_phase, a.clock_core, pre, &err);
    if (!pre_ok) fprintf(stderr, "gputrace: pre clocksync failed: %s\n", err.c_str());
    ev(EV_MARK, 0, 100, 0);   // strategy start
    const int64_t t_s0 = now_ns();
    if (a.strategy == "launch") s_launch();
    else if (a.strategy == "notify") s_notify();
    else if (a.strategy == "dispatch") s_dispatch();
    else if (a.strategy == "concurrency") s_concurrency();
    else if (a.strategy == "clocks") s_clocks();
    else if (a.strategy == "copy") s_copy();
    else if (a.strategy == "ramp") s_ramp();
    else if (a.strategy == "timeslice") s_timeslice();
    else if (a.strategy == "hog") s_hog();
    else usage();
    CK(cudaDeviceSynchronize());
    const int64_t t_s1 = now_ns();
    ev(EV_MARK, 0, 101, 0);   // strategy end
    if (!a.nosync) post_ok = clocksync(a.sync_rounds, a.sync_per_phase, a.clock_core, post, &err);
    if (!post_ok) fprintf(stderr, "gputrace: post clocksync failed: %s\n", err.c_str());
    const int64_t t_end = now_ns();

    unsigned count = g_dev.count(), n_rec = std::min<unsigned>(count, (unsigned)a.capacity);
    std::vector<GpuRec> recs(n_rec);
    if (n_rec) CK(cudaMemcpy(recs.data(), g_dev.td.recs, sizeof(GpuRec) * n_rec, cudaMemcpyDeviceToHost));
    std::sort(recs.begin(), recs.end(), [](const GpuRec &x, const GpuRec &y) { return x.g_begin < y.g_begin; });
    FILE *f = fopen((a.out + ".gpu.bin").c_str(), "wb");
    if (!f) { perror("gputrace: gpu.bin"); return 1; }
    fwrite(recs.data(), sizeof(GpuRec), recs.size(), f);
    fclose(f);
    f = fopen((a.out + ".host.bin").c_str(), "wb");
    if (!f) { perror("gputrace: host.bin"); return 1; }
    fwrite(g_ev.data(), sizeof(HostEvent), g_ev.size(), f);
    fclose(f);
    if (!a.nosync) { pre.dump(a.out + ".pre"); post.dump(a.out + ".post"); }

    int clock_khz = 0, mem_khz = 0, compute_mode = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, a.gpu);
    cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, a.gpu);
    cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, a.gpu);
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    const char *mps = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
    Json j;
    j.add("strategy", a.strategy).add("role", a.role).add("gpu", g_dev.prop.name).add("sms", g_dev.sms)
        .add("cc", std::to_string(g_dev.prop.major) + "." + std::to_string(g_dev.prop.minor))
        .add("pci_bus", g_dev.prop.pciBusID).add("clock_khz", clock_khz).add("mem_clock_khz", mem_khz)
        .add("compute_mode", compute_mode).add("mps_pct", mps ? mps : "").add("driver", drv).add("runtime", rt)
        .add("host", hostname()).add("time", iso_utc_now()).add("core", a.core).add("clock_core", a.clock_core)
        .add("iters", a.iters).add("idle_us", a.idle_us).add("idle_spin", a.idle_spin).add("timer_every", a.timer_every).add("depth", a.depth).add("graph", a.graph)
        .add("blocks", a.blocks).add("threads", a.threads).add("dur_us", a.dur_us).add("dur_b_us", a.dur_b_us)
        .add("smem", (long long)a.smem).add("reps", a.reps).add("waves", a.waves).add("offset_us", a.offset_us)
        .add("priority", a.priority).add("launch_dur_us", a.launch_dur_us).add("hog_dur_us", a.hog_dur_us).add("seconds", a.seconds).add("sample_us", a.sample_us).add("gap_us", a.gap_us)
        .add("sync_rounds", a.sync_rounds).add("sync_per_phase", a.sync_per_phase)
        .add("n_gpu_recs", (long long)n_rec).add("gpu_recs_dropped", (long long)(count > n_rec ? count - n_rec : 0))
        .add("n_host_events", (long long)g_ev.size()).add("kernels", (long long)g_kid)
        .add("pre_ok", pre_ok).add("post_ok", post_ok).add("bad_classic_pre", pre.bad_classic).add("bad_classic_post", post.bad_classic)
        .add("t_start", (long long)t_start).add("t_strategy_start", (long long)t_s0).add("t_strategy_end", (long long)t_s1)
        .add("t_end", (long long)t_end).add("seconds_total", (t_end - t_start) * 1e-9);
    std::string ticks;
    for (size_t i = 0; i < pre.tick_steps.size(); i++) ticks += (i ? "," : "") + std::to_string((long long)pre.tick_steps[i]);
    j.add_raw("timer_edge_steps_ns", "[" + ticks + "]");
    f = fopen((a.out + ".json").c_str(), "w");
    fprintf(f, "%s\n", j.str().c_str());
    fclose(f);
    printf("gputrace %s: %s sms=%d recs=%u%s events=%zu kernels=%u in %.2f s (sync pre %s post %s)\n", a.strategy.c_str(),
           g_dev.prop.name, g_dev.sms, n_rec, count > n_rec ? "(dropped)" : "", g_ev.size(), g_kid,
           (t_end - t_start) * 1e-9, pre_ok ? "ok" : "FAIL", post_ok ? "ok" : "FAIL");
    return 0;
}
