// Lockstep driver: who starts the 5G slot on time, the CPU or the GPU itself?
//
//   --mode cpu   the host spins until target T_k (CLOCK_MONOTONIC_RAW) and calls cudaGraphLaunch   (today's design)
//   --mode gpu   a resident 1-thread "slot executive" kernel watches %globaltimer and launches the slot
//                graph from the device (CUDA device graph launch) at T_k converted to GPU time; the CPU
//                is not on the timing path at all
//
// Both modes run the same captured slot graph (SlotPipeline::enqueue + a completion kernel) once per
// period, never overlapping: a boundary that arrives while the previous slot is still running is
// skipped and counted. Per slot: g_target (GPU time of T_k), g_launch (GPU time the launch was issued;
// cpu mode: host t0 mapped), g0/g1 (slot start/end stamps, GPU clock). Start error = g0 - g_target.
//
// Optional in-process load (--load stream): a thread keeps a low-priority stream busy with large FMA
// kernels, i.e. an AI job in the same CUDA context. Separate-process neighbours and MPS are set up by
// scripts/lockstep_matrix.sh around this binary.
//
// Build (Makefile target bin/lockstep_driver): nvcc -rdc=true ... -lcudadevrt
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "clock_fit.h"
#include "cuda_check.h"
#include "host_time.h"
#include "json_writer.h"
#include "slot_pipeline.h"

using namespace sb;

namespace {

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

struct LsRec {               // 56 bytes, little endian, written raw to --raw
    unsigned long long slot;
    long long t_target;      // host ns (CLOCK_MONOTONIC_RAW) of boundary k
    unsigned long long g_target;  // the same instant in GPU time (clock fit)
    unsigned long long g_launch;  // GPU time the launch was issued (gpu mode) / host t0 mapped (cpu mode)
    unsigned long long g0, g1;    // slot start / end stamps (GPU clock); 0 if skipped
    unsigned long long flags;     // 1 = boundary skipped (previous slot still running), 2 = launch error
};
static_assert(sizeof(LsRec) == 56, "LsRec");

// Appended to the slot graph: publishes the slot's stamps in device memory so the executive never has
// to spin on PCIe-mapped host memory. d[1] = start stamp (read once from the mapped stamps), d[0] = now.
__global__ void k_done(unsigned long long *d, const SlotStamps *stamps_dev) {
    d[1] = stamps_dev->start_t;
    __threadfence();
    d[0] = gtimer();
}

// The slot executive: one resident thread. For each boundary k it waits until the previous slot has
// finished (skipping boundaries that pass meanwhile), waits for g_target[k] on %globaltimer, then
// launches the slot graph from the device (fire-and-forget) and records when it did so.
// A graph execution may issue at most 120 fire-and-forget launches (CUDA Programming Guide, Device Graph
// Launch), so the executive processes `chunk` launches, saves its state and tail-launches itself; the
// tail launch starts once the chunk and its child slot have completed.
struct ExecState {
    int k;               // next boundary
    int launched;        // slot in flight (-1 none)
    int chunks;          // executions so far
    int finished;        // 1 when all boundaries are handled
    unsigned long long prev_done;
    unsigned long long n_launched, n_skipped, n_err, n_tail_err;
};

__global__ void k_executive(cudaGraphExec_t e0, cudaGraphExec_t e1, const unsigned long long *g_target, int n,
                            volatile unsigned long long *done, LsRec *rec, ExecState *state, int chunk,
                            volatile int *finished_host) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    ExecState st = *state;
    if (st.chunks == 0) st.prev_done = done[0];
    st.chunks++;
    int fired = 0;
    while (st.k < n) {
        // 1. previous slot must be complete before anything else is launched
        if (st.launched >= 0) {
            while (done[0] == st.prev_done) {
                if (st.k < n && gtimer() >= g_target[st.k]) {   // boundary passed while busy: skip it
                    rec[st.k].g_target = g_target[st.k];
                    rec[st.k].flags = 1ull;
                    st.n_skipped++;
                    st.k++;
                    if (st.k >= n) break;
                }
            }
            if (done[0] != st.prev_done) {
                st.prev_done = done[0];
                rec[st.launched].g0 = done[1];
                rec[st.launched].g1 = st.prev_done;
                st.launched = -1;
            }
            if (st.k >= n) break;
        }
        if (fired >= chunk) break;   // hand over to the next execution of this graph
        // 2. wait for the boundary
        unsigned long long T = g_target[st.k];
        while (gtimer() < T) {}
        // 3. launch from the device
        unsigned long long tl = gtimer();
        cudaError_t e = cudaGraphLaunch((st.n_launched & 1ull) ? e1 : e0, cudaStreamGraphFireAndForget);
        rec[st.k].g_target = T;
        rec[st.k].g_launch = tl;
        if (e != cudaSuccess) {
            rec[st.k].flags = 2ull;
            st.n_err++;
            st.k++;
            continue;
        }
        st.launched = st.k;
        st.n_launched++;
        fired++;
        st.k++;
    }
    if (st.k >= n && st.launched >= 0) {  // drain the last slot
        while (done[0] == st.prev_done) {}
        st.prev_done = done[0];
        rec[st.launched].g0 = done[1];
        rec[st.launched].g1 = st.prev_done;
        st.launched = -1;
    }
    if (st.k >= n) st.finished = 1;
    *state = st;
    __threadfence_system();
    if (st.finished) {
        *finished_host = 1;
    } else {
        cudaError_t e = cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
        if (e != cudaSuccess) {
            state->n_tail_err++;
            state->finished = 2;
            __threadfence_system();
            *finished_host = 2;
        }
    }
}

// calibration ping-pong (resident kernel answers host sequence writes over mapped memory)
constexpr unsigned kStop = 0xFFFFFFFEu;
__global__ void k_pingpong(volatile unsigned *ctl, unsigned long long *gt, int n, unsigned base) {
    for (int i = 0; i < n; i++) {
        unsigned want = base + i + 1, c;
        unsigned long long spins = 0;
        while ((c = ctl[0]) != want) {
            if (c == kStop || ++spins > 2000000000ull) return;
        }
        gt[i] = gtimer();
        __threadfence_system();
        ctl[1] = want;
        __threadfence_system();
    }
}

// in-process "AI job": large FMA kernels on a low-priority stream
__global__ void k_ai(float *a, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = a[i];
    for (int k = 0; k < iters; k++) v = v * 1.0000001f + 0.5f;
    a[i] = v;
}

struct Opts {
    std::string mode = "cpu", load = "none", out = "lockstep.json", raw, label;
    long slots = 20000, warmup = 500;
    double period_us = 500, deadline_us = 500, spin_us = 60;
    int gpu = 0, core = -1, calib = 20000;
    std::string prio = "high";
    PhyConfig phy;
};

[[noreturn]] void usage(const char *msg = nullptr) {
    if (msg) fprintf(stderr, "lockstep_driver: %s\n", msg);
    fprintf(stderr,
            "usage: lockstep_driver --mode cpu|gpu [--load none|stream] [--slots N] [--warmup N] [--period-us F]\n"
            "       [--deadline-us F] [--spin-us F] [--prio high|default|low] [--core N] [--gpu N] [--calib N]\n"
            "       [--out FILE.json] [--raw FILE.bin] [--label S] [sizes: --subcarriers --ldpc-cb --ldpc-iters ...]\n");
    exit(2);
}

Opts parse(int argc, char **argv) {
    Opts o;
    auto num = [&](const char *v) { char *e; double d = strtod(v, &e); if (*e) usage("bad number"); return d; };
    for (int i = 1; i < argc; i++) {
        std::string f = argv[i];
        if (i + 1 >= argc) usage(("missing value for " + f).c_str());
        const char *v = argv[++i];
        if (f == "--mode") o.mode = v;
        else if (f == "--load") o.load = v;
        else if (f == "--slots") o.slots = (long)num(v);
        else if (f == "--warmup") o.warmup = (long)num(v);
        else if (f == "--period-us") o.period_us = num(v);
        else if (f == "--deadline-us") o.deadline_us = num(v);
        else if (f == "--spin-us") o.spin_us = num(v);
        else if (f == "--prio") o.prio = v;
        else if (f == "--core") o.core = (int)num(v);
        else if (f == "--gpu") o.gpu = (int)num(v);
        else if (f == "--calib") o.calib = (int)num(v);
        else if (f == "--out") o.out = v;
        else if (f == "--raw") o.raw = v;
        else if (f == "--label") o.label = v;
        else if (f == "--fft") o.phy.fft = (int)num(v);
        else if (f == "--symbols") o.phy.symbols = (int)num(v);
        else if (f == "--subcarriers") o.phy.subcarriers = (int)num(v);
        else if (f == "--qam") o.phy.qam = (int)num(v);
        else if (f == "--ldpc-cb") o.phy.ldpc_cb = (int)num(v);
        else if (f == "--ldpc-iters") o.phy.ldpc_iters = (int)num(v);
        else if (f == "--ldpc-rows") o.phy.ldpc_rows = (int)num(v);
        else if (f == "--ldpc-z") o.phy.ldpc_z = (int)num(v);
        else usage(("unknown flag " + f).c_str());
    }
    if (o.mode != "cpu" && o.mode != "gpu") usage("--mode must be cpu or gpu");
    if (o.load != "none" && o.load != "stream") usage("--load must be none or stream");
    if (o.slots < 1 || o.period_us <= 0) usage("bad slots/period");
    return o;
}

// ---- clock calibration ----
std::vector<Bracket> calibrate(cudaStream_t s, volatile unsigned *ctl_h, unsigned *ctl_d, unsigned long long *gt_h,
                               unsigned long long *gt_d, int samples, unsigned &seq) {
    std::vector<Bracket> out;
    const int batch = 400;
    for (int done = 0; done < samples; done += batch) {
        unsigned base = seq;
        ctl_h[0] = base;
        ctl_h[1] = base;
        __sync_synchronize();
        k_pingpong<<<1, 1, 0, s>>>((volatile unsigned *)ctl_d, gt_d, batch, base);
        CK(cudaGetLastError());
        std::vector<std::pair<int64_t, int64_t>> tt;
        for (int i = 0; i < batch; i++) {
            unsigned w = base + i + 1;
            int64_t t0 = now_ns();
            ctl_h[0] = w;
            __sync_synchronize();
            int64_t lim = t0 + (i == 0 ? 2000000000LL : 200000000LL);
            bool ok = true;
            while (ctl_h[1] != w) {
                if (now_ns() > lim) { ok = false; break; }
            }
            int64_t t1 = now_ns();
            if (!ok) break;
            tt.push_back({t0, t1});
        }
        ctl_h[0] = kStop;
        __sync_synchronize();
        CK(cudaStreamSynchronize(s));
        for (size_t i = 0; i < tt.size(); i++) out.push_back({tt[i].first, tt[i].second, gt_h[i]});
        seq = base + batch + 1;
        if (tt.size() < (size_t)batch / 2) { fprintf(stderr, "lockstep_driver: calibration batch aborted\n"); break; }
    }
    return out;
}

double pct(std::vector<double> v, double p) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    size_t i = (size_t)std::ceil(p / 100.0 * (v.size() - 1));
    return v[std::min(i, v.size() - 1)];
}

std::string stats_json(const std::vector<double> &v) {
    Json j;
    j.add("n", (long long)v.size());
    if (!v.empty()) {
        double mx = *std::max_element(v.begin(), v.end()), mn = *std::min_element(v.begin(), v.end());
        j.add("min", mn).add("p50", pct(v, 50)).add("p90", pct(v, 90)).add("p99", pct(v, 99)).add("p99_9", pct(v, 99.9))
            .add("p99_99", pct(v, 99.99)).add("max", mx);
    }
    return j.str();
}

}  // namespace

int main(int argc, char **argv) {
    Opts o = parse(argc, argv);
    CK(cudaSetDevice(o.gpu));
    CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, o.gpu));
    int rt = 0, drv = 0;
    CK(cudaRuntimeGetVersion(&rt));
    CK(cudaDriverGetVersion(&drv));

    int lo, hi;
    CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    int prio = o.prio == "high" ? hi : o.prio == "low" ? lo : 0;
    cudaStream_t s, s_cal, s_load;
    CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, prio));
    CK(cudaStreamCreateWithPriority(&s_cal, cudaStreamNonBlocking, hi));
    CK(cudaStreamCreateWithPriority(&s_load, cudaStreamNonBlocking, lo));

    // calibration buffers
    volatile unsigned *ctl_h; unsigned *ctl_d; unsigned long long *gt_h, *gt_d;
    CK(cudaHostAlloc((void **)&ctl_h, 64, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void **)&ctl_d, (void *)ctl_h, 0));
    CK(cudaHostAlloc((void **)&gt_h, 400 * 8, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void **)&gt_d, gt_h, 0));
    unsigned cal_seq = 0;

    // the slot
    SlotPipeline pipe(o.phy);
    SlotStamps *st_dev;
    CK(cudaHostGetDevicePointer((void **)&st_dev, (void *)pipe.stamps(), 0));
    unsigned long long *done_d;
    CK(cudaMalloc(&done_d, 2 * sizeof(unsigned long long)));
    CK(cudaMemset(done_d, 0, 2 * sizeof(unsigned long long)));

    // capture one slot (+ completion kernel); device-launchable graphs may hold only kernel/memcpy/memset/
    // child nodes, so if the cuBLAS stages make instantiation fail, fall back to the slot without them
    // (FFT, gather, demod, de-match, LDPC, pack) and say so in the output.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t host_exec = nullptr, dev_exec[2] = {nullptr, nullptr}, exec_graph_exec = nullptr;
    size_t n_nodes = 0;
    std::string slot_variant = "full", dev_instantiate_error;
    SlotPipeline *pipe_p = &pipe;
    SlotPipeline *pipe_fallback = nullptr;
    for (int attempt = 0; attempt < 2; attempt++) {
        if (attempt == 1) {
            PhyConfig c2 = o.phy;
            c2.skip_blas = true;
            pipe_fallback = new SlotPipeline(c2);
            pipe_p = pipe_fallback;
            slot_variant = "no_cublas";
            CK(cudaHostGetDevicePointer((void **)&st_dev, (void *)pipe_p->stamps(), 0));
            if (host_exec) { CK(cudaGraphExecDestroy(host_exec)); host_exec = nullptr; }
            if (graph) { CK(cudaGraphDestroy(graph)); graph = nullptr; }
        }
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        pipe_p->enqueue(s);
        k_done<<<1, 1, 0, s>>>(done_d, st_dev);
        CK(cudaStreamEndCapture(s, &graph));
        CK(cudaGraphGetNodes(graph, nullptr, &n_nodes));
        CK(cudaGraphInstantiate(&host_exec, graph, 0));
        if (o.mode != "gpu") break;
        dev_instantiate_error.clear();
        for (int i = 0; i < 2; i++) {
            cudaError_t e = cudaGraphInstantiateWithFlags(&dev_exec[i], graph, cudaGraphInstantiateFlagDeviceLaunch);
            if (e != cudaSuccess) {
                dev_instantiate_error = cudaGetErrorString(e);
                (void)cudaGetLastError();
                fprintf(stderr, "lockstep_driver: device-launchable instantiate (%s slot) failed: %s\n", slot_variant.c_str(),
                        dev_instantiate_error.c_str());
                for (int j2 = 0; j2 < i; j2++) { cudaGraphExecDestroy(dev_exec[j2]); dev_exec[j2] = nullptr; }
                break;
            }
            CK(cudaGraphUpload(dev_exec[i], s));
        }
        CK(cudaStreamSynchronize(s));
        if (dev_instantiate_error.empty()) break;
    }
    if (o.mode == "gpu" && !dev_instantiate_error.empty()) {
        Json j;
        j.add("ok", false).add("mode", o.mode).add("error", "device graph launch not available for this slot graph: " + dev_instantiate_error)
            .add("graph_nodes", (long long)n_nodes).add("gpu", prop.name);
        FILE *f = fopen(o.out.c_str(), "w");
        if (f) { fputs(j.str().c_str(), f); fclose(f); }
        return 1;
    }
    const volatile SlotStamps *st = pipe_p->stamps();

    // prime: one host launch so the stamp sequence base is known
    CK(cudaGraphLaunch(host_exec, s));
    CK(cudaStreamSynchronize(s));
    unsigned long long seq_base = st->end_seq;

    // in-process load
    std::atomic<bool> stop{false};
    std::thread load_th;
    float *big = nullptr;
    size_t bigN = (size_t)prop.multiProcessorCount * 2048 * 8;
    if (o.load == "stream") {
        CK(cudaMalloc(&big, bigN * sizeof(float)));
        CK(cudaMemset(big, 0, bigN * sizeof(float)));
        load_th = std::thread([&] {
            while (!stop) {
                k_ai<<<(unsigned)(bigN / 256), 256, 0, s_load>>>(big, 20000);
                cudaStreamSynchronize(s_load);
            }
        });
        spin_until(now_ns() + 300000000LL);
    }

    pin_thread(o.core);
    std::vector<Bracket> br = calibrate(s_cal, ctl_h, ctl_d, gt_h, gt_d, o.calib, cal_seq);
    ClockFit fit = fit_clock(br);
    if (!fit.ok) { fprintf(stderr, "lockstep_driver: clock calibration failed\n"); return 3; }

    // targets
    const long N = o.slots, W = o.warmup, total = N + W;
    const int64_t period_ns = (int64_t)llround(o.period_us * 1000.0);
    const int64_t spin_ns = (int64_t)llround(o.spin_us * 1000.0);
    const int64_t t_start = now_ns() + 100000000LL;  // 100 ms from now
    std::vector<int64_t> T(total);
    std::vector<unsigned long long> G(total);
    for (long k = 0; k < total; k++) {
        T[k] = t_start + k * period_ns;
        G[k] = fit.gpu_of(T[k]);
    }
    std::vector<LsRec> rec(total);
    memset(rec.data(), 0, rec.size() * sizeof(LsRec));
    for (long k = 0; k < total; k++) { rec[k].slot = (unsigned long long)k; rec[k].t_target = T[k]; }
    unsigned long long stats[3] = {0, 0, 0};
    std::string exec_error;
    int exec_chunks = 0;
    constexpr int kChunk = 100;  // < 120 fire-and-forget launches per graph execution

    if (o.mode == "cpu") {
        unsigned long long seq = seq_base;
        long launched = -1;
        long k = 0;
        while (k < total) {
            if (launched >= 0) {
                while (st->end_seq != seq) {
                    if (k < total && now_ns() >= T[k]) { rec[k].g_target = G[k]; rec[k].flags = 1u; stats[1]++; k++; if (k >= total) break; }
                }
                if (st->end_seq == seq) { rec[launched].g0 = st->start_t; rec[launched].g1 = st->end_t; launched = -1; }
                if (k >= total) break;
            }
            sleep_until_raw(T[k] - spin_ns);
            spin_until(T[k]);
            int64_t t0 = now_ns();
            cudaError_t e = cudaGraphLaunch(host_exec, s);
            rec[k].g_target = G[k];
            rec[k].g_launch = fit.gpu_of(t0);
            if (e != cudaSuccess) { rec[k].flags = 2u; stats[2]++; (void)cudaGetLastError(); k++; continue; }
            seq++;
            launched = k;
            stats[0]++;
            k++;
        }
        if (launched >= 0) {
            int64_t lim = now_ns() + 5000000000LL;
            while (st->end_seq != seq && now_ns() < lim) {}
            if (st->end_seq == seq) { rec[launched].g0 = st->start_t; rec[launched].g1 = st->end_t; }
        }
    } else {
        unsigned long long *G_d;
        LsRec *rec_d;
        ExecState *state_d;
        volatile int *finished_h; int *finished_d;
        CK(cudaMalloc(&G_d, total * sizeof(unsigned long long)));
        CK(cudaMemcpy(G_d, G.data(), total * sizeof(unsigned long long), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&rec_d, total * sizeof(LsRec)));
        CK(cudaMemcpy(rec_d, rec.data(), total * sizeof(LsRec), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&state_d, sizeof(ExecState)));
        ExecState st0{};
        st0.launched = -1;
        CK(cudaMemcpy(state_d, &st0, sizeof st0, cudaMemcpyHostToDevice));
        CK(cudaHostAlloc((void **)&finished_h, 64, cudaHostAllocMapped));
        *finished_h = 0;
        CK(cudaHostGetDevicePointer((void **)&finished_d, (void *)finished_h, 0));
        // the executive itself runs as a device-launchable graph (device graph launch is only allowed
        // from kernels that are part of one) and tail-launches itself every `chunk` slots
        cudaGraph_t eg;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        k_executive<<<1, 32, 0, s>>>(dev_exec[0], dev_exec[1], G_d, (int)total, (volatile unsigned long long *)done_d, rec_d,
                                     state_d, kChunk, (volatile int *)finished_d);
        CK(cudaStreamEndCapture(s, &eg));
        cudaError_t e = cudaGraphInstantiateWithFlags(&exec_graph_exec, eg, cudaGraphInstantiateFlagDeviceLaunch);
        if (e != cudaSuccess) { exec_error = std::string("executive instantiate: ") + cudaGetErrorString(e); (void)cudaGetLastError(); }
        else {
            CK(cudaGraphUpload(exec_graph_exec, s));
            CK(cudaGraphLaunch(exec_graph_exec, s));
            // the finished flag covers the whole tail-launched chain; the stream query is a second check
            int64_t lim = now_ns() + (int64_t)((double)total * period_ns * 3) + 10000000000LL;
            while (*finished_h == 0) {
                if (now_ns() > lim) { exec_error = "executive did not finish (wall guard)"; break; }
                cudaError_t q = cudaStreamQuery(s);
                if (q != cudaSuccess && q != cudaErrorNotReady) { exec_error = std::string("executive: ") + cudaGetErrorString(q); break; }
                spin_until(now_ns() + 100000);
            }
            if (exec_error.empty() && *finished_h == 2) exec_error = "executive tail relaunch failed";
            if (exec_error.empty()) {
                int64_t lim2 = now_ns() + 5000000000LL;
                cudaError_t q;
                while ((q = cudaStreamQuery(s)) == cudaErrorNotReady && now_ns() < lim2) spin_until(now_ns() + 100000);
                if (q != cudaSuccess) exec_error = std::string("executive stream: ") + cudaGetErrorString(q);
            }
        }
        ExecState stf{};
        CK(cudaMemcpy(&stf, state_d, sizeof stf, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(rec.data(), rec_d, total * sizeof(LsRec), cudaMemcpyDeviceToHost));
        stats[0] = stf.n_launched; stats[1] = stf.n_skipped; stats[2] = stf.n_err;
        exec_chunks = stf.chunks;
        if (exec_error.empty() && stf.n_tail_err) exec_error = "executive tail relaunch error";
    }

    std::vector<Bracket> br2 = calibrate(s_cal, ctl_h, ctl_d, gt_h, gt_d, std::max(2000, o.calib / 4), cal_seq);
    ClockFit fit2 = fit_clock(br2);
    stop = true;
    if (load_th.joinable()) load_th.join();

    // ---- summarise recorded slots (after warm-up) ----
    std::vector<double> start_err, launch_err, launch_to_start, exec_us, lat_from_target;
    long recorded = 0, misses = 0, skipped = 0, errors = 0;
    const double deadline_ns = o.deadline_us * 1000.0;
    for (long k = W; k < total; k++) {
        const LsRec &r = rec[k];
        if (r.flags & 1u) { skipped++; misses++; continue; }
        if (r.flags & 2u) { errors++; misses++; continue; }
        if (!r.g0 || !r.g1) continue;
        recorded++;
        double se = ((double)r.g0 - (double)r.g_target) / 1000.0;
        double le = ((double)r.g_launch - (double)r.g_target) / 1000.0;
        start_err.push_back(se);
        launch_err.push_back(le);
        launch_to_start.push_back(((double)r.g0 - (double)r.g_launch) / 1000.0);
        exec_us.push_back(((double)r.g1 - (double)r.g0) / 1000.0);
        double lt = ((double)r.g1 - (double)r.g_target);
        lat_from_target.push_back(lt / 1000.0);
        if (lt > deadline_ns) misses++;
    }
    long total_slots = recorded + skipped + errors;

    if (!o.raw.empty()) {
        FILE *f = fopen(o.raw.c_str(), "wb");
        if (f) { fwrite(rec.data() + W, sizeof(LsRec), (size_t)N, f); fclose(f); }
    }
    Json j;
    j.add("ok", exec_error.empty()).add("error", exec_error).add("mode", o.mode).add("load", o.load).add("label", o.label)
        .add("gpu", prop.name).add("sm", prop.multiProcessorCount).add("driver", drv).add("runtime", rt)
        .add("slots", (long long)N).add("warmup", (long long)W).add("period_us", o.period_us).add("deadline_us", o.deadline_us)
        .add("stream_priority", prio).add("graph_nodes", (long long)n_nodes)
        .add_raw("phy", pipe_p->config().json())
        .add("recorded", (long long)recorded).add("skipped", (long long)skipped).add("launch_errors", (long long)errors)
        .add("total_slots", (long long)total_slots).add("misses", (long long)misses)
        .add("miss_rate", total_slots ? (double)misses / (double)total_slots : NAN)
        .add_raw("start_error_us", stats_json(start_err))      // g0 - g_target: when the slot really began vs when it should have
        .add_raw("launch_error_us", stats_json(launch_err))    // g_launch - g_target: when the launcher acted vs when it should have
        .add_raw("launch_to_start_us", stats_json(launch_to_start))
        .add_raw("exec_us", stats_json(exec_us))
        .add_raw("latency_from_target_us", stats_json(lat_from_target))
        .add_raw("clock_fit_pre", fit.json()).add_raw("clock_fit_post", fit2.json())
        .add("executive_launched", (long long)stats[0]).add("executive_skipped", (long long)stats[1])
        .add("executive_errors", (long long)stats[2]).add("executive_chunks", exec_chunks).add("slot_variant", slot_variant);
    FILE *f = fopen(o.out.c_str(), "w");
    if (!f) { perror("out"); return 2; }
    fputs(j.str().c_str(), f);
    fputs("\n", f);
    fclose(f);
    printf("%-4s load=%-6s %-24s start_err p50 %8.2f p99 %8.2f p99.9 %8.2f max %9.2f us | exec p50 %6.1f | miss %ld/%ld skipped %ld%s\n",
           o.mode.c_str(), o.load.c_str(), o.label.c_str(), pct(start_err, 50), pct(start_err, 99), pct(start_err, 99.9),
           start_err.empty() ? NAN : *std::max_element(start_err.begin(), start_err.end()), pct(exec_us, 50), misses, total_slots,
           skipped, exec_error.empty() ? "" : ("  ERROR " + exec_error).c_str());
    return exec_error.empty() ? 0 : 1;
}
