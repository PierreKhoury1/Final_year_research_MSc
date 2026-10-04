// Lockstep driver: who starts the 5G slot on time, the CPU or the GPU itself?
//
//   --mode cpu   the host spins until target T_k (CLOCK_MONOTONIC_RAW) and calls cudaGraphLaunch   (today's design)
//   --mode gpu   a resident 1-thread "slot executive" (lockstep_exec.cu) watches %globaltimer and launches the
//                slot graph from the device at T_k converted to GPU time; the CPU is off the timing path
//
// Both modes run the same captured slot graph (SlotPipeline::enqueue + stamps + a completion kernel) once per
// period, never overlapping: a boundary that arrives while the previous slot is still running is skipped and
// counted as a miss. Per slot (GPU clock unless noted): g_target = T_k mapped with the pre-run fit (the value
// the executive acts on), g_launch / g_launch_done around the launch call, g0 / g1 = slot start / end stamps.
//
// Metrics (all integer-subtracted before conversion; %globaltimer is ~1.7e18 ns):
//   launch_precision  = g0 - g_target             how well the launcher hit the instant it was given
//   start_error       = g0 - g2pt(T_k)            the same against a two-point host->GPU mapping anchored on the
//                                                 pre- and post-run calibrations (removes the pre-fit slope error)
//   target_pred_error = g_target - g2pt(T_k)      how far the pre-run fit's prediction of T_k was off (gpu mode's
//                                                 own boundary error, cpu mode's bias in launch_precision)
//   exec = g1 - g0, latency_from_target = g1 - g2pt(T_k), miss = latency_from_target > deadline, or skipped
//
// Optional in-process load (--load stream): a thread keeps a low-priority stream busy with large FMA kernels
// (an AI job in the same CUDA context). Separate-process neighbours and MPS are set up by
// scripts/lockstep_matrix.sh around this binary.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "clock_fit.h"
#include "cuda_check.h"
#include "host_time.h"
#include "json_writer.h"
#include "lockstep_common.h"
#include "slot_pipeline.h"
#include "slottrace.h"
#include "probe.cuh"

using namespace sb;

namespace {

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

// Appended to the slot graph: publishes the slot's stamps in device memory so the executive never spins on
// PCIe-mapped host memory. d[1] = start stamp, d[2] = end stamp (both read once from the mapped stamps,
// after k_stamp_end ran), then release-publish d[0] = completion sequence.
__global__ void k_done(unsigned long long *d, const SlotStamps *stamps_dev,
                       volatile unsigned long long *finished_host) {
    d[1] = stamps_dev->start_t;
    d[2] = stamps_dev->end_t;
    const auto seq = stamps_dev->end_seq;
    // Both launchers gate readiness on this final kernel. CPU polling must not use the earlier
    // k_stamp_end marker while the GPU waits for an additional kernel to become schedulable.
    __threadfence_system();
    *finished_host = seq;
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> completion(d[0]);
    completion.store(seq, cuda::memory_order_release);
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

// %globaltimer update granularity: smallest non-zero step seen by one thread over ~n reads
__global__ void k_tick(unsigned long long *out, int n) {
    unsigned long long last = gtimer(), best = ~0ull;
    for (int i = 0; i < n; i++) {
        unsigned long long t = gtimer();
        if (t != last) { if (t - last < best) best = t - last; last = t; }
    }
    out[0] = best;
}

struct Opts {
    std::string mode = "cpu", load = "none", out = "lockstep.json", raw, label, slot_variant = "full";
    long slots = 20000, warmup = 500;
    double period_us = 500, deadline_us = 500, spin_us = 200, calib_spread_s = 2.0;
    int gpu = 0, core = -1, fifo = 0, calib = 20000;
    std::string prio = "high";
    // slottrace: per-slot host records and the GPU residency probe (trace/README.md)
    std::string host_raw, probe_out;
    int probe = 0, probe_sleep_ns = 1000, probe_capacity = 1 << 20;
    double probe_gap_us = 20.0;
    PhyConfig phy;
};

[[noreturn]] void usage(const char *msg = nullptr) {
    if (msg) fprintf(stderr, "lockstep_driver: %s\n", msg);
    fprintf(stderr,
            "usage: lockstep_driver --mode cpu|gpu [--load none|stream] [--slots N] [--warmup N] [--period-us F]\n"
            "       [--deadline-us F] [--spin-us F] [--prio high|default|low] [--core N] [--fifo P] [--gpu N]\n"
            "       [--calib N] [--calib-spread-s F] [--out FILE.json] [--raw FILE.bin] [--label S]\n"
            "       [--slot-variant full|no_cublas]\n"
            "       [slottrace: --host-raw FILE.bin --probe 0|1 --probe-out FILE.bin --probe-gap-us F --probe-sleep-ns N]\n"
            "       [sizes: --fft --symbols --subcarriers --qam --ldpc-cb --ldpc-iters --ldpc-rows --ldpc-z]\n");
    exit(2);
}

Opts parse(int argc, char **argv) {
    Opts o;
    auto num = [&](const char *v) {
        char *e; errno = 0; double d = strtod(v, &e);
        if (e == v || *e || errno == ERANGE || !std::isfinite(d)) usage("bad number");
        return d;
    };
    auto integer = [&](const char *v) {
        double d = num(v);
        if (d < INT_MIN || d > INT_MAX || std::floor(d) != d) usage("integer out of range");
        return (int)d;
    };
    for (int i = 1; i < argc; i++) {
        std::string f = argv[i];
        if (i + 1 >= argc) usage(("missing value for " + f).c_str());
        const char *v = argv[++i];
        if (f == "--mode") o.mode = v;
        else if (f == "--load") o.load = v;
        else if (f == "--slots") o.slots = integer(v);
        else if (f == "--warmup") o.warmup = integer(v);
        else if (f == "--period-us") o.period_us = num(v);
        else if (f == "--deadline-us") o.deadline_us = num(v);
        else if (f == "--spin-us") o.spin_us = num(v);
        else if (f == "--prio") o.prio = v;
        else if (f == "--core") o.core = integer(v);
        else if (f == "--fifo") o.fifo = integer(v);
        else if (f == "--gpu") o.gpu = integer(v);
        else if (f == "--calib") o.calib = integer(v);
        else if (f == "--calib-spread-s") o.calib_spread_s = num(v);
        else if (f == "--out") o.out = v;
        else if (f == "--raw") o.raw = v;
        else if (f == "--label") o.label = v;
        else if (f == "--slot-variant") o.slot_variant = v;
        else if (f == "--host-raw") o.host_raw = v;
        else if (f == "--probe") o.probe = integer(v);
        else if (f == "--probe-out") o.probe_out = v;
        else if (f == "--probe-gap-us") o.probe_gap_us = num(v);
        else if (f == "--probe-sleep-ns") o.probe_sleep_ns = integer(v);
        else if (f == "--fft") o.phy.fft = integer(v);
        else if (f == "--symbols") o.phy.symbols = integer(v);
        else if (f == "--subcarriers") o.phy.subcarriers = integer(v);
        else if (f == "--qam") o.phy.qam = integer(v);
        else if (f == "--ldpc-cb") o.phy.ldpc_cb = integer(v);
        else if (f == "--ldpc-iters") o.phy.ldpc_iters = integer(v);
        else if (f == "--ldpc-rows") o.phy.ldpc_rows = integer(v);
        else if (f == "--ldpc-z") o.phy.ldpc_z = integer(v);
        else usage(("unknown flag " + f).c_str());
    }
    if (o.mode != "cpu" && o.mode != "gpu") usage("--mode must be cpu or gpu");
    if (o.load != "none" && o.load != "stream") usage("--load must be none or stream");
    if (o.slot_variant != "full" && o.slot_variant != "no_cublas") usage("bad slot variant");
    o.phy.skip_blas = o.slot_variant == "no_cublas";
    if (o.slots < 1 || o.warmup < 0 || o.slots + o.warmup > INT_MAX) usage("bad slots/warmup");
    if (o.period_us < 0.001 || o.period_us > 1e9 || o.deadline_us <= 0 || o.spin_us < 0 || o.spin_us > 1e9)
        usage("bad period/deadline/spin");
    if ((o.slots + o.warmup) * o.period_us > 1e12) usage("run duration exceeds 1,000,000 seconds");
    if (o.calib < 2 || o.calib > INT_MAX - 400 || o.calib_spread_s <= 0 || o.calib_spread_s > 3600)
        usage("bad calibration size/spread");
    if (o.prio != "high" && o.prio != "default" && o.prio != "low") usage("bad priority");
    if (o.gpu < 0 || o.core < -1 || o.core >= CPU_SETSIZE || o.fifo < 0 || o.fifo > 99)
        usage("bad gpu/core/fifo");
    if ((o.probe != 0 && o.probe != 1) || o.probe_gap_us < 2 || o.probe_gap_us > 1e6 || o.probe_sleep_ns < 0
        || o.probe_sleep_ns > 1000000 || (o.probe && o.probe_out.empty()))
        usage("bad probe options (--probe 1 needs --probe-out; gap >= 2 us)");
    return o;
}

// Never follow an expired stream guard with a blocking synchronization/copy.
cudaError_t wait_stream(cudaStream_t s, int64_t deadline) {
    cudaError_t e;
    while ((e = cudaStreamQuery(s)) == cudaErrorNotReady) {
        if (now_ns() >= deadline) return e;
        sleep_until_raw(now_ns() + 100000);
    }
    return e;
}

// ---- clock calibration: `samples` brackets spread over about `spread_s` seconds ----
std::vector<Bracket> calibrate(cudaStream_t s, volatile unsigned *ctl_h, unsigned *ctl_d, unsigned long long *gt_h,
                               unsigned long long *gt_d, int samples, double spread_s, unsigned &seq, std::string &error) {
    std::vector<Bracket> out;
    const int batch = 400;
    const int batches = std::max(1, (samples + batch - 1) / batch);
    const int64_t gap_ns = batches > 1 ? (int64_t)(spread_s * 1e9 / batches) : 0;
    for (int bi = 0; bi < batches; bi++) {
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
        cudaError_t e = wait_stream(s, now_ns() + 5000000000LL);
        if (e != cudaSuccess) {
            error = std::string("calibration did not complete: ") + cudaGetErrorString(e);
            return {};
        }
        for (size_t i = 0; i < tt.size(); i++) out.push_back({tt[i].first, tt[i].second, gt_h[i]});
        seq = base + batch + 1;
        if (tt.size() < (size_t)batch / 2) { fprintf(stderr, "lockstep_driver: calibration batch aborted\n"); break; }
        if (gap_ns > 0 && bi + 1 < batches) sleep_until_raw(now_ns() + gap_ns);
    }
    return out;
}

// two-point host->GPU mapping through the pre and post fits' anchors
struct TwoPoint {
    bool ok = false;
    int64_t h_a = 0, h_b = 0;
    unsigned long long g_a = 0, g_b = 0;
    double rate = 1.0;  // GPU ns per host ns
    unsigned long long gpu_of(int64_t t) const {
        double x = (double)(t - h_a) * rate;
        return g_a + (unsigned long long)(long long)std::llround(x);
    }
};
TwoPoint two_point(const ClockFit &a, const ClockFit &b) {
    TwoPoint tp;
    if (!a.ok || !b.ok) return tp;
    tp.g_a = a.g_ref;
    tp.h_a = (int64_t)std::llround(a.host_of(a.g_ref));
    tp.g_b = b.g_ref;
    tp.h_b = (int64_t)std::llround(b.host_of(b.g_ref));
    if (tp.h_b <= tp.h_a + 1000000) return tp;  // need a baseline
    tp.rate = (double)(long long)(tp.g_b - tp.g_a) / (double)(tp.h_b - tp.h_a);
    tp.ok = std::isfinite(tp.rate) && tp.rate > 0;
    return tp;
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

inline double dus(unsigned long long a, unsigned long long b) { return (double)(long long)(a - b) / 1000.0; }

std::string node_types_json(cudaGraph_t g) {
    size_t n = 0;
    if (cudaGraphGetNodes(g, nullptr, &n) != cudaSuccess) return "[]";
    std::vector<cudaGraphNode_t> nodes(n);
    cudaGraphGetNodes(g, nodes.data(), &n);
    int counts[16] = {0};
    for (size_t i = 0; i < n; i++) {
        cudaGraphNodeType t;
        if (cudaGraphNodeGetType(nodes[i], &t) == cudaSuccess && (int)t < 16) counts[(int)t]++;
    }
    static const char *names[] = {"kernel", "memcpy", "memset", "host", "graph", "empty", "wait_event", "event_record",
                                  "ext_semaphore_signal", "ext_semaphore_wait", "mem_alloc", "mem_free", "batch_memop",
                                  "conditional", "t14", "t15"};
    Json j;
    for (int i = 0; i < 16; i++) if (counts[i]) j.add(names[i], counts[i]);
    return j.str();
}

void write_json_and_exit(const std::string &path, const Json &j, int code) {
    FILE *f = fopen(path.c_str(), "w");
    if (f) { fputs(j.str().c_str(), f); fputs("\n", f); fclose(f); }
    fflush(stderr);
    _exit(code);
}

}  // namespace

int main(int argc, char **argv) {
    Opts o = parse(argc, argv);
    CK(cudaSetDevice(o.gpu));
    CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, o.gpu));
    auto fail_run = [&](const std::string &reason) {
        Json j;
        j.add("ok", false).add("mode", o.mode).add("load", o.load).add("label", o.label)
            .add("gpu", prop.name).add("slot_variant", o.slot_variant).add("error", reason);
        // A kernel may still be running; avoid local CUDA destructors and thread joins.
        write_json_and_exit(o.out, j, 1);
    };
    int rt = 0, drv = 0;
    CK(cudaRuntimeGetVersion(&rt));
    CK(cudaDriverGetVersion(&drv));
    const char *mps_pipe = getenv("CUDA_MPS_PIPE_DIRECTORY");
    bool mps_control_present = false;
    {
        std::string p = std::string(mps_pipe ? mps_pipe : "/tmp/nvidia-mps") + "/control";
        struct stat sb_;
        mps_control_present = stat(p.c_str(), &sb_) == 0;
    }

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

    // %globaltimer granularity
    unsigned long long tick_ns = 0;
    {
        unsigned long long *tk;
        CK(cudaMalloc(&tk, 8));
        k_tick<<<1, 1, 0, s>>>(tk, 200000);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(&tick_ns, tk, 8, cudaMemcpyDeviceToHost));
        cudaFree(tk);
    }

    // Capture the explicitly selected workload. A device-instantiation failure must not silently
    // remove stages from the GPU case while the CPU case retains the full workload.
    SlotPipeline pipe(o.phy);
    SlotStamps *st_dev = nullptr;
    volatile unsigned long long *slot_finished_h = nullptr;
    unsigned long long *slot_finished_d = nullptr;
    CK(cudaHostAlloc((void **)&slot_finished_h, sizeof(unsigned long long), cudaHostAllocMapped));
    *slot_finished_h = 0;
    CK(cudaHostGetDevicePointer((void **)&slot_finished_d, (void *)slot_finished_h, 0));
    unsigned long long *done_d;
    CK(cudaMalloc(&done_d, 3 * sizeof(unsigned long long)));
    CK(cudaMemsetAsync(done_d, 0, 3 * sizeof(unsigned long long), s));
    cudaGraph_t graph = nullptr;
    constexpr int kChunk = 100;  // below the per-execution fire-and-forget limit
    cudaGraphExec_t host_exec = nullptr, exec_graph_exec = nullptr;
    std::vector<cudaGraphExec_t> dev_exec(kChunk, nullptr);
    cudaGraphExec_t *exec_pool_d = nullptr;
    size_t n_nodes = 0;
    const std::string slot_variant = o.slot_variant;
    std::string dev_instantiate_error, node_types;
    const unsigned inst_flags = cudaGraphInstantiateFlagUseNodePriority;  // same priority semantics in both modes
    CK(cudaHostGetDevicePointer((void **)&st_dev, (void *)pipe.stamps(), 0));
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    pipe.enqueue(s);
    k_done<<<1, 1, 0, s>>>(done_d, st_dev, slot_finished_d);
    CK(cudaGetLastError());
    CK(cudaStreamEndCapture(s, &graph));
    CK(cudaGraphGetNodes(graph, nullptr, &n_nodes));
    node_types = node_types_json(graph);
    CK(cudaGraphInstantiateWithFlags(&host_exec, graph, inst_flags));
    if (o.mode == "gpu") {
        // Each handle is launched once per generation. Tail launch waits for all child graphs,
        // making reuse safe even when the completion kernel retires before its graph does.
        for (int i = 0; i < kChunk; i++) {
            cudaError_t e = cudaGraphInstantiateWithFlags(&dev_exec[i], graph, cudaGraphInstantiateFlagDeviceLaunch | inst_flags);
            if (e != cudaSuccess) {
                dev_instantiate_error = cudaGetErrorString(e);
                (void)cudaGetLastError();
                fprintf(stderr, "lockstep_driver: device-launchable instantiate (%s slot, nodes %s) failed: %s\n",
                        slot_variant.c_str(), node_types.c_str(), dev_instantiate_error.c_str());
                break;
            }
            CK(cudaGraphUpload(dev_exec[i], s));
        }
        CK(cudaStreamSynchronize(s));
    }
    if (o.mode == "gpu" && !dev_instantiate_error.empty()) {
        Json j;
        j.add("ok", false).add("mode", o.mode).add("label", o.label).add("gpu", prop.name)
            .add("error", "device graph launch not available for this slot graph: " + dev_instantiate_error)
            .add("graph_nodes", (long long)n_nodes).add_raw("graph_node_types", node_types).add("slot_variant", slot_variant);
        write_json_and_exit(o.out, j, 1);
    }
    const volatile SlotStamps *st = pipe.stamps();

    // prime: one host launch so the stamp sequence base is known and the done flag is non-zero
    CK(cudaGraphLaunch(host_exec, s));
    CK(cudaStreamSynchronize(s));
    unsigned long long seq_base = *slot_finished_h;

    const long N = o.slots, W = o.warmup, total = N + W;
    const int64_t period_ns = (int64_t)llround(o.period_us * 1000.0);
    const int64_t spin_ns = (int64_t)llround(o.spin_us * 1000.0);

    // gpu mode: everything the executive needs is built before any load starts and before the targets exist
    unsigned long long *G_d = nullptr;
    LsRec *rec_d = nullptr;
    ExecState *state_d = nullptr;
    volatile int *finished_h = nullptr;
    int *finished_d = nullptr;
    std::string exec_error;
    // GPU residency probe (allocated before targets exist; launched just before the first boundary)
    ProbeDev probe_dev{};
    cudaStream_t s_probe = nullptr;
    volatile int *probe_stop_h = nullptr;
    unsigned long long probe_stats[4] = {0, 0, 0, 0};
    std::vector<ProbeGap> probe_gaps;
    if (o.probe) {
        int *stop_d = nullptr;
        CK(cudaHostAlloc((void **)&probe_stop_h, sizeof(int), cudaHostAllocMapped));
        *probe_stop_h = 0;
        CK(cudaHostGetDevicePointer((void **)&stop_d, (void *)probe_stop_h, 0));
        CK(cudaMalloc(&probe_dev.gaps, (size_t)o.probe_capacity * sizeof(ProbeGap)));
        CK(cudaMalloc(&probe_dev.stats, 4 * sizeof(unsigned long long)));
        CK(cudaMemset(probe_dev.stats, 0, 4 * sizeof(unsigned long long)));
        probe_dev.capacity = (unsigned)o.probe_capacity;
        probe_dev.gap_ns = (unsigned long long)llround(o.probe_gap_us * 1000.0);
        probe_dev.sleep_ns = (unsigned)o.probe_sleep_ns;
        probe_dev.stop = stop_d;
        int lo_p, hi_p;
        CK(cudaDeviceGetStreamPriorityRange(&lo_p, &hi_p));
        CK(cudaStreamCreateWithPriority(&s_probe, cudaStreamNonBlocking, lo_p));
        CK(cudaDeviceSynchronize());
    }
    if (o.mode == "gpu") {
        CK(cudaMalloc(&exec_pool_d, kChunk * sizeof(cudaGraphExec_t)));
        CK(cudaMemcpyAsync(exec_pool_d, dev_exec.data(), kChunk * sizeof(cudaGraphExec_t), cudaMemcpyHostToDevice, s));
        CK(cudaMalloc(&G_d, total * sizeof(unsigned long long)));
        CK(cudaMalloc(&rec_d, total * sizeof(LsRec)));
        CK(cudaMalloc(&state_d, sizeof(ExecState)));
        CK(cudaHostAlloc((void **)&finished_h, 64, cudaHostAllocMapped));
        *finished_h = 0;
        CK(cudaHostGetDevicePointer((void **)&finished_d, (void *)finished_h, 0));
        // Capture and upload before targets exist. The absolute deadline is supplied in state_d.
        cudaGraph_t eg;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
        launch_executive(s, exec_pool_d, G_d, (int)total, done_d, rec_d, state_d, kChunk, finished_d);
        cudaError_t le = cudaGetLastError();
        cudaGraph_t eg_tmp = nullptr;
        cudaError_t ce = cudaStreamEndCapture(s, &eg_tmp);
        eg = eg_tmp;
        if (le != cudaSuccess || ce != cudaSuccess) {
            exec_error = std::string("executive launch/capture: ") + cudaGetErrorString(le != cudaSuccess ? le : ce);
        } else {
            cudaError_t e = cudaGraphInstantiateWithFlags(&exec_graph_exec, eg, cudaGraphInstantiateFlagDeviceLaunch | inst_flags);
            if (e != cudaSuccess) exec_error = std::string("executive instantiate: ") + cudaGetErrorString(e);
        }
        if (!exec_error.empty()) {
            (void)cudaGetLastError();
            Json j;
            j.add("ok", false).add("mode", o.mode).add("label", o.label).add("gpu", prop.name).add("error", exec_error)
                .add("mps_control_present", mps_control_present).add("slot_variant", slot_variant);
            write_json_and_exit(o.out, j, 1);
        }
        CK(cudaGraphDestroy(eg));
        CK(cudaGraphUpload(exec_graph_exec, s));
        CK(cudaStreamSynchronize(s));
    }

    // in-process load, on another core when the driver is pinned
    std::atomic<bool> stop{false};
    std::atomic<int> load_error{0};
    std::atomic<unsigned long long> load_completed{0};
    std::thread load_th;
    float *big = nullptr;
    size_t bigN = (size_t)prop.multiProcessorCount * 2048 * 8;
    if (o.load == "stream") {
        CK(cudaMalloc(&big, bigN * sizeof(float)));
        CK(cudaMemset(big, 0, bigN * sizeof(float)));
        int load_core = o.core >= 0 ? (o.core + 1) % (int)std::max(1u, std::thread::hardware_concurrency()) : -1;
        load_th = std::thread([&, load_core] {
            cudaError_t e = cudaSetDevice(o.gpu);
            if (e != cudaSuccess) { load_error = (int)e; return; }
            pin_thread(load_core);
            while (!stop) {
                k_ai<<<(unsigned)(bigN / 256), 256, 0, s_load>>>(big, 20000);
                e = cudaGetLastError();
                if (e == cudaSuccess) e = wait_stream(s_load, now_ns() + 5000000000LL);
                if (e != cudaSuccess) { load_error = (int)e; return; }
                load_completed++;
            }
        });
        const int64_t ready_limit = now_ns() + 10000000000LL;
        while (load_completed < 3 && load_error == 0 && now_ns() < ready_limit)
            sleep_until_raw(now_ns() + 1000000);
        if (load_error != 0 || load_completed < 3) fail_run("in-process load failed to produce work");
    }

    // Allocate host records before mlockall and before targets. Start the worker before setting
    // FIFO/affinity so it does not inherit the timing thread's real-time scheduling policy.
    std::vector<int64_t> T(total);
    std::vector<unsigned long long> G(total);
    std::vector<LsRec> rec(total);
    bool mlock_ok = lock_memory();
    bool pin_ok = pin_thread(o.core);
    bool fifo_ok = set_fifo(o.fifo);
    prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
    const long launcher_tid = syscall(SYS_gettid);
    const int launcher_cpu = sched_getcpu();
    std::vector<SlotHostRec> hrec(total);
    memset(hrec.data(), 0, hrec.size() * sizeof(SlotHostRec));

    std::vector<Bracket> br = calibrate(s_cal, ctl_h, ctl_d, gt_h, gt_d, o.calib, o.calib_spread_s, cal_seq, exec_error);
    if (!exec_error.empty()) fail_run(exec_error);
    ClockFit fit = fit_clock(br);
    if (!fit.ok || !std::isfinite(fit.a) || fit.a <= 0) fail_run("clock calibration failed");

    // targets
    const int64_t t_start = now_ns() + 100000000LL;  // 100 ms from now
    for (long k = 0; k < total; k++) {
        T[k] = t_start + k * period_ns;
        G[k] = fit.gpu_of(T[k]);
    }
    memset(rec.data(), 0, rec.size() * sizeof(LsRec));
    for (long k = 0; k < total; k++) {
        rec[k].slot = (unsigned long long)k; rec[k].t_target = T[k]; rec[k].g_target = G[k];
        hrec[k].slot = (uint64_t)k; hrec[k].t_target = T[k]; hrec[k].cpu = -1;
    }
    ExecState stf{};
    stf.launched = -1;
    stf.deadline_g = G.back() + 5000000000ull;
    const auto load_before_run = load_completed.load();
    const std::string run_start_utc = iso_utc_now();
    const int64_t run_wall_start = realtime_ns();
    if (now_ns() >= t_start) fail_run("host preparation overran first target; reduce slots or load");
    if (o.probe) {
        probe_dev.deadline_g = G.back() + 10000000000ull;  // the probe also stops on its own
        k_probe<<<1, 32, 0, s_probe>>>(probe_dev);
        CK(cudaGetLastError());
    }

    if (o.mode == "cpu") {
        unsigned long long seq = seq_base;
        long launched = -1;
        long k = 0;
        while (k < total) {
            if (launched >= 0) {
                while (*slot_finished_h != seq) {
                    if (k < total && now_ns() >= T[k]) { rec[k].g_target = G[k]; rec[k].flags = 1ull; stf.n_skipped++; k++; if (k >= total) break; }
                }
                if (*slot_finished_h == seq) {
                    std::atomic_thread_fence(std::memory_order_acquire);
                    rec[launched].g0 = st->start_t;
                    rec[launched].g1 = st->end_t;
                    launched = -1;
                }
                if (k >= total) break;
            }
            sleep_until_raw(T[k] - spin_ns);
            int64_t tw = now_ns();
            spin_until(T[k]);
            int64_t t0 = now_ns();
            cudaError_t e = cudaGraphLaunch(host_exec, s);
            int64_t t1 = now_ns();
            hrec[k].t_wake = tw; hrec[k].t_launch = t0; hrec[k].t_return = t1; hrec[k].cpu = sched_getcpu();
            rec[k].g_target = G[k];
            rec[k].g_launch = fit.gpu_of(t0);
            rec[k].g_launch_done = fit.gpu_of(t1);
            if (e != cudaSuccess) { rec[k].flags = 2ull; stf.n_err++; if (!stf.first_err) stf.first_err = (int)e; (void)cudaGetLastError(); k++; continue; }
            seq++;
            launched = k;
            stf.n_launched++;
            k++;
        }
        if (launched >= 0) {
            int64_t lim = now_ns() + 5000000000LL;
            while (*slot_finished_h != seq && now_ns() < lim) {}
            if (*slot_finished_h == seq) {
                std::atomic_thread_fence(std::memory_order_acquire);
                rec[launched].g0 = st->start_t;
                rec[launched].g1 = st->end_t;
            } else {
                rec[launched].flags |= 8ull;
                fail_run("CPU-launched slot did not complete before wall guard");
            }
        }
        cudaError_t q = wait_stream(s, now_ns() + 5000000000LL);
        if (q != cudaSuccess) fail_run(std::string("CPU slot stream did not finish: ") + cudaGetErrorString(q));
        stf.finished = 1;
    } else {
        CK(cudaMemcpyAsync(G_d, G.data(), total * sizeof(unsigned long long), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(rec_d, rec.data(), total * sizeof(LsRec), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(state_d, &stf, sizeof stf, cudaMemcpyHostToDevice, s));
        cudaError_t uploaded = wait_stream(s, now_ns() + 5000000000LL);
        if (uploaded != cudaSuccess) fail_run("executive input upload did not finish");
        if (now_ns() >= t_start) fail_run("executive setup overran first target; reduce slots or load");
        CK(cudaGraphLaunch(exec_graph_exec, s));
        // the finished flag covers the whole tail-launched chain; the stream query is a second check
        int64_t lim = now_ns() + (int64_t)((double)total * period_ns * 3) + 15000000000LL;
        while (*finished_h == 0) {
            if (now_ns() > lim) { exec_error = "executive did not finish (wall guard)"; break; }
            cudaError_t q = cudaStreamQuery(s);
            if (q != cudaSuccess && q != cudaErrorNotReady) fail_run(std::string("executive: ") + cudaGetErrorString(q));
            if (q == cudaSuccess && *finished_h == 0) fail_run("executive stream ended without completion flag");
            sleep_until_raw(now_ns() + 100000);
        }
        if (exec_error.empty() && *finished_h == 2) exec_error = "executive tail relaunch failed";
        if (exec_error.empty() && *finished_h == 3) exec_error = "executive in-kernel timeout (a slot never completed)";
        if (exec_error.find("wall guard") != std::string::npos) {
            Json j;  // the executive may still be resident: do not touch the device, report and leave
            j.add("ok", false).add("mode", o.mode).add("label", o.label).add("gpu", prop.name).add("error", exec_error)
                .add("slot_variant", slot_variant);
            write_json_and_exit(o.out, j, 1);
        }
        cudaError_t q = wait_stream(s, now_ns() + 10000000000LL);
        if (q != cudaSuccess) fail_run(std::string("executive stream did not finish: ") + cudaGetErrorString(q));
        CK(cudaMemcpy(&stf, state_d, sizeof stf, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(rec.data(), rec_d, total * sizeof(LsRec), cudaMemcpyDeviceToHost));
        if (exec_error.empty() && stf.n_tail_err) exec_error = "executive tail relaunch error";
    }
    const std::string run_end_utc = iso_utc_now();
    const int64_t run_wall_end = realtime_ns();
    if (o.probe) {
        *probe_stop_h = 1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        cudaError_t pq = wait_stream(s_probe, now_ns() + 5000000000LL);
        if (pq != cudaSuccess) fail_run(std::string("residency probe did not stop: ") + cudaGetErrorString(pq));
        CK(cudaMemcpy(probe_stats, probe_dev.stats, sizeof probe_stats, cudaMemcpyDeviceToHost));
        size_t keep = (size_t)std::min<unsigned long long>(probe_stats[0], (unsigned long long)o.probe_capacity);
        probe_gaps.resize(keep);
        if (keep) CK(cudaMemcpy(probe_gaps.data(), probe_dev.gaps, keep * sizeof(ProbeGap), cudaMemcpyDeviceToHost));
    }
    const auto load_during_run = load_completed.load() - load_before_run;

    std::string calibration_error;
    std::vector<Bracket> br2 = calibrate(s_cal, ctl_h, ctl_d, gt_h, gt_d, std::max(2000, o.calib / 4), o.calib_spread_s, cal_seq, calibration_error);
    if (!calibration_error.empty()) fail_run(calibration_error);
    ClockFit fit2 = fit_clock(br2);
    TwoPoint tp = two_point(fit, fit2);
    stop = true;
    if (load_th.joinable()) load_th.join();
    if (load_error != 0) fail_run(std::string("in-process load did not finish: ")
                                + cudaGetErrorString((cudaError_t)load_error.load()));
    if (!tp.ok && exec_error.empty()) exec_error = "post-run clock mapping failed";
    if (o.load == "stream" && load_during_run == 0)
        exec_error = "in-process load failed or completed no work during measurement";

    // ---- summarise recorded slots (after warm-up) ----
    std::vector<double> start_err, launch_prec, target_pred, launch_err, launch_call, launch_to_start, exec_us, lat_target, first_gen;
    long recorded = 0, misses = 0, skipped = 0, errors = 0, timeouts = 0;
    const double deadline_ns = o.deadline_us * 1000.0;
    for (long k = W; k < total; k++) {
        LsRec &r = rec[k];
        if (r.flags & 1ull) { skipped++; misses++; continue; }
        if (r.flags & 2ull) { errors++; misses++; continue; }
        if (r.flags & 8ull) { timeouts++; misses++; continue; }
        if (!r.g0 || !r.g1) { r.flags |= 8ull; timeouts++; misses++; continue; }
        recorded++;
        const unsigned long long g_true = tp.ok ? tp.gpu_of(r.t_target) : r.g_target;
        double se = dus(r.g0, g_true);
        start_err.push_back(se);
        launch_prec.push_back(dus(r.g0, r.g_target));
        target_pred.push_back(dus(r.g_target, g_true));
        launch_err.push_back(dus(r.g_launch, r.g_target));
        launch_call.push_back(dus(r.g_launch_done, r.g_launch));
        launch_to_start.push_back(dus(r.g0, r.g_launch));
        exec_us.push_back(dus(r.g1, r.g0));
        double lt = dus(r.g1, g_true);
        lat_target.push_back(lt);
        if (r.flags & 4ull) first_gen.push_back(se);
        if (lt * 1000.0 > deadline_ns) misses++;
    }
    long total_slots = recorded + skipped + errors + timeouts;
    bool ok = exec_error.empty() && errors == 0 && stf.n_err == 0 && recorded > 0 && timeouts == 0
        && tp.ok && total_slots == N && stf.finished == 1;

    if (!o.raw.empty()) {
        FILE *f = fopen(o.raw.c_str(), "wb");
        if (!f) { perror("raw"); return 2; }
        bool written = fwrite(rec.data() + W, sizeof(LsRec), (size_t)N, f) == (size_t)N;
        if (fclose(f) != 0) written = false;
        if (!written) { fprintf(stderr, "lockstep_driver: raw write failed\n"); return 2; }
    }
    if (!o.host_raw.empty()) {
        for (long k = 0; k < total; k++) hrec[k].flags = (uint32_t)rec[k].flags;
        FILE *hf = fopen(o.host_raw.c_str(), "wb");
        if (!hf) { perror("host-raw"); return 2; }
        bool written = fwrite(hrec.data() + W, sizeof(SlotHostRec), (size_t)N, hf) == (size_t)N;
        if (fclose(hf) != 0) written = false;
        if (!written) { fprintf(stderr, "lockstep_driver: host-raw write failed\n"); return 2; }
    }
    if (o.probe) {
        ProbeHeader ph{};
        memcpy(ph.magic, "SBPROBE1", 8);
        ph.gap_ns = probe_dev.gap_ns; ph.n_gaps = probe_stats[0]; ph.g_first = probe_stats[1];
        ph.g_last = probe_stats[2]; ph.iterations = probe_stats[3]; ph.capacity = probe_gaps.size();
        FILE *pf = fopen(o.probe_out.c_str(), "wb");
        if (!pf) { perror("probe-out"); return 2; }
        bool written = fwrite(&ph, sizeof ph, 1, pf) == 1
            && (probe_gaps.empty() || fwrite(probe_gaps.data(), sizeof(ProbeGap), probe_gaps.size(), pf) == probe_gaps.size());
        if (fclose(pf) != 0) written = false;
        if (!written) { fprintf(stderr, "lockstep_driver: probe-out write failed\n"); return 2; }
    }
    Json j;
    j.add("ok", ok).add("error", exec_error).add("mode", o.mode).add("load", o.load).add("label", o.label)
        .add("gpu", prop.name).add("sm", prop.multiProcessorCount).add("driver", drv).add("runtime", rt)
        .add("globaltimer_tick_ns", (unsigned long long)tick_ns)
        .add("slots", (long long)N).add("warmup", (long long)W).add("period_us", o.period_us).add("deadline_us", o.deadline_us)
        .add("spin_us", o.spin_us).add("stream_priority", prio).add("instantiate_use_node_priority", true)
        .add("graph_nodes", (long long)n_nodes).add_raw("graph_node_types", node_types).add("slot_variant", slot_variant)
        .add_raw("phy", pipe.config().json())
        .add("inproc_completed_during_run", load_during_run).add("inproc_error", load_error.load())
        .add("executive_pool_size", o.mode == "gpu" ? kChunk : 0)
        .add("core", o.core).add("pin_ok", pin_ok).add("fifo", o.fifo).add("fifo_ok", fifo_ok).add("mlock_ok", mlock_ok)
        .add("mps_pipe_env", mps_pipe ? mps_pipe : "").add("mps_control_present", mps_control_present)
        .add("run_start_utc", run_start_utc).add("run_end_utc", run_end_utc)
        .add("run_wall_start_ns", (long long)run_wall_start).add("run_wall_end_ns", (long long)run_wall_end)
        .add("recorded", (long long)recorded).add("skipped", (long long)skipped).add("launch_errors", (long long)errors)
        .add("timeouts", (long long)timeouts).add("total_slots", (long long)total_slots).add("misses", (long long)misses)
        .add("miss_rate", total_slots ? (double)misses / (double)total_slots : NAN)
        .add_raw("start_error_us", stats_json(start_err))           // g0 - g2pt(T_k)
        .add_raw("launch_precision_us", stats_json(launch_prec))    // g0 - g_target (the instant the launcher was given)
        .add_raw("target_pred_error_us", stats_json(target_pred))   // g_target - g2pt(T_k): pre-fit prediction error
        .add_raw("launch_error_us", stats_json(launch_err))         // g_launch - g_target: when the launcher acted
        .add_raw("launch_call_us", stats_json(launch_call))         // cost of the launch call itself
        .add_raw("launch_to_start_us", stats_json(launch_to_start))
        .add_raw("exec_us", stats_json(exec_us))
        .add_raw("latency_from_target_us", stats_json(lat_target))
        .add_raw("first_of_generation_start_error_us", stats_json(first_gen))
        .add_raw("clock_fit_pre", fit.json()).add_raw("clock_fit_post", fit2.json())
        .add("two_point_ok", tp.ok).add("two_point_rate_ppm", tp.ok ? (tp.rate - 1.0) * 1e6 : NAN)
        .add("executive_launched", (long long)stf.n_launched).add("executive_skipped", (long long)stf.n_skipped)
        .add("executive_errors", (long long)stf.n_err).add("executive_first_error", stf.first_err)
        .add("executive_chunks", stf.chunks).add("executive_finished", stf.finished)
        .add("launcher_tid", (long long)launcher_tid).add("launcher_cpu", launcher_cpu)
        .add("host_raw", o.host_raw).add("probe", o.probe).add("probe_out", o.probe_out)
        .add("probe_gap_us", o.probe_gap_us).add("probe_sleep_ns", o.probe_sleep_ns)
        .add("probe_gaps", probe_stats[0]).add("probe_gaps_stored", (unsigned long long)probe_gaps.size())
        .add("probe_iterations", probe_stats[3]).add("probe_g_first", probe_stats[1]).add("probe_g_last", probe_stats[2]);
    FILE *f = fopen(o.out.c_str(), "w");
    if (!f) { perror("out"); return 2; }
    fputs(j.str().c_str(), f);
    fputs("\n", f);
    fclose(f);
    printf("%-4s load=%-6s %-16s start_err p50 %8.2f p99 %8.2f p99.9 %8.2f max %9.2f us | launch_prec p50 %7.2f | exec p50 %6.1f | miss %ld/%ld skipped %ld err %ld%s%s\n",
           o.mode.c_str(), o.load.c_str(), o.label.c_str(), pct(start_err, 50), pct(start_err, 99), pct(start_err, 99.9),
           start_err.empty() ? NAN : *std::max_element(start_err.begin(), start_err.end()), pct(launch_prec, 50), pct(exec_us, 50),
           misses, total_slots, skipped, errors, ok ? "" : "  NOT OK", exec_error.empty() ? "" : (" " + exec_error).c_str());
    return ok ? 0 : 1;
}
