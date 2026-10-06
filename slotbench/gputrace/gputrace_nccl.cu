// gputrace_nccl: NCCL collectives across the GPUs of one host on one bounded time axis.
// Every GPU gets its own clock sync (clocksync.cuh) before and after; around each ncclAllReduce a one-block
// stamp kernel on each GPU's stream records %globaltimer before the collective is enqueued and after it
// completes. The analysis (analysis/gputrace.py, strategy "nccl") fits each GPU's timer to the host clock and
// reports, per collective size: duration seen by each GPU, start skew and end skew between GPUs (with the sum
// of the two bounds), and host launch-call → first GPU start.
// Output: PREFIX.gpu.bin (stamp records, flags = device, tag 1 before / 2 after, block = iteration, kernel_id =
// collective id), PREFIX.host.bin, PREFIX.gpu{d}.r{0,1}.{classic,up,down}.bin, PREFIX.json.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

#include "host_time.h"
#include "json_writer.h"
#include "gputrace.cuh"
#include "clocksync.cuh"

using namespace sb;
using namespace sb::gt;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "gputrace_nccl: %s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define NK(x) do { ncclResult_t e_ = (x); if (e_ != ncclSuccess) { \
    fprintf(stderr, "gputrace_nccl: %s: %s\n", #x, ncclGetErrorString(e_)); exit(1); } } while (0)

__global__ void k_stamp(TraceDev td, uint32_t kid, uint32_t tag, uint32_t dev, uint32_t iter) {
    if (threadIdx.x != 0) return;
    BlockTrace bt;
    trace_begin(td, bt);
    uint64_t g1 = gtimer();
    if (bt.slot >= td.capacity) return;
    GpuRec r{};
    r.g_begin = bt.g0; r.g_end = g1; r.clk_begin = bt.c0; r.clk_end = clock64(); r.smid = smid();
    r.kernel_id = kid; r.block = iter; r.tag = tag; r.flags = dev;
    td.recs[bt.slot] = r;
}

static std::vector<HostEvent> g_ev;
static inline void ev(uint32_t type, uint32_t kid, uint64_t a = 0, uint64_t b = 0) {
    g_ev.push_back(HostEvent{now_ns(), type, kid, a, b});
}

int main(int argc, char **argv) {
    int iters = 200, sync_rounds = 3, sync_per_phase = 1000, clock_core = -1, core = -1;
    std::string out = "gputrace_nccl", sizes_s = "8,65536,1048576,67108864";
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string f = argv[i], v = argv[i + 1];
        if (f == "--iters") iters = atoi(v.c_str());
        else if (f == "--sizes") sizes_s = v;
        else if (f == "--out") out = v;
        else if (f == "--core") core = atoi(v.c_str());
        else if (f == "--clock-core") clock_core = atoi(v.c_str());
        else if (f == "--sync-rounds") sync_rounds = atoi(v.c_str());
        else if (f == "--sync-per-phase") sync_per_phase = atoi(v.c_str());
        else { fprintf(stderr, "usage: gputrace_nccl [--iters N] [--sizes B,B,..] [--out PREFIX] [--core C] [--clock-core C]\n"); return 2; }
    }
    std::vector<size_t> sizes;
    for (size_t i = 0; i < sizes_s.size();) { size_t j = sizes_s.find(',', i); if (j == std::string::npos) j = sizes_s.size(); sizes.push_back((size_t)atol(sizes_s.substr(i, j - i).c_str())); i = j + 1; }
    size_t max_bytes = 0;
    for (size_t s : sizes) max_bytes = std::max(max_bytes, s);
    if (core >= 0) pin_thread(core);
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    if (n < 2) { fprintf(stderr, "gputrace_nccl: need at least 2 GPUs, found %d\n", n); return 3; }

    std::vector<TraceDev> td(n);
    std::vector<cudaStream_t> st(n);
    std::vector<float *> sbuf(n), rbuf(n);
    std::vector<ncclComm_t> comm(n);
    std::vector<int> devs(n);
    const unsigned capacity = (unsigned)(2u * (unsigned)iters * (unsigned)sizes.size() + 64);
    for (int d = 0; d < n; d++) {
        devs[d] = d;
        CK(cudaSetDevice(d));
        CK(cudaSetDeviceFlags(cudaDeviceMapHost));
        CK(cudaMalloc(&td[d].recs, sizeof(GpuRec) * capacity));
        CK(cudaMalloc(&td[d].count, sizeof(unsigned)));
        CK(cudaMemset(td[d].count, 0, sizeof(unsigned)));
        td[d].capacity = capacity;
        CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking));
        CK(cudaMalloc(&sbuf[d], max_bytes + 64));
        CK(cudaMalloc(&rbuf[d], max_bytes + 64));
        CK(cudaMemset(sbuf[d], 0, max_bytes + 64));
        k_stamp<<<1, 32, 0, st[d]>>>(td[d], 0, 0, (uint32_t)d, 0);   // warm
        CK(cudaStreamSynchronize(st[d]));
        CK(cudaMemset(td[d].count, 0, sizeof(unsigned)));
    }
    NK(ncclCommInitAll(comm.data(), n, devs.data()));
    // one warm collective (NCCL lazily sets up channels on the first call)
    NK(ncclGroupStart());
    for (int d = 0; d < n; d++) { CK(cudaSetDevice(d)); NK(ncclAllReduce(sbuf[d], rbuf[d], 16, ncclFloat, ncclSum, comm[d], st[d])); }
    NK(ncclGroupEnd());
    for (int d = 0; d < n; d++) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(st[d])); }

    const int64_t t_start = now_ns();
    auto sync_all = [&](int r) {
        for (int d = 0; d < n; d++) {
            CK(cudaSetDevice(d));
            ClockSamples cs; std::string err;
            ev(EV_MARK, (uint32_t)d, 200, (uint64_t)r);
            bool ok = clocksync(sync_rounds, sync_per_phase, clock_core, cs, &err);
            ev(EV_MARK, (uint32_t)d, 201, ok);
            if (!ok) fprintf(stderr, "gputrace_nccl: gpu %d clocksync round %d: %s\n", d, r, err.c_str());
            else cs.dump(out + ".gpu" + std::to_string(d) + ".r" + std::to_string(r));
        }
    };
    sync_all(0);
    ev(EV_MARK, 0, 100, 0);
    uint32_t kid = 0;
    for (size_t s : sizes) {
        const size_t count = std::max<size_t>(1, s / sizeof(float));
        for (int i = 0; i < iters; i++) {
            ++kid;
            for (int d = 0; d < n; d++) { CK(cudaSetDevice(d)); k_stamp<<<1, 32, 0, st[d]>>>(td[d], kid, 1, (uint32_t)d, (uint32_t)i); }
            ev(EV_LAUNCH_ENTER, kid, (uint64_t)s, (uint64_t)n);
            NK(ncclGroupStart());
            for (int d = 0; d < n; d++) { CK(cudaSetDevice(d)); NK(ncclAllReduce(sbuf[d], rbuf[d], count, ncclFloat, ncclSum, comm[d], st[d])); }
            NK(ncclGroupEnd());
            ev(EV_LAUNCH_RETURN, kid, (uint64_t)s, (uint64_t)n);
            for (int d = 0; d < n; d++) { CK(cudaSetDevice(d)); k_stamp<<<1, 32, 0, st[d]>>>(td[d], kid, 2, (uint32_t)d, (uint32_t)i); }
            ev(EV_SYNC_ENTER, kid);
            for (int d = 0; d < n; d++) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(st[d])); }
            ev(EV_SYNC_RETURN, kid);
        }
    }
    ev(EV_MARK, 0, 101, 0);
    sync_all(1);
    const int64_t t_end = now_ns();

    std::vector<GpuRec> all;
    for (int d = 0; d < n; d++) {
        CK(cudaSetDevice(d));
        unsigned c; CK(cudaMemcpy(&c, td[d].count, sizeof c, cudaMemcpyDeviceToHost));
        c = std::min(c, capacity);
        std::vector<GpuRec> recs(c);
        if (c) CK(cudaMemcpy(recs.data(), td[d].recs, sizeof(GpuRec) * c, cudaMemcpyDeviceToHost));
        all.insert(all.end(), recs.begin(), recs.end());
    }
    FILE *f = fopen((out + ".gpu.bin").c_str(), "wb");
    fwrite(all.data(), sizeof(GpuRec), all.size(), f); fclose(f);
    f = fopen((out + ".host.bin").c_str(), "wb");
    fwrite(g_ev.data(), sizeof(HostEvent), g_ev.size(), f); fclose(f);
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int ncclv = 0; ncclGetVersion(&ncclv);
    Json j;
    j.add("strategy", "nccl").add("gpu", prop.name).add("sms", prop.multiProcessorCount).add("n_gpus", n).add("reps", 2)
        .add("iters", iters).add("sizes", sizes_s).add("nccl_version", ncclv).add("host", hostname()).add("time", iso_utc_now())
        .add("n_gpu_recs", (long long)all.size()).add("n_host_events", (long long)g_ev.size()).add("kernels", (long long)kid)
        .add("sync_rounds", sync_rounds).add("sync_per_phase", sync_per_phase).add("seconds_total", (t_end - t_start) * 1e-9)
        .add("t_start", (long long)t_start).add("t_end", (long long)t_end);
    f = fopen((out + ".json").c_str(), "w"); fprintf(f, "%s\n", j.str().c_str()); fclose(f);
    printf("gputrace_nccl: %d x %s, %u collectives, %zu stamps in %.1f s\n", n, prop.name, kid, all.size(), (t_end - t_start) * 1e-9);
    for (int d = 0; d < n; d++) ncclCommDestroy(comm[d]);
    return 0;
}
