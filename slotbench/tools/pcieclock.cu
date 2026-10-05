// pcieclock: how precisely can the GPU's %globaltimer be placed on the host clock (CLOCK_MONOTONIC_RAW)
// over plain PCIe, with no NIC timestamping? Three samplers run interleaved in one process:
//
//   classic  host-initiated round trip (what slotbench's clock_fit uses): host stamps t0, raises a request,
//            the GPU answers with %globaltimer g, host stamps t1. Constraint: host(g) in [t0, t1 + tick].
//   up       GPU -> host, tick-edge aligned: the GPU spins until %globaltimer changes, so the new value E is
//            the GPU time of that edge, then writes E to pinned host memory. The host thread spins on the
//            line and stamps h when it changes. Constraint: host(E) <= h.
//   down     host -> GPU, tick-edge aligned: a host thread keeps writing its clock t into pinned memory. The
//            GPU loads t (the value existed before the load sampled memory), then spins to the next timer
//            edge E. Constraint: host(E) >= t.
//
// For a linear clock relation host = a*E + b, "up" samples bound b from above and "down" samples from below;
// the gap between the tightest bounds is a hard bound on the mapping error that needs no symmetry assumption.
// Output: one binary file per sampler (int64 pairs/triples) plus a JSON header; analysis/pcieclock.py fits.
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "host_time.h"
#include "json_writer.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "pcieclock: %s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

// Shared pinned (mapped) memory layout. Each field on its own 64-byte line.
struct alignas(64) Line { volatile uint64_t v; char pad[56]; };
struct Shared {
    Line host_clock;   // down: host thread keeps writing CLOCK_MONOTONIC_RAW here
    Line req;          // classic: host request sequence
    Line ack_seq;      // classic: GPU answer sequence
    Line ack_g;        // classic: GPU %globaltimer for the answer
    Line up_seq;       // up: GPU sample sequence
    Line up_e;         // up: edge value
    Line phase;        // 0 idle, 1 classic, 2 up, 3 down, 9 stop
    Line gpu_phase;    // GPU's acknowledgement of the phase
};

__device__ __forceinline__ uint64_t gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ uint64_t ld_sys(const volatile uint64_t *p) {
    uint64_t v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_sys(volatile uint64_t *p, uint64_t v) {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// Spin until %globaltimer changes; return the new value (the GPU time of that edge).
__device__ __forceinline__ uint64_t next_edge() {
    uint64_t a = gtimer(), b;
    do { b = gtimer(); } while (b == a);
    return b;
}

// One thread. Follows the phase word; writes "down" results to device-visible pinned arrays.
__global__ void k_clock(Shared *s, uint64_t *down_t, uint64_t *down_e, int n_down_max, int *n_down_out,
                        uint64_t *tick_probe) {
    for (int i = 0; i < 64; i++) tick_probe[i] = next_edge();   // 64 consecutive timer edges
    int n_down = 0;
    uint64_t up_seq = 0, last_req = 0, acked = ~0ull;
    for (;;) {
        uint64_t ph = ld_sys(&s->phase.v);
        if (ph == 9) break;
        if (ph != acked) { st_sys(&s->gpu_phase.v, ph); __threadfence_system(); acked = ph; }
        if (ph == 1) {                       // classic: poll only the request line (one PCIe read per poll)
            for (;;) {
                uint64_t r = ld_sys(&s->req.v);
                if (r == ~0ull) break;           // host leaves the classic phase
                if (r != last_req) {
                    uint64_t g = gtimer();
                    // one 64-bit store, so value and sequence can never arrive torn or reordered
                    st_sys(&s->ack_g.v, ((g >> 10) << 12) | (r & 0xfff) | ((g & 1023) ? (1ull << 63) : 0));
                    __threadfence_system();
                    last_req = r;
                }
            }
        } else if (ph == 2) {                // up: edge, then publish immediately
            uint64_t e = next_edge();
            st_sys(&s->up_e.v, e);            // single store: the value is its own sequence number
            __threadfence_system();
            ++up_seq;
            // let the host see it before the next sample (spin ~3 us of GPU time)
            uint64_t until = e + 3000;
            while (gtimer() < until) {}
        } else if (ph == 3) {                // down: read host clock, then wait for the next edge
            if (n_down < n_down_max) {
                uint64_t t = ld_sys(&s->host_clock.v);
                // The SM does not wait for a load before reading %globaltimer unless something consumes the
                // value: branch on it so the edge search starts only after the host clock value arrived.
                if (t == ~0ull) asm volatile("trap;");
                uint64_t e = next_edge();
                down_t[n_down] = t;
                down_e[n_down] = e;
                n_down++;
            }
        }
    }
    *n_down_out = n_down;
    __threadfence_system();
}

static int64_t now() { return sb::now_ns(); }

int main(int argc, char **argv) {
    int rounds = 20, per_phase = 5000, gpu = 0, core = -1, clock_core = -1;
    std::string out = "pcieclock";
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string f = argv[i], v = argv[i + 1];
        if (f == "--rounds") rounds = atoi(v.c_str());
        else if (f == "--per-phase") per_phase = atoi(v.c_str());
        else if (f == "--gpu") gpu = atoi(v.c_str());
        else if (f == "--core") core = atoi(v.c_str());
        else if (f == "--clock-core") clock_core = atoi(v.c_str());
        else if (f == "--out") out = v;
        else { fprintf(stderr, "usage: pcieclock [--rounds N] [--per-phase N] [--gpu N] [--core C] [--clock-core C] [--out PREFIX]\n"); return 2; }
    }
    CK(cudaSetDevice(gpu));
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, gpu));
    if (core >= 0) sb::pin_thread(core);

    Shared *s = nullptr, *s_d = nullptr;
    CK(cudaHostAlloc((void **)&s, sizeof(Shared), cudaHostAllocMapped));
    memset((void *)s, 0, sizeof(Shared));
    CK(cudaHostGetDevicePointer((void **)&s_d, s, 0));
    const int n_down_max = rounds * per_phase;
    uint64_t *down_t = nullptr, *down_e = nullptr, *down_t_d, *down_e_d;
    int *n_down = nullptr, *n_down_d;
    CK(cudaHostAlloc((void **)&down_t, sizeof(uint64_t) * n_down_max, cudaHostAllocMapped));
    CK(cudaHostAlloc((void **)&down_e, sizeof(uint64_t) * n_down_max, cudaHostAllocMapped));
    CK(cudaHostAlloc((void **)&n_down, sizeof(int), cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void **)&down_t_d, down_t, 0));
    CK(cudaHostGetDevicePointer((void **)&down_e_d, down_e, 0));
    CK(cudaHostGetDevicePointer((void **)&n_down_d, n_down, 0));

    // Host clock writer for "down" (its own core when given): writes CLOCK_MONOTONIC_RAW continuously.
    std::atomic<bool> clock_run{true};
    std::thread clk([&] {
        if (clock_core >= 0) sb::pin_thread(clock_core);
        while (clock_run.load(std::memory_order_relaxed)) s->host_clock.v = (uint64_t)now();
    });

    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    uint64_t *tick_probe = nullptr, *tick_probe_d;
    CK(cudaHostAlloc((void **)&tick_probe, sizeof(uint64_t) * 64, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void **)&tick_probe_d, tick_probe, 0));
    k_clock<<<1, 1, 0, st>>>(s_d, down_t_d, down_e_d, n_down_max, n_down_d, tick_probe_d);
    CK(cudaGetLastError());

    auto set_phase = [&](uint64_t p) {
        s->phase.v = p;
        int64_t limit = now() + 2000000000LL;
        while (s->gpu_phase.v != p) {
            if (now() > limit) { fprintf(stderr, "pcieclock: GPU did not enter phase %llu\n", (unsigned long long)p); exit(1); }
        }
    };
    std::vector<int64_t> c_t0, c_t1, up_h;
    std::vector<uint64_t> c_g, up_e;
    c_t0.reserve(rounds * per_phase); c_t1.reserve(rounds * per_phase); c_g.reserve(rounds * per_phase);
    up_h.reserve(rounds * per_phase); up_e.reserve(rounds * per_phase);
    const int64_t t_start = now();
    uint64_t req = 0, up_seen = 0;
    long long bad_classic = 0;
    for (int r = 0; r < rounds; r++) {
        s->req.v = req;                                  // last answered request: GPU waits for req+1
        set_phase(1);                                    // classic
        for (int i = 0; i < per_phase; i++) {
            int64_t t0 = now();
            s->req.v = ++req;
            uint64_t w;
            while (((w = s->ack_g.v) & 0xfff) != (req & 0xfff)) {}
            int64_t t1 = now();
            if (w >> 63) bad_classic++;
            c_t0.push_back(t0); c_t1.push_back(t1); c_g.push_back(((w & ~(1ull << 63)) >> 12) << 10);
        }
        s->req.v = ~0ull;                                // release the GPU from its classic poll loop
        set_phase(2);                                    // up
        up_seen = s->up_e.v;
        for (int i = 0; i < per_phase; i++) {
            uint64_t e;
            while ((e = s->up_e.v) == up_seen) {}
            int64_t h = now();
            up_seen = e;
            up_h.push_back(h); up_e.push_back(e);
        }
        set_phase(3);                                    // down: GPU samples while the clock thread writes
        int64_t until = now() + (int64_t)per_phase * 3000;  // ~3 us per sample
        while (now() < until) {}
    }
    s->phase.v = 9;
    CK(cudaStreamSynchronize(st));
    const int64_t t_end = now();
    clock_run = false;
    clk.join();

    auto dump = [&](const std::string &name, const std::vector<std::vector<int64_t>> &cols, size_t n) {
        FILE *f = fopen((out + "." + name + ".bin").c_str(), "wb");
        if (!f) { perror("fopen"); exit(1); }
        for (size_t i = 0; i < n; i++)
            for (auto &c : cols) fwrite(&c[i], sizeof(int64_t), 1, f);
        fclose(f);
    };
    std::vector<int64_t> cg(c_g.begin(), c_g.end()), ue(up_e.begin(), up_e.end());
    std::vector<int64_t> dt(down_t, down_t + *n_down), de(down_e, down_e + *n_down);
    dump("classic", {c_t0, c_t1, cg}, c_t0.size());
    dump("up", {up_h, ue}, up_h.size());
    dump("down", {dt, de}, dt.size());

    sb::Json j;
    j.add("gpu", prop.name).add("sm", prop.multiProcessorCount)
        .add("pci_bus", prop.pciBusID).add("rounds", rounds).add("per_phase", per_phase)
        .add("n_classic", (long long)c_t0.size()).add("n_up", (long long)up_h.size()).add("n_down", (long long)*n_down)
        .add("core", core).add("clock_core", clock_core).add("seconds", (t_end - t_start) * 1e-9)
        .add("host", sb::hostname()).add("time", sb::iso_utc_now()).add("bad_classic_unaligned_g", bad_classic);
    std::string ticks;
    for (int i = 1; i < 64; i++) ticks += (i > 1 ? "," : "") + std::to_string((long long)(tick_probe[i] - tick_probe[i - 1]));
    j.add_raw("timer_edge_steps_ns", "[" + ticks + "]");
    FILE *f = fopen((out + ".json").c_str(), "w");
    fprintf(f, "%s\n", j.str().c_str());
    fclose(f);
    printf("pcieclock: %s classic=%zu up=%zu down=%d in %.2f s\n", prop.name, c_t0.size(), up_h.size(), *n_down,
           (t_end - t_start) * 1e-9);
    return 0;
}
