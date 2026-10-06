// Host <-> GPU clock sync for gputrace: the tools/pcieclock.cu samplers (classic brackets, tick-edge "up" and
// "down") as a class, run before and after every strategy. The kernel and the host protocol are the ones
// validated in data/2026-10-05_a100_pcieclock (0 constraint violations across 11 runs on 3 hosts); only the
// packaging differs. Output files use pcieclock's layouts so analysis/pcieclock.py's edge_fit/classic_fit apply.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "host_time.h"
#include "gputrace.cuh"

namespace sb {
namespace gt {

struct alignas(64) Line { volatile uint64_t v; char pad[56]; };
struct ClockShared {
    Line host_clock;   // down: host thread keeps writing CLOCK_MONOTONIC_RAW here
    Line req;          // classic: host request sequence
    Line ack_g;        // classic: GPU %globaltimer for the answer (packed with the sequence)
    Line up_e;         // up: edge value
    Line phase;        // 0 idle, 1 classic, 2 up, 3 down, 9 stop
    Line gpu_phase;    // GPU's acknowledgement of the phase
};

// One thread. Identical protocol to tools/pcieclock.cu k_clock.
__global__ void k_clocksync(ClockShared *s, uint64_t *down_t, uint64_t *down_e, int n_down_max, int *n_down_out,
                            uint64_t *tick_probe) {
    for (int i = 0; i < 64; i++) tick_probe[i] = next_edge();
    int n_down = 0;
    uint64_t last_req = 0, acked = ~0ull;
    for (;;) {
        uint64_t ph = ld_sys(&s->phase.v);
        if (ph == 9) break;
        if (ph != acked) { st_sys(&s->gpu_phase.v, ph); __threadfence_system(); acked = ph; }
        if (ph == 1) {
            for (;;) {
                uint64_t r = ld_sys(&s->req.v);
                if (r == ~0ull) break;
                if (r != last_req) {
                    uint64_t g = gtimer();
                    st_sys(&s->ack_g.v, ((g >> 10) << 12) | (r & 0xfff) | ((g & 1023) ? (1ull << 63) : 0));
                    __threadfence_system();
                    last_req = r;
                }
            }
        } else if (ph == 2) {
            uint64_t e = next_edge();
            st_sys(&s->up_e.v, e);
            __threadfence_system();
            uint64_t until = e + 3000;
            while (gtimer() < until) {}
        } else if (ph == 3) {
            if (n_down < n_down_max) {
                uint64_t t = ld_sys(&s->host_clock.v);
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

struct ClockSamples {
    std::vector<int64_t> c_t0, c_t1, up_h;
    std::vector<uint64_t> c_g, up_e, down_t, down_e;
    std::vector<int64_t> tick_steps;   // 63 consecutive timer-edge steps
    long long bad_classic = 0;
    int64_t t_start = 0, t_end = 0;

    // Write PREFIX.classic.bin (t0,t1,g), PREFIX.up.bin (h,E), PREFIX.down.bin (t,E) as int64 columns.
    bool dump(const std::string &prefix) const {
        auto wr = [&](const std::string &name, const std::vector<std::vector<int64_t>> &cols, size_t n) {
            FILE *f = fopen((prefix + "." + name + ".bin").c_str(), "wb");
            if (!f) return false;
            for (size_t i = 0; i < n; i++)
                for (auto &c : cols) fwrite(&c[i], sizeof(int64_t), 1, f);
            fclose(f);
            return true;
        };
        std::vector<int64_t> cg(c_g.begin(), c_g.end()), ue(up_e.begin(), up_e.end());
        std::vector<int64_t> dt(down_t.begin(), down_t.end()), de(down_e.begin(), down_e.end());
        return wr("classic", {c_t0, c_t1, cg}, c_t0.size()) && wr("up", {up_h, ue}, up_h.size()) &&
               wr("down", {dt, de}, dt.size());
    }
};

// Runs the resident kernel on its own non-blocking stream for `rounds` x (classic, up, down) phases.
// clock_core: CPU for the host clock writer thread (-1: unpinned). Returns false on protocol timeout.
inline bool clocksync(int rounds, int per_phase, int clock_core, ClockSamples &out, std::string *err = nullptr) {
    ClockShared *s = nullptr, *s_d = nullptr;
    if (cudaHostAlloc((void **)&s, sizeof(ClockShared), cudaHostAllocMapped) != cudaSuccess) { if (err) *err = "hostalloc"; return false; }
    memset((void *)s, 0, sizeof(ClockShared));
    cudaHostGetDevicePointer((void **)&s_d, s, 0);
    const int n_down_max = rounds * per_phase;
    uint64_t *down_t, *down_e, *down_t_d, *down_e_d, *tick, *tick_d;
    int *n_down, *n_down_d;
    cudaHostAlloc((void **)&down_t, sizeof(uint64_t) * n_down_max, cudaHostAllocMapped);
    cudaHostAlloc((void **)&down_e, sizeof(uint64_t) * n_down_max, cudaHostAllocMapped);
    cudaHostAlloc((void **)&n_down, sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void **)&tick, sizeof(uint64_t) * 64, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&down_t_d, down_t, 0);
    cudaHostGetDevicePointer((void **)&down_e_d, down_e, 0);
    cudaHostGetDevicePointer((void **)&n_down_d, n_down, 0);
    cudaHostGetDevicePointer((void **)&tick_d, tick, 0);
    *n_down = 0;

    std::atomic<bool> clock_run{true};
    std::thread clk([&] {
        if (clock_core >= 0) pin_thread(clock_core);
        while (clock_run.load(std::memory_order_relaxed)) s->host_clock.v = (uint64_t)now_ns();
    });
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    k_clocksync<<<1, 1, 0, st>>>(s_d, down_t_d, down_e_d, n_down_max, n_down_d, tick_d);
    bool ok = cudaGetLastError() == cudaSuccess;
    auto set_phase = [&](uint64_t p) {
        s->phase.v = p;
        int64_t limit = now_ns() + 3000000000LL;
        while (s->gpu_phase.v != p)
            if (now_ns() > limit) return false;
        return true;
    };
    out = ClockSamples{};
    out.t_start = now_ns();
    uint64_t req = 0, up_seen = 0;
    for (int r = 0; ok && r < rounds; r++) {
        s->req.v = req;
        if (!(ok = set_phase(1))) break;
        for (int i = 0; i < per_phase; i++) {
            int64_t t0 = now_ns();
            s->req.v = ++req;
            uint64_t w;
            int64_t limit = t0 + 1000000000LL;
            while (((w = s->ack_g.v) & 0xfff) != (req & 0xfff))
                if (now_ns() > limit) { ok = false; break; }
            if (!ok) break;
            int64_t t1 = now_ns();
            if (w >> 63) out.bad_classic++;
            out.c_t0.push_back(t0); out.c_t1.push_back(t1); out.c_g.push_back(((w & ~(1ull << 63)) >> 12) << 10);
        }
        s->req.v = ~0ull;
        if (!ok || !(ok = set_phase(2))) break;
        up_seen = s->up_e.v;
        for (int i = 0; i < per_phase; i++) {
            uint64_t e;
            int64_t limit = now_ns() + 1000000000LL;
            while ((e = s->up_e.v) == up_seen)
                if (now_ns() > limit) { ok = false; break; }
            if (!ok) break;
            int64_t h = now_ns();
            up_seen = e;
            out.up_h.push_back(h); out.up_e.push_back(e);
        }
        if (!ok || !(ok = set_phase(3))) break;
        int64_t until = now_ns() + (int64_t)per_phase * 3000;
        while (now_ns() < until) {}
    }
    s->phase.v = 9;
    if (ok) ok = cudaStreamSynchronize(st) == cudaSuccess;
    else { cudaStreamSynchronize(st); if (err) *err = "clocksync protocol timeout"; }
    out.t_end = now_ns();
    clock_run = false;
    clk.join();
    out.down_t.assign(down_t, down_t + *n_down);
    out.down_e.assign(down_e, down_e + *n_down);
    for (int i = 1; i < 64; i++) out.tick_steps.push_back((int64_t)(tick[i] - tick[i - 1]));
    cudaStreamDestroy(st);
    cudaFreeHost(s); cudaFreeHost(down_t); cudaFreeHost(down_e); cudaFreeHost(n_down); cudaFreeHost(tick);
    return ok;
}

}  // namespace gt
}  // namespace sb
