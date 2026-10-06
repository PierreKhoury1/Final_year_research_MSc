// gputrace device API: hardware-timer records per block. Thread 0 of a block calls trace_begin() at entry
// and trace_end() at exit; records go to a device array with an atomic slot counter and are copied out by
// the host after the strategy. Costs per block: two %globaltimer reads, two clock64 reads, one atomicAdd,
// one 64-byte store.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "gputrace.h"

namespace sb {
namespace gt {

struct TraceDev {
    GpuRec *recs;
    unsigned int *count;    // next free slot (may exceed capacity: records past it are dropped and counted)
    unsigned int capacity;
};

__device__ __forceinline__ uint64_t gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
    return t;
}
__device__ __forceinline__ uint32_t smid() {
    uint32_t s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s;
}
__device__ __forceinline__ uint64_t ld_sys(const volatile uint64_t *p) {
    uint64_t v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_sys(volatile uint64_t *p, uint64_t v) {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t next_edge() {
    uint64_t a = gtimer(), b;
    do { b = gtimer(); } while (b == a);
    return b;
}

struct BlockTrace {
    unsigned int slot;
    uint64_t g0, c0;
};

// Thread 0 only. Returns false if the ring is full (the block still runs; nothing is recorded).
__device__ __forceinline__ bool trace_begin(const TraceDev &td, BlockTrace &bt) {
    bt.g0 = gtimer();
    bt.c0 = clock64();
    bt.slot = atomicAdd(td.count, 1u);
    return bt.slot < td.capacity;
}

__device__ __forceinline__ void trace_end(const TraceDev &td, const BlockTrace &bt, uint32_t kernel_id, uint32_t tag,
                                          uint64_t max_gap_ns, uint32_t n_iters, uint32_t flags) {
    uint64_t c1 = clock64();
    uint64_t g1 = gtimer();
    if (bt.slot >= td.capacity) return;
    GpuRec r;
    r.g_begin = bt.g0; r.g_end = g1; r.clk_begin = bt.c0; r.clk_end = c1; r.max_gap_ns = max_gap_ns;
    r.smid = smid(); r.kernel_id = kernel_id;
    r.block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    r.tag = tag; r.n_iters = n_iters; r.flags = flags;
    td.recs[bt.slot] = r;
}

// Spin for dur_ns of GPU time from g0, tracking the largest jump between consecutive %globaltimer reads
// (a jump much longer than one read means the block was not running: time-sliced out or preempted).
// All threads spin so the block keeps its SM resources busy the way a real kernel would.
__device__ __forceinline__ void spin_ns(uint64_t g0, uint64_t dur_ns, uint64_t &max_gap, uint32_t &iters) {
    uint64_t last = gtimer(), until = g0 + dur_ns;
    max_gap = 0; iters = 0;
    while (last < until) {
        uint64_t t = gtimer();
        if (t - last > max_gap) max_gap = t - last;
        last = t;
        iters++;
    }
}

}  // namespace gt
}  // namespace sb
