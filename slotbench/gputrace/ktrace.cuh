// Kernel timeline: a staged kernel in which every warp stamps the SM cycle counter and %globaltimer at fixed
// checkpoints, so the flow of execution inside the kernel (loads, barrier arrival and release per warp, a compute
// chain, store + fence, the grid-wide atomic ticket, exit) is laid out on one time axis. Stamps are kept in
// registers and written as records only at the end of the warp (one atomicAdd per warp), so a checkpoint costs
// three timer reads and nothing else; checkpoint 1 follows checkpoint 0 immediately and measures that cost.
//
// A stamp is %globaltimer, clock64, %globaltimer: the cycle read is bracketed by two timer reads, so the true
// GPU time of the cycle lies in [g0, g1 + tick) whatever happens between the reads (a warp descheduled between
// two reads, with 32 warps on the SM, would otherwise put the pair outside its window). Per SM, clock64 is one
// counter shared by every warp on that SM (checked on the data: no per-warp or per-sub-partition offset), so
// stamps of different warps and blocks on the same SM compare at cycle resolution; the analysis bounds each SM's
// cycle-to-ns line by all of that SM's windows, which places every checkpoint on the GPU's common axis with a
// resolution far below the timer tick, and then on the host axis through the run's clock bound.
//
// Two things the SASS forced (both checked with cuobjdump, tools/sass_phases.py):
//  - the "loaded" stamp follows a volatile shared store of the loaded value: the hardware wait for load data
//    attaches to the first instruction that reads the register, and an empty asm consumer emits nothing;
//  - __syncthreads is BAR.SYNC.DEFER_BLOCKING: the warp keeps issuing non-memory instructions (a clock read
//    included) until it reaches a memory instruction, so a stamp right after the barrier reads 13 cycles after
//    the warp's own arrival whether or not the other warps have arrived (first RTX 3060 run). The release stamps
//    therefore follow a shared load issued after the barrier, consumed by a volatile shared store; the same
//    load + store + stamp sequence without a barrier (checkpoint 2 -> 3) calibrates its cost.
#pragma once
#include "gputrace.cuh"

namespace sb {
namespace gt {

enum KtCheckpoint : int {
    KT_ENTRY = 0,        // first instructions of the warp
    KT_CAL = 1,          // immediately after ENTRY: the cost of one checkpoint (3 timer reads)
    KT_LOADED = 2,       // two dependent coalesced global loads landed (consumed by a volatile shared store)
    KT_BAR1_ARRIVE = 3,  // after LOADED: one shared load + volatile shared store (2 -> 3 calibrates that chain); about to enter __syncthreads
    KT_BAR1_RELEASE = 4, // after __syncthreads and the same shared load + store: the barrier has completed for this warp
    KT_COMPUTED = 5,     // ffma_n dependent FFMA done
    KT_BAR2_ARRIVE = 6,  // after COMPUTED: shared load + store, about to enter the second __syncthreads
    KT_BAR2_RELEASE = 7, // after the second barrier and its shared load + store
    KT_STORED = 8,       // global store + __threadfence
    KT_TICKET = 9,       // warp 0: grid-wide atomicAdd returned (n_iters = ticket); other warps: nothing between 8 and 9
    KT_EXIT = 10,        // after the third __syncthreads (ticket shared) and the last block's flag write
    KT_N = 11
};
#define KT_TAG_BASE 100u   // record tag = KT_TAG_BASE + checkpoint

struct KtStamp { uint64_t g0, c, g1; };

__device__ __forceinline__ void kt_stamp(KtStamp &s) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(s.g0) :: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(s.c) :: "memory");
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(s.g1) :: "memory");
}

// A shared load whose value is consumed by a volatile shared store: issued after a barrier it waits for the
// barrier to complete (deferred blocking), and the store waits for the load's data, so a stamp after it is
// after the barrier release. Returns the loaded value for use by the compute chain.
__device__ __forceinline__ float4 kt_touch(const float4 *tile, unsigned idx, unsigned own) {
    float4 u;
    unsigned src = (unsigned)__cvta_generic_to_shared(tile + idx), dst = (unsigned)__cvta_generic_to_shared(tile + own);
    asm volatile("ld.volatile.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(u.x), "=f"(u.y), "=f"(u.z), "=f"(u.w) : "r"(src) : "memory");
    asm volatile("st.volatile.shared.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(dst), "f"(u.x), "f"(u.y), "f"(u.z), "f"(u.w) : "memory");
    return u;
}

// in: n_blocks * blockDim + 1024 float4 (zeros), out: n_blocks * blockDim float4; ticket: one zeroed counter;
// flag: mapped host word the last block writes (its %globaltimer) or nullptr.
__global__ void __launch_bounds__(256) k_ktrace(TraceDev td, uint32_t kid, const float4 *__restrict__ in, float4 *out,
                                                unsigned *ticket, volatile uint64_t *flag, uint32_t ffma_n, uint32_t n_blocks) {
    __shared__ float4 tile[256];
    __shared__ unsigned s_ticket;
    KtStamp s[KT_N];
    kt_stamp(s[KT_ENTRY]);
    kt_stamp(s[KT_CAL]);
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    // LOAD: a coalesced load, then a second load whose address depends on the first (the data holds zeros, which
    // the compiler cannot know): two dependent DRAM/L2 round trips, consumed by a volatile shared store
    float4 v = in[i];
    float4 w = in[i + (__float_as_uint(v.w) & 1023u)];
    asm volatile("st.volatile.shared.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"((unsigned)__cvta_generic_to_shared(&tile[threadIdx.x])),
                 "f"(w.x), "f"(w.y), "f"(w.z), "f"(w.w) : "memory");
    kt_stamp(s[KT_LOADED]);
    kt_touch(tile, threadIdx.x, threadIdx.x);                     // calibration of the touch chain (no barrier)
    kt_stamp(s[KT_BAR1_ARRIVE]);
    __syncthreads();
    float4 u = kt_touch(tile, (threadIdx.x + 32) & 255, threadIdx.x);   // waits for the barrier, then for the data
    kt_stamp(s[KT_BAR1_RELEASE]);
    // COMPUTE: a dependent FFMA chain seeded from the other warp's tile entry
    float x = u.x + 1.0f;
#pragma unroll 16
    for (uint32_t k = 0; k < ffma_n; k++) x = fmaf(x, 1.0001f, 0.5f);
    asm volatile("" : "+f"(x));
    kt_stamp(s[KT_COMPUTED]);
    kt_touch(tile, threadIdx.x, threadIdx.x);
    kt_stamp(s[KT_BAR2_ARRIVE]);
    __syncthreads();
    u = kt_touch(tile, (threadIdx.x + 64) & 255, threadIdx.x);
    kt_stamp(s[KT_BAR2_RELEASE]);
    // STORE: one coalesced global store, made visible device-wide
    out[i] = make_float4(x, u.y, u.z, u.w);
    __threadfence();
    kt_stamp(s[KT_STORED]);
    unsigned t = 0;
    if (threadIdx.x == 0) { t = atomicAdd(ticket, 1u); s_ticket = t; }
    kt_stamp(s[KT_TICKET]);
    __syncthreads();
    t = s_ticket;
    if (threadIdx.x == 0 && flag && t == n_blocks - 1) { st_sys(flag, gtimer()); __threadfence_system(); }
    kt_stamp(s[KT_EXIT]);
    if ((threadIdx.x & 31) == 0) {
        const unsigned base = atomicAdd(td.count, (unsigned)KT_N);
        const uint32_t sm = smid(), warp = threadIdx.x >> 5;
#pragma unroll
        for (int k = 0; k < KT_N; k++) {
            if (base + k >= td.capacity) break;
            GpuRec r;
            r.g_begin = s[k].g0; r.g_end = s[k].g1; r.clk_begin = r.clk_end = s[k].c; r.max_gap_ns = 0;
            r.smid = sm; r.kernel_id = kid; r.block = blockIdx.x; r.tag = KT_TAG_BASE + k;
            r.n_iters = (k == KT_TICKET) ? t : ffma_n; r.flags = warp;
            td.recs[base + k] = r;
        }
    }
}

}  // namespace gt
}  // namespace sb
