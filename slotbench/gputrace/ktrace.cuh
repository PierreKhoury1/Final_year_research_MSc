// Kernel timeline: a staged kernel in which every warp stamps (%globaltimer, clock64) at fixed checkpoints, so the
// flow of execution inside the kernel (loads, barrier arrival and release per warp, a compute chain, store + fence,
// the grid-wide atomic ticket, exit) is laid out on one time axis. Stamps are kept in registers and written as
// records only at the end of the warp (one atomicAdd per warp), so the checkpoints cost one %globaltimer read and
// one clock64 read each and nothing else; checkpoint 1 follows checkpoint 0 immediately and measures that cost.
//
// Per SM, clock64 is one counter shared by every warp on that SM, so stamps of different warps and blocks on the
// same SM compare at cycle resolution; the analysis fits each SM's cycle counter to %globaltimer over all of that
// SM's stamps (hundreds of points over the kernel), which places every checkpoint on the GPU's common axis with a
// resolution far below the timer tick, and then on the host axis through the run's clock bound.
#pragma once
#include "gputrace.cuh"

namespace sb {
namespace gt {

enum KtCheckpoint : int {
    KT_ENTRY = 0,        // first instruction of the warp
    KT_CAL = 1,          // immediately after ENTRY: the cost of one checkpoint
    KT_LOADED = 2,       // two dependent coalesced global loads landed (consumed by a volatile shared store)
    KT_BAR1_ARRIVE = 3,  // immediately after 2: about to enter __syncthreads (2->3 is one checkpoint's cost)
    KT_BAR1_RELEASE = 4, // out of __syncthreads
    KT_COMPUTED = 5,     // ffma_n dependent FFMA done
    KT_BAR2_ARRIVE = 6,
    KT_BAR2_RELEASE = 7,
    KT_STORED = 8,       // global store + __threadfence
    KT_TICKET = 9,       // warp 0: grid-wide atomicAdd returned (n_iters = ticket); other warps: nothing between 8 and 9
    KT_EXIT = 10,        // last instruction before the records are written
    KT_N = 11
};
#define KT_TAG_BASE 100u   // record tag = KT_TAG_BASE + checkpoint

struct KtStamp { uint64_t g, c; };

__device__ __forceinline__ void kt_stamp(KtStamp &s) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(s.g) :: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(s.c) :: "memory");
}

// in: n_blocks * blockDim + 1024 float4 (zeros), out: n_blocks * blockDim float4; ticket: one zeroed counter; flag: mapped host word the last block
// writes (its %globaltimer) or nullptr.
__global__ void __launch_bounds__(256) k_ktrace(TraceDev td, uint32_t kid, const float4 *__restrict__ in, float4 *out,
                                                unsigned *ticket, volatile uint64_t *flag, uint32_t ffma_n, uint32_t n_blocks) {
    __shared__ float4 tile[256];
    __shared__ unsigned s_ticket;
    KtStamp s[KT_N];
    kt_stamp(s[KT_ENTRY]);
    kt_stamp(s[KT_CAL]);
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    // LOAD: a coalesced load, then a second load whose address depends on the first (the data holds zeros, which
    // the compiler cannot know), so the phase is two dependent DRAM/L2 round trips, consumed before the stamp
    float4 v = in[i];
    float4 w = in[i + (__float_as_uint(v.w) & 1023u)];   // in has 1024 float4 of slack past n_blocks * blockDim
    // The loaded value is consumed by a volatile shared store *before* the stamp: the hardware's wait for the load
    // data attaches to the first instruction that reads it, and an empty asm consumer emits no instruction, so
    // without this the stamp was issued before the data had arrived (seen in SASS: STS after CS2R).
    asm volatile("st.volatile.shared.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"((unsigned)__cvta_generic_to_shared(&tile[threadIdx.x])),
                 "f"(w.x), "f"(w.y), "f"(w.z), "f"(w.w) : "memory");
    kt_stamp(s[KT_LOADED]);
    kt_stamp(s[KT_BAR1_ARRIVE]);
    __syncthreads();
    kt_stamp(s[KT_BAR1_RELEASE]);
    // COMPUTE: a dependent FFMA chain seeded from another warp's tile entry (so it waits for the barrier by data too)
    const float4 u = tile[(threadIdx.x + 32) & 255];
    float x = u.x + 1.0f;
#pragma unroll 16
    for (uint32_t k = 0; k < ffma_n; k++) x = fmaf(x, 1.0001f, 0.5f);
    asm volatile("" : "+f"(x));
    kt_stamp(s[KT_COMPUTED]);
    kt_stamp(s[KT_BAR2_ARRIVE]);
    __syncthreads();
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
            r.g_begin = r.g_end = s[k].g; r.clk_begin = r.clk_end = s[k].c; r.max_gap_ns = 0;
            r.smid = sm; r.kernel_id = kid; r.block = blockIdx.x; r.tag = KT_TAG_BASE + k;
            r.n_iters = (k == KT_TICKET) ? t : ffma_n; r.flags = warp;
            td.recs[base + k] = r;
        }
    }
}

}  // namespace gt
}  // namespace sb
