// Per-instruction brackets: a chain of N dependent instructions of one kind between two clock64 reads, with the
// chain's result consumed before the closing read so the hardware cannot retire the bracket early. N is a template
// parameter, so the chain is straight-line SASS that tools/sass_check.py can verify (exactly N target opcodes between
// the CS2R pair). Per sample one GpuRec: clk = cycles for the whole bracket, g = %globaltimer ns, tag = kind,
// block = N, flags = working-set index (loads) or 0. Fitting cycles = a + b*N over N gives the instruction latency
// (b) separated from the bracket overhead (a).
#pragma once
#include "gputrace.cuh"

namespace sb {
namespace gt {

enum InstrKind : int {
    I_LDG_CA = 0, I_LDG_CG = 1, I_LDG_CS = 2, I_LDG_NC = 3,   // dependent global loads over a pointer-chase buffer
    I_LDS = 4,          // dependent shared-memory loads
    I_FADD = 5, I_FFMA = 6, I_IMAD = 7,   // dependent arithmetic
    I_SHFL = 8,         // dependent warp shuffle
    I_ATOM_RET = 9,     // global atomics with return, address dependent on the returned value
    I_RED = 10,         // global atomics without return (issue cost), then one fence
    I_STG_FENCE = 11,   // N global stores then one membar.gl: store issue + visibility
    I_BAR = 12,         // __syncthreads, whole block of 256 threads
    I_N_KINDS = 13
};

__device__ __forceinline__ unsigned ldg_k(int K, const unsigned *p) {
    unsigned v;
    if (K == I_LDG_CA) asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    else if (K == I_LDG_CG) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    else if (K == I_LDG_CS) asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    else asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// One sample: returns the bracket cycle count; g0/g1 are the timer reads outside the clock reads.
template <int K, int N>
__device__ __forceinline__ uint64_t bracket(const unsigned *buf, unsigned *sh, unsigned *out, unsigned &idx, float &x, uint64_t &g0, uint64_t &g1) {
    g0 = gtimer();
    uint64_t c0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c0) :: "memory");
    // Tie the chain to the opening clock read: ptxas keeps volatile loads in order with the clock read and the
    // chain depends on the loaded value (0 in memory, unknown to the compiler). Without this, ptxas hoisted the
    // SHFL and short STG chains above the clock read (seen in SASS on the first A100 run). Costs one L1 hit,
    // the same for every N, so it lands in the fitted intercept and not in the per-instruction slope.
    unsigned seed;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(seed) : "l"(out + 4095) : "memory");
    idx += seed; x += (float)seed;
    if (K <= I_LDG_NC) {
#pragma unroll
        for (int i = 0; i < N; i++) idx = ldg_k(K, buf + idx);
        if (idx == 0xffffffffu) asm volatile("trap;");
    } else if (K == I_LDS) {
#pragma unroll
        for (int i = 0; i < N; i++) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(idx) : "r"((unsigned)__cvta_generic_to_shared(sh) + idx) : "memory");
        if (idx == 0xffffffffu) asm volatile("trap;");
    } else if (K == I_FADD) {
#pragma unroll
        for (int i = 0; i < N; i++) asm volatile("add.f32 %0, %0, %1;" : "+f"(x) : "f"(1.0f));
        if (x == -12345.f) asm volatile("trap;");
    } else if (K == I_FFMA) {
#pragma unroll
        for (int i = 0; i < N; i++) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(1.0001f), "f"(0.5f));
        if (x == -12345.f) asm volatile("trap;");
    } else if (K == I_IMAD) {
#pragma unroll
        for (int i = 0; i < N; i++) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(idx) : "r"(3u), "r"(7u));
        if (idx == 0xffffffffu) asm volatile("trap;");
    } else if (K == I_SHFL) {
#pragma unroll
        for (int i = 0; i < N; i++) asm volatile("shfl.sync.idx.b32 %0, %0, %1, 31, 0xffffffff;" : "+r"(idx) : "r"(idx & 31));
        if (idx == 0xffffffffu) asm volatile("trap;");
    } else if (K == I_ATOM_RET) {
#pragma unroll
        for (int i = 0; i < N; i++) asm volatile("atom.global.add.u32 %0, [%1], 1;" : "=r"(idx) : "l"(out + (idx & 1023)) : "memory");
        if (idx == 0xffffffffu) asm volatile("trap;");
    } else if (K == I_RED) {
#pragma unroll
        for (int i = 0; i < N; i++) asm volatile("red.global.add.u32 [%0], 1;" ::"l"(out + ((idx + i) & 1023)) : "memory");
        asm volatile("membar.gl;" ::: "memory");
    } else if (K == I_STG_FENCE) {
#pragma unroll
        for (int i = 0; i < N; i++) asm volatile("st.global.u32 [%0], %1;" ::"l"(out + ((idx + i) & 1023)), "r"(idx) : "memory");
        asm volatile("membar.gl;" ::: "memory");
    } else if (K == I_BAR) {
#pragma unroll
        for (int i = 0; i < N; i++) __syncthreads();
    }
    uint64_t c1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c1) :: "memory");
    g1 = gtimer();
    return c1 - c0;
}

template <int K, int N>
__global__ void k_instr(TraceDev td, uint32_t kid, const unsigned *buf, unsigned *out, uint32_t reps, uint32_t ws_idx) {
    __shared__ unsigned sh[2048];
    if (K == I_LDS) { for (int i = threadIdx.x; i < 2048; i += blockDim.x) sh[i] = (unsigned)(((i * 613) + 1) & 2047) * 4; __syncthreads(); }
    const bool rec = threadIdx.x == 0;
    if (!rec && K != I_BAR) return;
    unsigned idx = 0; float x = 1.0f;
    uint64_t g0, g1;
    bracket<K, N>(buf, sh, out, idx, x, g0, g1);   // warm: instruction cache, TLB, the chain's lines
    for (uint32_t r = 0; r < reps; r++) {
        uint64_t cyc = bracket<K, N>(buf, sh, out, idx, x, g0, g1);
        if (rec) {
            unsigned s = atomicAdd(td.count, 1u);
            if (s < td.capacity) {
                GpuRec rr{};
                rr.g_begin = g0; rr.g_end = g1; rr.clk_begin = 0; rr.clk_end = cyc; rr.smid = smid(); rr.kernel_id = kid;
                rr.block = N; rr.tag = (uint32_t)K; rr.n_iters = N; rr.flags = ws_idx;
                td.recs[s] = rr;
            }
        }
    }
    if (rec && (idx == 0xfffffffeu || x == -1.f)) out[0] = idx;
}

}  // namespace gt
}  // namespace sb
