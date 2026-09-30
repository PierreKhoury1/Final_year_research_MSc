// Maths shared by the GPU kernels and the host reference/tests (no CUDA headers: g++ compiles it).
// Everything that must give bit-identical results on host and device lives here:
//  - complex helpers on a plain {float x, y} (same layout as float2 / cuComplex / cufftComplex)
//  - Gray square QAM mapping (38.211 5.1) and per-axis max-log LLRs, positive LLR = bit 0
//  - normalized min-sum check-node helpers and one layered check-node update (ms_row_update)
//  - LLR clamping and a software IEEE fp16 round-to-nearest-even (matches __float2half_rn)
//  - slot geometry helpers (FFT bin of a used subcarrier, DMRS/data symbol indices)
// Multiplies/adds in the decoder use fmul_rn/fadd_rn so nvcc cannot contract them into FMAs,
// which would make the GPU decoder differ from the host reference.
#pragma once
#include <cmath>
#include <cstdint>

#include "hd.h"

namespace sb {

// ---------------- complex ----------------
struct alignas(8) Cf {
    float x, y;
};
HD Cf cf(float x, float y) { return Cf{x, y}; }
HD Cf cadd(Cf a, Cf b) { return Cf{a.x + b.x, a.y + b.y}; }
HD Cf csub(Cf a, Cf b) { return Cf{a.x - b.x, a.y - b.y}; }
HD Cf cmul(Cf a, Cf b) { return Cf{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
HD Cf cconj(Cf a) { return Cf{a.x, -a.y}; }
HD Cf cmulconj(Cf a, Cf b) { return cmul(a, cconj(b)); }  // a * conj(b)
HD Cf cscale(Cf a, float s) { return Cf{a.x * s, a.y * s}; }
HD float cabs2(Cf a) { return a.x * a.x + a.y * a.y; }

// ---------------- non-contracted float ops ----------------
HD float fmul_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
HD float fadd_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
HD float fsub_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}

// ---------------- LLR clamp / fp16 ----------------
// Every LLR that is stored (channel input, a posteriori) is clamped to +-kLlrClamp so fp16 storage
// never overflows (fp16 max 65504) and nothing becomes inf. Min-sum is scale invariant, so the
// absolute value only matters for saturation.
constexpr float kLlrClamp = 8192.0f;
HD float llr_clamp(float v) { return fminf(fmaxf(v, -kLlrClamp), kLlrClamp); }  // NaN -> -kLlrClamp

union F32Bits {
    float f;
    uint32_t u;
};

// IEEE binary16 bits of f, round to nearest even, overflow -> inf, NaN kept quiet.
HD uint16_t f32_to_f16_bits(float f) {
    F32Bits b;
    b.f = f;
    uint32_t x = b.u;
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    int exp = (int)((x >> 23) & 0xffu);
    uint32_t mant = x & 0x7fffffu;
    if (exp == 0xff) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    int e = exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {                        // half subnormal (or zero)
        if (e < -10) return sign;        // below half the smallest subnormal -> +-0
        uint32_t m = mant | 0x800000u;
        int shift = 14 - e;              // 14..24
        uint32_t h = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;  // carry may reach inf: correct
    return (uint16_t)(sign | h);
}

HD float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ffu;
    F32Bits b;
    if (exp == 0) {
        float v = (float)mant * 5.9604644775390625e-8f;  // mant * 2^-24, exact
        b.f = v;
        b.u |= sign;
        return b.f;
    }
    if (exp == 31) {
        b.u = sign | 0x7f800000u | (mant << 13);
        return b.f;
    }
    b.u = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    return b.f;
}

HD float f16_round(float f) { return f16_bits_to_f32(f32_to_f16_bits(f)); }

// ---------------- Gray square QAM (38.211 5.1) ----------------
// Bits b0..b(Q-1) of one symbol: I axis uses b0, b2, b4, b6 and Q axis b1, b3, b5, b7.
// With per-axis bits c0, c1, ... (c0 = sign bit) the amplitude is, e.g. for 256-QAM,
//   (1-2c0) * (8 - (1-2c1) * (4 - (1-2c2) * (2 - (1-2c3))))
// scaled to unit average energy. QPSK is (1-2c0)/sqrt(2) per axis.
HD int qam_bits(int qam) { return qam == 4 ? 2 : qam == 16 ? 4 : qam == 64 ? 6 : qam == 256 ? 8 : 0; }
HD float qam_scale(int qam) {  // 1 / sqrt(2 (M - 1) / 3)
    return qam == 4 ? 0.70710678118654752f : qam == 16 ? 0.31622776601683793f
         : qam == 64 ? 0.15430334996209191f : 0.076696498884737041f;
}
// Unscaled odd-integer amplitude of per-axis bit pattern c (bit i of c = c_i), m bits per axis.
HD int qam_axis_level(unsigned c, int m) {
    int v = 1 - 2 * (int)((c >> (m - 1)) & 1u);
    for (int i = m - 2; i >= 0; i--) v = (1 - 2 * (int)((c >> i) & 1u)) * ((1 << (m - 1 - i)) - v);
    return v;
}

// bits: qam_bits(qam) values in {0,1}, 38.211 order.
HD Cf qam_map(const uint8_t *bits, int qam) {
    int q = qam_bits(qam), m = q / 2;
    unsigned ci = 0, cq = 0;
    for (int i = 0; i < m; i++) {
        ci |= (unsigned)(bits[2 * i] & 1u) << i;
        cq |= (unsigned)(bits[2 * i + 1] & 1u) << i;
    }
    float s = qam_scale(qam);
    return Cf{s * (float)qam_axis_level(ci, m), s * (float)qam_axis_level(cq, m)};
}

// Max-log LLR of the m bits of one axis: (min over c_i=1 of (y-a)^2 - min over c_i=0) * inv_n0,
// clamped. inv_n0 = 1 / (complex noise variance) = 1 / (2 * per-axis variance).
// out[i] is written with stride `stride` (2 for interleaved I/Q bits).
HD void qam_llr_axis(float y, int m, float scale, float inv_n0, float *out, int stride) {
    float d0[4], d1[4];
    for (int i = 0; i < 4; i++) { d0[i] = 3.0e38f; d1[i] = 3.0e38f; }
    for (unsigned c = 0; c < (1u << m); c++) {
        float e = y - scale * (float)qam_axis_level(c, m);
        float d = e * e;
        for (int i = 0; i < m; i++) {
            if ((c >> i) & 1u) d1[i] = fminf(d1[i], d);
            else d0[i] = fminf(d0[i], d);
        }
    }
    for (int i = 0; i < m; i++) out[i * stride] = llr_clamp((d1[i] - d0[i]) * inv_n0);
}

// All qam_bits(qam) LLRs of symbol y in 38.211 bit order; positive = bit 0.
HD void qam_llr(Cf y, int qam, float inv_n0, float *llr) {
    int m = qam_bits(qam) / 2;
    float s = qam_scale(qam);
    qam_llr_axis(y.x, m, s, inv_n0, llr, 2);
    qam_llr_axis(y.y, m, s, inv_n0, llr + 1, 2);
}

// ---------------- normalized min-sum ----------------
// Accumulates |t| minima and the sign parity of the variable-to-check messages of one check node.
struct MsAcc {
    float min1, min2;  // smallest and second smallest |t|
    int idx1;          // edge position of min1
    unsigned neg;      // parity of the number of negative t (t == 0 counts as positive)
};
HD void ms_init(MsAcc &a) {
    a.min1 = 1.0e30f;
    a.min2 = 1.0e30f;
    a.idx1 = -1;
    a.neg = 0;
}
HD void ms_add(MsAcc &a, float t, int j) {
    float m = fabsf(t);
    a.neg ^= (t < 0.0f) ? 1u : 0u;
    if (m < a.min1) {
        a.min2 = a.min1;
        a.min1 = m;
        a.idx1 = j;
    } else if (m < a.min2) {
        a.min2 = m;
    }
}
// Check-to-variable message for edge j: alpha * (product of the other signs) * (min over the others).
HD float ms_out(const MsAcc &a, float t, int j, float alpha) {
    float v = fmul_rn(alpha, j == a.idx1 ? a.min2 : a.min1);
    unsigned s = a.neg ^ ((t < 0.0f) ? 1u : 0u);
    return s ? -v : v;
}

// One layered update of check node (row, z). Acc supplies the storage:
//   int col(e), int shift(e)       edge e's base column and V mod Z
//   float app(v), void set_app(v, x)   a posteriori LLR of lifted variable v (set_app clamps and
//                                      rounds to the storage precision)
//   float msg(e), void set_msg(e, x)   check-to-variable message of edge e for this z (fp32)
// first: first iteration, old messages are taken as 0 without reading them.
// Two passes recompute t = app - old instead of keeping per-edge arrays; within a row every edge
// touches a different variable, so pass 2 sees the same app values as pass 1.
#if defined(__CUDACC__)
#pragma nv_exec_check_disable
#endif
template <class Acc>
HD void ms_row_update(Acc &s, int e0, int deg, int z, int Z, float alpha, bool first) {
    MsAcc a;
    ms_init(a);
    for (int j = 0; j < deg; j++) {
        int e = e0 + j;
        int sh = z + s.shift(e);
        int v = s.col(e) * Z + (sh >= Z ? sh - Z : sh);
        float old = first ? 0.0f : s.msg(e);
        ms_add(a, fsub_rn(s.app(v), old), j);
    }
    for (int j = 0; j < deg; j++) {
        int e = e0 + j;
        int sh = z + s.shift(e);
        int v = s.col(e) * Z + (sh >= Z ? sh - Z : sh);
        float old = first ? 0.0f : s.msg(e);
        float t = fsub_rn(s.app(v), old);
        float n = ms_out(a, t, j, alpha);
        s.set_msg(e, n);
        s.set_app(v, fadd_rn(t, n));
    }
}

// ---------------- slot geometry ----------------
// DMRS pilots on OFDM symbols 2 and 11; the rest are data symbols.
constexpr int kDmrsSym0 = 2;
constexpr int kDmrsSym1 = 11;
// OFDM symbol of data symbol t (t = 0 .. symbols-3).
HD int data_symbol(int t) { return t + (t >= 2 ? 1 : 0) + (t >= 10 ? 1 : 0); }
// FFT bin of used subcarrier k (0..S-1): the S used bins are centred on DC.
HD int subcarrier_bin(int k, int S, int fft) {
    int b = k - S / 2;
    return b < 0 ? b + fft : b;
}
// Pilot column j (0..3) of the 4x4 pilot block of subcarrier k: pairs (k & ~1, k | 1) share one
// block; j = 2 * (subcarrier parity) + (symbol 11 ? 1 : 0). Both subcarriers of a pair gather
// the columns in this same order, so one fixed orthogonal Xp estimates H for both.
HD int pilot_subcarrier(int k, int j) { return (k & ~1) + (j >> 1); }
HD int pilot_symbol(int j) { return (j & 1) ? kDmrsSym1 : kDmrsSym0; }
// Fixed orthogonal pilot matrix (4x4 Walsh-Hadamard, layers x pilot columns): Xp Xp^H = 4 I.
HD float pilot_value(int layer, int j) {
    unsigned x = (unsigned)(layer & j) & 3u;
    return ((x ^ (x >> 1)) & 1u) ? -1.0f : 1.0f;
}

}  // namespace sb
