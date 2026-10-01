// Device kernels of the slot pipeline and their launch wrappers (see phy_kernels.cuh).
#include "phy_kernels.cuh"

#include <cuda_fp16.h>

#include "cuda_check.h"

namespace sb {

namespace {

constexpr int kAnt = 4;  // antennas == layers == 4 (validated by SlotPipeline)
constexpr int kThreads = 256;

inline unsigned blocks_for(long long n) { return (unsigned)((n + kThreads - 1) / kThreads); }

__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
    return t;
}

// ---- S0 / S12 ----
__global__ void k_stamp_start(SlotStamps *st, unsigned long long *counter) {
    volatile unsigned long long *c = counter;
    volatile SlotStamps *v = st;
    unsigned long long seq = *c + 1;
    *c = seq;
    v->start_t = globaltimer();
    __threadfence_system();
    v->start_seq = seq;
    __threadfence_system();
}

__global__ void k_stamp_end(SlotStamps *st, const unsigned long long *counter) {
    volatile const unsigned long long *c = counter;
    volatile SlotStamps *v = st;
    v->end_t = globaltimer();
    __threadfence_system();
    v->end_seq = *c;
    __threadfence_system();
}

// ---- S2 ----
__global__ void k_pilot_gather(const Cf *__restrict__ freq, Cf *__restrict__ Yp, Cf *__restrict__ Yd, int fft,
                               int S, int n_data) {
    const int per_k = kAnt * (4 + n_data);
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)S * per_k) return;
    int k = (int)(idx / per_k), r = (int)(idx % per_k);
    int a = r % kAnt, c = r / kAnt;
    if (c < 4) {
        int kk = pilot_subcarrier(k, c), sym = pilot_symbol(c);
        Yp[(size_t)k * 16 + a + 4 * c] = freq[((size_t)sym * kAnt + a) * fft + subcarrier_bin(kk, S, fft)];
    } else {
        int t = c - 4;
        Yd[(size_t)k * kAnt * n_data + a + 4 * t] =
            freq[((size_t)data_symbol(t) * kAnt + a) * fft + subcarrier_bin(k, S, fft)];
    }
}

// ---- S4b ----
__global__ void k_add_sigma2(Cf *G, int batch, float sigma2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * 4) return;
    G[(size_t)(idx >> 2) * 16 + (idx & 3) * 5].x += sigma2;
}

// ---- S8 ----
// Bits per axis is a template parameter so qam_llr's small per-bit arrays stay in registers (with a
// runtime qam they lived in local memory and S8 took ~150 us at 100 MHz on an RTX 3060).
template <int M>
__global__ void k_demod(const Cf *__restrict__ X, float *__restrict__ llr, int S, int n_data, float inv_n0) {
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= (long long)S * n_data * kAnt) return;
    int l = (int)(q % kAnt);
    long long kt = q / kAnt;
    int t = (int)(kt % n_data), k = (int)(kt / n_data);
    Cf y = X[(size_t)k * kAnt * n_data + l + 4 * t];
    qam_llr_m<M>(y, inv_n0, llr + q * (2 * M));
}

// ---- S9 ----
__global__ void k_rate_dematch(const float *__restrict__ llr, long long n_llr, float *__restrict__ cw, int n_cw,
                               int n_bits, int punct) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n_cw * n_bits) return;
    long long c = idx / n_bits;
    int j = (int)(idx % n_bits);
    float v = 0.0f;
    if (j >= punct) v = llr[(c * (n_bits - punct) + (j - punct)) % n_llr];
    cw[idx] = v;
}

// ---- S10 ----
__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }
template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }
template <> __device__ __forceinline__ __half from_f<__half>(float x) { return __float2half_rn(x); }

// Storage accessor for ms_row_update (phy_ops.h); the host reference has the same interface.
template <typename T>
struct DevAcc {
    T *appv;                 // shared memory, cols * Z
    CnWord *cnv;             // this codeword's [row][z] check-node state
    const int *ecol, *eshift;
    int Z, z;
    __device__ int col(int e) const { return __ldg(ecol + e); }
    __device__ int shift(int e) const { return __ldg(eshift + e); }
    __device__ float app(int v) const { return to_f(appv[v]); }
    __device__ void set_app(int v, float x) { appv[v] = from_f<T>(llr_clamp(x)); }
    __device__ CnWord cn(int r) const {
        uint2 w = reinterpret_cast<const uint2 *>(cnv)[(size_t)r * Z + z];
        return CnWord{w.x, w.y};
    }
    __device__ void set_cn(int r, CnWord w) { reinterpret_cast<uint2 *>(cnv)[(size_t)r * Z + z] = make_uint2(w.lo, w.hi); }
};

// One block per codeword, thread z owns check node (row, z) of every layer. In a layer the
// variables col*Z + (z+shift)%Z are distinct across threads and edges, so no atomics; layers are
// separated by __syncthreads().
template <typename T>
__global__ void __launch_bounds__(384) k_ldpc_decode(LdpcDevCode code, const float *__restrict__ cw_llr,
                                                     float *__restrict__ msg, float *__restrict__ app_out, int iters,
                                                     float alpha, int serial = 0) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T *app = reinterpret_cast<T *>(smem_raw);
    const int Z = code.Z, z = threadIdx.x, n = code.cols * Z;
    const size_t c = blockIdx.x;
    const float *in = cw_llr + c * n;
    for (int v = z; v < n; v += Z) app[v] = from_f<T>(llr_clamp(in[v]));
    __syncthreads();
    DevAcc<T> acc{app, reinterpret_cast<CnWord *>(msg) + c * code.rows * Z, code.edge_col, code.edge_shift, Z, z};
    for (int it = 0; it < iters; it++) {
        for (int r = 0; r < code.rows; r++) {
            int e0 = __ldg(code.row_start + r), deg = __ldg(code.row_start + r + 1) - e0;
            if (!serial) {
                ms_row_update(acc, r, e0, deg, z, Z, alpha, it == 0);
            } else {  // debug: one check node at a time, in z order, like the host reference
                for (int zz = 0; zz < Z; zz++) {
                    if (z == zz) ms_row_update(acc, r, e0, deg, z, Z, alpha, it == 0);
                    __syncthreads();
                }
            }
            __syncthreads();
        }
    }
    float *out = app_out + c * n;
    for (int v = z; v < n; v += Z) out[v] = to_f(app[v]);
}

// ---- S11 ----
__global__ void k_hard_pack(const float *__restrict__ app, int n_bits, int k_bits, uint32_t *__restrict__ bits,
                            int n_cw) {
    const int words = (k_bits + 31) / 32;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;  // one thread per bit slot
    long long total = (long long)n_cw * words * 32;                    // multiple of 32: whole warps
    if (idx >= total) return;
    long long w_all = idx >> 5;
    int lane = (int)(idx & 31);
    int c = (int)(w_all / words), w = (int)(w_all % words);
    int b = w * 32 + lane;
    bool one = b < k_bits && app[(size_t)c * n_bits + b] < 0.0f;
    unsigned m = __ballot_sync(0xffffffffu, one);
    if (lane == 0) bits[w_all] = m;
}

}  // namespace

void launch_stamp_start(SlotStamps *stamps_dev, unsigned long long *counter, cudaStream_t s) {
    k_stamp_start<<<1, 1, 0, s>>>(stamps_dev, counter);
    CK(cudaGetLastError());
}

void launch_stamp_end(SlotStamps *stamps_dev, const unsigned long long *counter, cudaStream_t s) {
    k_stamp_end<<<1, 1, 0, s>>>(stamps_dev, counter);
    CK(cudaGetLastError());
}

void launch_pilot_gather(const Cf *freq, Cf *Yp, Cf *Yd, int fft, int subcarriers, int n_data, cudaStream_t s) {
    long long n = (long long)subcarriers * kAnt * (4 + n_data);
    k_pilot_gather<<<blocks_for(n), kThreads, 0, s>>>(freq, Yp, Yd, fft, subcarriers, n_data);
    CK(cudaGetLastError());
}

void launch_add_sigma2(Cf *G, int batch, float sigma2, cudaStream_t s) {
    k_add_sigma2<<<blocks_for((long long)batch * 4), kThreads, 0, s>>>(G, batch, sigma2);
    CK(cudaGetLastError());
}

void launch_demod(const Cf *X, float *llr, int subcarriers, int n_data, int qam, float inv_n0, cudaStream_t s) {
    long long n = (long long)subcarriers * n_data * kAnt;
    switch (qam_bits(qam) / 2) {
        case 1: k_demod<1><<<blocks_for(n), kThreads, 0, s>>>(X, llr, subcarriers, n_data, inv_n0); break;
        case 2: k_demod<2><<<blocks_for(n), kThreads, 0, s>>>(X, llr, subcarriers, n_data, inv_n0); break;
        case 3: k_demod<3><<<blocks_for(n), kThreads, 0, s>>>(X, llr, subcarriers, n_data, inv_n0); break;
        default: k_demod<4><<<blocks_for(n), kThreads, 0, s>>>(X, llr, subcarriers, n_data, inv_n0); break;
    }
    CK(cudaGetLastError());
}

void launch_rate_dematch(const float *llr, long long n_llr, float *cw_llr, int n_cw, int n_bits, int punct,
                         cudaStream_t s) {
    long long n = (long long)n_cw * n_bits;
    k_rate_dematch<<<blocks_for(n), kThreads, 0, s>>>(llr, n_llr, cw_llr, n_cw, n_bits, punct);
    CK(cudaGetLastError());
}

bool ldpc_decoder_setup(int device, int cols, int Z, bool *app_fp16, size_t *smem_bytes, int *optin_max) {
    int optin = 0;
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    *optin_max = optin;
    size_t b32 = (size_t)4 * cols * Z, b16 = (size_t)2 * cols * Z;
    if (b32 <= (size_t)optin) {
        *app_fp16 = false;
        *smem_bytes = b32;
        CK(cudaFuncSetAttribute(k_ldpc_decode<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)b32));
        return true;
    }
    if (b16 <= (size_t)optin) {
        *app_fp16 = true;
        *smem_bytes = b16;
        CK(cudaFuncSetAttribute(k_ldpc_decode<__half>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)b16));
        return true;
    }
    return false;
}

void launch_ldpc_decode(const LdpcDevCode &code, const float *cw_llr, float *msg, float *app_out, int n_cw,
                        int iters, float alpha, bool app_fp16, size_t smem_bytes, cudaStream_t s) {
    if (app_fp16)
        k_ldpc_decode<__half><<<n_cw, code.Z, smem_bytes, s>>>(code, cw_llr, msg, app_out, iters, alpha);
    else
        k_ldpc_decode<float><<<n_cw, code.Z, smem_bytes, s>>>(code, cw_llr, msg, app_out, iters, alpha);
    CK(cudaGetLastError());
}

void launch_ldpc_decode_debug(const LdpcDevCode &code, const float *cw_llr, float *msg, float *app_out, int n_cw,
                              int iters, float alpha, bool app_fp16, size_t smem_bytes, int serial, cudaStream_t s) {
    if (app_fp16)
        k_ldpc_decode<__half><<<n_cw, code.Z, smem_bytes, s>>>(code, cw_llr, msg, app_out, iters, alpha, serial);
    else
        k_ldpc_decode<float><<<n_cw, code.Z, smem_bytes, s>>>(code, cw_llr, msg, app_out, iters, alpha, serial);
    CK(cudaGetLastError());
}

void launch_hard_pack(const float *app, int n_bits, int k_bits, uint32_t *bits, int n_cw, cudaStream_t s) {
    long long n = (long long)n_cw * ((k_bits + 31) / 32) * 32;
    k_hard_pack<<<blocks_for(n), kThreads, 0, s>>>(app, n_bits, k_bits, bits, n_cw);
    CK(cudaGetLastError());
}

}  // namespace sb
