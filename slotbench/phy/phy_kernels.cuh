// Device kernels of the slot pipeline (DESIGN.md section 2) and their host launch wrappers.
// Every wrapper only enqueues on the given stream: no allocation, no synchronisation, so all of
// them can be captured into a CUDA graph. Launch errors are checked with cudaGetLastError.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "phy_ops.h"
#include "slot_pipeline.h"

namespace sb {

// S0: seq = ++*counter; stamps->start_t = %globaltimer; fence; stamps->start_seq = seq; fence.
void launch_stamp_start(SlotStamps *stamps_dev, unsigned long long *counter, cudaStream_t s);
// S12: stamps->end_t = %globaltimer; fence; stamps->end_seq = *counter; fence.
void launch_stamp_end(SlotStamps *stamps_dev, const unsigned long long *counter, cudaStream_t s);

// S2: freq is [symbol][antenna][fft] (cuFFT batch order). For each used subcarrier k:
//   Yp[k] (4x4, column-major, antennas x pilot columns j, see phy_ops.h pilot_subcarrier/symbol)
//   Yd[k] (4 x n_data, column-major, antennas x data symbols t, OFDM symbol data_symbol(t))
void launch_pilot_gather(const Cf *freq, Cf *Yp, Cf *Yd, int fft, int subcarriers, int n_data, cudaStream_t s);

// S4b: G[b] += sigma2 * I for batch of 4x4 matrices.
void launch_add_sigma2(Cf *G, int batch, float sigma2, cudaStream_t s);

// S8: X is per subcarrier a 4 x n_data column-major matrix (layers x data symbols). Symbol index
// q = (k * n_data + t) * 4 + l writes qam_bits(qam) LLRs at llr[q * qam_bits].
void launch_demod(const Cf *X, float *llr, int subcarriers, int n_data, int qam, float inv_n0, cudaStream_t s);

// S9: cw_llr[c][j] for c < n_cw, j < n_bits: 0 for j < punct (= 2Z), else
//   llr[(c * (n_bits - punct) + j - punct) mod n_llr].
void launch_rate_dematch(const float *llr, long long n_llr, float *cw_llr, int n_cw, int n_bits, int punct,
                         cudaStream_t s);

// Lifted code in device memory (edge lists as in LdpcCode, rows in order).
struct LdpcDevCode {
    const int *row_start = nullptr;   // rows + 1
    const int *edge_col = nullptr;    // n_edges
    const int *edge_shift = nullptr;  // n_edges
    int rows = 0, cols = 0, Z = 0, n_edges = 0;
};

// Chooses the decoder's a posteriori storage for this device and sets the kernel's dynamic shared
// memory attribute (call once at init): fp32 if 4*cols*Z bytes fit the opt-in maximum, else fp16.
// Returns false if even fp16 does not fit. *smem_bytes gets the dynamic shared memory per block.
bool ldpc_decoder_setup(int device, int cols, int Z, bool *app_fp16, size_t *smem_bytes, int *optin_max);

// S10: one block per codeword, blockDim = Z. Check-to-variable messages are fp32 in msg laid out
// [codeword][edge][z]; they are (re)initialised by the kernel itself (iteration 0 treats them as 0).
// app_out[c][v] gets the final a posteriori LLR of every variable (fp32).
void launch_ldpc_decode(const LdpcDevCode &code, const float *cw_llr, float *msg, float *app_out, int n_cw,
                        int iters, float alpha, bool app_fp16, size_t smem_bytes, cudaStream_t s);

// S11: bits[c][w] bit b = (app[c][32 w + b] < 0) for the first k_bits variables of each codeword;
// words per codeword = (k_bits + 31) / 32, unused high bits of the last word are 0.
void launch_hard_pack(const float *app, int n_bits, int k_bits, uint32_t *bits, int n_cw, cudaStream_t s);

}  // namespace sb
