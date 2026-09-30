// Fail-fast error checks for CUDA runtime, cuBLAS and cuFFT calls. CUDA translation units only.
// CUBLAS_STATUS_SUCCESS and CUFFT_SUCCESS are both 0, so the library checks need no library headers.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

#define CUBLAS_CK(x) do { int s_ = (int)(x); if (s_ != 0) { \
    fprintf(stderr, "%s:%d %s: cuBLAS status %d\n", __FILE__, __LINE__, #x, s_); exit(2); } } while (0)

#define CUFFT_CK(x) do { int r_ = (int)(x); if (r_ != 0) { \
    fprintf(stderr, "%s:%d %s: cuFFT result %d\n", __FILE__, __LINE__, #x, r_); exit(2); } } while (0)
