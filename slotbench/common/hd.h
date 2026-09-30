// HD marks maths shared by CUDA kernels and host-only tests.
#pragma once
#if defined(__CUDACC__)
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif
