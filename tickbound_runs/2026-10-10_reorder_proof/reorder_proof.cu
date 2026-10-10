// Reordering proof (10 Oct 2026): does hoisting a load above independent work make a whole kernel faster than the
// compiler's own schedule, by the amount the measured latency law predicts?
//   base : natural loop, x = load(i); acc += chain_N(x)      -> the chain waits for the load every iteration
//   pipe : same instructions, load(i+1) issued before chain_N(x_i) (one extra load at the end)
//   u4   : base with #pragma unroll 4: the compiler's own chance to hoist loads
// All variants read the same lines in the same order (paired) and must give bit-identical outputs.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ uint32_t line_of(uint32_t i, uint32_t w, uint32_t nlines) {
    return (i * 2654435761u + w * 40503u + 12345u) % nlines;
}
template <int N> __device__ __forceinline__ float chain(float v) {
#pragma unroll
    for (int k = 0; k < N; k++) v = fmaf(v, 1.0001f, 0.5f);
    return v;
}
template <int N> __global__ void k_base(const float* __restrict__ a, float* out, long long* cyc, int iters, uint32_t nlines) {
    int lane = threadIdx.x & 31, w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    float acc = 0.f;
    long long t0 = clock64();
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
        float x = __ldcg(&a[(size_t)line_of(i, w, nlines) * 32 + lane]);
        acc += chain<N>(x);
    }
    long long t1 = clock64();
    out[w * 32 + lane] = acc;
    if (lane == 0) cyc[w] = t1 - t0;
}
template <int N> __global__ void k_u4(const float* __restrict__ a, float* out, long long* cyc, int iters, uint32_t nlines) {
    int lane = threadIdx.x & 31, w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    float acc = 0.f;
    long long t0 = clock64();
#pragma unroll 4
    for (int i = 0; i < iters; i++) {
        float x = __ldcg(&a[(size_t)line_of(i, w, nlines) * 32 + lane]);
        acc += chain<N>(x);
    }
    long long t1 = clock64();
    out[w * 32 + lane] = acc;
    if (lane == 0) cyc[w] = t1 - t0;
}
template <int N> __global__ void k_pipe(const float* __restrict__ a, float* out, long long* cyc, int iters, uint32_t nlines) {
    int lane = threadIdx.x & 31, w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    float acc = 0.f;
    long long t0 = clock64();
    float x = __ldcg(&a[(size_t)line_of(0, w, nlines) * 32 + lane]);
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
        uint32_t ni = (i + 1 < iters) ? (uint32_t)(i + 1) : (uint32_t)i;
        float xn = __ldcg(&a[(size_t)line_of(ni, w, nlines) * 32 + lane]);   // next iteration's load, hoisted
        acc += chain<N>(x);
        x = xn;
    }
    long long t1 = clock64();
    out[w * 32 + lane] = acc;
    if (lane == 0) cyc[w] = t1 - t0;
}
template <int N> __global__ void k_p4(const float* __restrict__ a, float* out, long long* cyc, int iters, uint32_t nlines) {
    // our reorder on top of unroll x4: the next group's 4 loads are issued before this group's 4 chains
    int lane = threadIdx.x & 31, w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    float acc = 0.f;
    long long t0 = clock64();
#define LD(j) __ldcg(&a[(size_t)line_of((uint32_t)(j), w, nlines) * 32 + lane])
    float x0 = LD(0), x1 = LD(1), x2 = LD(2), x3 = LD(3);
#pragma unroll 1
    for (int i = 0; i < iters; i += 4) {
        int b = (i + 4 < iters) ? i + 4 : i;
        float n0 = LD(b), n1 = LD(b + 1), n2 = LD(b + 2), n3 = LD(b + 3);
        acc += chain<N>(x0); acc += chain<N>(x1); acc += chain<N>(x2); acc += chain<N>(x3);
        x0 = n0; x1 = n1; x2 = n2; x3 = n3;
    }
#undef LD
    long long t1 = clock64();
    out[w * 32 + lane] = acc;
    if (lane == 0) cyc[w] = t1 - t0;
}
typedef void (*kfn)(const float*, float*, long long*, int, uint32_t);
template <int N> void kernels(kfn* k) { k[0] = k_base<N>; k[1] = k_pipe<N>; k[2] = k_u4<N>; k[3] = k_p4<N>; }

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int sms = p.multiProcessorCount;
    printf("# gpu %s sms %d clock_khz %d\n", p.name, sms, p.clockRate);
    const size_t maxBytes = (size_t)512 << 20;
    std::vector<float> h(maxBytes / 4);
    for (size_t i = 0; i < h.size(); i++) h[i] = 1.0f + 1e-4f * (float)(i & 1023);
    float *a, *out; long long* cyc;
    const int maxW = 16, iters = 1024, reps = 7;
    CK(cudaMalloc(&a, maxBytes)); CK(cudaMemcpy(a, h.data(), maxBytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&out, (size_t)sms * maxW * 32 * 4)); CK(cudaMalloc(&cyc, (size_t)sms * maxW * 8));
    const char* names[4] = {"base", "pipe", "u4", "p4"};
    int Ns[4] = {0, 16, 64, 128};
    size_t sets_mb[2] = {8, 512};
    int Ws[3] = {1, 4, 16};
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    printf("kernel,N,set_MB,warps_per_SM,ms_median,cycles_per_iter_median,bitmatch_vs_base\n");
    for (int ni = 0; ni < 4; ni++) for (int si = 0; si < 2; si++) for (int wi = 0; wi < 3; wi++) {
        kfn k[4];
        switch (Ns[ni]) { case 0: kernels<0>(k); break; case 16: kernels<16>(k); break; case 64: kernels<64>(k); break; default: kernels<128>(k); }
        uint32_t nlines = (uint32_t)(sets_mb[si] * (1u << 20) / 128);
        int W = Ws[wi], nw = sms * W;
        std::vector<float> ref(nw * 32), got(nw * 32);
        for (int v = 0; v < 4; v++) {
            std::vector<float> ms; std::vector<double> cpi; std::vector<long long> hc(nw);
            k[v]<<<sms, 32 * W>>>(a, out, cyc, iters, nlines); CK(cudaDeviceSynchronize());   // warm-up
            for (int r = 0; r < reps; r++) {
                CK(cudaEventRecord(e0)); k[v]<<<sms, 32 * W>>>(a, out, cyc, iters, nlines); CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1)); float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t);
                CK(cudaMemcpy(hc.data(), cyc, nw * 8, cudaMemcpyDeviceToHost));
                std::vector<long long> s(hc); std::sort(s.begin(), s.end()); cpi.push_back((double)s[nw / 2] / iters);
            }
            CK(cudaMemcpy(v == 0 ? ref.data() : got.data(), out, nw * 32 * 4, cudaMemcpyDeviceToHost));
            bool match = v == 0 ? true : memcmp(ref.data(), got.data(), nw * 32 * 4) == 0;
            std::sort(ms.begin(), ms.end()); std::sort(cpi.begin(), cpi.end());
            printf("%s,%d,%zu,%d,%.4f,%.1f,%s\n", names[v], Ns[ni], sets_mb[si], W, ms[reps / 2], cpi[reps / 2], match ? "yes" : "NO");
            fflush(stdout);
        }
    }
    return 0;
}
