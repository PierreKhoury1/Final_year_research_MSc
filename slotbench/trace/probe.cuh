// GPU residency probe: one resident thread in its own stream that watches %globaltimer and records
// every interval in which it did not run for longer than gap_ns. Under cross-process time-slicing the
// whole context is switched out, so the probe's gaps are the times the slot's context did not own the
// GPU. The probe sleeps ~sleep_ns between readings (__nanosleep, sm_70+) to keep its own footprint
// small; it still keeps one SM slot and the GPU "active" (which can hold clocks up — record it).
#pragma once
#include <cuda_runtime.h>
#include "slottrace.h"

namespace sb {

__device__ __forceinline__ unsigned long long probe_gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

struct ProbeDev {
    ProbeGap *gaps;              // device ring of gap records
    unsigned long long *stats;   // [0] n_gaps, [1] g_first, [2] g_last, [3] iterations
    unsigned int capacity;
    unsigned long long gap_ns;
    unsigned int sleep_ns;
    unsigned long long deadline_g;  // stop on its own after this GPU time (watchdog)
    volatile int *stop;             // mapped host flag
};

__global__ void k_probe(ProbeDev p) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned long long last = probe_gtimer(), first = last, it = 0, n = 0;
    for (;;) {
#if __CUDA_ARCH__ >= 700
        if (p.sleep_ns) __nanosleep(p.sleep_ns);
#endif
        unsigned long long t = probe_gtimer();
        if (t - last > p.gap_ns) {
            if (n < p.capacity) { p.gaps[n].g_from = last; p.gaps[n].g_to = t; }
            n++;
        }
        last = t;
        if ((++it & 63ull) == 0 && (*p.stop || t > p.deadline_g)) break;
    }
    p.stats[0] = n; p.stats[1] = first; p.stats[2] = last; p.stats[3] = it;
    __threadfence_system();
}

}  // namespace sb
