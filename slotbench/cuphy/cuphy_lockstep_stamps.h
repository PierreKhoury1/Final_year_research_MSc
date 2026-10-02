#ifndef SB_CUPHY_LOCKSTEP_STAMPS_H
#define SB_CUPHY_LOCKSTEP_STAMPS_H

#include <cuda_runtime.h>

// One instrumented pipeline per worker/context. The mapped allocation lives until process exit,
// because the owning cuPHY graph can outlive the benchmark helper.
struct SbCuPhyStamps {
    unsigned long long start;
    unsigned long long end;
    unsigned long long sequence;
};

cudaError_t sb_cuphy_lockstep_attach_stamps(cudaGraph_t graph, int priority);
cudaError_t sb_cuphy_lockstep_get_stamps(SbCuPhyStamps** host, SbCuPhyStamps** device);

#endif
