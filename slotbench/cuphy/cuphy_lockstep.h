#ifndef SB_CUPHY_LOCKSTEP_H
#define SB_CUPHY_LOCKSTEP_H

#include <cuda_runtime.h>

// graph is the already configured, instrumented and device-launchable full-slot graph owned by cuPHY.
// The single-pipeline caller keeps its input/configuration/buffers unchanged throughout this call.
// validate performs output copies and validates decoded bits and CRCs; zero means success.
int sb_cuphy_lockstep_run(cudaGraphExec_t graph, cudaStream_t stream,
                         int (*validate)(void*), void* validation_context);

#endif
