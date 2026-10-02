// Shared between lockstep_driver.cu (host logic, no rdc) and lockstep_exec.cu (the device-launching executive,
// compiled with -rdc=true for device graph launch).
#pragma once
#include <cuda_runtime.h>

namespace sb {

struct LsRec {                   // 64 bytes, little endian, written raw to --raw (after warm-up)
    unsigned long long slot;
    long long t_target;          // host ns (CLOCK_MONOTONIC_RAW) of boundary k
    unsigned long long g_target; // the same instant in GPU time (pre-run clock fit)
    unsigned long long g_launch; // GPU time just before the launch call (gpu mode) / host t0 mapped (cpu mode)
    unsigned long long g_launch_done;  // GPU time just after the launch call returned (gpu mode) / host t_after mapped
    unsigned long long g0, g1;   // slot start / end stamps (GPU clock); 0 if skipped
    unsigned long long flags;    // 1 skipped (previous slot still running), 2 launch error, 4 first slot of an executive generation, 8 timeout
};
static_assert(sizeof(LsRec) == 64, "LsRec");

struct ExecState {
    int k;                 // next boundary
    int launched;          // slot in flight (-1 none)
    int chunks;            // executive executions so far
    int finished;          // 1 all boundaries handled, 2 tail relaunch failed, 3 in-kernel timeout
    unsigned long long prev_done;
    unsigned long long deadline_g;  // whole-run GPU deadline; set before launching the captured executive
    unsigned long long n_launched, n_skipped, n_err, n_tail_err;
    int first_err;         // first cudaError_t from a device launch (0 none)
    int pad;
};

// lockstep_exec.cu: launches the executive kernel into stream s (the caller captures it into a graph).
// exec_pool: device array of `chunk` distinct, uploaded slot graph handles. Each is used at most once
// per executive generation; the tail launch completes every child graph before any handle is reused.
// done: device memory {completion sequence, start stamp, end stamp}; the sequence is published with
// a device-scope atomic release store and consumed with acquire loads. finished_host: mapped host int.
void launch_executive(cudaStream_t s, const cudaGraphExec_t *exec_pool, const unsigned long long *g_target, int n,
                      unsigned long long *done, LsRec *rec, ExecState *state, int chunk, int *finished_host);

}  // namespace sb
