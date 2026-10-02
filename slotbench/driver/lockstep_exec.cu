// The slot executive: one resident GPU thread that launches the slot graph itself (CUDA device graph
// launch) at target instants on %globaltimer. Compiled with -rdc=true and linked with cudadevrt.
//
// For each boundary k: wait until the previous slot has finished (boundaries that pass meanwhile are
// skipped and counted), wait for g_target[k], launch the slot graph fire-and-forget and record the
// times. CUDA allows 120 fire-and-forget launches per execution of a graph, so after `chunk` attempts
// the executive saves its state and tail-launches itself; the tail launch starts once this execution
// and all its children are complete. Each attempt uses a distinct slot graph handle in that generation;
// the final kernel's completion marker alone does not prove CUDA has finished with its graph handle.
#include <cuda/atomic>

#include "lockstep_common.h"

namespace sb {
namespace {

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

__global__ void k_executive(const cudaGraphExec_t *exec_pool, const unsigned long long *g_target, int n,
                            unsigned long long *done, LsRec *rec, ExecState *state, int chunk,
                            volatile int *finished_host) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    ExecState st = *state;
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> completion(done[0]);
    if (st.chunks == 0) st.prev_done = completion.load(cuda::memory_order_acquire);
    st.chunks++;
    int fired = 0;
    bool timeout = false;
    while (st.k < n && !timeout) {
        // 1. the previous slot must be complete before anything else is launched
        if (st.launched >= 0) {
            unsigned long long completed;
            while ((completed = completion.load(cuda::memory_order_acquire)) == st.prev_done) {
                unsigned long long now = gtimer();
                if (st.k < n && now >= g_target[st.k]) {   // boundary passed while busy: skip it
                    rec[st.k].g_target = g_target[st.k];
                    rec[st.k].flags = 1ull;
                    st.n_skipped++;
                    st.k++;
                    if (st.k >= n) break;
                }
                if (now > st.deadline_g) { timeout = true; break; }
            }
            if (completed != st.prev_done) {
                st.prev_done = completed;
                rec[st.launched].g0 = done[1];
                rec[st.launched].g1 = done[2];
                st.launched = -1;
            } else if (timeout) {
                rec[st.launched].flags |= 8ull;
                break;
            }
            if (st.k >= n) break;
        }
        if (fired >= chunk) break;   // hand over to the next execution of this graph
        // 2. wait for the boundary
        unsigned long long T = g_target[st.k];
        while (gtimer() < T) {}
        // 3. launch from the device
        unsigned long long tl = gtimer();
        cudaError_t e = cudaGraphLaunch(exec_pool[fired], cudaStreamGraphFireAndForget);
        unsigned long long tl2 = gtimer();
        rec[st.k].g_target = T;
        rec[st.k].g_launch = tl;
        rec[st.k].g_launch_done = tl2;
        if (fired == 0) rec[st.k].flags |= 4ull;
        fired++;  // never reuse a handle within this generation, including after an error
        if (e != cudaSuccess) {
            rec[st.k].flags |= 2ull;
            if (!st.first_err) st.first_err = (int)e;
            st.n_err++;
            st.k++;
            continue;
        }
        st.launched = st.k;
        st.n_launched++;
        st.k++;
    }
    if (st.k >= n && st.launched >= 0 && !timeout) {  // drain the last slot
        unsigned long long completed;
        while ((completed = completion.load(cuda::memory_order_acquire)) == st.prev_done) {
            if (gtimer() > st.deadline_g) { timeout = true; break; }
        }
        if (!timeout) {
            st.prev_done = completed;
            rec[st.launched].g0 = done[1];
            rec[st.launched].g1 = done[2];
            st.launched = -1;
        }
    }
    if (timeout) {
        if (st.launched >= 0) rec[st.launched].flags |= 8ull;
        // Every requested boundary must remain in the result denominator, including work abandoned
        // when the deadline expires before all boundaries have been processed.
        while (st.k < n) {
            rec[st.k].g_target = g_target[st.k];
            rec[st.k].flags |= 8ull;
            st.k++;
        }
        st.finished = 3;
    }
    else if (st.k >= n) st.finished = 1;
    *state = st;
    __threadfence_system();
    if (st.finished) {
        *finished_host = st.finished;
    } else {
        cudaError_t e = cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
        if (e != cudaSuccess) {
            state->n_tail_err++;
            state->finished = 2;
            if (!state->first_err) state->first_err = (int)e;
            __threadfence_system();
            *finished_host = 2;
        }
    }
}

}  // namespace

void launch_executive(cudaStream_t s, const cudaGraphExec_t *exec_pool, const unsigned long long *g_target, int n,
                      unsigned long long *done, LsRec *rec, ExecState *state, int chunk, int *finished_host) {
    k_executive<<<1, 32, 0, s>>>(exec_pool, g_target, n, done, rec, state, chunk,
                                 (volatile int *)finished_host);
}

}  // namespace sb
