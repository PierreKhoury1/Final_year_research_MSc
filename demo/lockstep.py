"""How precisely can the CPU make the GPU start work at a chosen instant T?

Methods: (1) normal launch at T, (2) pre-enqueued kernel released by a user event at T,
(3) persistent kernel released by a shared-memory flag at T, (4) persistent kernel that waits on its
own clock for T (converted to GPU ticks with the certified mapping). Each run idle and under GPU load.
Start error = (GPU clock read at kernel start, mapped to host time) - T.
"""
import json
import threading
import time

import numpy as np
import pyopencl as cl

from phy import GPU, fit_clock

N = 250            # targets per method/condition
GAP_NS = 2_000_000  # 2 ms between targets

SRC = r"""
inline ulong gclk(void){ return __builtin_amdgcn_s_sendmsg_rtnl(131); }
__kernel void work(__global ulong *st, uint idx, __global float *a){
  if(get_global_id(0) == 0) st[idx] = gclk();
  float v = a[get_global_id(0)]; for(int k = 0; k < 200; k++) v = v*1.0000001f + 0.5f; a[get_global_id(0)] = v;
}
__kernel void flagged(__global volatile uint *ctl, __global ulong *st, uint n){
  for(uint i = 0; i < n; i++){ ulong sp = 0; uint c;
    while((c = ctl[0]) != i + 1){ if(c == 0xFFFFFFFEu || ++sp > 2000000000ul) return; }
    st[i] = gclk(); mem_fence(CLK_GLOBAL_MEM_FENCE); ctl[1] = i + 1; }
}
__kernel void selftimed(__global const ulong *tgt, __global ulong *st, uint n){
  for(uint i = 0; i < n; i++){ ulong t; while((t = gclk()) < tgt[i]) ; st[i] = t; }
}
__kernel void ai_job(__global float *a, int iters){
  int i = get_global_id(0); float v = a[i]; for(int k = 0; k < iters; k++) v = v*1.0000001f + 0.5f; a[i] = v;
}
"""


def spin_until(t):
    pc = time.perf_counter_ns
    while pc() < t:
        pass


class Load:
    """Background 'AI job': keeps the GPU busy with large FMA kernels on its own queue."""
    def __init__(self, g, prg):
        self.q = cl.CommandQueue(g.ctx)
        self.k = cl.Kernel(prg, "ai_job")
        self.buf = cl.Buffer(g.ctx, cl.mem_flags.READ_WRITE, 4 << 22)
        self.k.set_args(self.buf, np.int32(3000))
        self.stop = threading.Event()
        self.th = threading.Thread(target=self.run)

    def run(self):
        while not self.stop.is_set():
            cl.enqueue_nd_range_kernel(self.q, self.k, (1 << 22,), None)
            self.q.finish()

    def __enter__(self):
        self.th.start(); time.sleep(0.3); return self

    def __exit__(self, *a):
        self.stop.set(); self.th.join()


def main():
    g = GPU()
    prg = cl.Program(g.ctx, SRC).build(options=["-cl-std=CL2.0"])
    q = cl.CommandQueue(g.ctx)
    f = cl.svm_mem_flags.READ_WRITE | cl.svm_mem_flags.SVM_FINE_GRAIN_BUFFER
    st = cl.SVM(cl.svm_empty(g.ctx, f, N, np.uint64)); stm = st.mem
    ctl = cl.SVM(cl.svm_empty(g.ctx, f, 16, np.uint32)); c = ctl.mem
    tgt = cl.SVM(cl.svm_empty(g.ctx, f, N, np.uint64)); tg = tgt.mem
    a = cl.Buffer(g.ctx, cl.mem_flags.READ_WRITE, 4 * 16384)
    k_work, k_flag, k_self = (cl.Kernel(prg, n) for n in ("work", "flagged", "selftimed"))

    br = [g.calibrate(batches=6)]
    pc = time.perf_counter_ns
    results = {}

    def run_method(name):
        T = pc() + 50_000_000 + np.arange(N, dtype=np.int64) * GAP_NS   # targets, host ns
        stm[:] = 0
        if name == "normal launch":
            for i in range(N):
                spin_until(T[i])
                k_work.set_args(st, np.uint32(i), a)
                cl.enqueue_nd_range_kernel(q, k_work, (4096,), (256,)); q.flush()
            q.finish()
        elif name == "pre-loaded launch":
            evs = []
            for i in range(N):
                ue = cl.UserEvent(g.ctx)
                k_work.set_args(st, np.uint32(i), a)
                cl.enqueue_nd_range_kernel(q, k_work, (4096,), (256,), wait_for=[ue]); evs.append(ue)
            q.flush()
            for i in range(N):
                spin_until(T[i]); evs[i].set_status(cl.command_execution_status.COMPLETE)
            q.finish()
        elif name == "persistent kernel":
            c[0] = 0; c[1] = 0
            k_flag.set_args(ctl, st, np.uint32(N))
            cl.enqueue_nd_range_kernel(q, k_flag, (1,), None); q.flush()
            for i in range(N):
                spin_until(T[i]); c[0] = i + 1
            spin_until(T[-1] + 2_000_000); c[0] = 0xFFFFFFFE; q.finish()
        elif name == "GPU self-timed":
            A, B, _, tick0, _ = fit_clock(np.vstack(br))
            tg[:] = np.round((T - B) / A + tick0).astype(np.uint64)
            k_self.set_args(tgt, st, np.uint32(N))
            cl.enqueue_nd_range_kernel(q, k_self, (1,), None); q.finish()
        return T, np.array(stm, dtype=np.float64)

    methods = ["normal launch", "pre-loaded launch", "persistent kernel", "GPU self-timed"]
    raw = {}
    for cond in ("idle", "GPU busy with AI job"):
        for m in methods:
            if cond == "idle":
                raw[(m, cond)] = run_method(m)
            else:
                with Load(g, prg):
                    raw[(m, cond)] = run_method(m)
            br.append(g.calibrate(batches=2))
            print(f"done: {m} / {cond}", flush=True)

    A, B, eps, tick0, nsel = fit_clock(np.vstack(br))
    for (m, cond), (T, ticks) in raw.items():
        ok = ticks > 0
        if not ok.any():
            results[f"{m}|{cond}"] = {"n": 0}; continue
        err_us = ((A * (ticks[ok] - tick0) + B) - T[ok]) / 1e3
        results[f"{m}|{cond}"] = {
            "n": int(ok.sum()), "median_us": float(np.median(err_us)),
            "p99_us": float(np.percentile(err_us, 99)), "max_us": float(err_us.max()),
            "min_us": float(err_us.min()),
            "jitter_us": float(np.percentile(err_us, 99) - np.percentile(err_us, 1)),
        }
    out = {"clock_bound_us": eps / 1e3, "calib_brackets_used": nsel, "results": results}
    json.dump(out, open("lockstep_results.json", "w"), indent=1)
    print(f"clock mapping bound: +/-{eps/1e3:.2f} us ({nsel} tight brackets)")
    print(f"{'method':20s} {'condition':22s} {'median':>9s} {'p99':>9s} {'worst':>9s} {'jitter':>9s}  (us, start error vs T)")
    for key, r in results.items():
        m, cond = key.split("|")
        if r["n"] == 0:
            print(f"{m:20s} {cond:22s} NO SAMPLES"); continue
        print(f"{m:20s} {cond:22s} {r['median_us']:9.1f} {r['p99_us']:9.1f} {r['max_us']:9.1f} {r['jitter_us']:9.1f}")


if __name__ == "__main__":
    main()
