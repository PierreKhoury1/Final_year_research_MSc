import pyopencl as cl, numpy as np, time, threading, warnings
warnings.filterwarnings("ignore")
ctx = cl.create_some_context(interactive=False); dev = ctx.devices[0]
q = cl.CommandQueue(ctx)
SRC = """
__kernel void pp(__global volatile uint *ctl, __global ulong *gt, uint n, uint base){
  for(uint i=0;i<n;i++){ uint want=base+i+1; uint c; ulong s=0;
    while((c=ctl[0])!=want){ if(c==0xFFFFFFFEu||++s>400000000ul) return; }
    gt[i]=__builtin_amdgcn_s_sendmsg_rtnl(131); mem_fence(CLK_GLOBAL_MEM_FENCE); ctl[1]=want; } }"""
prg = cl.Program(ctx, SRC).build(options=["-cl-std=CL2.0"])
flags = cl.svm_mem_flags.READ_WRITE | cl.svm_mem_flags.SVM_FINE_GRAIN_BUFFER
ctl = cl.SVM(cl.svm_empty(ctx, flags, 64, np.uint32)); gt = cl.SVM(cl.svm_empty(ctx, flags, 400, np.uint64))
c = ctl.mem; g = gt.mem
k = cl.Kernel(prg, "pp"); N = 400; base = 0
c[0] = base; c[1] = base
k.set_args(ctl, gt, np.uint32(N), np.uint32(base)); cl.enqueue_nd_range_kernel(q, k, (1,), None); q.flush()
pc = time.perf_counter_ns; t0s = []; t1s = []
for i in range(N):
    w = base + i + 1; a = pc(); c[0] = w
    while c[1] != w: pass
    b = pc(); t0s.append(a); t1s.append(b)
c[0] = 0xFFFFFFFE; q.finish()
wd = np.array(t1s) - np.array(t0s)
print("bracket ns: min", wd.min(), "p10", np.percentile(wd, 10), "median", np.median(wd), "p99", np.percentile(wd, 99))
print("gpu ticks ok:", g[:3], np.all(np.diff(g.astype(np.int64)) > 0))
