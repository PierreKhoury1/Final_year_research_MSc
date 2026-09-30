import pyopencl as cl, numpy as np, time
ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
print("device:", dev.name, "timer res ns:", dev.profiling_timer_resolution)
try:
    print("device_and_host_timer:", dev.device_and_host_timer, "host_timer:", dev.host_timer)
except Exception as e: print("pair timer FAIL:", e)
src = """
__kernel void k(__global ulong *o){
  if(get_global_id(0)==0){ o[0]=__builtin_amdgcn_s_memrealtime(); o[1]=__builtin_amdgcn_s_memtime(); }
}"""
try:
    prg = cl.Program(ctx, src).build()
    buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 16)
    out = np.zeros(2, np.uint64)
    for i in range(3):
        ev = prg.k(q, (64,), None, buf); ev.wait(); cl.enqueue_copy(q, out, buf)
        print("memrealtime", out[0], "memtime", out[1], "evt start", ev.profile.start)
        time.sleep(0.5)
except Exception as e: print("kernel FAIL:", e)
