import pyopencl as cl, numpy as np, time
ctx = cl.create_some_context(interactive=False); dev = ctx.devices[0]
q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
for i in range(3):
    try: print("pair", dev.device_and_host_timer(), "host", dev.host_timer(), "perf_ns", time.perf_counter_ns())
    except Exception as e: print("pair FAIL", e); break
src = """__kernel void k(__global ulong *o){ if(get_global_id(0)==0){ o[0]=__builtin_amdgcn_s_sendmsg_rtnl(131); } }"""
prg = cl.Program(ctx, src).build()
buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 8); out = np.zeros(1, np.uint64)
prev=None
for i in range(4):
    ev = prg.k(q, (64,), None, buf); ev.wait(); cl.enqueue_copy(q, out, buf); t=time.perf_counter_ns()
    if prev: print("realtime delta ticks", int(out[0])-prev[0], "host delta ns", t-prev[1], "evt", ev.profile.start)
    prev=(int(out[0]),t); time.sleep(0.5)
