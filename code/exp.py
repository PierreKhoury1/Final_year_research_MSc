# CPU-GPU clock experiment: bracket GPU REALTIME reads between host QPC reads.
import pyopencl as cl, numpy as np, time, multiprocessing as mp, threading, json, os, warnings
warnings.filterwarnings("ignore")
def spin(stop):
    x=0
    while not stop.is_set(): x+=1
SRC = """
__kernel void stamp(__global ulong *o){ if(get_global_id(0)==0) o[0]=__builtin_amdgcn_s_sendmsg_rtnl(131); }
__kernel void burn(__global float *a){ float v=a[get_global_id(0)]; for(int i=0;i<20000;i++) v=v*1.0000001f+0.5f; a[get_global_id(0)]=v; }
"""
def phase(name, secs, ctx, q, k, buf):
    out=np.zeros(1,np.uint64); rows=[]; t_end=time.perf_counter()+secs
    while time.perf_counter()<t_end:
        h0=time.perf_counter_ns(); ev=k(q,(64,),None,buf); ev.wait(); h1=time.perf_counter_ns()
        cl.enqueue_copy(q,out,buf)
        rows.append((h0,h1,int(out[0]),ev.profile.start,ev.profile.end))
        time.sleep(0.004)
    print(name, len(rows), "samples", flush=True)
    return np.array(rows,dtype=np.float64)
if __name__=="__main__":
    ctx=cl.create_some_context(interactive=False)
    q=cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    q2=cl.CommandQueue(ctx)
    prg=cl.Program(ctx,SRC).build(); k=cl.Kernel(prg,"stamp"); kb=cl.Kernel(prg,"burn")
    buf=cl.Buffer(ctx,cl.mem_flags.READ_WRITE,8)
    big=cl.Buffer(ctx,cl.mem_flags.READ_WRITE,4*1<<20)
    res={}
    res["idle"]=phase("idle",60,ctx,q,k,buf)
    stop=mp.Event(); ps=[mp.Process(target=spin,args=(stop,)) for _ in range(os.cpu_count())]
    [p.start() for p in ps]; res["cpu_load"]=phase("cpu_load",60,ctx,q,k,buf); stop.set(); [p.join() for p in ps]
    gstop=threading.Event()
    def gburn():
        while not gstop.is_set(): kb.set_arg(0,big); cl.enqueue_nd_range_kernel(q2,kb,(1<<20,),None); q2.finish()
    th=threading.Thread(target=gburn); th.start()
    res["gpu_load"]=phase("gpu_load",60,ctx,q,k,buf); gstop.set(); th.join()
    np.savez("data.npz",**res)
