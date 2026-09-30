import numpy as np
d=np.load("data.npz"); ph=["idle","cpu_load","gpu_load"]
A=np.vstack([d[p] for p in ph]); lab=np.concatenate([[p]*len(d[p]) for p in ph])
h0,h1,tk,es,ee=A.T; w=h1-h0; mid=(h0+h1)/2
t0=h0[0]; T0=tk[0]
# robust fit on narrowest 10% brackets per phase
sel=np.zeros(len(A),bool)
for p in ph:
    m=lab==p; sel|= m & (w<=np.percentile(w[m],10))
c=np.polyfit(tk[sel]-T0, mid[sel]-t0, 1)
ns_per_tick=c[0]; f=1e9/ns_per_tick
print(f"GPU REALTIME freq vs host QPC: {f/1e6:.6f} MHz  -> {(f/1e8-1)*1e6:+.2f} ppm vs nominal 100 MHz")
pred=np.polyval(c,tk-T0)+t0            # host-time estimate of GPU read
err=pred-mid
inside=(pred>=h0)&(pred<=h1)
for p in ph:
    m=lab==p
    print(f"\n[{p}] bracket width ns: median {np.median(w[m]):.0f}, p99 {np.percentile(w[m],99):.0f}, max {w[m].max():.0f}")
    print(f"  single-fit estimate inside bracket: {inside[m].mean()*100:.1f}%   |pred-mid| p50 {np.median(abs(err[m])):.0f} ns p99 {np.percentile(abs(err[m]),99):.0f} ns")
    # driver's own timestamps vs independent clock
    ins=(pred[m]>=es[m])&(pred[m]<=ee[m])
    print(f"  driver event [start,end] contains GPU read: {ins.mean()*100:.1f}%   start-pred p50 {np.median(es[m]-pred[m]):.0f} ns, end-start p50 {np.median(ee[m]-es[m]):.0f} ns")
    evb=(es[m]>=h0[m])&(ee[m]<=h1[m]); print(f"  driver event inside host bracket: {evb.mean()*100:.1f}%")
# drift wander: slope per 10 s window
print("\nfreq per 10 s window (ppm vs fit):")
tt=(h0-t0)/1e9; out=[]
for s in np.arange(0,tt.max(),10):
    m=(tt>=s)&(tt<s+10)&sel
    if m.sum()>20:
        cc=np.polyfit(tk[m]-T0,mid[m]-t0,1); out.append((s,(ns_per_tick/cc[0]-1)*1e6, lab[m][0]))
for s,p,l in out: print(f"  t={s:5.0f}s {l:9s} {p:+.3f} ppm")
res=err[sel]; q=np.polyfit(tt[sel],res,2); print(f"\nresidual trend over run (quadratic coef): {q[0]:.3f} ns/s^2; residual range {res.min():.0f}..{res.max():.0f} ns")
