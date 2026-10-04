import numpy as np
def pz(f,PSD0,fz,az,fp,ap):
    S=10**(PSD0/10)*np.ones_like(f)
    for z,a in zip(fz,az): S*=1+(f/z)**a
    for p,a in zip(fp,ap): S/=1+(f/p)**a
    return S
f=np.logspace(0,8,200000)
# TR 38.803 Table 6.1.10-1 (29.55 GHz) as quoted by MathWorks / arXiv 2412.05841
S=pz(f,32,[3e3,0.55e6,280e6],[2.37,2.7,2.53],[1,1.6e6,30e6],[3.3,3.3,1])
for o in [1e3,1e4,1e5,1e6,1e7]:
    i=np.argmin(abs(f-o)); print("offset %.0e : %.1f dBc/Hz"%(o,10*np.log10(S[i])))
for lo,hi in [(1e3,1e7),(1e4,1e7),(1e3,1e6)]:
    m=(f>=lo)&(f<=hi); print("int %g-%g Hz: %.2f deg rms (SSB*2)"%(lo,hi,np.degrees(np.sqrt(2*np.trapezoid(S[m],f[m])))))
print("Model A 30 GHz")
S=pz(f,-79.4,[1.8e6,2.2e6,40e6],[2,2,2],[0.1e6,0.2e6,8e6],[2,2,2])
for o in [1e3,1e4,1e5,1e6,1e7]:
    i=np.argmin(abs(f-o)); print("offset %.0e : %.1f dBc/Hz"%(o,10*np.log10(S[i])))
m=(f>=1e3)&(f<=1e8); print("int 1k-100M %.2f deg"%np.degrees(np.sqrt(2*np.trapezoid(S[m],f[m]))))
print("Model B 60 GHz")
S=pz(f,-70,[0.02e6,6e6,10e6],[2,2,2],[0.005e6,0.4e6,0.6e6],[2,2,2])
for o in [1e3,1e4,1e5,1e6,1e7]:
    i=np.argmin(abs(f-o)); print("offset %.0e : %.1f dBc/Hz"%(o,10*np.log10(S[i])))
m=(f>=1e3)&(f<=1e8); print("int 1k-100M %.2f deg"%np.degrees(np.sqrt(2*np.trapezoid(S[m],f[m]))))
