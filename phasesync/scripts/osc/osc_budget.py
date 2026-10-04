import numpy as np
np.set_printoptions(suppress=True)
# ---- reference-oscillator SSB phase noise L(f) at 10 MHz (dBc/Hz) ----
off = np.array([1,10,100,1e3,1e4,1e5,1e6])
REF = {
 # cheap XO (scaled to 10 MHz), typical crystal XO datasheets
 'XO'   : [-50,-85,-115,-135,-148,-152,-152],
 # good 10 MHz TCXO (-100 @10 Hz per rfessentials; floor -150..-155)
 'TCXO' : [-65,-100,-125,-142,-150,-155,-155],
 # low-noise 10 MHz OCXO (KLNE: -120 @10 Hz)
 'OCXO' : [-90,-120,-140,-150,-155,-160,-160],
 # GPSDO: short-term = its disciplined OCXO (GPS loop tau >~100 s)
 'GPSDO': [-90,-120,-140,-150,-155,-160,-160],
 # Microchip SA65 CSAC datasheet
 'CSAC' : [-44,-64,-110,-128,-135,-140,-140],
}
def Lref(name,f):
    lf=np.log10(f); y=np.interp(lf,np.log10(off),REF[name])
    # extrapolate below 1 Hz with 1/f^3 (flicker FM, -30 dB/dec)
    y=np.where(f<1, REF[name][0]-30*np.log10(np.maximum(f,1e-9)), y)
    return y
def Lcarrier(name,fc,f,pll_bw=None,fpd=100e6,highpass=None):
    """SSB phase noise of PLL-synthesised carrier (dBc/Hz).
    reference multiplied by 20log(fc/10MHz) inside PLL BW (2nd-order LP),
    + synth in-band floor (FOM -236 dBc/Hz, LMX2594-class) + 1/f (-129 normalised) ,
    + VCO 1/f^2 outside loop BW. highpass: freq-sync (SyncE/common ref) removes
    relative noise below EEC/DPLL bandwidth (models common-ref case)."""
    if pll_bw is None: pll_bw = 200e3 if fc<10e9 else 1e6
    N=fc/fpd
    ref = 10**((Lref(name,f)+20*np.log10(fc/10e6))/10)
    h = 1/(1+(f/pll_bw)**4)              # closed-loop LP magnitude^2
    syn_inband = 10**((-236+10*np.log10(fpd)+20*np.log10(N))/10)
    flick = 10**((-129+20*np.log10(fc/1e9)-10*np.log10(f/1e4))/10)
    inb = (syn_inband+flick)*h
    # VCO: set = inband level at BW, falls 20 dB/dec above BW, floor -150
    vco_bw = syn_inband+10**((-129+20*np.log10(fc/1e9)-10*np.log10(pll_bw/1e4))/10)
    vco = vco_bw*(pll_bw/f)**2*(1-h) + 10**(-150/10)
    S = ref*h + inb + vco
    if highpass: S = S * (f/highpass)**4/(1+(f/highpass)**4)
    return 10*np.log10(S)

f = np.logspace(-3,7.5,40000)   # 1 mHz .. 31.6 MHz
def Sphi(name,fc,**kw):  # one-sided phase PSD rad^2/Hz, single radio
    return 2*10**(Lcarrier(name,fc,f,**kw)/10)

def rms_err_deg(S, T, D=0.0, mode='hold', two=True):
    """rms phase error between estimate and truth, averaged over the update
    interval [D, D+T] after the measurement (latency D). two=True -> two independent radios."""
    taus = np.linspace(D, D+T, 25)[1:] if T>0 else np.array([D])
    v=[]
    for tau in taus:
        w=2*np.pi*f
        if mode=='hold':
            H2 = 4*np.sin(w*tau/2)**2
        else:  # linear extrapolation, freq from last 2 estimates spaced T
            Tp = T if T>0 else 1e-3
            H = np.exp(1j*w*tau)-1-(tau/Tp)*(1-np.exp(-1j*w*Tp))
            H2 = np.abs(H)**2
        v.append(np.trapezoid(S*H2,f))
    var=np.mean(v)*(2 if two else 1)
    return np.degrees(np.sqrt(var))

def adev(S_phi, fc, tau, fh=1e4):
    Sy = (f**2/fc**2)*S_phi
    m = f<=fh
    x = np.pi*f[m]*tau
    return np.sqrt(np.trapezoid(2*Sy[m]*np.sin(x)**4/x**2, f[m]))

carriers = {'FR1 3.5 GHz':3.5e9,'FR2 28 GHz':28e9,'subTHz 140 GHz':140e9}
print("=== single-carrier phase noise L(f) dBc/Hz (free-running ref) ===")
for cn,fc in carriers.items():
    for o in ['XO','TCXO','OCXO','CSAC']:
        L=Lcarrier(o,fc,np.array([1e3,1e4,1e5,1e6]))
        print(f"{cn:15s} {o:6s} 1k {L[0]:7.1f} 10k {L[1]:7.1f} 100k {L[2]:7.1f} 1M {L[3]:7.1f}")
print("\n=== 10 MHz ref ADEV from model (fh=10 kHz) ===")
for o in ['XO','TCXO','OCXO','CSAC']:
    S=2*10**(Lref(o,f)/10)
    print(o, ["%.1e"%adev(S,10e6,t) for t in [1e-3,1e-2,1e-1,1]])

print("\n=== deterministic drift from residual frequency offset, deg per ms ===")
for eps,lab in [(40e-6,'2xXO 20ppm'),(5e-6,'2xTCXO 2.5ppm (X310)'),(0.1e-6,'3GPP 2x0.05ppm (PTP-only bound)'),
                (40e-9,'2x GPSDO unlocked 20ppb'),(10e-9,'10 ppb'),(1e-9,'1 ppb'),(0.39e-9,'Rel-19 CJTC-F finest step 0.1ppm/255'),(0.2e-9,'0.2 ppb (RFClock common-ref residual)'),(0.02e-9,'2x GPSDO locked 0.01ppb')]:
    print(f"{lab:40s}", " ".join(f"{cn.split()[0]}: {360*eps*fc*1e-3:10.3g} deg/ms" for cn,fc in carriers.items()))
print("\ntime to accumulate 10/30 deg (ms):")
for eps in [0.1e-6,10e-9,1e-9,0.1e-9]:
    print(f"eps={eps*1e9:6.2f} ppb", " ".join(f"{cn.split()[0]}: {10/(360*eps*fc)*1e3:8.3g}/{30/(360*eps*fc)*1e3:8.3g} ms" for cn,fc in carriers.items()))

print("\n=== irreducible intra-update jitter (integrated 1/T..10MHz noise, two radios) ===")
Ts=[0.125e-3,0.5e-3,1e-3,2.5e-3,5e-3,10e-3,20e-3,50e-3,100e-3,1.0]
def table(title,kw,modes=('hold','lin'),osc=('XO','TCXO','OCXO','CSAC'),D=0.0):
    print("\n=== "+title+f" (latency D={D*1e3} ms) rms inter-radio phase error [deg] vs update interval T ===")
    print(" "*28+"".join(f"{t*1e3:>8.3g}" for t in Ts))
    for cn,fc in carriers.items():
        for o in osc:
            S=Sphi(o,fc,**kw)
            for m in modes:
                row=[rms_err_deg(S,T,D=D,mode=m) for T in Ts]
                print(f"{cn.split()[0]:6s}{o:6s}{m:5s}{'':11s}"+"".join(f"{x:8.1f}" for x in row))
table("FREE-RUNNING refs, CFO removed (PTP time sync only; deterministic CFO tracked)",{})
table("FREQ-SYNC (SyncE/common freq), relative noise above 10 Hz EEC BW only",{'highpass':10.0},modes=('hold',))
table("FREQ-SYNC, latency 1 ms",{'highpass':10.0},modes=('hold',),D=1e-3)
table("FREE-RUNNING, latency 1 ms",{},modes=('lin',),D=1e-3)

# floor: total integrated phase jitter 1 kHz-10 MHz per radio
print("\n=== per-radio integrated rms phase jitter 1 kHz - 10 MHz (deg) ===")
for cn,fc in carriers.items():
    for o in ['TCXO','OCXO']:
        S=Sphi(o,fc); m=(f>=1e3)&(f<=1e7)
        print(cn,o, "%.2f deg"%np.degrees(np.sqrt(np.trapezoid(S[m],f[m]))))

# max update interval meeting 10 and 30 deg
print("\n=== max update interval T (ms) for rms inter-radio error < 10 / 30 deg ===")
Tg=np.logspace(-4.5,0.5,60)
def tmax(S,thr,mode,D=0):
    ok=[T for T in Tg if rms_err_deg(S,T,D=D,mode=mode)<thr]
    return max(ok)*1e3 if ok else float('nan')
for cn,fc in carriers.items():
    for o in ['XO','TCXO','OCXO','CSAC']:
        for lab,kw,m in [('free+CFO-hold',{},'hold'),('free+lin-extrap',{},'lin'),('SyncE-hold',{'highpass':10.0},'hold')]:
            S=Sphi(o,fc,**kw)
            print(f"{cn:15s}{o:6s}{lab:16s} 10deg: {tmax(S,10,m):9.3g} ms   30deg: {tmax(S,30,m):9.3g} ms")

print("\n=== coherent combining gain loss (dB) vs per-TX iid Gaussian phase error sigma ===")
for s in [5,7.07,10,15,20,21.2,30,45,60]:
    r=np.radians(s)
    print(f"sigma={s:5.1f} deg  "+"  ".join(f"N={N}: {-10*np.log10(1/N+(1-1/N)*np.exp(-r**2)):5.2f} dB" for N in [2,4,8]))
print("uniform +-D:")
for d in [10,30,60,90]:
    r=np.radians(d); g=(np.sin(r)/r)**2
    print(f"D={d}", "  ".join(f"N={N}: {-10*np.log10(1/N+(1-1/N)*g):5.2f} dB" for N in [2,4,8]))
print("N=2 deterministic difference delta: loss = -20log10(cos(delta/2))")
for d in [10,20,30,60,90,120]:
    print(d, "%.2f dB"%(-20*np.log10(np.cos(np.radians(d)/2))))
print("ZF/MU leakage SIR ceiling ~ e^-s2/(1-e^-s2):")
for s in [5,10,20,30]:
    r=np.radians(s); print(s, "%.1f dB"%(10*np.log10(np.exp(-r**2)/(1-np.exp(-r**2)))))
