import numpy as np
fc=3.5e9
for name,ppm in [("XO 10ppm",10),("XO 2ppm",2),("TCXO 0.5ppm",0.5),("OCXO 0.01ppm",0.01),("SyncE-locked ~16ppb",0.016)]:
    df=fc*ppm*1e-6
    print(f"{name:22s} CFO={df:9.1f} Hz  deg/slot(0.5ms)={360*df*0.5e-3:10.1f}  frac of 30kHz SCS={df/30e3:.4f}  ICI SIR~{-10*np.log10((np.pi*df/30e3)**2/3+1e-30):.1f} dB")
for s in [5,10,20,30,45]:
    r=np.deg2rad(s)
    print(f"sigma={s}deg coherent gain loss (large N) = {10*np.log10(np.exp(-r**2)):.2f} dB ; ZF leakage SINR cap ~ {10*np.log10(1/r**2):.1f} dB")
# Monte Carlo N=4
rng=np.random.default_rng(0)
for s in [10,20,30]:
    th=rng.normal(0,np.deg2rad(s),(200000,4))
    g=np.abs(np.exp(1j*th).sum(1))**2/16
    print(f"N=4 sigma={s}: mean gain {10*np.log10(g.mean()):.2f} dB")
