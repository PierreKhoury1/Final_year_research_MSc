# Improvement checks on existing data: naive vs min-filter calibration, and how many samples a good calibration needs.
import numpy as np
from ana2 import load, PH
for cfg in ("default", "tuned"):
    hz, r = load(f"pp_{cfg}.bin")
    t0 = (r["t0"] - r["t0"][0]) / hz * 1e9; t1 = (r["t1"] - r["t0"][0]) / hz * 1e9
    G = (r["g"] - r["g"][0]).astype(float); w = t1 - t0; mid = (t0 + t1) / 2
    tight = w < 3000
    def viol(c):  # how far a mapping lands outside the brackets it must satisfy (tight brackets only)
        p = np.polyval(c, G[tight]); v = np.maximum(t0[tight] - p, p - t1[tight]); return np.percentile(np.maximum(v, 0), 99), np.max(v)
    naive = np.polyfit(G, mid, 1)                                  # average every sample
    minf = np.polyfit(G[w <= np.percentile(w, 1)], mid[w <= np.percentile(w, 1)], 1)  # keep narrowest 1%
    print(f"== {cfg}")
    for name, c in (("naive (all samples)", naive), ("min-filter (best 1%)", minf)):
        p99, mx = viol(c); print(f"  {name:22s} lands outside tight brackets: p99 {p99/1e3:8.2f} us  worst {mx/1e3:8.2f} us")
    # calibration speed: from N consecutive samples, the best bracket bounds the offset error to half its width
    for ph in ("tight", "cpu_load", "gpu_load"):
        m = np.where(r["ph"] == PH.index(ph))[0]; wm = w[m]; tm = t0[m]
        out = []
        for N in (10, 100, 1000, 10000):
            k = len(wm) // N
            if k < 3: continue
            best = wm[: k * N].reshape(k, N).min(1) / 2
            dur = np.median(tm[N - 1 : k * N : N] - tm[0 : k * N : N]) / 1e6
            out.append(f"N={N:5d} ({dur:6.1f} ms): bound p50 {np.median(best)/1e3:6.2f} us, worst {best.max()/1e3:7.2f} us")
        print(f"  [{ph}]"); [print("    " + o) for o in out]
