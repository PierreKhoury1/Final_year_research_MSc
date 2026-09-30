"""Analyse ping-pong runs: bracket widths, GPU-vs-TSC rate, offset wander, stall attribution."""
import json
import numpy as np

PH = ["tight", "sleepy", "cpu_load", "gpu_load", "tight2"]
DT = np.dtype([("ph", "u4"), ("ok", "u4"), ("t0", "u8"), ("t1", "u8"), ("g", "u8")])


def load(fn):
    b = open(fn, "rb").read()
    hz = np.frombuffer(b[:8], np.float64)[0]
    r = np.frombuffer(b[8:], DT)
    return hz, r[r["ok"] == 1]


def analyse(fn):
    hz, r = load(fn)
    t0 = r["t0"].astype(np.float64)
    t1 = r["t1"].astype(np.float64)
    g = r["g"].astype(np.float64)
    ns0 = (t0 - t0[0]) / hz * 1e9
    ns1 = (t1 - t0[0]) / hz * 1e9
    w = ns1 - ns0
    mid = (ns0 + ns1) / 2
    G = g - g[0]
    # fit GPU ticks -> host ns on the tightest brackets, then refine to points the model keeps inside
    sel = w <= np.percentile(w, 5)
    for _ in range(3):
        c = np.polyfit(G[sel], mid[sel], 1)
        pred = np.polyval(c, G)
        sel = sel & (np.abs(pred - mid) <= w / 2 + 50)
    c = np.polyfit(G[sel], mid[sel], 1)
    pred = np.polyval(c, G)
    f_gpu = 1e9 / c[0]
    out = {"file": fn, "tsc_mhz": hz / 1e6, "gpu_mhz": f_gpu / 1e6,
           "gpu_ppm_vs_100": (f_gpu / 1e8 - 1) * 1e6, "n": int(len(r)),
           "inside_pct": float(np.mean((pred >= ns0 - 1) & (pred <= ns1 + 1)) * 100), "phases": {}}
    for i, p in enumerate(PH):
        m = r["ph"] == i
        if not m.any():
            continue
        wm = w[m]
        before = (pred - ns0)[m]
        after = (ns1 - pred)[m]
        big = wm > 20000
        out["phases"][p] = {
            "n": int(m.sum()),
            "p1": float(np.percentile(wm, 1)), "p50": float(np.median(wm)),
            "p90": float(np.percentile(wm, 90)), "p99": float(np.percentile(wm, 99)),
            "max": float(wm.max()),
            "gap_ms": float(np.median(np.diff(ns0[m])) / 1e6),
            "frac_over_20us": float(big.mean()),
            # where the time went in slow round trips: before GPU saw the write vs after GPU read
            "stall_before_ms": float(before[big].sum() / 1e6), "stall_after_ms": float(after[big].sum() / 1e6),
        }
    # rate wander: GPU-vs-TSC frequency per 10 s window, from tight brackets only
    tt = ns0 / 1e9
    win = []
    for s in np.arange(0, tt.max() - 5, 10):
        m = sel & (tt >= s) & (tt < s + 10)
        if m.sum() > 50:
            cc = np.polyfit(G[m], mid[m], 1)
            win.append({"t": float(s), "ppm": float((c[0] / cc[0] - 1) * 1e6), "phase": PH[int(np.bincount(r["ph"][m]).argmax())]})
    out["windows"] = win
    ppm = np.array([x["ppm"] for x in win])
    out["wander_ppm_std"] = float(ppm.std())
    out["wander_ppm_range"] = [float(ppm.min()), float(ppm.max())]
    # offset residual over time after single linear fit (thinned for plotting)
    res = (pred - mid)[sel]
    ts = tt[sel]
    k = max(1, len(ts) // 300)
    out["residual"] = {"t": ts[::k].round(2).tolist(), "ns": res[::k].round(0).tolist(),
                       "p99_abs_ns": float(np.percentile(np.abs(res), 99))}
    # quadratic term: does the rate itself drift over the run
    q = np.polyfit(tt[sel], (pred - mid)[sel], 2)
    out["quad_ns_per_s2"] = float(q[0])
    return out


if __name__ == "__main__":
    res = {k: analyse(f"pp_{k}.bin") for k in ("default", "tuned")}
    json.dump(res, open("summary.json", "w"), indent=1)
    for k, v in res.items():
        print(f"== {k}: n={v['n']} GPU {v['gpu_mhz']:.6f} MHz ({v['gpu_ppm_vs_100']:+.1f} ppm) TSC {v['tsc_mhz']:.3f} MHz inside {v['inside_pct']:.1f}%")
        print(f"   wander std {v['wander_ppm_std']:.4f} ppm range {v['wander_ppm_range']} quad {v['quad_ns_per_s2']:.3f} ns/s^2 resid p99 {v['residual']['p99_abs_ns']:.0f} ns")
        for p, s in v["phases"].items():
            print(f"   {p:9s} n={s['n']:7d} p1 {s['p1']/1e3:7.2f}us p50 {s['p50']/1e3:8.2f}us p90 {s['p90']/1e3:8.2f}us p99 {s['p99']/1e3:9.1f}us max {s['max']/1e6:6.1f}ms >20us {s['frac_over_20us']*100:5.1f}%  stall before/after {s['stall_before_ms']:.0f}/{s['stall_after_ms']:.0f} ms")
