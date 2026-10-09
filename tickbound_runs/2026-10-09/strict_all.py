"""Strict (envelope) vs reported (chord) clock bound for every run with pre/post sync files under a data root.
Strict = half-extent of the projection of the whole feasible (rate, offset) set, measured from the fit's estimate,
max over the pre->post gap (the half-extent is convex in time, so its max over the gap is at one of the gap's ends).
Usage: strict_all.py SRC_DIR DATA_ROOT OUT_JSON"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from tickbound.analysis.clockfit import load, edge_fit   # noqa: E402


def strict(pre, post):
    up = load(pre + ".up.bin", 2) + load(post + ".up.bin", 2)
    down = load(pre + ".down.bin", 2) + load(post + ".down.bin", 2)
    fit = edge_fit(up, down)
    if not fit["feasible"]:
        return None
    E0, H0, a_fit, mid = fit["model"]
    E0 = min(e for _, e in up + down); H0 = min(h for h, _ in up)
    U = np.array([(h - H0, e - E0) for h, e in up], dtype=float); D = np.array([(t - H0, e - E0) for t, e in down], dtype=float)
    hu, eu, td, ed = U[:, 0], U[:, 1], D[:, 0], D[:, 1]
    HI = lambda a: (hu - a * eu).min(); LO = lambda a: (td - a * ed).max(); W = lambda a: HI(a) - LO(a)
    def edge(lo, hi, inside_is_hi):
        for _ in range(200):
            m = 0.5 * (lo + hi); ok = W(m) >= 0
            if inside_is_hi: lo, hi = (lo, m) if ok else (m, hi)
            else: lo, hi = (m, hi) if ok else (lo, m)
        return hi if inside_is_hi else lo
    a_lo = edge(1 - 500e-6, a_fit, True) if W(1 - 500e-6) < 0 else 1 - 500e-6
    a_hi = edge(a_fit, 1 + 500e-6, False) if W(1 + 500e-6) < 0 else 1 + 500e-6
    B = W(a_fit) / 2
    # the fit's own estimate line, on the same (H0, E0) origin: est(e) = a_fit * e + c, c chosen so that est sits mid-gap at e
    c = 0.5 * (HI(a_fit) + LO(a_fit))
    def tern(f, lo, hi, maximize):
        for _ in range(200):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if (f(m1) < f(m2)) == maximize: lo = m1
            else: hi = m2
        return f(0.5 * (lo + hi))
    pre_e = [e for _, e in load(pre + ".up.bin", 2)]; post_e = [e for _, e in load(post + ".up.bin", 2)]
    out = []
    for e in (float(max(pre_e) - E0), float(min(post_e) - E0)):
        est = a_fit * e + c
        vmax = tern(lambda a: a * e + HI(a), a_lo, a_hi, True); vmin = tern(lambda a: a * e + LO(a), a_lo, a_hi, False)
        out.append(max(vmax - est, est - vmin))
    return dict(chord_ns=B, strict_ns=max(out), ratio=max(out) / B if B else None, gap_s=(min(post_e) - max(pre_e)) / 1e9)


res = {}
for pre_up in sorted(glob.glob(os.path.join(sys.argv[2], "**", "*.pre.up.bin"), recursive=True)):
    prefix = pre_up[:-len(".pre.up.bin")]
    rel = os.path.relpath(prefix, sys.argv[2]).replace(os.sep, "/")
    try:
        r = strict(prefix + ".pre", prefix + ".post")
    except Exception as ex:
        r = dict(error=str(ex)[:120])
    res[rel] = r
    if r and "chord_ns" in r:
        print(f"{rel:62s} chord {r['chord_ns']:7.1f}  strict {r['strict_ns']:7.1f}  x{r['ratio']:.2f}")
    else:
        print(f"{rel:62s} {r}")
json.dump(res, open(sys.argv[3], "w"), indent=1)
