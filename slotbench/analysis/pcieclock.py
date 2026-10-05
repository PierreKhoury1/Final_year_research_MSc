#!/usr/bin/env python3
"""Analyse tools/pcieclock output: how tightly is %globaltimer pinned to the host clock?

classic  brackets (t0, t1, g): host(g) lies in [t0, t1 + tick] (g is a truncated tick count).
         Reported as slotbench's clock_fit does it: regress midpoints of the tightest 1 %, error bound
         eps = half the kept median width (+ any bracket the fit misses) - plus the tick it ignores.
edge     up (h, E): host(E) <= h.  down (t, E): host(E) >= t.  For host = a*E + b (constant rate over the
         window) the feasible offsets at the best rate form [max_i(t_i - aE_i), min_j(h_j - aE_j)]; half its
         width is a hard error bound with no symmetry assumption. Negative width = constraints violated
         (clock relation not linear over the window, or a broken assumption) and is reported as such.
Usage: pcieclock.py PREFIX [--windows N] [--json OUT]
"""
import argparse
import json
import struct
import sys


def load(path, ncol):
    raw = open(path, "rb").read()
    s = struct.Struct("<" + "q" * ncol)
    return [s.unpack_from(raw, i) for i in range(0, len(raw) - len(raw) % s.size, s.size)]


def tick_of(values):
    vs = sorted(set(values))
    diffs = sorted(b - a for a, b in zip(vs, vs[1:]) if b > a)
    return diffs[0] if diffs else 0


def classic_fit(br, tick, keep_frac=0.01, min_keep=50):
    br = [b for b in br if b[1] >= b[0] and b[2]]
    widths = sorted(b[1] - b[0] for b in br)
    keep = min(len(br), max(min_keep, int(keep_frac * len(br) + 0.999)))
    thr = widths[keep - 1]
    kept = [b for b in br if b[1] - b[0] <= thr]
    g0, t0 = kept[0][2], kept[0][0]
    X = [b[2] - g0 for b in kept]
    Y = [0.5 * (b[0] - t0) + 0.5 * (b[1] - t0) for b in kept]
    mx, my = sum(X) / len(X), sum(Y) / len(Y)
    sxx = sum((x - mx) ** 2 for x in X)
    a = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sxx if sxx else 1.0
    b = my - a * mx
    viol = max(max((bb[0] - t0) - (b + a * x), (b + a * x) - (bb[1] - t0)) for bb, x in zip(kept, X))
    kw = sorted(bb[1] - bb[0] for bb in kept)
    eps = kw[len(kw) // 2] / 2 + max(viol, 0.0)
    return dict(n=len(br), n_kept=len(kept), min_width_ns=widths[0], median_width_ns=widths[len(widths) // 2],
                eps_ns=eps, eps_with_tick_ns=eps + tick / 2, rate_ppm=(a - 1) * 1e6,
                model=(g0, t0, a, b))


def edge_fit(up, down, iters=200, trim=0):
    """trim: ignore the `trim` most extreme constraints on each side (reported separately from the strict fit)."""
    E0 = min(e for _, e in up + down)
    H0 = min(h for h, _ in up)
    U = [(h - H0, e - E0) for h, e in up]
    D = [(t - H0, e - E0) for t, e in down]

    def bounds(a):
        his = sorted(h - a * e for h, e in U)
        los = sorted((t - a * e for t, e in D), reverse=True)
        return los[trim], his[trim]

    def width(a):
        lo, hi = bounds(a)
        return hi - lo

    lo_a, hi_a = 1 - 500e-6, 1 + 500e-6   # width(a) is concave (min of lines minus max of lines): ternary max
    for _ in range(iters):
        m1, m2 = lo_a + (hi_a - lo_a) / 3, hi_a - (hi_a - lo_a) / 3
        if width(m1) < width(m2):
            lo_a = m1
        else:
            hi_a = m2
    a = (lo_a + hi_a) / 2
    lo, hi = bounds(a)
    mid = (lo + hi) / 2
    viol_up = sum(1 for h, e in U if h - (a * e + mid) < -(hi - lo) / 2 - 1e-6)
    viol_down = sum(1 for t, e in D if (a * e + mid) - t < -(hi - lo) / 2 - 1e-6)
    return dict(n_up=len(U), n_down=len(D), trim=trim, rate_ppm=(a - 1) * 1e6, width_ns=hi - lo, bound_ns=(hi - lo) / 2,
                feasible=hi >= lo, violations_up=viol_up, violations_down=viol_down, model=(E0, H0, a, mid),
                up_slack_min_ns=min(h - (a * e + (lo + hi) / 2) for h, e in U),
                down_slack_min_ns=min((a * e + (lo + hi) / 2) - t for t, e in D))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--json")
    a = ap.parse_args()
    meta = json.load(open(a.prefix + ".json"))
    classic = load(a.prefix + ".classic.bin", 3)
    up = load(a.prefix + ".up.bin", 2)
    down = load(a.prefix + ".down.bin", 2)
    tick = tick_of([g for _, _, g in classic])
    steps = meta.get("timer_edge_steps_ns")
    if steps:
        tick = min(steps)
    res = dict(meta=meta, tick_ns=tick, classic=classic_fit(classic, tick), edge=edge_fit(up, down),
               edge_trim2=edge_fit(up, down, trim=2), windows=[])
    # stability: fit each time window separately (drift is not exactly linear over long spans)
    k = a.windows
    for w in range(k):
        cu = up[w * len(up) // k:(w + 1) * len(up) // k]
        cd = down[w * len(down) // k:(w + 1) * len(down) // k]
        cc = classic[w * len(classic) // k:(w + 1) * len(classic) // k]
        if len(cu) > 10 and len(cd) > 10 and len(cc) > 60:
            e, c = edge_fit(cu, cd), classic_fit(cc, tick)
            # offset disagreement between the two methods at the window's middle GPU time
            E0, H0, ae, be = e["model"]
            g0, t0, ac, bc = c["model"]
            mid = sorted(x for _, x in cu)[len(cu) // 2]
            host_edge = H0 + be + ae * (mid - E0)
            host_classic = t0 + bc + ac * (mid - g0)
            res["windows"].append(dict(edge_bound_ns=e["bound_ns"], edge_feasible=e["feasible"],
                                       classic_eps_ns=c["eps_ns"], classic_minus_edge_ns=host_classic - host_edge))
    for d in (res["classic"], res["edge"], res["edge_trim2"]):
        d.pop("model", None)
    print(json.dumps(res, indent=2))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())
