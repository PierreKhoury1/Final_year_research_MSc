"""Audit (multi): recompute every number of the 9 Oct 8-GPU timeline claims from the raw traces, with the exact code
that analysed them (box_9oct/stage/tb/src). Read-only on the data; writes only to this audit folder.

Per GPU: chord bound (run.clock), strict bound (projection of the whole feasible (rate, offset) set, max over the
pre->post gap, as strict_all.py), and the projection half-extent AT each launch's first entry.
Per 108-block launch: launch-enter host time, first warp-0 entry (as multi_fig.py), first entry over all warps,
the stamp bound used by multi_fig (sm_fit.bound_ns_max) and the exact interval of the first entry.
Per grid instant: launch spread, start spread, proved pairs under several rules.
Usage: python recompute_multi.py SRC MULTI_DIR OUT_JSON"""
import json, os, sys, time
import numpy as np

SRC, MDIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, SRC)
from tickbound.analysis.core import Run                    # noqa: E402
from tickbound.analysis.ktrace import kernel_timeline     # noqa: E402
from tickbound.analysis.clockfit import edge_fit          # noqa: E402

GRID = 2_000_000


def tern(f, lo, hi, maximize, it=200):
    for _ in range(it):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if (f(m1) < f(m2)) == maximize:
            lo = m1
        else:
            hi = m2
    return f(0.5 * (lo + hi))


class Proj:
    """Projection of the feasible (rate, offset) set of the run's pre+post edge constraints (rebased GPU axis)."""
    def __init__(self, run):
        _, u_pre, d_pre = run._samples("pre")
        _, u_post, d_post = run._samples("post")
        up, down = u_pre + u_post, d_pre + d_post
        fit = edge_fit(up, down)
        assert fit["feasible"]
        self.E0, self.H0, self.a_fit, self.mid = fit["model"]
        U = np.array([(h - self.H0, e - self.E0) for h, e in up], float)
        D = np.array([(t - self.H0, e - self.E0) for t, e in down], float)
        hu, eu, td, ed = U[:, 0], U[:, 1], D[:, 0], D[:, 1]
        self.HI = lambda a: (hu - a * eu).min()
        self.LO = lambda a: (td - a * ed).max()
        W = lambda a: self.HI(a) - self.LO(a)
        def edge(lo, hi, inside_is_hi):
            for _ in range(200):
                m = 0.5 * (lo + hi); ok = W(m) >= 0
                if inside_is_hi: lo, hi = (lo, m) if ok else (m, hi)
                else: lo, hi = (m, hi) if ok else (lo, m)
            return hi if inside_is_hi else lo
        a_fit = self.a_fit
        self.a_lo = edge(1 - 500e-6, a_fit, True) if W(1 - 500e-6) < 0 else 1 - 500e-6
        self.a_hi = edge(a_fit, 1 + 500e-6, False) if W(1 + 500e-6) < 0 else 1 + 500e-6
        self.chord = W(a_fit) / 2
        self.c = 0.5 * (self.HI(a_fit) + self.LO(a_fit))
        self.pre_end = max(e for _, e in u_pre) - self.E0
        self.post_start = min(e for _, e in u_post) - self.E0
        self.pre_first = min(e for _, e in u_pre) - self.E0
        self.post_last = max(e for _, e in u_post) - self.E0
        # per-window edge fits (rate inside each window alone)
        self.w_pre = edge_fit(u_pre, d_pre); self.w_post = edge_fit(u_post, d_post)

    def at(self, g_rel):
        """(est, vmin, vmax) host times (absolute ns) of GPU time g (run's rebased axis)."""
        e = float(g_rel) - self.E0
        est = self.a_fit * e + self.c
        vmax = tern(lambda a: a * e + self.HI(a), self.a_lo, self.a_hi, True)
        vmin = tern(lambda a: a * e + self.LO(a), self.a_lo, self.a_hi, False)
        return est + self.H0, vmin + self.H0, vmax + self.H0

    def half(self, g_rel):
        est, lo, hi = self.at(g_rel)
        return max(hi - est, est - lo)

    def strict(self):
        return max(self.half(self.pre_end + self.E0), self.half(self.post_start + self.E0))


t0 = time.time()
res = dict(gpus={}, launches={})
runs = {}
for g in range(8):
    p = os.path.join(MDIR, f"gpu{g}")
    r = Run(p)
    runs[g] = r
    P = Proj(r)
    m = r.ev["type"] == 1
    le = {int(k): (int(t), int(a)) for t, k, a in zip(r.ev["t"][m], r.ev["kernel_id"][m], r.ev["a"][m])}
    info = dict(chord_ns=float(r.clock["bound_ns"]), chord_proj=float(P.chord), strict_ns=float(P.strict()),
                rate_ppm=float(r.clock["rate_ppm"]), a_lo_ppm=(P.a_lo - 1) * 1e6, a_hi_ppm=(P.a_hi - 1) * 1e6,
                gap_s=(P.post_start - P.pre_end) / 1e9, pre_window_s=(P.pre_end - P.pre_first) / 1e9,
                post_window_s=(P.post_last - P.post_start) / 1e9,
                half_at_pre_end=float(P.half(P.pre_end + P.E0)), half_at_post_start=float(P.half(P.post_start + P.E0)),
                w_pre=dict(bound=P.w_pre["bound_ns"], rate_ppm=P.w_pre["rate_ppm"], feasible=P.w_pre["feasible"]),
                w_post=dict(bound=P.w_post["bound_ns"], rate_ppm=P.w_post["rate_ppm"], feasible=P.w_post["feasible"]),
                core=r.meta.get("core"), clock_core=r.meta.get("clock_core"))
    res["gpus"][g] = info
    L = []
    for kid, (t_le, B) in sorted(le.items()):
        if B != 108:
            continue
        tl = kernel_timeline(r, kid)
        ix = {n: i for i, n in enumerate(tl["names"])}
        E = ix["entry"]
        rows = [x for x in tl["rows"] if "t_host" in x]
        w0 = [x for x in rows if x["warp"] == 0]
        f0 = min(w0, key=lambda x: x["t_host"][E])
        fa = min(rows, key=lambda x: x["t_host"][E])
        # exact interval of the first warp-0 entry: true GPU time in [min t_lo, min t_hi]; host = feasible line of it
        glo0 = min(float(x["t_lo"][E]) for x in w0); ghi0 = min(float(x["t_hi"][E]) for x in w0)
        glo_a = min(float(x["t_lo"][E]) for x in rows); ghi_a = min(float(x["t_hi"][E]) for x in rows)
        est0, vmin_lo0, _ = P.at(glo0); _, _, vmax_hi0 = P.at(ghi0)
        _, vmin_loa, _ = P.at(glo_a); _, _, vmax_hia = P.at(ghi_a)
        g_first0 = float(f0["t_gpu"][E])
        hinst = P.half(g_first0)
        exit_last = max(float(x["t_host"][ix["exit"]]) for x in w0)
        L.append(dict(kid=kid, slot=int(round(t_le / GRID)), le=t_le, le_mod_grid=t_le % GRID,
                      first0=float(f0["t_host"][E]), first_all=float(fa["t_host"][E]),
                      first0_block=int(f0["block"]), first_all_block=int(fa["block"]), first_all_warp=int(fa["warp"]),
                      stamp_max=float(tl["sm_fit"]["bound_ns_max"]),
                      stamp_entry_w0first=float(f0["bounds_ns"][E]),
                      stamp_entry_max_w0=float(max(x["bounds_ns"][E] for x in w0)),
                      stamp_entry_max_all=float(max(x["bounds_ns"][E] for x in rows)),
                      exact0=[float(vmin_lo0), float(vmax_hi0)], exact_all=[float(vmin_loa), float(vmax_hia)],
                      clock_half_at_instant=float(hinst), g_first0=g_first0,
                      t_from_post_start_ms=(P.post_start + P.E0 - g_first0) / 1e6,
                      rate_incons=float(tl["sm_fit"].get("rate_inconsistency_ns") or 0.0),
                      ticket_viol=int(tl["ticket"]["violations"]), ticket_viol_rt=int(tl["ticket"]["violations_roundtrip"]),
                      exit_last0=exit_last, n_warps=len(rows), n_w0=len(w0)))
    res["launches"][g] = L
    print(f"GPU {g}: chord {info['chord_ns']:.1f} strict {info['strict_ns']:.1f} (pre end {info['half_at_pre_end']:.1f}, post start "
          f"{info['half_at_post_start']:.1f}) gap {info['gap_s']:.2f} s; rate {info['rate_ppm']:.3f} ppm [{info['a_lo_ppm']:.4f}, {info['a_hi_ppm']:.4f}]; "
          f"{len(L)} launches; at-instant clock half {min(x['clock_half_at_instant'] for x in L):.1f}-{max(x['clock_half_at_instant'] for x in L):.1f}; "
          f"{time.time() - t0:.0f} s", flush=True)

json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
