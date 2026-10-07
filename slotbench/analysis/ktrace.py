"""Kernel timeline analysis (strategy `ktrace`): every warp of k_ktrace stamps (%globaltimer, clock64) at eleven
checkpoints; this module places every stamp on one axis and extracts the flow of execution inside the kernel.

Clocks. clock64 is one counter per SM, exact to the cycle for every warp and block on that SM. %globaltimer is the
GPU-wide ns timer, quantised to its tick (1024 ns on A100 / RTX 3060, 64 ns on H100). Each stamp reads both, so
every stamp says: the true GPU time of cycle c lies in [g, g + tick). Per SM and per kernel, the set of such
constraints bounds the cycle-to-ns line (offset and rate) exactly as the tick-edge method bounds the host-GPU
mapping, with %globaltimer in the role of the host clock and the SM cycle counter in the role of the GPU timer:
the feasible region of all constraints is a convex set of (rate, offset) pairs, and its projection at each stamp is
that stamp's hard interval. The unit of the fit is one block (its warps share the SM's counter), with the rate
shared across the launch: every SM runs from the one GPU clock, so the launch's rate interval is the intersection
of its blocks' exact intervals (widened by 1e-3 for the clock's modulation), and an empty intersection is reported
as the smallest window widening that restores it (rate_inconsistency_ns: 0 on 5 us launches, <= 47 ns on 28-59 us
ones in the A100 / RTX 3060 datasets), never hidden.
Consecutive checkpoints are tens of cycles apart, so a warp straddling a tick edge pins that edge to within those
few cycles, and the half-width ends up far below the tick. Within an SM nothing needs the fit: differences are in
cycles. Across SMs, times come from the fitted lines and carry the two SMs' half-widths. From the GPU axis to the
host axis the run's own clock bound applies on top.

Checks that must hold if the placement is right: every warp's barrier release is at or after the last warp's
arrival in its block (cycles, exact); the grid-wide ticket order is consistent with the ticket stamps across SMs
(ticket k's atomic returned no earlier than one SM->L2->SM round trip after ticket k-1's was issued, within the
projected bounds; a violation would falsify a line); the host sees the flag after the last block wrote it. The
ticket check is the only cross-SM check and its floor (worst margin over the bounds) is reported: a cross-SM
%globaltimer offset below it is not excluded by this data.
"""
import numpy as np

from analysis.pcieclock import edge_fit

KT_NAMES = ["entry", "cal", "loaded", "bar1_arrive", "bar1_release", "computed", "bar2_arrive", "bar2_release", "stored",
            "ticket", "exit"]
KT_N = len(KT_NAMES)
# phase = (name, from checkpoint, to checkpoint, calibration); cycles are c[to] - c[from] minus the calibration:
# "cal" = one checkpoint's own cost (1 - 0); "touch" = shared load + volatile shared store + checkpoint (3 - 2),
# the chain that follows each barrier so that the release stamp waits for the barrier (deferred blocking).
KT_PHASES = [("load", 1, 2, "cal"), ("barrier1", 3, 4, "touch"), ("compute", 4, 5, "cal"), ("barrier2", 6, 7, "touch"),
             ("store_fence", 7, 8, "cal"), ("ticket", 8, 9, "cal"), ("tail", 9, 10, "cal")]


def kt_stamps(run):
    r = run.recs[(run.recs["tag"] >= 100) & (run.recs["tag"] < 100 + KT_N)]
    S = dict(kid=r["kernel_id"].astype(int), block=r["block"].astype(int), warp=r["flags"].astype(int),
             ck=(r["tag"] - 100).astype(int), sm=r["smid"].astype(int), g=r["g_begin"].astype(float),
             g1=r["g_end"].astype(float), c=r["clk_begin"].astype(float), n=r["n_iters"].astype(int))
    return S


def warp_table(S, kid):
    """(block, warp) -> dict(sm, c[KT_N], g[KT_N], n[KT_N]) for one kernel; warps with a missing checkpoint are dropped."""
    m = S["kid"] == kid
    out = {}
    for b, w, ck, sm, g, g1, c, n in zip(S["block"][m], S["warp"][m], S["ck"][m], S["sm"][m], S["g"][m], S["g1"][m], S["c"][m], S["n"][m]):
        d = out.setdefault((int(b), int(w)), dict(sm=int(sm), c=np.full(KT_N, np.nan), g=np.full(KT_N, np.nan), g1=np.full(KT_N, np.nan), n=np.zeros(KT_N, int)))
        d["c"][ck] = c; d["g"][ck] = g; d["g1"][ck] = g1; d["n"][ck] = n
    return {k: v for k, v in out.items() if not np.isnan(v["c"]).any()}


# ---------------------------------------------------------------------------------------------------------------
# The cycle -> ns line. Every stamp i gives two affine constraints on (rate r [ns per cycle], offset o [ns]):
#     g0_i <= o + r * e_i   and   o + r * e_i < g1_i + tick,  with e_i = c_i - c_ref.
# For a fixed r the feasible offsets are [LO(r), HI(r)] with LO = max_i(g0_i - r e_i) (convex in r) and
# HI = min_i(g1_i + tick - r e_i) (concave in r); width(r) = HI - LO is concave, so the feasible rate set is one
# interval found exactly by ternary search on width and bisection of its ends. The rate is searched over the
# physical range of SM clocks (0.5-2.5 GHz) with no prior around a least-squares slope: a slope fitted to a
# 1024 ns staircase with one or two edges is wrong by far more than any small window (an earlier version
# clipped the search to +-5 % of that slope and then needed a spurious "guard" to make the data fit).
R_LO, R_HI = 1.0 / 2.5, 1.0 / 0.5   # ns per cycle: 2.5 GHz .. 0.5 GHz


def _rate_interval(e, g0, g1, tick, lo=R_LO, hi=R_HI, iters=100):
    """Exact feasible rate interval [r_lo, r_hi] (None if empty) of the constraints above."""
    def LO(r): return (g0 - r * e).max()
    def HI(r): return (g1 + tick - r * e).min()
    def width(r): return HI(r) - LO(r)
    x, y = lo, hi
    for _ in range(iters):   # ternary search for the maximum of the concave width
        m1, m2 = x + (y - x) / 3, y - (y - x) / 3
        if width(m1) < width(m2): x = m1
        else: y = m2
    r_star = (x + y) / 2
    if width(r_star) < 0:
        return None
    x, y = lo, r_star                     # left end: width crosses 0 from below
    if width(x) < 0:
        for _ in range(iters):
            m = (x + y) / 2
            if width(m) >= 0: y = m
            else: x = m
        r_lo = y
    else:
        r_lo = x
    x, y = r_star, hi
    if width(y) < 0:
        for _ in range(iters):
            m = (x + y) / 2
            if width(m) >= 0: x = m
            else: y = m
        r_hi = x
    else:
        r_hi = y
    return r_lo, r_hi


def _projection(e, g0, g1, tick, r_lo, r_hi, n=193):
    """Per-stamp exact-envelope functions over rates in [r_lo, r_hi] (sampled, ends included; the envelope of
    LO(r) + r e is convex and of HI(r) + r e concave in r, so the sampling error is at most the spacing times the
    curvature; 193 samples keep it below ~2 ns here)."""
    R = np.linspace(r_lo, r_hi, n)
    LO = np.array([(g0 - r * e).max() for r in R]); HI = np.array([(g1 + tick - r * e).min() for r in R])
    def t_lo(ee): return (LO[:, None] + R[:, None] * ee[None, :]).min(0)
    def t_hi(ee): return (HI[:, None] + R[:, None] * ee[None, :]).max(0)
    return t_lo, t_hi


def window_line(c, g0, g1, tick, rate_lo=R_LO, rate_hi=R_HI):
    """One block alone: exact feasible rate interval within [rate_lo, rate_hi] ns/cycle and the projection of the
    whole feasible set at any cycle. None if the block's windows admit no line."""
    c = np.asarray(c, float); g0 = np.asarray(g0, float); g1 = np.asarray(g1, float)
    cm = c.mean(); e = c - cm
    ri = _rate_interval(e, g0, g1, tick, rate_lo, rate_hi)
    if ri is None:
        return None
    r_lo, r_hi = ri
    t_lo, t_hi = _projection(e, g0, g1, tick, r_lo, r_hi)
    return dict(t_lo=lambda cyc: t_lo(np.asarray(cyc, float) - cm), t_hi=lambda cyc: t_hi(np.asarray(cyc, float) - cm),
                t_of=lambda cyc: 0.5 * (t_lo(np.asarray(cyc, float) - cm) + t_hi(np.asarray(cyc, float) - cm)),
                r_lo=float(r_lo), r_hi=float(r_hi), ghz_lo=float(1 / r_hi), ghz_hi=float(1 / r_lo), block_ghz_lo=float(1 / r_hi), block_ghz_hi=float(1 / r_lo),
                n=int(c.size), span_ns=float(np.ptp(g0)), feasible=True)


def fit_block_lines(W, tick, rate_tol=1e-3):
    """One bounded line per block, with the SM clock shared across the launch.
    Every SM of the GPU runs from one clock, so within one launch (a few us to tens of us) all blocks share one
    rate up to the clock's modulation (parts in 1e4 over tens of us; rate_tol = 1e-3 covers it with margin).
    Step 1: each block's exact feasible rate interval alone (its windows only, physical rate range, no prior).
    Step 2: the launch's common rate interval = the intersection of the blocks' intervals, widened by rate_tol;
    if the intersection is empty, the smallest uniform widening of the windows (ns) that makes it non-empty is
    found by bisection and reported as rate_inconsistency_ns (a clock change inside the launch, or a timer
    inconsistency; 0 on the 5 us launches and <= 47 ns on the 28-59 us launches of the A100 and RTX 3060 datasets).
    Step 3: each block is projected over the common interval intersected with its own.
    Returns ({block: line or None}, info) with info = dict(ghz_lo, ghz_hi, eps_ns, n_blocks, n_infeasible)."""
    by_b = {}
    for (b, w), d in W.items():
        by_b.setdefault(b, []).append(d)
    data = {}
    for b, ds in by_b.items():
        c = np.concatenate([d["c"] for d in ds]); g0 = np.concatenate([d["g"] for d in ds]); g1 = np.concatenate([d["g1"] for d in ds])
        if c.size >= 4 and np.ptp(c) > 0:
            data[b] = (c, g0, g1)
    def intervals(eps):
        out = {}
        for b, (c, g0, g1) in data.items():
            e = c - c.mean()
            out[b] = _rate_interval(e, g0 - eps, g1 + eps, tick)
        return out
    def common(iv):
        ok = [v for v in iv.values() if v is not None]
        if len(ok) < len(iv) or not ok:
            return None
        lo = max(v[0] for v in ok); hi = min(v[1] for v in ok)
        return (lo, hi) if lo <= hi else None
    eps = 0.0
    iv = intervals(0.0); R = common(iv)
    if R is None:
        lo_e, hi_e = 0.0, 4096.0
        if common(intervals(hi_e)) is not None:
            for _ in range(24):
                m = (lo_e + hi_e) / 2
                if common(intervals(m)) is not None: hi_e = m
                else: lo_e = m
            eps = hi_e; iv = intervals(eps); R = common(iv)
    lines = {}
    if R is None:   # no common rate even with widening: fall back to each block alone (reported)
        for b, (c, g0, g1) in data.items():
            lines[b] = window_line(c, g0, g1, tick)
        info = dict(ghz_lo=None, ghz_hi=None, eps_ns=None, common_rate=False, n_blocks=len(data), n_infeasible=sum(1 for v in lines.values() if v is None))
        return lines, info
    mid = 0.5 * (R[0] + R[1]); Rw = (R[0] - rate_tol * mid, R[1] + rate_tol * mid)
    for b, (c, g0, g1) in data.items():
        cm = c.mean(); e = c - cm
        bi = iv[b]
        r_lo, r_hi = max(Rw[0], bi[0]), min(Rw[1], bi[1])
        if r_lo > r_hi:
            r_lo, r_hi = bi
        t_lo, t_hi = _projection(e, g0 - eps, g1 + eps, tick, r_lo, r_hi)
        lines[b] = dict(t_lo=lambda cyc, f=t_lo, cm=cm: f(np.asarray(cyc, float) - cm), t_hi=lambda cyc, f=t_hi, cm=cm: f(np.asarray(cyc, float) - cm),
                        t_of=lambda cyc, f=t_lo, g=t_hi, cm=cm: 0.5 * (f(np.asarray(cyc, float) - cm) + g(np.asarray(cyc, float) - cm)),
                        r_lo=float(r_lo), r_hi=float(r_hi), ghz_lo=float(1 / r_hi), ghz_hi=float(1 / r_lo), block_ghz_lo=float(1 / bi[1]), block_ghz_hi=float(1 / bi[0]),
                        n=int(c.size), span_ns=float(np.ptp(g0)), feasible=True)
    info = dict(ghz_lo=float(1 / Rw[1]), ghz_hi=float(1 / Rw[0]), ghz_lo_exact=float(1 / R[1]), ghz_hi_exact=float(1 / R[0]), eps_ns=float(eps),
                common_rate=True, rate_tol=rate_tol, n_blocks=len(data), n_infeasible=0)
    return lines, info


def kernel_timeline(run, kid, tick=None):
    """Everything about one launch: per warp the eleven checkpoints in cycles and on the GPU axis (ns, from the
    SM's fitted line) and on the host axis (if the run has a clock fit), per-phase cycles, barrier decomposition
    per block, block/kernel spans, the ticket check, the host events."""
    tick = tick if tick is not None else (run.clock.get("tick_ns") or 1024)
    S = kt_stamps(run)
    W = warp_table(S, kid)
    lines, finfo = fit_block_lines(W, tick)
    rows = []
    for (b, w), d in sorted(W.items()):
        f = lines.get(b)
        if f is None:
            continue
        cal = d["c"][1] - d["c"][0]
        touch = d["c"][3] - d["c"][2]
        calib = dict(cal=cal, touch=touch)
        t = f["t_of"](d["c"]); lo = f["t_lo"](d["c"]); hi = f["t_hi"](d["c"])
        phases = {name: float(d["c"][j] - d["c"][i] - calib[k]) for name, i, j, k in KT_PHASES}
        if w != 0:
            phases["ticket"] = None   # only warp 0 issues the atomic; the other warps branch around it (~33 cycles)
        row = dict(block=b, warp=w, sm=d["sm"], c=d["c"].copy(), g=d["g"].copy(), g1=d["g1"].copy(), t_gpu=t, t_lo=lo, t_hi=hi,
                   bound_ns=float((hi - lo).max() / 2), bounds_ns=(hi - lo) / 2, cal=cal, touch=touch,
                   ticket=int(d["n"][9]) if w == 0 else None, ffma=int(d["n"][5]), phases=phases)
        if run.host_of is not None:
            row["t_host"] = run.host_of(t); row["t_host_lo"] = run.host_of(lo); row["t_host_hi"] = run.host_of(hi)
        rows.append(row)
    # barriers per block and per barrier: who waited for whom (cycles, exact within the SM). The release stamp
    # follows a shared load + store chain; the logical check is raw (release stamp >= last arrival stamp); the
    # latency subtracts the least-contended cost of that chain in the launch (its p10). The last-arriving warp's
    # wait is 0 by construction and is excluded from wait_per_warp (kept in wait_all).
    touch_cal = float(np.percentile([r["touch"] for r in rows], 10)) if rows else 0.0
    blocks = {}
    for r in rows:
        blocks.setdefault(r["block"], []).append(r)
    bars = []
    for b, rs in sorted(blocks.items()):
        for name, ia, ir in (("barrier1", 3, 4), ("barrier2", 6, 7)):
            arr = np.array([r["c"][ia] for r in rs]); rel = np.array([r["c"][ir] for r in rs])
            last = arr.max(); il = int(arr.argmax())
            waits = (last - arr).tolist()
            bars.append(dict(block=b, barrier=name, sm=rs[0]["sm"], n_warps=len(rs), arrival_spread=float(last - arr.min()),
                             last_warp=int(rs[il]["warp"]), wait_all=waits, wait_per_warp=[x for i, x in enumerate(waits) if i != il],
                             latency_per_warp=(rel - last - touch_cal).tolist(),
                             raw_margin_min=float((rel - last).min()), release_before_last_arrival=int((rel < last).sum())))
    # ticket order across blocks. The ticket is the order in which the L2 processed the blocks' atomics; a block's
    # atomic was issued after its stamp 8 and had returned by its stamp 9, so for consecutive tickets k-1 and k:
    # issue(k-1) <= processed(k-1) < processed(k) <= return(k), i.e. t9(k) - t8(k-1) >= 0. Sharpened: the atomic
    # of ticket k cannot return less than one SM->L2->SM round trip after ticket k-1's was issued; the shortest
    # warp-0 ticket phase of the launch (minus ~30 cycles of non-atomic instructions) at the launch's highest
    # feasible clock is that round trip, so t9(k) - t8(k-1) >= rt_min. Both are evaluated on the projected
    # bounds; the worst margin is the floor below which a cross-SM timer offset is not excluded by this check.
    tk = sorted([(r["ticket"], r["t_gpu"][8], r["t_lo"][8], r["t_gpu"][9], r["t_hi"][9], r["block"], r["sm"], r["phases"]["ticket"]) for r in rows if r["warp"] == 0 and r["ticket"] is not None])
    ghz_hi = finfo.get("ghz_hi") or max((f["ghz_hi"] for f in lines.values() if f), default=2.5)
    rt_min_ns = (max(0.0, min((x[7] for x in tk), default=0.0) - 30.0)) / ghz_hi if tk else 0.0
    viol, viol_rt, slack, slack_b = 0, 0, [], []
    for (k0, t8_0, lo8_0, _, _, _, _, _), (k1, _, _, t9_1, hi9_1, _, _, _) in zip(tk, tk[1:]):
        slack.append(t9_1 - t8_0); slack_b.append(hi9_1 - lo8_0)
        if hi9_1 < lo8_0:
            viol += 1
        if hi9_1 - lo8_0 < rt_min_ns:
            viol_rt += 1
    entry = np.array([r["t_gpu"][0] for r in rows]); exit_ = np.array([r["t_gpu"][10] for r in rows])
    fl = [f for f in lines.values() if f]
    bounds_all = np.concatenate([r["bounds_ns"] for r in rows]) if rows else np.array([0.0])
    bounds_w0 = np.array([r["bound_ns"] for r in rows if r["warp"] == 0]) if rows else np.array([0.0])
    out = dict(kid=kid, n_blocks=len(blocks), n_warps=len(rows), n_sms=len(set(r["sm"] for r in rows)), tick_ns=tick,
               sm_fit=dict(n=len(fl), bound_ns_p50=float(np.median(bounds_all)), bound_ns_max=float(bounds_all.max()),
                           block_max_bound_p50=float(np.median(bounds_w0)), block_max_bound_max=float(bounds_w0.max()),
                           ghz_lo=finfo.get("ghz_lo"), ghz_hi=finfo.get("ghz_hi"), ghz_lo_exact=finfo.get("ghz_lo_exact"), ghz_hi_exact=finfo.get("ghz_hi_exact"),
                           ghz_p50=0.5 * ((finfo.get("ghz_lo") or 0) + (finfo.get("ghz_hi") or 0)) if finfo.get("common_rate") else None,
                           common_rate=finfo.get("common_rate"), rate_inconsistency_ns=finfo.get("eps_ns"),
                           infeasible=int(finfo.get("n_infeasible", 0)) + int(sum(1 for b in W if b[0] not in lines) > 0 and 0)),
               span_gpu_ns=float(exit_.max() - entry.min()) if rows else None,
               block_start_spread_ns=float(np.ptp([min(r["t_gpu"][0] for r in rs) for rs in blocks.values()])) if blocks else None,
               ticket=dict(n=len(tk), violations=viol, violations_roundtrip=viol_rt, rt_min_ns=float(rt_min_ns),
                           slack_min_ns=float(min(slack)) if slack else None, slack_p50_ns=float(np.median(slack)) if slack else None,
                           worst_margin_ns=float(min(slack_b)) if slack_b else None,
                           cross_sm_floor_ns=float(min(slack_b)) if slack_b else None),
               touch_cal_cycles=touch_cal, phase_bias_cycles=3.0, barriers=bars, rows=rows)
    # host side (flag: the last block writes its %globaltimer into the mapped word; the host records when it saw it)
    ev = run.ev
    def one(t, k):
        m = (ev["type"] == t) & (ev["kernel_id"] == kid)
        return int(ev["t"][m][0]) if m.any() else None
    h = dict(launch_enter=one(1, kid), launch_return=one(2, kid), flag_seen=one(6, kid), sync_enter=one(3, kid), sync_return=one(4, kid))
    m = (ev["type"] == 6) & (ev["kernel_id"] == kid)
    h["flag_value_gpu_ns"] = int(ev["a"][m][0]) if m.any() else None
    out["host"] = h
    if run.host_of is not None and rows:
        hb = run.clock["bound_ns"]
        first_entry = min(r["t_host"][0] for r in rows); last_exit = max(r["t_host"][10] for r in rows)
        warm = [r for r in rows if r["warp"] == 0 and r["ticket"] is not None and r["ticket"] == max(x["ticket"] for x in rows if x["warp"] == 0 and x["ticket"] is not None)]
        hv = dict(launch_to_first_entry_ns=first_entry - h["launch_enter"] if h["launch_enter"] else None,
                  call_ns=(h["launch_return"] - h["launch_enter"]) if h["launch_return"] else None,
                  last_exit_to_sync_return_ns=(h["sync_return"] - last_exit) if h["sync_return"] else None, bound_ns=hb)
        if warm and h.get("flag_seen") and h.get("flag_value_gpu_ns"):
            # the flag is written by warp 0 of the last-ticket block after its stamp 9 (third barrier + shared load,
            # ~60-100 cycles) and before its stamp 10 (which follows the system-scope fence); the written value is
            # the %globaltimer truncated to the tick, so the write lies in [value, value + tick) as well
            r = warm[0]; fv = h["flag_value_gpu_ns"] - run.g_ref
            w_lo = max(float(r["t_lo"][9]), float(fv)); w_hi = min(float(r["t_hi"][10]), float(fv + tick))
            if w_lo <= w_hi:
                hv["flag_write_gpu_window_ns"] = [w_lo, w_hi]
                hv["flag_write_to_flag_seen_ns"] = float(h["flag_seen"] - run.host_of(0.5 * (w_lo + w_hi)))
                hv["flag_write_to_flag_seen_min_ns"] = float(h["flag_seen"] - run.host_of(w_hi))
                hv["flag_write_to_flag_seen_max_ns"] = float(h["flag_seen"] - run.host_of(w_lo))
                hv["flag_seen_after_write"] = bool(h["flag_seen"] + hb >= float(run.host_of(w_lo)))
        out["host_view"] = hv
    return out


def a_ktrace(run):
    """Per launch: the timeline (summaries only in the JSON); per block count: phase cycles over all warps (ticket:
    warp 0 only), barrier 1 and 2 separately (wait excludes the last-arriving warp), placement bounds, checks, and
    the host-side latencies over the warm launches (the first launch of a run is cold and reported apart)."""
    from analysis.gputrace import q
    S = kt_stamps(run)
    kids = sorted(set(S["kid"].tolist()))
    enter = run.events("LAUNCH_ENTER")
    B_of = {int(k): int(a) for k, a in zip(enter["kernel_id"], enter["a"])}
    per_B, kernels = {}, []
    for kid in kids:
        T = kernel_timeline(run, kid)
        if not T["rows"]:
            continue
        B = B_of.get(kid, T["n_blocks"])
        acc = per_B.setdefault(B, dict(phases={n: [] for n, _, _, _ in KT_PHASES}, cal=[], touch=[], b1_wait=[], b1_lat=[], b1_spread=[],
                                       b2_wait=[], b2_lat=[], b2_spread=[], span=[], start_spread=[], launch_to_first=[], launch_to_first_cold=[],
                                       flag=[], flag_lo=[], flag_hi=[], exit_to_sync=[], ghz_lo=[], ghz_hi=[], eps=[], bound_p50=[], bound_max=[],
                                       ticket_viol=0, ticket_viol_rt=0, ticket_n=0, floor=[], bar_release_early=0, kernels=0))
        for r in T["rows"]:
            for n in acc["phases"]:
                if r["phases"][n] is not None:
                    acc["phases"][n].append(r["phases"][n])
            acc["cal"].append(r["cal"]); acc["touch"].append(r["touch"])
        for b in T["barriers"]:
            k = "b1" if b["barrier"] == "barrier1" else "b2"
            acc[k + "_wait"] += b["wait_per_warp"]; acc[k + "_lat"] += b["latency_per_warp"]; acc[k + "_spread"].append(b["arrival_spread"])
            acc["bar_release_early"] += b["release_before_last_arrival"]
        acc["span"].append(T["span_gpu_ns"]); acc["start_spread"].append(T["block_start_spread_ns"])
        sf = T["sm_fit"]
        if sf.get("ghz_lo"): acc["ghz_lo"].append(sf["ghz_lo"]); acc["ghz_hi"].append(sf["ghz_hi"])
        if sf.get("rate_inconsistency_ns") is not None: acc["eps"].append(sf["rate_inconsistency_ns"])
        acc["bound_p50"].append(sf["block_max_bound_p50"]); acc["bound_max"].append(sf["block_max_bound_max"])
        acc["ticket_viol"] += T["ticket"]["violations"]; acc["ticket_viol_rt"] += T["ticket"]["violations_roundtrip"]; acc["ticket_n"] += T["ticket"]["n"]
        if T["ticket"]["cross_sm_floor_ns"] is not None: acc["floor"].append(T["ticket"]["cross_sm_floor_ns"])
        acc["kernels"] += 1
        hv = T.get("host_view")
        cold = kid == kids[0]
        if hv:
            if hv.get("launch_to_first_entry_ns") is not None:
                (acc["launch_to_first_cold"] if cold else acc["launch_to_first"]).append(hv["launch_to_first_entry_ns"])
            if hv.get("flag_write_to_flag_seen_ns") is not None:
                acc["flag"].append(hv["flag_write_to_flag_seen_ns"]); acc["flag_lo"].append(hv["flag_write_to_flag_seen_min_ns"]); acc["flag_hi"].append(hv["flag_write_to_flag_seen_max_ns"])
            if hv.get("last_exit_to_sync_return_ns") is not None and not cold:
                acc["exit_to_sync"].append(hv["last_exit_to_sync_return_ns"])
        kernels.append(dict(kid=kid, blocks=B, cold=cold, span_gpu_ns=T["span_gpu_ns"], block_start_spread_ns=T["block_start_spread_ns"],
                            sm_fit=T["sm_fit"], ticket=T["ticket"], host_view=hv))
    out = dict(ffma=run.meta.get("ffma"), tick_ns=run.clock.get("tick_ns"), bound_ns=run.clock.get("bound_ns"), by_blocks={}, kernels=kernels,
               notes=["phase cycles subtract the entry checkpoint cost (c1 - c0, 14-16 cy), which overstates the in-phase checkpoint cost by ~3 cycles: phases are ~3 cy short",
                      "load = S2R x2 + address arithmetic + a 4 B .nc load + a dependent 16 B .nc load + the consuming shared store (L2-resident input)",
                      "compute = 256 dependent FFMA in a 16x-unrolled loop: ~16 taken branches of loop control are included (~200 cycles at 1 block/SM)",
                      "ticket = warp 0 only (the atomic with return); the other 7 warps branch around it in ~33 cycles",
                      "host-axis latencies carry the run's host<->GPU bound (bound_ns) in addition to the block bound"])
    for B, acc in sorted(per_B.items()):
        out["by_blocks"][B] = dict(kernels=acc["kernels"], checkpoint_cost_cycles=q(acc["cal"]), touch_chain_cycles=q(acc["touch"]),
                                   phases_cycles={n: q(v) for n, v in acc["phases"].items()},
                                   barrier1=dict(wait_cycles=q(acc["b1_wait"]), release_cycles=q(acc["b1_lat"]), arrival_spread_cycles=q(acc["b1_spread"])),
                                   barrier2=dict(wait_cycles=q(acc["b2_wait"]), release_cycles=q(acc["b2_lat"]), arrival_spread_cycles=q(acc["b2_spread"])),
                                   barrier_release_before_last_arrival=acc["bar_release_early"],
                                   kernel_span_ns=q(acc["span"]), block_start_spread_ns=q(acc["start_spread"]),
                                   sm_ghz_lo=q(acc["ghz_lo"]), sm_ghz_hi=q(acc["ghz_hi"]), rate_inconsistency_ns=q(acc["eps"]),
                                   block_bound_ns=dict(p50_over_launches=q(acc["bound_p50"]), max_over_launches=q(acc["bound_max"])),
                                   ticket=dict(n=acc["ticket_n"], violations=acc["ticket_viol"], violations_roundtrip=acc["ticket_viol_rt"], cross_sm_floor_ns=q(acc["floor"])),
                                   launch_to_first_entry_ns=q(acc["launch_to_first"]), launch_to_first_entry_cold_ns=q(acc["launch_to_first_cold"]),
                                   flag_write_to_flag_seen_ns=q(acc["flag"]), flag_write_to_flag_seen_min_ns=q(acc["flag_lo"]), flag_write_to_flag_seen_max_ns=q(acc["flag_hi"]),
                                   last_exit_to_sync_return_ns=q(acc["exit_to_sync"]))
    return out


def one_line_ktrace(res, bs):
    parts = []
    for B, d in res.get("by_blocks", {}).items():
        ph = d["phases_cycles"]; b1 = d["barrier1"]; bb = d["block_bound_ns"]
        g = lambda dd, k="p50": (dd.get(k) if dd and dd.get(k) is not None else float("nan"))
        parts.append(f"B={B}: load {g(ph['load']):.0f} | bar1 wait {g(b1['wait_cycles']):.0f}/{g(b1['wait_cycles'], 'p99'):.0f} rel {g(b1['release_cycles']):.0f} "
                     f"| compute {g(ph['compute']):.0f} | store+fence {g(ph['store_fence']):.0f} | atomic {g(ph['ticket']):.0f} cy; span {g(d['kernel_span_ns']) / 1e3:.1f} us, "
                     f"block bound {g(bb['p50_over_launches']):.0f}/{g(bb['max_over_launches'], 'p100'):.0f} ns p50/max, SM {g(d['sm_ghz_lo']):.3f}-{g(d['sm_ghz_hi']):.3f} GHz, "
                     f"rate incons {g(d['rate_inconsistency_ns'], 'p100'):.0f} ns, ticket {d['ticket']['violations']}+{d['ticket']['violations_roundtrip']}/{d['ticket']['n']} viol (floor {g(d['ticket']['cross_sm_floor_ns'], 'p100'):.0f} ns), "
                     f"launch->first {g(d['launch_to_first_entry_ns']) / 1e3:.1f} us warm, flag->seen {g(d['flag_write_to_flag_seen_ns']) / 1e3:.2f} us")
    return f"ktrace {bs}: " + " || ".join(parts)
