"""Kernel timeline analysis (strategy `ktrace`): every warp of k_ktrace stamps (%globaltimer, clock64) at eleven
checkpoints; this module places every stamp on one axis and extracts the flow of execution inside the kernel.

Clocks. clock64 is one counter per SM, exact to the cycle for every warp and block on that SM. %globaltimer is the
GPU-wide ns timer, quantised to its tick (1024 ns on A100 / RTX 3060, 64 ns on H100). Each stamp reads both, so
every stamp says: the true GPU time of cycle c lies in [g, g + tick). Per SM and per kernel, the set of such
constraints bounds the cycle-to-ns line (offset and rate) exactly as the tick-edge method bounds the host-GPU
mapping, with %globaltimer in the role of the host clock and the SM cycle counter in the role of the GPU timer:
the feasible region of all constraints is a convex set of (rate, offset) pairs, and its projection at each stamp is
that stamp's hard interval. The unit of the fit is one block (its warps share the SM's counter and its lifetime is
short): pooling blocks that ran at different times on one SM is not valid, because the SM clock is modulated by a
few parts in 1e4 over tens of us (spread-spectrum), seen as ~20 ns inconsistencies between the blocks of a 28 us
kernel.
Consecutive checkpoints are tens of cycles apart, so a warp straddling a tick edge pins that edge to within those
few cycles, and the half-width ends up far below the tick. Within an SM nothing needs the fit: differences are in
cycles. Across SMs, times come from the fitted lines and carry the two SMs' half-widths. From the GPU axis to the
host axis the run's own clock bound applies on top.

Checks that must hold if the placement is right: every warp's barrier release is at or after the last warp's
arrival in its block (cycles, exact); the grid-wide ticket order equals the time order of the ticket stamps across
SMs (fitted lines; a violation beyond the two half-widths would falsify a line); the host sees the flag after the
last block's exit.
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


def window_line(c, g0, g1, tick, rate_range=0.05, n_rate=241):
    """The set of lines t = off + rate * c consistent with every window g0 <= t(c) < g1 + tick, and the projection
    of that set onto each stamp: t_lo(c) <= t(c) <= t_hi(c) over all feasible (rate, offset) pairs. The rate is
    pre-scaled by a least-squares slope, searched over +-rate_range, and the offset interval at each rate is
    [max(g0 - rate*e), min(g1 + tick - rate*e)] (an exact 2-D linear-programming region, sampled in rate).
    Returns None if no line fits (the stamps of this unit do not share one counter / one clock)."""
    c = np.asarray(c, float); g0 = np.asarray(g0, float); g1 = np.asarray(g1, float)
    cm, gm = c.mean(), g0.mean()
    b0 = ((c - cm) * (g0 - gm)).sum() / max(((c - cm) ** 2).sum(), 1e-9)
    if not (0.05 < 1 / b0 < 20):
        return None
    e = b0 * (c - cm)                      # ns at the nominal rate, centred
    rates = 1 + np.linspace(-rate_range, rate_range, n_rate)
    lo = np.array([(g0 - a * e).max() for a in rates]); hi = np.array([(g1 + tick - a * e).min() for a in rates])
    ok = lo <= hi
    if not ok.any():
        return None
    # refine the feasible rate interval's ends by bisection (width(a) = hi - lo is concave)
    def width(a): return (g1 + tick - a * e).min() - (g0 - a * e).max()
    i0, i1 = int(np.argmax(ok)), int(len(ok) - 1 - np.argmax(ok[::-1]))
    a_lo, a_hi = rates[i0], rates[i1]
    if i0 > 0:
        x, y = rates[i0 - 1], rates[i0]
        for _ in range(40):
            m = (x + y) / 2
            if width(m) >= 0: y = m
            else: x = m
        a_lo = y
    if i1 < len(rates) - 1:
        x, y = rates[i1], rates[i1 + 1]
        for _ in range(40):
            m = (x + y) / 2
            if width(m) >= 0: x = m
            else: y = m
        a_hi = x
    A = np.linspace(a_lo, a_hi, 97)
    LO = np.array([(g0 - a * e).max() for a in A]); HI = np.array([(g1 + tick - a * e).min() for a in A])
    def t_lo(cyc):
        ee = b0 * (np.asarray(cyc, float) - cm)
        return (LO[:, None] + A[:, None] * ee[None, :]).min(0)
    def t_hi(cyc):
        ee = b0 * (np.asarray(cyc, float) - cm)
        return (HI[:, None] + A[:, None] * ee[None, :]).max(0)
    mid_a = (a_lo + a_hi) / 2
    return dict(t_lo=t_lo, t_hi=t_hi, t_of=lambda cyc: 0.5 * (t_lo(cyc) + t_hi(cyc)),
                ghz_lo=float(1 / (a_hi * b0)), ghz_hi=float(1 / (a_lo * b0)), ghz=float(1 / (mid_a * b0)),
                n=int(c.size), span_ns=float(np.ptp(g0)), feasible=True)


def fit_block_lines(W, tick, guard_ns=None):
    """One bounded line per block: a block's warps share the SM's cycle counter (checked: barrier releases never
    precede the last arrival, margins within ~10 cycles) and the block's lifetime is short enough for one rate.
    Pooling blocks that ran at different times on the same SM is not valid: the SM clock moves by a few parts in
    1e4 over tens of us and between launches (DVFS), seen as ~20 ns inconsistencies between the blocks of a 28 us
    kernel on the RTX 3060.
    guard_ns: the %globaltimer readings of different warps on one SM disagree by up to tens of ns (RTX 3060: up to
    ~80 ns, with no tick edge involved), so every window is widened by a guard. None: the smallest guard (ns) that
    makes every block's windows consistent is found by bisection and applied to all blocks, so the bounds carry it.
    Returns ({block: line or None}, guard_ns)."""
    by_b = {}
    for (b, w), d in W.items():
        by_b.setdefault(b, []).append(d)
    data = {}
    for b, ds in by_b.items():
        c = np.concatenate([d["c"] for d in ds]); g0 = np.concatenate([d["g"] for d in ds]); g1 = np.concatenate([d["g1"] for d in ds])
        if c.size >= 4 and np.ptp(c) > 0:
            data[b] = (c, g0, g1)
    def fit_all(eps):
        return {b: window_line(c, g0 - eps, g1 + eps, tick) for b, (c, g0, g1) in data.items()}
    if guard_ns is None:
        lines = fit_all(0.0)
        if all(v is not None for v in lines.values()):
            guard_ns = 0.0
        else:
            lo, hi = 0.0, 4096.0
            if all(v is not None for v in fit_all(hi).values()):
                for _ in range(24):
                    m = (lo + hi) / 2
                    if all(v is not None for v in fit_all(m).values()): hi = m
                    else: lo = m
                guard_ns = hi
            else:
                guard_ns = hi
            lines = fit_all(guard_ns)
    else:
        lines = fit_all(float(guard_ns))
    return lines, float(guard_ns)


def kernel_timeline(run, kid, tick=None):
    """Everything about one launch: per warp the eleven checkpoints in cycles and on the GPU axis (ns, from the
    SM's fitted line) and on the host axis (if the run has a clock fit), per-phase cycles, barrier decomposition
    per block, block/kernel spans, the ticket check, the host events."""
    tick = tick if tick is not None else (run.clock.get("tick_ns") or 1024)
    S = kt_stamps(run)
    W = warp_table(S, kid)
    lines, guard_ns = fit_block_lines(W, tick)
    rows = []
    for (b, w), d in sorted(W.items()):
        f = lines.get(b)
        if f is None:
            continue
        cal = d["c"][1] - d["c"][0]
        touch = d["c"][3] - d["c"][2]
        calib = dict(cal=cal, touch=touch)
        t = f["t_of"](d["c"]); lo = f["t_lo"](d["c"]); hi = f["t_hi"](d["c"])
        row = dict(block=b, warp=w, sm=d["sm"], c=d["c"].copy(), g=d["g"].copy(), g1=d["g1"].copy(), t_gpu=t, t_lo=lo, t_hi=hi,
                   bound_ns=float((hi - lo).max() / 2), bounds_ns=(hi - lo) / 2, cal=cal, touch=touch,
                   ticket=int(d["n"][9]) if w == 0 else None, ffma=int(d["n"][5]),
                   phases={name: float(d["c"][j] - d["c"][i] - calib[k]) for name, i, j, k in KT_PHASES})
        if run.host_of is not None:
            row["t_host"] = run.host_of(t); row["t_host_lo"] = run.host_of(lo); row["t_host_hi"] = run.host_of(hi)
        rows.append(row)
    # barriers per block: who waited for whom (cycles, exact within the SM). The release stamp follows a shared
    # load + store chain, so the logical check is raw (release stamp >= last arrival stamp); the latency subtracts
    # the least-contended cost of that chain in the launch (its p10), because the per-warp calibration at
    # checkpoint 2 -> 3 is taken under a different load than the post-barrier chain.
    touch_cal = float(np.percentile([r["touch"] for r in rows], 10)) if rows else 0.0
    blocks = {}
    for r in rows:
        blocks.setdefault(r["block"], []).append(r)
    bars = []
    for b, rs in sorted(blocks.items()):
        for name, ia, ir in (("barrier1", 3, 4), ("barrier2", 6, 7)):
            arr = np.array([r["c"][ia] for r in rs]); rel = np.array([r["c"][ir] for r in rs])
            last = arr.max()
            bars.append(dict(block=b, barrier=name, sm=rs[0]["sm"], n_warps=len(rs), arrival_spread=float(last - arr.min()),
                             wait_per_warp=(last - arr).tolist(), latency_per_warp=(rel - last - touch_cal).tolist(),
                             raw_margin_min=float((rel - last).min()), release_before_last_arrival=int((rel < last).sum())))
    # ticket order across blocks: time order of the warp-0 ticket stamps must match the ticket numbers, within the
    # projected bounds of the two blocks' lines at those stamps
    tk = sorted([(r["ticket"], r["t_gpu"][9], r["t_lo"][9], r["t_hi"][9], r["block"], r["sm"]) for r in rows if r["warp"] == 0 and r["ticket"] is not None])
    viol, slack, slack_b = 0, [], []
    for (k0, t0, lo0, hi0, _, _), (k1, t1, lo1, hi1, _, _) in zip(tk, tk[1:]):
        slack.append(t1 - t0); slack_b.append(hi1 - lo0)   # the latest ticket k could be minus the earliest k-1 could be
        if hi1 < lo0:
            viol += 1
    entry = np.array([r["t_gpu"][0] for r in rows]); exit_ = np.array([r["t_gpu"][10] for r in rows])
    fl = [f for f in lines.values() if f]
    bounds_all = np.concatenate([r["bounds_ns"] for r in rows]) if rows else np.array([0.0])
    out = dict(kid=kid, n_blocks=len(blocks), n_warps=len(rows), n_sms=len(set(r["sm"] for r in rows)), tick_ns=tick,
               sm_fit=dict(n=len(fl), bound_ns_p50=float(np.median(bounds_all)), bound_ns_max=float(bounds_all.max()),
                           ghz_p50=float(np.median([f["ghz"] for f in fl])) if fl else None,
                           ghz_min=float(min(f["ghz_lo"] for f in fl)) if fl else None, ghz_max=float(max(f["ghz_hi"] for f in fl)) if fl else None,
                           infeasible=int(sum(1 for f in lines.values() if f is None)), timer_guard_ns=guard_ns),
               span_gpu_ns=float(exit_.max() - entry.min()) if rows else None,
               block_start_spread_ns=float(np.ptp([min(r["t_gpu"][0] for r in rs) for rs in blocks.values()])) if blocks else None,
               # cross-SM guard: %globaltimer is distributed to the SMs and two SMs' copies can disagree by more than
               # the within-SM guard; the ticket order measures it: the smallest extra widening that makes every
               # consecutive-ticket pair consistent is the cross-SM timer skew this launch exhibits
               ticket=dict(n=len(tk), violations=viol, slack_min_ns=float(min(slack)) if slack else None,
                           slack_p50_ns=float(np.median(slack)) if slack else None,
                           worst_margin_ns=float(min(slack_b)) if slack_b else None,
                           cross_sm_guard_ns=float(max(0.0, -min(slack_b))) if slack_b else 0.0),
               touch_cal_cycles=touch_cal, timer_guard_ns=guard_ns, barriers=bars, rows=rows)
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
        flag_write_host = float(run.host_of(h["flag_value_gpu_ns"] - run.g_ref)) if h.get("flag_value_gpu_ns") else None
        out["host_view"] = dict(launch_to_first_entry_ns=first_entry - h["launch_enter"] if h["launch_enter"] else None,
                                call_ns=(h["launch_return"] - h["launch_enter"]) if h["launch_return"] else None,
                                flag_write_to_flag_seen_ns=(h["flag_seen"] - flag_write_host) if (h["flag_seen"] and flag_write_host) else None,
                                last_exit_to_flag_seen_ns=(h["flag_seen"] - last_exit) if h["flag_seen"] else None,
                                last_exit_to_sync_return_ns=(h["sync_return"] - last_exit) if h["sync_return"] else None,
                                bound_ns=hb)
    return out


def a_ktrace(run):
    """Per launch: the timeline (summaries only in the JSON); per block count: phase cycles over all warps, barrier
    waits, placement bounds, checks."""
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
        acc = per_B.setdefault(B, dict(phases={n: [] for n, _, _, _ in KT_PHASES}, cal=[], bar_wait=[], bar_latency=[], bar_spread=[],
                                       span=[], start_spread=[], launch_to_first=[], exit_to_flag=[], exit_to_sync=[], ghz=[],
                                       sm_bound=[], ticket_viol=0, ticket_n=0, bar_release_early=0, kernels=0))
        for r in T["rows"]:
            for n in acc["phases"]:
                acc["phases"][n].append(r["phases"][n])
            acc["cal"].append(r["cal"]); acc.setdefault("touch", []).append(r["touch"])
        for b in T["barriers"]:
            acc["bar_wait"] += b["wait_per_warp"]; acc["bar_latency"] += b["latency_per_warp"]; acc["bar_spread"].append(b["arrival_spread"])
            acc["bar_release_early"] += b["release_before_last_arrival"]
        acc["span"].append(T["span_gpu_ns"]); acc["start_spread"].append(T["block_start_spread_ns"])
        acc["ghz"].append(T["sm_fit"]["ghz_p50"]); acc["sm_bound"].append(T["sm_fit"]["bound_ns_p50"])
        acc["ticket_viol"] += T["ticket"]["violations"]; acc["ticket_n"] += T["ticket"]["n"]; acc["kernels"] += 1
        acc.setdefault("guards", []).append(T["timer_guard_ns"]); acc.setdefault("xsm", []).append(T["ticket"]["cross_sm_guard_ns"])
        hv = T.get("host_view")
        if hv:
            for k, dst in (("launch_to_first_entry_ns", "launch_to_first"), ("flag_write_to_flag_seen_ns", "exit_to_flag"), ("last_exit_to_sync_return_ns", "exit_to_sync")):
                if hv.get(k) is not None:
                    acc[dst].append(hv[k])
        kernels.append(dict(kid=kid, blocks=B, span_gpu_ns=T["span_gpu_ns"], block_start_spread_ns=T["block_start_spread_ns"],
                            sm_fit=T["sm_fit"], ticket=T["ticket"], host_view=hv))
    out = dict(ffma=run.meta.get("ffma"), tick_ns=run.clock.get("tick_ns"), bound_ns=run.clock.get("bound_ns"), by_blocks={}, kernels=kernels)
    for B, acc in sorted(per_B.items()):
        out["by_blocks"][B] = dict(kernels=acc["kernels"], checkpoint_cost_cycles=q(acc["cal"]), touch_chain_cycles=q(acc.get("touch", [])),
                                   phases_cycles={n: q(v) for n, v in acc["phases"].items()},
                                   barrier_wait_cycles=q(acc["bar_wait"]), barrier_latency_cycles=q(acc["bar_latency"]),
                                   barrier_arrival_spread_cycles=q(acc["bar_spread"]), barrier_release_before_last_arrival=acc["bar_release_early"],
                                   kernel_span_ns=q(acc["span"]), block_start_spread_ns=q(acc["start_spread"]),
                                   sm_ghz=q(acc["ghz"]), sm_line_bound_ns=q(acc["sm_bound"]),
                                   ticket=dict(n=acc["ticket_n"], violations=acc["ticket_viol"], cross_sm_guard_ns_max=float(max(acc.get("xsm", [0.0])))),
                                   timer_guard_ns=q(acc.get("guards", [])),
                                   launch_to_first_entry_ns=q(acc["launch_to_first"]), flag_write_to_flag_seen_ns=q(acc["exit_to_flag"]),
                                   last_exit_to_sync_return_ns=q(acc["exit_to_sync"]))
    return out


def one_line_ktrace(res, bs):
    parts = []
    for B, d in res.get("by_blocks", {}).items():
        ph = d["phases_cycles"]
        parts.append(f"B={B}: load {ph['load']['p50']:.0f} | bar1 wait {d['barrier_wait_cycles']['p50']:.0f}/{d['barrier_wait_cycles']['p99']:.0f} "
                     f"lat {d['barrier_latency_cycles']['p50']:.0f} | compute {ph['compute']['p50']:.0f} | store+fence {ph['store_fence']['p50']:.0f} "
                     f"| ticket {ph['ticket']['p50']:.0f} cy; span {d['kernel_span_ns']['p50'] / 1e3:.1f} us, starts spread {d['block_start_spread_ns']['p50'] / 1e3:.1f} us, "
                     f"block lines ±{d['sm_line_bound_ns']['p50']:.0f} ns (timer guard {d['timer_guard_ns'].get('p100') or 0:.0f} ns, cross-SM {d['ticket']['cross_sm_guard_ns_max']:.0f} ns), "
                     f"launch->first {d['launch_to_first_entry_ns'].get('p50') or float('nan'):.0f} ns, flag write->seen {d['flag_write_to_flag_seen_ns'].get('p50') or float('nan'):.0f} ns")
    return f"ktrace {bs}: " + " || ".join(parts)
