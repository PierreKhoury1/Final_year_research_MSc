"""Kernel timeline analysis (strategy `ktrace`): every warp of k_ktrace stamps (%globaltimer, clock64) at eleven
checkpoints; this module places every stamp on one axis and extracts the flow of execution inside the kernel.

Clocks. clock64 is one counter per SM, exact to the cycle for every warp and block on that SM. %globaltimer is the
GPU-wide ns timer, quantised to its tick (1024 ns on A100 / RTX 3060, 64 ns on H100). Each stamp reads both, so
every stamp says: the true GPU time of cycle c lies in [g, g + tick). Per SM and per kernel, the set of such
constraints bounds the SM's cycle-to-ns line (offset and rate) exactly as the tick-edge method bounds the host-GPU
mapping (`analysis/pcieclock.edge_fit`, with %globaltimer in the role of the host clock and the SM cycle counter
in the role of the GPU timer): the feasible region of all constraints gives the line and a hard half-width.
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


def fit_sm_lines(W, tick):
    """Per SM: the cycle->ns line bounded by every stamp's [g0, g1 + tick) window (edge_fit), for one kernel.
    Returns {sm: dict(t_of (callable cycles -> GPU ns), ghz, bound_ns, width_ns, n, feasible, span_ns)}."""
    by_sm = {}
    for (b, w), d in W.items():
        by_sm.setdefault(d["sm"], []).append(d)
    fits = {}
    for sm, ds in by_sm.items():
        c = np.concatenate([d["c"] for d in ds]); g = np.concatenate([d["g"] for d in ds]); g1 = np.concatenate([d["g1"] for d in ds])
        if c.size < 4 or np.ptp(c) <= 0:
            continue
        # least squares first (for the scale), then the exact feasible region around it
        cm, gm = c.mean(), g.mean()
        b0 = ((c - cm) * (g - gm)).sum() / ((c - cm) ** 2).sum()
        if not (0.1 < 1 / b0 < 10):   # cycles per ns outside any SM clock: the stamps are not a line
            continue
        E = (b0 * c).tolist()
        up = [(float(gi) + tick, ei) for gi, ei in zip(g1, E)]     # true time of the cycle read < g1 + tick
        down = [(float(gi), ei) for gi, ei in zip(g, E)]           # true time of the cycle read >= g0
        e = edge_fit(up, down, iters=160, rate_range=0.05)   # the lstsq scale can be off by ~1 % over a short kernel
        E0, H0, a, mid = e["model"]
        fits[sm] = dict(t_of=lambda cyc, E0=E0, H0=H0, a=a, mid=mid, b0=b0: H0 + mid + a * (b0 * np.asarray(cyc, dtype=float) - E0),
                        ghz=float(1 / (a * b0)), bound_ns=float(e["bound_ns"]), width_ns=float(e["width_ns"]), n=int(c.size),
                        feasible=bool(e["feasible"]), span_ns=float(np.ptp(g)), lstsq_ghz=float(1 / b0))
    return fits


def kernel_timeline(run, kid, tick=None):
    """Everything about one launch: per warp the eleven checkpoints in cycles and on the GPU axis (ns, from the
    SM's fitted line) and on the host axis (if the run has a clock fit), per-phase cycles, barrier decomposition
    per block, block/kernel spans, the ticket check, the host events."""
    tick = tick if tick is not None else (run.clock.get("tick_ns") or 1024)
    S = kt_stamps(run)
    W = warp_table(S, kid)
    fits = fit_sm_lines(W, tick)
    rows = []
    for (b, w), d in sorted(W.items()):
        f = fits.get(d["sm"])
        if f is None:
            continue
        cal = d["c"][1] - d["c"][0]
        touch = d["c"][3] - d["c"][2]
        calib = dict(cal=cal, touch=touch)
        t = f["t_of"](d["c"])
        row = dict(block=b, warp=w, sm=d["sm"], c=d["c"].copy(), g=d["g"].copy(), g1=d["g1"].copy(), t_gpu=t, bound_ns=f["bound_ns"], cal=cal, touch=touch,
                   ticket=int(d["n"][9]) if w == 0 else None, ffma=int(d["n"][5]),
                   phases={name: float(d["c"][j] - d["c"][i] - calib[k]) for name, i, j, k in KT_PHASES})
        if run.host_of is not None:
            row["t_host"] = run.host_of(t)
        rows.append(row)
    # barriers per block: who waited for whom (cycles, exact within the SM)
    blocks = {}
    for r in rows:
        blocks.setdefault(r["block"], []).append(r)
    bars = []
    for b, rs in sorted(blocks.items()):
        for name, ia, ir in (("barrier1", 3, 4), ("barrier2", 6, 7)):
            arr = np.array([r["c"][ia] for r in rs]); rel = np.array([r["c"][ir] for r in rs]); touch = np.array([r["touch"] for r in rs])
            last = arr.max()
            # the release stamp follows the touch chain (shared load + store + checkpoint), whose cost is calibrated
            # per warp by checkpoint 2 -> 3; a release earlier than the last arrival plus that chain would be a
            # warp leaving the barrier before every warp arrived
            bars.append(dict(block=b, barrier=name, sm=rs[0]["sm"], n_warps=len(rs), arrival_spread=float(last - arr.min()),
                             wait_per_warp=(last - arr).tolist(), latency_per_warp=(rel - last - touch).tolist(),
                             release_before_last_arrival=int((rel - touch < last).sum())))
    # ticket order across SMs: time order of the warp-0 ticket stamps must match the ticket numbers
    tk = sorted([(r["ticket"], r["t_gpu"][9], r["bound_ns"], r["block"], r["sm"]) for r in rows if r["warp"] == 0 and r["ticket"] is not None])
    viol, slack = 0, []
    for (k0, t0, b0, _, _), (k1, t1, b1, _, _) in zip(tk, tk[1:]):
        s = t1 - t0
        slack.append(s)
        if s < -(b0 + b1):
            viol += 1
    # spans
    entry = np.array([r["t_gpu"][0] for r in rows]); exit_ = np.array([r["t_gpu"][10] for r in rows])
    out = dict(kid=kid, n_blocks=len(blocks), n_warps=len(rows), n_sms=len(set(r["sm"] for r in rows)), tick_ns=tick,
               sm_fit=dict(n=len(fits), bound_ns_p50=float(np.median([f["bound_ns"] for f in fits.values()])) if fits else None,
                           bound_ns_max=float(max(f["bound_ns"] for f in fits.values())) if fits else None,
                           ghz_p50=float(np.median([f["ghz"] for f in fits.values()])) if fits else None,
                           ghz_min=float(min(f["ghz"] for f in fits.values())) if fits else None,
                           ghz_max=float(max(f["ghz"] for f in fits.values())) if fits else None,
                           infeasible=int(sum(1 for f in fits.values() if not f["feasible"]))),
               span_gpu_ns=float(exit_.max() - entry.min()) if rows else None,
               block_start_spread_ns=float(np.ptp([min(r["t_gpu"][0] for r in rs) for rs in blocks.values()])) if blocks else None,
               ticket=dict(n=len(tk), violations=viol, slack_min_ns=float(min(slack)) if slack else None,
                           slack_p50_ns=float(np.median(slack)) if slack else None),
               barriers=bars, rows=rows)
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
        flag_write_host = float(run.host_of(h["flag_value_gpu_ns"])) if h.get("flag_value_gpu_ns") else None
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
        acc["ghz"].append(T["sm_fit"]["ghz_p50"]); acc["sm_bound"].append(T["sm_fit"]["bound_ns_max"])
        acc["ticket_viol"] += T["ticket"]["violations"]; acc["ticket_n"] += T["ticket"]["n"]; acc["kernels"] += 1
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
                                   ticket=dict(n=acc["ticket_n"], violations=acc["ticket_viol"]),
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
                     f"SM lines ±{d['sm_line_bound_ns']['p50']:.0f} ns, ticket order {d['ticket']['violations']}/{d['ticket']['n']} viol, "
                     f"launch->first {d['launch_to_first_entry_ns'].get('p50') or float('nan'):.0f} ns, flag write->seen {d['flag_write_to_flag_seen_ns'].get('p50') or float('nan'):.0f} ns")
    return f"ktrace {bs}: " + " || ".join(parts)
