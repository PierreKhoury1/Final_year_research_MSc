"""Multi-GPU instruction-level timeline from the 9 Oct 2026 8-GPU ktrace run (box_9oct/onbox_a100x8.sh step 2).

Every GPU ran the same ktrace kernel in its own process; all processes started together and launched on one shared
grid of the host clock (CLOCK_MONOTONIC_RAW, TB_ALIGN_US). For one grid instant at which every GPU launched at
1 block per SM, this draws each block's warp 0 coloured by phase (the kernel's real SASS phases between checkpoints)
on that one host axis, one lane per GPU, with each GPU's placement bound; a lower panel puts each GPU's first warp
entry with its bound, and says which GPU pairs' start order the bounds prove.

Usage: python multi_fig.py SRC_DIR MULTI_DIR OUT_DIR [GRID_US [STRICT_JSON]]  (STRICT_JSON: {gpu: clock bound ns})
  SRC_DIR   the tickbound src directory the run was made with (for tickbound.analysis)
  MULTI_DIR res/multi (gpu<g>.json, .gpu.bin, .host.bin, sync files)
"""
import json, os, sys

import numpy as np


def main(src, mdir, out, grid_us=2000.0, strict=None):
    sys.path.insert(0, src)
    from tickbound.analysis.core import Run
    from tickbound.analysis.ktrace import kernel_timeline
    from tickbound.viewer.figures import _plt, PHASE_COLOR, PROBE, INK, INK2, MUTED, _phase_label, SLOTS

    grid_ns = int(grid_us * 1000)
    runs, bus = {}, {}
    for g in range(16):
        p = os.path.join(mdir, f"gpu{g}")
        if os.path.exists(p + ".json") and os.path.exists(p + ".gpu.bin"):
            try:
                runs[g] = Run(p)
            except Exception as e:   # a GPU whose run failed is reported, not drawn
                print(f"GPU {g}: cannot load ({e})")
    csv = os.path.join(os.path.dirname(mdir.rstrip("/\\")), "gpus.csv")
    if os.path.exists(csv):
        for line in open(csv).read().splitlines()[1:]:
            f = [x.strip() for x in line.split(",")]
            if f and f[0].isdigit():
                bus[int(f[0])] = f[1]

    # launches on the grid: slot -> {gpu: (kid, launch_enter host ns, blocks)}
    slots, bmax = {}, {}
    for g, r in runs.items():
        m = r.ev["type"] == 1
        for t, kid, a in zip(r.ev["t"][m], r.ev["kernel_id"][m], r.ev["a"][m]):
            slots.setdefault(int(round(int(t) / grid_ns)), {})[g] = (int(kid), int(t), int(a))
            bmax[g] = max(bmax.get(g, 0), int(a))
    full = {s: d for s, d in slots.items() if all(g in d and d[g][2] == bmax[g] for g in runs)}
    n_all = len(full)
    if not full:   # fall back to the slot with the most GPUs at 1 block per SM
        best = max(sum(1 for g, v in d.items() if v[2] == bmax[g]) for d in slots.values())
        full = {s: {g: v for g, v in d.items() if v[2] == bmax[g]} for s, d in slots.items()
                if sum(1 for g, v in d.items() if v[2] == bmax[g]) == best}
    cand = sorted(full)
    slot = cand[len(cand) // 2]   # a warm one in the middle of the run
    T0 = slot * grid_ns
    print(f"{len(runs)} GPUs loaded; {n_all} grid instants with every GPU launching at 1 block/SM; drawing slot {slot}")

    spans = [("load", "cal", "loaded"), ("barrier1", "bar1_arrive", "bar1_release"), ("compute", "bar1_release", "computed"),
             ("barrier2", "bar2_arrive", "bar2_release"), ("store_fence", "bar2_release", "stored"), ("ticket", "stored", "ticket"),
             ("tail", "ticket", None), ("flag_store", "flag", "exit")]
    probe = [("entry", "cal"), ("loaded", "bar1_arrive"), ("computed", "bar2_arrive")]

    lanes, table = [], []
    for g in sorted(full[slot]):
        r = runs[g]
        if r.host_of is None:
            print(f"GPU {g}: no clock fit, not drawn"); continue
        kid, le, B = full[slot][g]
        tl = kernel_timeline(r, kid)
        names = tl["names"]; ix = {n: i for i, n in enumerate(names)}
        w0 = sorted([x for x in tl["rows"] if x["warp"] == 0 and "t_host" in x], key=lambda x: x["t_host"][ix["entry"]])
        if not w0:
            continue
        # strict: the per-GPU clock bound from the projection of the whole feasible (rate, offset) set, max over the
        # pre->post gap (bound_projection); else the edge fit's chord bound
        cb = float(strict[str(g)]) if strict else float(r.clock["bound_ns"]); sb = float(tl["sm_fit"]["bound_ns_max"])
        first = min(x["t_host"][ix["entry"]] for x in w0); last = max(x["t_host"][ix["exit"]] for x in w0)
        lanes.append(dict(g=g, kid=kid, rows=w0, ix=ix, layout=tl.get("kt_layout", 1), le=le, cb=cb, sb=sb, first=first, last=last))
        table.append(dict(gpu=g, pci=bus.get(g), kernel_id=kid, blocks=B, warps_drawn=len(w0),
                          launch_call_us=(le - T0) / 1e3, first_warp_entry_us=(first - T0) / 1e3, last_warp_exit_us=(last - T0) / 1e3,
                          span_us=(last - first) / 1e3, launch_to_first_entry_us=(first - le) / 1e3,
                          clock_bound_ns=cb, clock_method=("strict envelope" if strict else r.clock.get("method")), clock_rate_ppm=r.clock.get("rate_ppm"),
                          stamp_bound_max_ns=sb, rate_inconsistency_ns=tl["sm_fit"].get("rate_inconsistency_ns"),
                          ticket_violations=tl["ticket"]["violations"]))

    # which pairs' kernel-start order is proved: |delta| > bound_i + bound_j (each bound = clock + stamp)
    pairs, proved = [], 0
    for i in range(len(lanes)):
        for j in range(i + 1, len(lanes)):
            a, b = lanes[i], lanes[j]
            d = b["first"] - a["first"]; s = a["cb"] + a["sb"] + b["cb"] + b["sb"]
            ok = bool(abs(d) > s); proved += int(ok)
            pairs.append(dict(a=a["g"], b=b["g"], delta_ns=d, bound_sum_ns=s, order_proved=bool(ok)))

    plt = _plt()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.5, 9.6), gridspec_kw=dict(height_ratios=[3.4, 1.25], hspace=0.32))
    used = set()
    H = 0.84
    for li, L in enumerate(lanes):
        ix, rows = L["ix"], L["rows"]
        n = len(rows); h = H / n
        ys = li + np.arange(n) * h
        def x(r, name):
            return (r["t_host"][ix[name]] - T0) / 1e3
        for a, b in probe:
            left = np.array([x(r, a) for r in rows]); w = np.array([x(r, b) for r in rows]) - left
            ax.barh(ys, w, left=left, height=h, align="edge", color=PROBE, linewidth=0,
                    label="probe's own calibration steps" if "probe" not in used else None)
            used.add("probe")
        for p, a, b in spans:
            b = b or ("flag" if "flag" in ix else "exit")
            if a not in ix or b not in ix:
                continue
            left = np.array([x(r, a) for r in rows]); w = np.array([x(r, b) for r in rows]) - left
            ax.barh(ys, w, left=left, height=h, align="edge", color=PHASE_COLOR[p], linewidth=0,
                    label=_phase_label(p, L["layout"]) if p not in used else None)
            used.add(p)
        lc = (L["le"] - T0) / 1e3
        ax.plot([lc, lc], [li - 0.04, li + H + 0.04], color=INK, lw=1.4, label="host launch call (cudaLaunchKernel entry)" if "lc" not in used else None)
        used.add("lc")
        e = (L["cb"] + L["sb"]) / 1e3; f = (L["first"] - T0) / 1e3
        ax.errorbar([f], [li + H / 2], xerr=[[e], [e]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3,
                    label="placement bound of this GPU on the host axis (±)" if "eb" not in used else None)
        used.add("eb")
    ax.set_yticks([li + H / 2 for li in range(len(lanes))])
    ax.set_yticklabels([f"GPU {L['g']}" + (f"  {bus[L['g']][-7:]}" if L["g"] in bus else "") + f"\n±{(L['cb'] + L['sb']) / 1e3:.2f} µs"
                        for L in lanes], fontsize=8.5)
    ax.set_ylim(-0.15, len(lanes) - 0.1); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("µs after the shared launch-grid instant (host clock CLOCK_MONOTONIC_RAW, one axis for all GPUs)")
    nB = lanes[0]["rows"] and len(lanes[0]["rows"])
    ax.set_title(f"{len(lanes)} GPUs, one host clock: every block's warp 0, phase by phase (one launch per GPU, {nB} blocks each)")

    for li, L in enumerate(lanes):
        e = (L["cb"] + L["sb"]) / 1e3
        bx.errorbar([(L["first"] - T0) / 1e3], [li], xerr=[[e], [e]], fmt="o", color=SLOTS[li % len(SLOTS)], ecolor=SLOTS[li % len(SLOTS)],
                    capsize=3, ms=5)
        bx.plot([(L["le"] - T0) / 1e3], [li], marker="|", color=INK, ms=10)
    bx.set_yticks(range(len(lanes))); bx.set_yticklabels([f"GPU {L['g']}" for L in lanes], fontsize=8)
    bx.invert_yaxis(); bx.grid(axis="y", visible=False)
    bx.set_xlabel("first warp entry per GPU (dot, ± " + ("strict " if strict else "") + "bound) and its launch call (|), µs after the grid instant")
    bx.set_title(f"Kernel start per GPU: start order proved for {proved} of {len(pairs)} GPU pairs (|Δ| > sum of the two bounds)", fontsize=10.5)

    h, l = ax.get_legend_handles_labels()
    bx.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=8)
    os.makedirs(out, exist_ok=True)
    png = os.path.join(out, "multi_gpu_instruction_timeline.png")
    fig.savefig(png, dpi=200, bbox_inches="tight"); fig.savefig(png[:-4] + ".svg", bbox_inches="tight")
    json.dump(dict(slot=slot, grid_us=grid_us, grid_instants_all_gpus=n_all, gpus=table, pairs=pairs, pairs_order_proved=proved,
                   note=("bounds: clock = strict (projection of the feasible rate/offset set, max over the pre->post gap); " if strict else
                         "bounds: clock = edge-fit (chord) bound of the run's pre+post sync; ") + "stamp = max over the drawn warps of the "
                        "SM line placement bound; host axis = CLOCK_MONOTONIC_RAW shared by all processes"),
              open(os.path.join(out, "multi_gpu_instruction_timeline.json"), "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for t in table:
        print(f"GPU {t['gpu']} {t['pci'] or ''}: launch {t['launch_call_us']:8.2f} us  first entry {t['first_warp_entry_us']:8.2f}  "
              f"last exit {t['last_warp_exit_us']:8.2f}  span {t['span_us']:6.2f}  clock ±{t['clock_bound_ns']:.0f} ns ({t['clock_method']})  "
              f"stamp ±{t['stamp_bound_max_ns']:.0f}  widen {t['rate_inconsistency_ns']}  ticket viol {t['ticket_violations']}")
    print(f"order proved for {proved}/{len(pairs)} pairs; wrote {png}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 3:
        sys.exit(__doc__)
    main(a[0], a[1], a[2], float(a[3]) if len(a) > 3 else 2000.0, json.load(open(a[4])) if len(a) > 4 else None)
