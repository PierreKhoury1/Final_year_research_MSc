#!/usr/bin/env python3
"""Analyse one gputrace run (gputrace/gputrace.cu): put GPU and host events on one axis and summarise the
latency angle the strategy measured.

Clock: the pre and post clock-sync samples are fitted together with the pcieclock edge method
(host = H0 + b + a*(E - E0), feasible offsets from up/down constraints; half the feasible width is a hard
bound under a constant rate over the run). If the combined constraints are infeasible the classic bracket
fit is used and the result is flagged. Every host-vs-GPU latency below is reported with that bound.

Usage: gputrace.py PREFIX [--json OUT] [--csv-dir DIR]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.pcieclock import classic_fit, edge_fit, load as load_cols  # noqa: E402

GPU_DT = np.dtype([("g_begin", "<u8"), ("g_end", "<u8"), ("clk_begin", "<u8"), ("clk_end", "<u8"),
                   ("max_gap_ns", "<u8"), ("smid", "<u4"), ("kernel_id", "<u4"), ("block", "<u4"), ("tag", "<u4"),
                   ("n_iters", "<u4"), ("flags", "<u4")])
HOST_DT = np.dtype([("t", "<i8"), ("type", "<u4"), ("kernel_id", "<u4"), ("a", "<u8"), ("b", "<u8")])
assert GPU_DT.itemsize == 64 and HOST_DT.itemsize == 32

EV = dict(LAUNCH_ENTER=1, LAUNCH_RETURN=2, SYNC_ENTER=3, SYNC_RETURN=4, EVENT_SEEN=5, FLAG_SEEN=6, MARK=7,
          COPY_ENTER=8, COPY_RETURN=9, COPY_DONE=10, IDLE_END=11, GRAPH_ENTER=12, GRAPH_RETURN=13)


def q(x, ps=(50, 90, 99, 100)):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {f"p{p}": None for p in ps} | {"n": 0}
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} | {"n": int(x.size), "mean": float(x.mean())}


class Run:
    def __init__(self, prefix):
        self.prefix = prefix
        self.meta = json.load(open(prefix + ".json"))
        self.recs = np.fromfile(prefix + ".gpu.bin", dtype=GPU_DT)
        self.ev = np.fromfile(prefix + ".host.bin", dtype=HOST_DT)
        self.clock = self.fit_clock()

    def _samples(self, tag):
        try:
            return (load_cols(f"{self.prefix}.{tag}.classic.bin", 3), load_cols(f"{self.prefix}.{tag}.up.bin", 2),
                    load_cols(f"{self.prefix}.{tag}.down.bin", 2))
        except FileNotFoundError:
            return [], [], []

    def fit_clock(self):
        c_pre, u_pre, d_pre = self._samples("pre")
        c_post, u_post, d_post = self._samples("post")
        steps = self.meta.get("timer_edge_steps_ns") or []
        tick = min(steps) if steps else 0
        res = dict(tick_ns=tick, method=None, bound_ns=None, rate_ppm=None, feasible=None, windows={})
        for name, (c, u, d) in dict(pre=(c_pre, u_pre, d_pre), post=(c_post, u_post, d_post)).items():
            if len(u) > 10 and len(d) > 10:
                e = edge_fit(u, d)
                res["windows"][name] = dict(edge_bound_ns=e["bound_ns"], edge_feasible=e["feasible"], n_up=e["n_up"],
                                            n_down=e["n_down"], rate_ppm=e["rate_ppm"])
        up, down, classic = u_pre + u_post, d_pre + d_post, c_pre + c_post
        model = None
        if len(up) > 10 and len(down) > 10:
            e = edge_fit(up, down)
            if e["feasible"]:
                E0, H0, a, mid = e["model"]
                model = lambda g, E0=E0, H0=H0, a=a, mid=mid: H0 + mid + a * (np.asarray(g, dtype=float) - E0)
                res.update(method="edge", bound_ns=e["bound_ns"], rate_ppm=e["rate_ppm"], feasible=True,
                           width_ns=e["width_ns"])
            else:
                res["feasible"] = False
                res["edge_width_ns"] = e["width_ns"]
        if model is None and len(classic) > 60:
            c = classic_fit(classic, tick)
            g0, t0, a, b = c["model"]
            model = lambda g, g0=g0, t0=t0, a=a, b=b: t0 + b + a * (np.asarray(g, dtype=float) - g0)
            res.update(method="classic", bound_ns=c["eps_with_tick_ns"], rate_ppm=c["rate_ppm"])
        self.host_of = model
        return res

    # ---- helpers
    def events(self, typ, kid=None):
        m = self.ev["type"] == EV[typ]
        if kid is not None:
            m &= self.ev["kernel_id"] == kid
        return self.ev[m]

    def ev_time(self, typ, kid):
        e = self.events(typ, kid)
        return int(e["t"][0]) if e.size else None

    def recs_of(self, kid):
        return self.recs[self.recs["kernel_id"] == kid]

    def freq_ghz(self, r):
        dg = r["g_end"].astype(float) - r["g_begin"].astype(float)
        dc = r["clk_end"].astype(float) - r["clk_begin"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(dg > 0, dc / dg, np.nan)


# ---- strategies

def a_launch(run):
    """host launch call -> block start (bounded), call duration, kernel end -> sync return, SM clock at start."""
    enter = run.events("LAUNCH_ENTER")
    graph = False
    if enter.size == 0:
        enter, graph = run.events("GRAPH_ENTER"), True
    ret = run.events("GRAPH_RETURN" if graph else "LAUNCH_RETURN")
    recs = run.recs[(run.recs["tag"] == 0) & (run.recs["kernel_id"] != 0)]
    if graph:   # the graph kernel carries a constant id: match i-th launch with i-th record (one stream, in order)
        recs = run.recs[np.argsort(run.recs["g_begin"])]
        # drop the warm-up launch made before the first traced one (it precedes the first host enter time)
        slack = 2 * (run.clock["bound_ns"] or 0)
        recs = recs[run.host_of(recs["g_end"]) >= float(enter["t"][0]) - slack]
        n = min(len(recs), len(enter))
        recs, enter, ret = recs[:n], enter[:n], ret[:n]
        g_begin, g_end = recs["g_begin"], recs["g_end"]
    else:
        by_kid = {int(r["kernel_id"]): r for r in recs}
        keep = [i for i, e in enumerate(enter) if int(e["kernel_id"]) in by_kid]
        enter, ret = enter[keep], ret[keep]
        rs = np.array([by_kid[int(e["kernel_id"])] for e in enter], dtype=GPU_DT)
        recs, g_begin, g_end = rs, rs["g_begin"], rs["g_end"]
    if len(enter) == 0:
        return dict(error="no launches matched")
    hb, he = run.host_of(g_begin), run.host_of(g_end)
    lat = hb - enter["t"].astype(float)
    call = ret["t"].astype(float) - enter["t"].astype(float)
    out = dict(graph=graph, n=int(len(enter)), launch_to_start_ns=q(lat), call_ns=q(call),
               bound_ns=run.clock["bound_ns"], exec_ns=q(g_end.astype(float) - g_begin.astype(float)),
               sm_clock_ghz=q(run.freq_ghz(recs)[np.isfinite(run.freq_ghz(recs))]))
    depth = int(run.meta.get("depth", 1))
    if depth > 1:   # inter-kernel gap inside one submission burst (records are in launch order on one stream)
        gaps = [float(g_begin[i]) - float(g_end[i - 1]) for i in range(1, len(g_begin)) if i % depth != 0]
        out["inter_kernel_gap_ns"] = q(gaps)
        out["launch_to_start_first_of_burst_ns"] = q(lat[::depth])
    sync_ret = run.events("SYNC_RETURN")
    if sync_ret.size and not graph:
        # completion -> sync return for the last kernel of each iteration
        last_kid = {int(k): t for k, t in zip(sync_ret["kernel_id"], sync_ret["t"])}
        ends = [(last_kid[int(k)] - float(e)) for k, e in zip(enter["kernel_id"], he) if int(k) in last_kid]
        out["end_to_sync_return_ns"] = q(ends)
    return out


def a_notify(run):
    """kernel end -> host observation, per mechanism (tag 0 flag, 1 event, 2 stream sync)."""
    out = dict(bound_ns=run.clock["bound_ns"])
    for tag, name, typ in ((0, "flag_poll", "FLAG_SEEN"), (1, "event_query", "EVENT_SEEN"), (2, "stream_sync", "SYNC_RETURN")):
        recs = run.recs[run.recs["tag"] == tag]
        seen = run.events(typ)
        tmap = {int(k): int(t) for k, t in zip(seen["kernel_id"], seen["t"])}
        lat = [tmap[int(r["kernel_id"])] - float(run.host_of(r["g_end"])) for r in recs if int(r["kernel_id"]) in tmap]
        out[name] = q(lat)
        if tag == 0 and recs.size:   # the flag carries the GPU time of the write: distance between record end and flag write
            fl = {int(k): int(a) for k, a in zip(seen["kernel_id"], seen["a"])}
            out["flag_write_after_end_ns"] = q([fl[int(r["kernel_id"])] - float(r["g_end"]) for r in recs if int(r["kernel_id"]) in fl])
    return out


def _max_concurrent(starts, ends):
    ev = sorted([(s, 1) for s in starts] + [(e, -1) for e in ends])
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def a_dispatch(run):
    """per kernel: SM usage, first-wave dispatch span and rate, waves, kernel span vs ideal, launch latency."""
    sms = int(run.meta["sms"])
    kernels = []
    for e in run.events("LAUNCH_ENTER"):
        kid = int(e["kernel_id"])
        r = run.recs_of(kid)
        if r.size == 0:
            continue
        r = r[np.argsort(r["g_begin"], kind="stable")]
        per_sm = {}
        for rec in r:
            per_sm.setdefault(int(rec["smid"]), []).append((int(rec["g_begin"]), int(rec["g_end"])))
        conc = max(_max_concurrent([a for a, _ in v], [b for _, b in v]) for v in per_sm.values())
        slots = sms * conc
        wave1 = r[:min(len(r), slots)]
        g0, g1 = float(r["g_begin"].min()), float(r["g_end"].max())
        dur = float(np.median(r["g_end"].astype(float) - r["g_begin"].astype(float)))
        waves = int(np.ceil(len(r) / slots))
        span1 = float(wave1["g_begin"].max() - wave1["g_begin"].min())
        kernels.append(dict(
            kernel_id=kid, blocks=int(len(r)), threads=int(e["b"]), sms_used=len(per_sm), sms=sms,
            max_blocks_per_sm_concurrent=conc, blocks_per_sm_min=min(len(v) for v in per_sm.values()),
            blocks_per_sm_max=max(len(v) for v in per_sm.values()), first_wave_span_ns=span1,
            first_wave_rate_blocks_per_us=(len(wave1) - 1) / (span1 / 1000) if span1 > 0 else None,
            waves=waves, block_dur_ns=dur, kernel_span_ns=g1 - g0, ideal_span_ns=waves * dur,
            launch_to_first_block_ns=float(run.host_of(g0)) - float(e["t"]),
            first_wave_sm_order=[int(x) for x in wave1["smid"][:64]],
            # a block's largest gap between two consecutive timer reads: one tick when it ran undisturbed;
            # much longer when its warp was not issued (contention from other resident warps, or preemption)
            max_gap_ns=q(r["max_gap_ns"]), blocks_with_gap_gt_5us=int((r["max_gap_ns"] > 5000).sum()),
            timer_reads_per_us=q(r["n_iters"].astype(float) / np.maximum(1.0, (r["g_end"].astype(float) - r["g_begin"].astype(float)) / 1000)),
            block_overrun_ns=q(r["g_end"].astype(float) - r["g_begin"].astype(float) - dur)))
    return dict(bound_ns=run.clock["bound_ns"], kernels=kernels)


def a_concurrency(run):
    """second stream: time to first block after its launch, co-residency with the first kernel, preemption."""
    reps = []
    enters = run.events("LAUNCH_ENTER")
    kids = [int(k) for k in enters["kernel_id"]]
    for i in range(0, len(kids) - 1, 2):
        ka, kb = kids[i], kids[i + 1]
        A, B = run.recs_of(ka), run.recs_of(kb)
        if A.size == 0 or B.size == 0:
            continue
        tb = float(run.ev_time("LAUNCH_ENTER", kb))
        b_first = float(B["g_begin"].min())
        a_last_end = float(A["g_end"].max())
        a_running_at_b = A[(A["g_begin"].astype(float) <= b_first) & (A["g_end"].astype(float) >= b_first)]
        co_sms = set(int(s) for s in a_running_at_b["smid"]) & set(int(s) for s in B[B["g_begin"].astype(float) <= b_first + 1000]["smid"])
        reps.append(dict(
            kernel_a=ka, kernel_b=kb, blocks_a=int(A.size), blocks_b=int(B.size),
            b_launch_to_first_block_ns=float(run.host_of(b_first)) - tb,
            b_blocks_started_before_a_end=int((B["g_begin"].astype(float) < a_last_end).sum()),
            a_blocks_running_when_b_started=int(a_running_at_b.size),
            sms_shared_at_b_start=len(co_sms), b_sms=len(set(int(s) for s in B["smid"])),
            a_blocks_with_gap_gt_5us=int((A["max_gap_ns"] > 5000).sum()), a_max_gap_ns=int(A["max_gap_ns"].max()),
            b_blocks_with_gap_gt_5us=int((B["max_gap_ns"] > 5000).sum()), b_max_gap_ns=int(B["max_gap_ns"].max()),
            a_block_dur_us=float(run.meta.get("dur_us", 0)), b_timer_reads_per_us=float(np.median(B["n_iters"] / np.maximum(1.0, (B["g_end"].astype(float) - B["g_begin"].astype(float)) / 1000))),
            b_span_ns=float(B["g_end"].max() - B["g_begin"].min()), a_span_ns=float(A["g_end"].max() - A["g_begin"].min()),
            b_block_dur_median_ns=float(np.median(B["g_end"].astype(float) - B["g_begin"].astype(float)))))
    return dict(bound_ns=run.clock["bound_ns"], priority=int(run.meta.get("priority", 0)), reps=reps)


def a_clocks(run):
    """SM clock over time from (globaltimer, clock64) samples; before, during and after the load."""
    s = run.recs[run.recs["tag"] == 1]
    s = s[np.argsort(s["g_begin"])]
    if s.size < 3:
        return dict(error="no samples")
    g, c = s["g_begin"].astype(float), s["clk_begin"].astype(float)
    f = np.diff(c) / np.diff(g)   # GHz
    tmid = run.host_of(g[1:])
    marks = run.events("MARK")
    t_on = [int(t) for t, a in zip(marks["t"], marks["a"]) if int(a) == 1]
    t_off = [int(t) for t, a in zip(marks["t"], marks["a"]) if int(a) == 2]
    out = dict(n_samples=int(s.size), sample_us=run.meta.get("sample_us"), ghz_all=q(f))
    if t_on and t_off:
        on, off = t_on[0], t_off[0]
        before, during, after = f[tmid < on], f[(tmid >= on) & (tmid <= off)], f[tmid > off]
        out.update(ghz_before=q(before), ghz_during=q(during), ghz_after=q(after))
        if during.size:
            target = 0.95 * np.percentile(during, 90)
            idx = np.where((tmid >= on) & (f >= target))[0]
            out["ramp_to_95pct_ns"] = float(tmid[idx[0]] - on) if idx.size else None
    # a timeline for plotting / CSV: host time, GHz
    out["timeline"] = [(float(t), float(x)) for t, x in zip(tmid[::max(1, len(tmid) // 2000)], f[::max(1, len(f) // 2000)])]
    return out


def a_ramp(run):
    """SM clock vs time since block start, after an idle gap of idle_us; averaged over reps."""
    s = run.recs[run.recs["tag"] == 1]
    if s.size < 3:
        return dict(error="no samples")
    kids = sorted(set(int(k) for k in s["kernel_id"]))
    curves, first, launch = [], [], []
    for kid in kids:
        r = s[s["kernel_id"] == kid]
        r = r[np.argsort(r["g_begin"])]
        if r.size < 3:
            continue
        g, c = r["g_begin"].astype(float), r["clk_begin"].astype(float)
        f = np.diff(c) / np.diff(g)
        t = (g[1:] - g[0])   # ns since the block's first sample
        curves.append((t, f))
        first.append(f[0])
        te = run.ev_time("LAUNCH_ENTER", kid)
        if te is not None:
            launch.append(float(run.host_of(g[0])) - te)
    if not curves:
        return dict(error="no curves")
    # common grid: median across reps at each sample index
    n = min(len(t) for t, _ in curves)
    grid_t = np.median(np.array([t[:n] for t, _ in curves]), axis=0)
    grid_f = np.median(np.array([f[:n] for _, f in curves]), axis=0)
    fmax = float(np.percentile(grid_f, 95))
    idx = np.where(grid_f >= 0.95 * fmax)[0]
    return dict(idle_us=run.meta.get("idle_us"), reps=len(curves), ghz_first_sample=q(first), ghz_steady=fmax,
                ramp_to_95pct_ns=float(grid_t[idx[0]]) if idx.size else None,
                launch_to_start_ns=q(launch), bound_ns=run.clock["bound_ns"],
                curve=[(float(a), float(b)) for a, b in zip(grid_t[::max(1, n // 200)], grid_f[::max(1, n // 200)])])


def per_gpu_fits(run, n, reps):
    """Edge fit of each GPU's %globaltimer to the host clock from PREFIX.gpu{d}.r{r}.* (all rounds pooled).
    Returns {d: dict(edge, classic, t_first, t_last, host_of, gpu_of, bound_ns)}."""
    fits = {}
    for d in range(n):
        up, down, classic = [], [], []
        for r in range(reps):
            pre = f"{run.prefix}.gpu{d}.r{r}"
            try:
                classic += load_cols(pre + ".classic.bin", 3)
                up += load_cols(pre + ".up.bin", 2)
                down += load_cols(pre + ".down.bin", 2)
            except FileNotFoundError:
                continue
        if len(up) > 10 and len(down) > 10:
            e = edge_fit(up, down)
            c = classic_fit(classic, run.clock["tick_ns"]) if len(classic) > 60 else None
            E0, H0, a, mid = e["model"]
            fits[d] = dict(edge=e, classic=c, t_first=min(h for h, _ in up), t_last=max(h for h, _ in up),
                           bound_ns=e["bound_ns"],
                           host_of=lambda g, E0=E0, H0=H0, a=a, mid=mid: H0 + mid + a * (np.asarray(g, dtype=float) - E0),
                           gpu_of=lambda t, E0=E0, H0=H0, a=a, mid=mid: E0 + (np.asarray(t, dtype=float) - H0 - mid) / a)
    return fits


def a_gpus(run):
    """Every GPU's %globaltimer on the host axis (edge fit per GPU over all rounds): offset and rate of each GPU
    relative to GPU 0 at the middle of the run, with the sum of the two bounds."""
    n = int(run.meta.get("n_gpus", 1))
    reps = int(run.meta.get("reps", 1))
    fits = per_gpu_fits(run, n, reps)
    if not fits:
        return dict(error="no per-GPU samples", n_gpus=n)
    t_mid = 0.5 * (min(f["t_first"] for f in fits.values()) + max(f["t_last"] for f in fits.values()))
    out = dict(n_gpus=n, rounds=reps, t_mid=t_mid, gpus={})
    for d, f in fits.items():
        e = f["edge"]
        g = dict(bound_ns=e["bound_ns"], feasible=e["feasible"], rate_ppm=e["rate_ppm"], n_up=e["n_up"], n_down=e["n_down"],
                 classic_eps_ns=f["classic"]["eps_with_tick_ns"] if f["classic"] else None)
        if 0 in fits:
            g["timer_minus_gpu0_ns"] = float(f["gpu_of"](t_mid) - fits[0]["gpu_of"](t_mid))
            g["offset_bound_ns"] = e["bound_ns"] + fits[0]["edge"]["bound_ns"]
            g["rate_minus_gpu0_ppm"] = e["rate_ppm"] - fits[0]["edge"]["rate_ppm"]
        out["gpus"][d] = g
    return out


def a_nccl(run):
    """ncclAllReduce across the GPUs of one host. Stamps: tag 1 just before the collective on each GPU's stream,
    tag 2 just after (flags = device, kernel_id = collective id, block = iteration). Per collective size:
      span      after.g_begin - before.g_end on each GPU: the collective plus two inter-kernel gaps (GPU-local ns)
      start skew  GPU d's before-stamp end minus GPU 0's, on the host axis (bound = sum of the two GPU bounds);
                  includes the host's sequential enqueue over devices
      end skew    the same for the after-stamp start: when each GPU finished the collective
      causality   no GPU can finish an all-reduce before every other GPU has started it, so
                  host(after_d.g_begin) >= host(before_e.g_end) for all d != e. A violation larger than the two bounds
                  would falsify the per-GPU clock mappings; the smallest slack is reported.
      algbw/busbw size / span, busbw = algbw * 2 (n-1) / n (NCCL's convention)."""
    n = int(run.meta.get("n_gpus", 1))
    fits = per_gpu_fits(run, n, int(run.meta.get("reps", 2)))
    if len(fits) < n:
        return dict(error=f"clock fits for {len(fits)} of {n} GPUs")
    enter = run.events("LAUNCH_ENTER")
    size_of = {int(k): int(a) for k, a in zip(enter["kernel_id"], enter["a"])}
    t_enter = {int(k): int(t) for k, t in zip(enter["kernel_id"], enter["t"])}
    ret = {int(k): int(t) for k, t in zip(run.events("LAUNCH_RETURN")["kernel_id"], run.events("LAUNCH_RETURN")["t"])}
    sync_ret = {int(k): int(t) for k, t in zip(run.events("SYNC_RETURN")["kernel_id"], run.events("SYNC_RETURN")["t"])}
    stamps = {}
    for r in run.recs:
        stamps[(int(r["kernel_id"]), int(r["flags"]), int(r["tag"]))] = r
    release = {int(k): int(t) for k, t, c in zip(run.events("MARK")["kernel_id"], run.events("MARK")["t"], run.events("MARK")["a"]) if int(c) == 300}
    bsum = {d: fits[d]["bound_ns"] + fits[0]["bound_ns"] for d in fits}
    causal_bound = max(fits[d]["bound_ns"] + fits[e]["bound_ns"] for d in fits for e in fits if d != e)
    per_size = {}
    for kid, size in sorted(size_of.items()):
        try:
            b = {d: stamps[(kid, d, 1)] for d in range(n)}
            a = {d: stamps[(kid, d, 2)] for d in range(n)}
        except KeyError:
            continue
        hb = {d: float(fits[d]["host_of"](b[d]["g_end"])) for d in range(n)}
        ha = {d: float(fits[d]["host_of"](a[d]["g_begin"])) for d in range(n)}
        span = {d: float(a[d]["g_begin"]) - float(b[d]["g_end"]) for d in range(n)}
        slack = min(ha[d] - hb[e] for d in range(n) for e in range(n) if d != e)
        ps = per_size.setdefault(size, dict(span={d: [] for d in range(n)}, start_skew={d: [] for d in range(1, n)},
                                            end_skew={d: [] for d in range(1, n)}, slack=[], call=[], end_to_sync=[]))
        for d in range(n):
            ps["span"][d].append(span[d])
        for d in range(1, n):
            ps["start_skew"][d].append(hb[d] - hb[0])
            ps["end_skew"][d].append(ha[d] - ha[0])
        ps["slack"].append(slack)
        if kid in release:
            ps.setdefault("release_to_stamp", {d: [] for d in range(n)})
            for d in range(n):
                ps["release_to_stamp"][d].append(float(fits[d]["host_of"](b[d]["g_begin"])) - release[kid])
        if kid in ret:
            ps["call"].append(ret[kid] - t_enter[kid])
        if kid in sync_ret:
            ps["end_to_sync"].append(sync_ret[kid] - max(float(fits[d]["host_of"](a[d]["g_end"])) for d in range(n)))
    out = dict(n_gpus=n, nccl_version=run.meta.get("nccl_version"), gate=run.meta.get("gate"),
               clock={d: dict(bound_ns=f["bound_ns"], feasible=f["edge"]["feasible"], rate_ppm=f["edge"]["rate_ppm"]) for d, f in fits.items()},
               skew_bound_ns={d: bsum[d] for d in range(1, n)}, causal_bound_ns=causal_bound, sizes={})
    for size, ps in sorted(per_size.items()):
        med_span = float(np.median([np.median(v) for v in ps["span"].values()]))
        algbw = size / med_span if med_span > 0 else None   # bytes per ns = GB/s
        out["sizes"][size] = dict(
            n=len(ps["slack"]), span_ns={d: q(v) for d, v in ps["span"].items()},
            start_skew_ns={d: q(v) for d, v in ps["start_skew"].items()},
            end_skew_ns={d: q(v) for d, v in ps["end_skew"].items()},
            causal_slack_min_ns=float(min(ps["slack"])), causal_violations=int(sum(1 for x in ps["slack"] if x < -causal_bound)),
            call_ns=q(ps["call"]), end_to_sync_ns=q(ps["end_to_sync"]),
            algbw_GBps=algbw, busbw_GBps=(algbw * 2 * (n - 1) / n) if algbw else None,
            release_to_stamp_ns={d: q(v) for d, v in ps.get("release_to_stamp", {}).items()})
    return out


MODS = {0: ".ca", 1: ".cg", 2: ".cs", 3: ".nc"}


def a_memory(run):
    """Pointer-chase latency per working set and PTX modifier: cycles per load (clock64), ns per load (%globaltimer)
    and the SM clock, each from the same record; the co-tenant's streaming blocks (tag 20) are summarised too.
    Tiers are read off the latency curve: a 'knee' is a working set whose latency is > 1.6x the previous one."""
    sizes = {}
    enter = run.events("LAUNCH_ENTER")
    ws_of_kid = {int(k): int(a) for k, a in zip(enter["kernel_id"], enter["a"])}
    r = run.recs[(run.recs["tag"] < 4) & (run.recs["n_iters"] > 0)]
    cond = int(run.meta.get("cotenant", 0))
    for rec in r:
        ws = ws_of_kid.get(int(rec["kernel_id"]))
        if ws is None:
            continue
        n = float(rec["n_iters"])
        cyc = (float(rec["clk_end"]) - float(rec["clk_begin"])) / n
        ns = (float(rec["g_end"]) - float(rec["g_begin"])) / n
        d = sizes.setdefault((int(rec["tag"]), ws), dict(cyc=[], ns=[], ghz=[]))
        d["cyc"].append(cyc); d["ns"].append(ns)
        if ns > 0: d["ghz"].append(cyc / ns)
    out = dict(cotenant=cond, batch=run.meta.get("batch"), modifiers={}, bound_ns=run.clock["bound_ns"])
    for mod in sorted(set(m for m, _ in sizes)):
        curve = []
        for (m, ws) in sorted(k for k in sizes if k[0] == mod):
            d = sizes[(m, ws)]
            curve.append(dict(ws=ws, cycles=q(d["cyc"]), ns=q(d["ns"]), ghz=float(np.median(d["ghz"])) if d["ghz"] else None))
        knees = [c["ws"] for i, c in enumerate(curve) if i and c["cycles"]["p50"] > 1.6 * curve[i - 1]["cycles"]["p50"]]
        out["modifiers"][MODS.get(mod, str(mod))] = dict(curve=curve, knees=knees,
            tiers=dict(smallest=curve[0]["cycles"]["p50"] if curve else None, largest=curve[-1]["cycles"]["p50"] if curve else None))
    co = run.recs[run.recs["tag"] == 20]
    if co.size:
        out["cotenant_blocks"] = dict(n=int(co.size), sms=len(set(int(x) for x in co["smid"])),
                                      span_ms=float((co["g_end"].max() - co["g_begin"].min()) / 1e6),
                                      passes=q(co["n_iters"]))
    return out


KINDS = {0: "LDG.ca", 1: "LDG.cg", 2: "LDG.cs", 3: "LDG.nc", 4: "LDS", 5: "FADD", 6: "FFMA", 7: "IMAD", 8: "SHFL",
         9: "ATOM(ret)", 10: "RED+fence", 11: "STG+fence", 12: "BAR"}


def a_instr(run):
    """Instruction table: per kind (and working set for loads), bracket cycles for chains of N = 1..32 dependent
    instructions; a least-squares fit cycles = a + b*N over the per-N medians gives latency b (cycles per
    instruction) and bracket overhead a. ns per instruction from the longest chain's %globaltimer span (tick-limited
    for short brackets), and the SM clock from the same record."""
    enter = run.events("LAUNCH_ENTER")
    ws_of = {int(k): int(a) for k, a in zip(enter["kernel_id"], enter["a"])}
    r = run.recs[(run.recs["tag"] < 13) & (run.recs["clk_begin"] == 0) & (run.recs["n_iters"] > 0)]
    groups = {}
    for rec in r:
        key = (int(rec["tag"]), ws_of.get(int(rec["kernel_id"]), 0))
        g = groups.setdefault(key, {})
        g.setdefault(int(rec["n_iters"]), []).append((float(rec["clk_end"]), float(rec["g_end"]) - float(rec["g_begin"])))
    table = []
    for (kind, ws), byN in sorted(groups.items()):
        Ns = sorted(byN)
        med = {N: float(np.median([c for c, _ in byN[N]])) for N in Ns}
        p99 = {N: float(np.percentile([c for c, _ in byN[N]], 99)) for N in Ns}
        if len(Ns) >= 2:
            A = np.vstack([np.ones(len(Ns)), np.array(Ns, dtype=float)]).T
            (a, b), *_ = np.linalg.lstsq(A, np.array([med[N] for N in Ns]), rcond=None)
        else:
            a, b = float("nan"), med[Ns[0]]
        Nmax = Ns[-1]
        ns_total = float(np.median([g for _, g in byN[Nmax]]))
        # the SM clock from one bracket is only meaningful when the bracket spans many timer ticks
        ghz = med[Nmax] / ns_total if ns_total >= 8 * max(1, run.clock["tick_ns"]) else float("nan")
        table.append(dict(kind=KINDS.get(kind, str(kind)), ws=ws, latency_cycles=float(b), bracket_overhead_cycles=float(a),
                          single_bracket_cycles=med.get(1), per_N_median=med, per_N_p99=p99, samples=sum(len(v) for v in byN.values()),
                          ns_per_instr_at_Nmax=ns_total / Nmax, sm_ghz=ghz, Nmax=Nmax))
    return dict(cotenant=run.meta.get("cotenant"), reps=run.meta.get("reps"), table=table, bound_ns=run.clock["bound_ns"])


def a_copy(run):
    """host-visible memcpy latency per size and direction; GPU-side dependent PCIe read latency."""
    out = dict(memcpy={})
    enter, ret, done = run.events("COPY_ENTER"), run.events("COPY_RETURN"), run.events("COPY_DONE")
    dmap = {int(k): int(t) for k, t in zip(done["kernel_id"], done["t"])}
    rmap = {int(k): int(t) for k, t in zip(ret["kernel_id"], ret["t"])}
    groups = {}
    for e in enter:
        kid = int(e["kernel_id"])
        if kid in dmap:
            key = f"{'h2d' if int(e['b']) == 0 else 'd2h'}_{int(e['a'])}B"
            groups.setdefault(key, ([], []))
            groups[key][0].append(dmap[kid] - int(e["t"]))
            groups[key][1].append(rmap[kid] - int(e["t"]))
    for k, (tot, call) in groups.items():
        out["memcpy"][k] = dict(enter_to_done_ns=q(tot), call_ns=q(call))
    r = run.recs[run.recs["tag"] == 4]
    if r.size:
        dg = r["g_end"].astype(float) - r["g_begin"].astype(float)
        dc = r["clk_end"].astype(float) - r["clk_begin"].astype(float)
        ghz = dc.sum() / dg.sum() if dg.sum() > 0 else float("nan")
        out["gpu_read_host_mem"] = dict(n=int(r.size), sm_clock_ghz_est=float(ghz), ns_by_clock=q(dc / ghz),
                                         ns_by_globaltimer=q(dg), tick_ns=run.clock["tick_ns"])
    return out


def a_timeslice(run):
    """gaps in our resident thread (quanta given to the other process) and the hog's view."""
    gaps = run.recs[run.recs["tag"] == 2]
    fin = run.recs[run.recs["tag"] == 3]
    marks = run.events("MARK")
    t_on = [int(t) for t, a in zip(marks["t"], marks["a"]) if int(a) == 1]
    out = dict(gap_threshold_us=run.meta.get("gap_us"), n_gaps=int(gaps.size))
    if gaps.size:
        gl = gaps["g_end"].astype(float) - gaps["g_begin"].astype(float)
        hs = run.host_of(gaps["g_begin"])
        out["gap_ns"] = q(gl)
        if t_on:
            on = t_on[0]
            before = gl[hs < on]
            after = gl[hs >= on]
            out["gaps_alone"] = q(before)
            out["gaps_with_hog"] = q(after)
            if after.size and fin.size:
                win_end = float(run.host_of(fin["g_end"][0]))
                out["hog_window_s"] = (win_end - on) * 1e-9
                out["fraction_not_running_with_hog"] = float(after.sum() / max(1.0, win_end - on))
                out["switches_per_s_with_hog"] = float(after.size / max(1e-9, (win_end - on) * 1e-9))
                # between consecutive gaps: how long we ran = our quantum (GPU timestamps, no mapping needed)
                gw = gaps[hs >= on]
                ours = gw["g_begin"][1:].astype(float) - gw["g_end"][:-1].astype(float)
                out["our_quantum_ns"] = q(ours)
    if fin.size:
        out["resident_window_ns"] = float(fin["g_end"][0] - fin["g_begin"][0])
        out["resident_iters"] = int(fin["n_iters"][0])
    hog_prefix = run.prefix + ".hog"
    if os.path.exists(hog_prefix + ".json"):
        hog = Run(hog_prefix)
        hr = hog.recs[hog.recs["tag"] == 9]
        out["hog"] = dict(blocks=int(hr.size), sms_used=len(set(int(s) for s in hr["smid"])), sms=int(hog.meta["sms"]),
                          block_dur_ns=q(hr["g_end"].astype(float) - hr["g_begin"].astype(float)),
                          max_gap_ns=q(hr["max_gap_ns"]), blocks_with_gap_gt_5us=int((hr["max_gap_ns"] > 5000).sum()),
                          hog_dur_us=hog.meta.get("hog_dur_us"),
                          mps_pct=hog.meta.get("mps_pct"))
        # Cross-process validation: the other process can only finish a kernel while it holds the GPU, so every hog
        # kernel's end (its last block, on the hog's own clock fit) must fall inside one of our not-running intervals
        # (on ours), within the sum of the two bounds. A miss would falsify one of the two mappings.
        if gaps.size and hog.host_of is not None and hr.size:
            order = np.argsort(run.host_of(gaps["g_begin"]))
            gs, ge = run.host_of(gaps["g_begin"])[order], run.host_of(gaps["g_end"])[order]
            ends = {}
            for k, e in zip(hr["kernel_id"], hog.host_of(hr["g_end"])):
                ends[int(k)] = max(ends.get(int(k), -1e30), float(e))
            E = np.array(sorted(ends.values()))
            E = E[(E > gs[0]) & (E < ge[-1])]
            bsum = run.clock["bound_ns"] + hog.clock["bound_ns"]
            idx = np.maximum(np.searchsorted(gs, E) - 1, 0)
            inside = (E >= gs[idx] - bsum) & (E <= ge[idx] + bsum)
            L = ge - gs
            out["hog"]["kernel_ends_checked"] = int(E.size)
            out["hog"]["kernel_ends_inside_our_gaps"] = int(inside.sum())
            out["hog"]["cross_check_bound_ns"] = float(bsum)
            out["gap_short_lt_1500us"] = q(L[L < 1.5e6])
            out["gap_long_ge_1500us"] = q(L[L >= 1.5e6])
    return out


STRATEGIES = dict(launch=a_launch, notify=a_notify, dispatch=a_dispatch, concurrency=a_concurrency, clocks=a_clocks,
                  copy=a_copy, timeslice=a_timeslice, ramp=a_ramp, gpus=a_gpus, nccl=a_nccl, memory=a_memory, instr=a_instr)


def analyse(prefix):
    run = Run(prefix)
    s = run.meta["strategy"]
    res = dict(prefix=prefix, strategy=s, gpu=run.meta.get("gpu"), sms=run.meta.get("sms"), host=run.meta.get("host"),
               clock=run.clock, n_gpu_recs=int(run.recs.size), gpu_recs_dropped=run.meta.get("gpu_recs_dropped"),
               n_host_events=int(run.ev.size))
    if run.host_of is None and s not in ("gpus", "nccl"):   # multi-GPU strategies fit each GPU themselves
        res["error"] = "no clock model"
        return res
    fn = STRATEGIES.get(s)
    res["result"] = fn(run) if fn else dict(error=f"unknown strategy {s}")
    return res


def one_line(res):
    s, r, c = res["strategy"], res.get("result", {}), res["clock"]
    b = c.get("bound_ns")
    bs = f"±{b:.0f} ns ({c.get('method')})" if b is not None else "no clock"
    def P(d, k="p50"):
        return "-" if not d or d.get(k) is None else f"{d[k] / 1000:.1f}"
    if s == "launch":
        return (f"launch: start {P(r.get('launch_to_start_ns'))}/{P(r.get('launch_to_start_ns'), 'p99')} us p50/p99 {bs}; "
                f"call {P(r.get('call_ns'))} us; end->sync {P(r.get('end_to_sync_return_ns'))} us; "
                f"SM {r.get('sm_clock_ghz', {}).get('p50', float('nan')):.2f} GHz; graph={r.get('graph')} n={r.get('n')}")
    if s == "notify":
        return (f"notify: flag {P(r.get('flag_poll'))}/{P(r.get('flag_poll'), 'p99')} | event {P(r.get('event_query'))}/"
                f"{P(r.get('event_query'), 'p99')} | sync {P(r.get('stream_sync'))}/{P(r.get('stream_sync'), 'p99')} us p50/p99 {bs}")
    if s == "dispatch":
        ks = r.get("kernels", [])
        parts = [f"{k['blocks']}b:{k['sms_used']}sm x{k['max_blocks_per_sm_concurrent']} "
                 f"{(k['first_wave_rate_blocks_per_us'] or 0):.1f}b/us w{k['waves']} "
                 f"span {k['kernel_span_ns'] / 1000:.0f}/{k['ideal_span_ns'] / 1000:.0f}us" for k in ks[::max(1, len(ks) // 5)]]
        return "dispatch: " + " | ".join(parts) + f"; gaps>5us in {sum(k['blocks_with_gap_gt_5us'] for k in ks)}/{sum(k['blocks'] for k in ks)} blocks {bs}"
    if s == "concurrency":
        reps = r.get("reps", [])
        if not reps:
            return "concurrency: no reps"
        lat = [x["b_launch_to_first_block_ns"] for x in reps]
        return (f"concurrency(prio={r.get('priority')}): B launch->first block {np.median(lat) / 1000:.1f} us median "
                f"(max {max(lat) / 1000:.1f}) {bs}; B before A end {np.median([x['b_blocks_started_before_a_end'] for x in reps]):.0f}/"
                f"{reps[0]['blocks_b']} blocks; A blocks with >5us gap {sum(x['a_blocks_with_gap_gt_5us'] for x in reps)}; "
                f"A max gap {max(x['a_max_gap_ns'] for x in reps) / 1000:.0f} us")
    if s == "clocks":
        g = lambda k: (r.get(k) or {}).get("p50")
        return (f"clocks: before {g('ghz_before')} during {g('ghz_during')} after {g('ghz_after')} GHz; "
                f"ramp to 95% {None if r.get('ramp_to_95pct_ns') is None else r['ramp_to_95pct_ns'] / 1000:.0f} us; n={r.get('n_samples')}")
    if s == "instr":
        rows = []
        for t in r.get("table", []):
            w = f"@{t['ws'] >> 10}K" if 0 < t["ws"] < (1 << 20) else (f"@{t['ws'] >> 20}M" if t["ws"] else "")
            rows.append(f"{t['kind']}{w} {t['latency_cycles']:.0f}cy (ovh {t['bracket_overhead_cycles']:.0f})")
        return f"instr(cotenant={r.get('cotenant')}): " + " | ".join(rows)
    if s == "memory":
        parts = []
        for mod, m in r.get("modifiers", {}).items():
            c = m["curve"]
            def at(ws):
                x = [p for p in c if p["ws"] == ws]
                return f"{x[0]['cycles']['p50']:.0f}cy/{x[0]['ns']['p50']:.0f}ns" if x else "-"
            parts.append(f"{mod}: 16k {at(16384)} 1m {at(1 << 20)} 16m {at(16 << 20)} 128m {at(128 << 20)} knees {[k >> 10 for k in m['knees']]}K")
        return f"memory(cotenant={r.get('cotenant')}): " + " | ".join(parts)
    if s == "nccl":
        szs = r.get("sizes", {})
        def fmt(sz, v):
            sk = v["end_skew_ns"].get(1) or v["end_skew_ns"].get("1") or {}
            return (f"{sz}B: {v['span_ns'][next(iter(v['span_ns']))]['p50'] / 1000:.1f} us, end skew {(sk.get('p50') or 0) / 1000:+.2f} us, "
                    f"busbw {v['busbw_GBps'] or 0:.1f} GB/s, causal viol {v['causal_violations']}/{v['n']}")
        return (f"nccl x{r.get('n_gpus')}: " + " | ".join(fmt(k, v) for k, v in szs.items()) +
                f" (skew bound ±{max((r.get('skew_bound_ns') or {0: 0}).values()) / 1000:.2f} us)")
    if s == "gpus":
        gs = r.get("gpus", {})
        return "gpus: " + " | ".join(f"gpu{d}: {g.get('timer_minus_gpu0_ns', 0) / 1000:+.2f} us vs gpu0 ±{g.get('offset_bound_ns', 0) / 1000:.2f} "
                                     f"({g.get('rate_minus_gpu0_ppm', 0):+.2f} ppm, bound {g['bound_ns']:.0f} ns{'' if g['feasible'] else ' INFEASIBLE'})"
                                     for d, g in gs.items())
    if s == "ramp":
        rr = r.get("ramp_to_95pct_ns")
        return (f"ramp(idle {r.get('idle_us')} us): first sample {(r.get('ghz_first_sample') or {}).get('p50', float('nan')):.2f} GHz, "
                f"steady {r.get('ghz_steady', float('nan')):.2f} GHz, to 95% in {'-' if rr is None else f'{rr / 1000:.0f}'} us; "
                f"launch->start {P(r.get('launch_to_start_ns'))} us {bs}; reps={r.get('reps')}")
    if s == "copy":
        m = r.get("memcpy", {})
        gr = r.get("gpu_read_host_mem", {})
        small = {k: v for k, v in m.items() if k.endswith("_8B")}
        return ("copy: " + " ".join(f"{k} {P(v['enter_to_done_ns'])}us" for k, v in small.items()) +
                f"; GPU read host mem {P(gr.get('ns_by_clock'))}/{P(gr.get('ns_by_clock'), 'p99')} us p50/p99 (n={gr.get('n')})")
    if s == "timeslice":
        return (f"timeslice: gaps alone {P(r.get('gaps_alone'), 'p99')} us p99 | with hog {P(r.get('gaps_with_hog'))}/"
                f"{P(r.get('gaps_with_hog'), 'p99')} us p50/p99 n={(r.get('gaps_with_hog') or {}).get('n')}; "
                f"our quantum {P(r.get('our_quantum_ns'))} us; not running {100 * (r.get('fraction_not_running_with_hog') or 0):.0f}%; "
                f"hog SMs {(r.get('hog') or {}).get('sms_used')}/{(r.get('hog') or {}).get('sms')}")
    return f"{s}: {json.dumps(r)[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    res = analyse(a.prefix)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=float)
    print(one_line(res))
    if not a.quiet and not a.json:
        print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
