#!/usr/bin/env python3
"""Export a gputrace run to the Chrome trace-event format (open in https://ui.perfetto.dev or chrome://tracing).

Everything is placed on the host clock (CLOCK_MONOTONIC_RAW, microseconds from the run's first exported event):
  process "host"            launch calls, synchronize waits, copies (slices); flag/event observations, marks (instants)
  process "GPU d (±b ns)"   one track per SM, one slice per traced block (kernel id, block, tag, largest timer gap);
                            for timeslice runs, the resident thread's not-running intervals; for nccl runs, one
                            "allreduce" slice per GPU per collective (between the stamp kernels)
The GPU slices are mapped with the run's edge fit (per GPU for gpus/nccl runs); the bound is in the process name and
in every slice's args, so a viewer can see how far a GPU slice may be from its drawn position.

Usage: gputrace_export.py PREFIX [--out FILE] [--max-kernels N] [--start-ms X --window-ms Y]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.gputrace import KINDS as INSTR_KINDS, EV, Run, per_gpu_fits  # noqa: E402

EV_NAME = {v: k for k, v in EV.items()}
HOST_PID, GPU_PID0 = 1, 10


def export(prefix, max_kernels=200, start_ms=None, window_ms=None, max_events=400_000):
    run = Run(prefix)
    strategy = run.meta.get("strategy")
    n = int(run.meta.get("n_gpus", 1))
    if strategy in ("gpus", "nccl"):
        fits = per_gpu_fits(run, n, int(run.meta.get("reps", 2)))
        mappers = {d: (f["host_of"], f["bound_ns"]) for d, f in fits.items()}
    else:
        if run.host_of is None:
            raise SystemExit("no clock model for this run")
        mappers = {0: (run.host_of, run.clock["bound_ns"])}

    ev = run.ev
    kids = sorted(set(int(k) for k in run.recs["kernel_id"]))
    keep_kids = set(kids[:max_kernels]) if max_kernels else set(kids)
    recs = run.recs[np.isin(run.recs["kernel_id"], list(keep_kids)) | np.isin(run.recs["tag"], [2, 3] if strategy != "instr" else [])]
    enter = run.events("LAUNCH_ENTER")
    ws_of = {int(k): int(a) for k, a in zip(enter["kernel_id"], enter["a"])}   # instr: working set per bracket kernel
    dev_of = (recs["flags"] if strategy == "nccl" else np.zeros(len(recs), dtype=np.uint32)).astype(int)

    # time origin: first host event or first block, whichever is earlier
    t_candidates = [float(ev["t"].min())] if ev.size else []
    for d, (host_of, _) in mappers.items():
        rd = recs[dev_of == d]
        if rd.size:
            t_candidates.append(float(host_of(rd["g_begin"].min())))
    t0 = min(t_candidates)
    lo = t0 + (start_ms or 0) * 1e6
    hi = lo + window_ms * 1e6 if window_ms else float("inf")

    out = []
    us = lambda t: (t - t0) / 1000.0
    inside = lambda a, b: b >= lo and a <= hi

    def meta(pid, name, tid=None, tname=None, sort=None):
        if tid is None:
            out.append(dict(ph="M", pid=pid, name="process_name", args=dict(name=name)))
            if sort is not None:
                out.append(dict(ph="M", pid=pid, name="process_sort_index", args=dict(sort_index=sort)))
        else:
            out.append(dict(ph="M", pid=pid, tid=tid, name="thread_name", args=dict(name=tname)))

    # ---- host
    meta(HOST_PID, f"host ({run.meta.get('host', '')})", sort=0)
    for tid, tname in ((1, "launch calls"), (2, "synchronize / poll"), (3, "copies"), (4, "marks")):
        meta(HOST_PID, None, tid, tname)
    pairs = ((EV["LAUNCH_ENTER"], EV["LAUNCH_RETURN"], 1, "launch"), (EV["GRAPH_ENTER"], EV["GRAPH_RETURN"], 1, "graph launch"),
             (EV["SYNC_ENTER"], EV["SYNC_RETURN"], 2, "synchronize"), (EV["COPY_ENTER"], EV["COPY_DONE"], 3, "copy"))
    for t_open, t_close, tid, name in pairs:
        opens = ev[ev["type"] == t_open]
        closes = {int(k): int(t) for k, t in zip(ev[ev["type"] == t_close]["kernel_id"], ev[ev["type"] == t_close]["t"])}
        for e in opens:
            k = int(e["kernel_id"])
            if keep_kids and k not in keep_kids and k != 0 and strategy not in ("copy",):
                continue
            if k in closes and inside(float(e["t"]), float(closes[k])):
                args = dict(kernel=k)
                if t_open in (EV["LAUNCH_ENTER"], EV["COPY_ENTER"]):
                    args.update(a=int(e["a"]), b=int(e["b"]))
                out.append(dict(ph="X", pid=HOST_PID, tid=tid, name=f"{name} k{k}", ts=us(float(e["t"])),
                                dur=(closes[k] - float(e["t"])) / 1000.0, args=args))
    for typ in (EV["FLAG_SEEN"], EV["EVENT_SEEN"], EV["MARK"], EV["IDLE_END"]):
        for e in ev[ev["type"] == typ]:
            if lo <= float(e["t"]) <= hi:
                out.append(dict(ph="i", s="t", pid=HOST_PID, tid=4 if typ in (EV["MARK"], EV["IDLE_END"]) else 2,
                                name=f"{EV_NAME[typ].lower()} k{int(e['kernel_id'])}", ts=us(float(e["t"])),
                                args=dict(a=int(e["a"]), b=int(e["b"]))))

    # ---- GPUs
    for d, (host_of, bound) in sorted(mappers.items()):
        pid = GPU_PID0 + d
        meta(pid, f"GPU {d}: {run.meta.get('gpu', '')}  (on host axis, ±{bound:.0f} ns)", sort=1 + d)
        rd = recs[dev_of == d]
        if not rd.size:
            continue
        hb, he = host_of(rd["g_begin"]), host_of(rd["g_end"])
        seen_sm = set()
        for r, a, b in zip(rd, hb, he):
            if not inside(float(a), float(b)):
                continue
            tag = int(r["tag"])
            if strategy == "nccl":
                continue   # stamps are drawn as collective slices below
            if strategy == "instr" and tag < 13 and int(r["clk_begin"]) == 0:   # one bracket sample: kind, chain length N
                ws = int(ws_of.get(int(r["kernel_id"]), 0))
                tid = int(r["smid"])
                name = INSTR_KINDS.get(tag, str(tag)) + (f" {ws >> 10}K" if 0 < ws < 1 << 20 else (f" {ws >> 20}M" if ws else ""))
            elif tag == 20:
                tid, name = int(r["smid"]), "co-tenant stream"
            elif tag == 2 and strategy != "instr":
                tid, name = 100000, "not running (time-sliced out)"
            elif tag == 3 and strategy != "instr":
                continue
            else:
                tid, name = int(r["smid"]), f"k{int(r['kernel_id'])} b{int(r['block'])}"
            if tid not in seen_sm:
                meta(pid, None, tid, "resident thread" if tid == 100000 else f"SM {tid}")
                seen_sm.add(tid)
            out.append(dict(ph="X", pid=pid, tid=tid, name=name, ts=us(float(a)), dur=max(0.001, (float(b) - float(a)) / 1000.0),
                            args=dict(kernel=int(r["kernel_id"]), block=int(r["block"]), tag=tag, sm=int(r["smid"]), n=int(r["n_iters"]),
                                      max_gap_ns=int(r["max_gap_ns"]), sm_cycles=int(r["clk_end"]) - int(r["clk_begin"]),
                                      bound_ns=round(bound))))
            if len(out) > max_events:
                break
        if strategy == "nccl":
            meta(pid, None, 1, "collectives")
            size_of = {int(k): int(a) for k, a in zip(ev[ev["type"] == EV["LAUNCH_ENTER"]]["kernel_id"], ev[ev["type"] == EV["LAUNCH_ENTER"]]["a"])}
            before = {int(r["kernel_id"]): r for r in rd if int(r["tag"]) == 1}
            after = {int(r["kernel_id"]): r for r in rd if int(r["tag"]) == 2}
            for k in sorted(set(before) & set(after)):
                a, b = float(host_of(before[k]["g_end"])), float(host_of(after[k]["g_begin"]))
                if inside(a, b):
                    out.append(dict(ph="X", pid=pid, tid=1, name=f"allreduce {size_of.get(k, 0)} B", ts=us(a), dur=(b - a) / 1000.0,
                                    args=dict(kernel=k, bytes=size_of.get(k, 0), bound_ns=round(bound))))
    # ---- the other process of a timeslice run (PREFIX.hog.*), on its own clock fit, as a separate process
    hog = prefix + ".hog"
    if strategy == "timeslice" and os.path.exists(hog + ".json") and os.path.exists(hog + ".gpu.bin"):
        hr = Run(hog)
        if hr.host_of is not None:
            pid = GPU_PID0 + 50
            meta(pid, f"GPU 0: other process (hog, {hr.meta.get('hog_dur_us', 0):.0f} us blocks; ±{hr.clock['bound_ns']:.0f} ns)", sort=60)
            r9 = hr.recs[hr.recs["tag"] == 9]
            hb, he = hr.host_of(r9["g_begin"]), hr.host_of(r9["g_end"])
            m = (he >= lo) & (hb <= hi)
            sms = sorted(set(int(x) for x in r9["smid"][m]))[:4]   # four SMs are enough to see the pattern
            for sm in sms:
                meta(pid, None, sm, f"SM {sm}")
            for r, a_, b_ in zip(r9[m], hb[m], he[m]):
                if int(r["smid"]) in sms:
                    out.append(dict(ph="X", pid=pid, tid=int(r["smid"]), name=f"hog k{int(r['kernel_id'])} b{int(r['block'])}",
                                    ts=us(float(a_)), dur=(float(b_) - float(a_)) / 1000.0,
                                    args=dict(max_gap_ns=int(r["max_gap_ns"]), bound_ns=round(hr.clock["bound_ns"]))))
    return dict(traceEvents=out, displayTimeUnit="ns",
                otherData=dict(strategy=strategy, gpu=run.meta.get("gpu"), host=run.meta.get("host"),
                               clock={str(d): dict(bound_ns=b) for d, (_, b) in mappers.items()}, prefix=os.path.basename(prefix)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--out")
    ap.add_argument("--max-kernels", type=int, default=200, help="export only the first N kernels (0: all)")
    ap.add_argument("--start-ms", type=float)
    ap.add_argument("--window-ms", type=float)
    a = ap.parse_args()
    tr = export(a.prefix, a.max_kernels, a.start_ms, a.window_ms)
    path = a.out or a.prefix + ".trace.json"
    with open(path, "w") as f:
        json.dump(tr, f, separators=(",", ":"))
    print(f"{path}: {len(tr['traceEvents'])} events; open in https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
