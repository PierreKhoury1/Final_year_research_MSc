#!/usr/bin/env python3
"""slottrace analyzer: merge per-slot records, kernel scheduler/interrupt events and GPU residency-probe gaps on
one CLOCK_MONOTONIC_RAW timeline, and attribute every late slot to a cause.

  analyze.py --run RUN.json --raw RUN.bin --host-raw RUN.host.bin [--trace RUN/trace.txt] [--probe RUN.probe.bin]
             [--faults faults.jsonl] [--deadline-us F] --out-dir DIR

Per slot k (CPU mode): target T, wake (sleep returned), launch (launch call entered), ret (call returned), GPU start
and end mapped to the host axis with the run's two-point clock mapping. Components:
  wake_us   = wake - (T - spin)        sleep overshoot (timer + run-queue delay)
  host_us   = launch - T               how late the launch call was entered (0 if the spin absorbed the wake delay)
  call_us   = ret - launch             launch call duration
  queue_us  = start - ret              host return to GPU start (GPU mode: start - T)
  exec_us   = end - start              GPU execution
A slot is late if it was skipped or end - T > deadline. Its delay is attributed to the component with the largest
excess over that component's median among on-time slots, and the cause is refined from the evidence in the slot's
window on the launcher's CPU:
  host side  : rt_throttle       launcher (SCHED_FIFO) runnable but not running while lower-priority tasks or idle ran
               preempt_rt        a higher-priority RT task ran instead of the launcher
               cpu_contention    another SCHED_OTHER task ran instead of the launcher (launcher SCHED_OTHER)
               irq               hard/soft interrupt time on the launcher's CPU explains the delay
               timer_late        the launcher's wake-up itself fired late (no competitor)
               invisible         no event in the trace explains it (hypervisor steal, SMI, firmware; or untraced)
               no_trace          no kernel trace was given
  launch call: driver_call       the launch call itself was slow (with the same host refinements if it was preempted)
  GPU side   : gpu_timeslice     the residency probe saw the slot's context switched out during queue/exec
               previous_overrun  the GPU was still running the previous slot
               gpu_slow          execution longer than usual with no probe gap (in-context contention, clock state)
               gpu_queue         GPU start late with no probe gap and no previous overrun
Skipped boundaries inherit the root cause of the late slot that blocked them (cause 'carryover:<root cause>').
With --faults (ground truth from faults.py), each late slot gets the injected fault whose window overlaps it, and a
confusion matrix of injected fault vs attributed cause is written.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import struct
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slottrace  # noqa: E402

LSREC = struct.Struct("<Qq6Q")
HOSTREC = struct.Struct("<QqqqqiI")  # slot, t_target, t_wake, t_launch, t_return, cpu, flags
PROBE_HDR = struct.Struct("<8s7Q")
PROBE_GAP = struct.Struct("<QQ")

# fault type (faults.py) -> causes that count as a correct attribution
FAULT_EXPECT = {
    "cfs_hog": {"cpu_contention"},
    "rt_hog": {"preempt_rt"},
    "rt_throttle": {"rt_throttle"},
    "irq_storm": {"irq"},
    "gpu_neighbor": {"gpu_timeslice"},
    "clock_step": {"clock"},
}


def two_point(run: dict):
    pre, post = run["clock_fit_pre"], run["clock_fit_post"]
    ha = int(pre["t_ref"]) + round(Fraction(str(pre["b_ns"])))
    hb = int(post["t_ref"]) + round(Fraction(str(post["b_ns"])))
    ga, gb = int(pre["g_ref"]), int(post["g_ref"])
    rate = (gb - ga) / (hb - ha) if hb != ha else 1.0  # GPU ns per host ns

    def host_of(g: int) -> float:
        return ha + (g - ga) / rate

    return host_of


def load_records(raw: str, host_raw: str | None):
    recs = [dict(zip(("slot", "t_target", "g_target", "g_launch", "g_launch_done", "g0", "g1", "flags"), r))
            for r in LSREC.iter_unpack(Path(raw).read_bytes())]
    if host_raw:
        hs = list(HOSTREC.iter_unpack(Path(host_raw).read_bytes()))
        if len(hs) != len(recs):
            raise SystemExit(f"analyze: {len(hs)} host records vs {len(recs)} slot records")
        for r, h in zip(recs, hs):
            if h[0] != r["slot"]:
                raise SystemExit(f"analyze: host record slot {h[0]} != {r['slot']}")
            r.update(t_wake=h[2], t_launch=h[3], t_return=h[4], cpu=h[5])
    return recs


def load_probe(path: str | None):
    if not path:
        return None
    b = Path(path).read_bytes()
    magic, gap_ns, g_first, g_last, n_gaps, cap, iters, _ = PROBE_HDR.unpack_from(b, 0)
    if magic != b"SBPROBE1":
        raise SystemExit("analyze: bad probe file")
    gaps = [PROBE_GAP.unpack_from(b, PROBE_HDR.size + i * PROBE_GAP.size) for i in range(cap)]
    return {"gap_ns": gap_ns, "g_first": g_first, "g_last": g_last, "n_gaps": n_gaps, "stored": cap,
            "iterations": iters, "gaps": gaps}


def load_faults(path: str | None):
    if not path:
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out.append((int(d["t_start"]), int(d["t_end"]), d["type"]))
    out.sort()
    return out


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


class Intervals:
    """Sorted, non-overlapping-by-construction list of (start, end, info) with overlap queries."""

    def __init__(self, items):
        self.items = sorted(items, key=lambda x: x[0])
        self.starts = [x[0] for x in self.items]
        # max end prefix for overlap search
        self.maxend = []
        m = -1
        for s, e, _ in self.items:
            m = max(m, e)
            self.maxend.append(m)

    def overlapping(self, a, b):
        """items with start < b and end > a"""
        i = bisect.bisect_left(self.starts, b)
        out = []
        j = i - 1
        while j >= 0 and self.maxend[j] > a:
            s, e, info = self.items[j]
            if e > a and s < b:
                out.append(self.items[j])
            j -= 1
        return out

    def overlap_ns(self, a, b):
        return sum(min(e, b) - max(s, a) for s, e, _ in self.overlapping(a, b))


def build_kernel_view(events, tid: int, cpu: int):
    """From parsed ftrace events: launcher off-CPU intervals split into sleep / runnable wait, who ran during the
    runnable wait, interrupt and softirq busy intervals on the launcher's CPU, and the launcher's wake-up times."""
    launcher_prio = None
    off_start = off_state = None
    wakings = []
    runnable = []  # (start, end, {'by': [(comm, prio, ns)], 'state': prev_state})
    sleeps = []
    cpu_switches = []  # (t, prev_comm, prev_prio, next_comm, next_prio) on the launcher's cpu
    irq_open, soft_open = {}, {}
    irqs, softs = [], []
    for e in events:
        ev = e["ev"]
        if ev == "sched_switch":
            if e.get("cpu") == cpu:
                cpu_switches.append((e["t"], e.get("prev_comm"), e.get("prev_prio"), e.get("next_comm"), e.get("next_prio"),
                                     e.get("prev_pid"), e.get("next_pid")))
            if e.get("prev_pid") == tid:
                launcher_prio = e.get("prev_prio", launcher_prio)
                off_start, off_state = e["t"], e.get("prev_state", "?")
            elif e.get("next_pid") == tid and off_start is not None:
                launcher_prio = e.get("next_prio", launcher_prio)
                on = e["t"]
                st = off_state or "?"
                if st.startswith("R"):
                    runnable.append((off_start, on, {"state": st}))
                else:
                    # sleep until the waking event, runnable after it
                    i = bisect.bisect_right(wakings, on) - 1
                    w = wakings[i] if i >= 0 and wakings[i] >= off_start else None
                    if w is None:
                        sleeps.append((off_start, on, {"state": st}))
                    else:
                        sleeps.append((off_start, w, {"state": st}))
                        runnable.append((w, on, {"state": "woken"}))
                off_start = None
        elif ev in ("sched_waking",) and e.get("target_pid") == tid:
            wakings.append(e["t"])
        elif e.get("cpu") == cpu and ev == "irq_handler_entry":
            irq_open[e.get("irq")] = (e["t"], e.get("irq_name", str(e.get("irq"))))
        elif e.get("cpu") == cpu and ev == "irq_handler_exit":
            o = irq_open.pop(e.get("irq"), None)
            if o:
                irqs.append((o[0], e["t"], {"name": o[1]}))
        elif e.get("cpu") == cpu and ev == "softirq_entry":
            soft_open[e.get("vec")] = (e["t"], e.get("action", str(e.get("vec"))))
        elif e.get("cpu") == cpu and ev == "softirq_exit":
            o = soft_open.pop(e.get("vec"), None)
            if o:
                softs.append((o[0], e["t"], {"name": o[1]}))
    # who ran during each runnable interval
    sw_t = [s[0] for s in cpu_switches]
    for s, e, info in runnable:
        by = Counter()
        i = bisect.bisect_right(sw_t, s) - 1
        # task running at s is next_* of the last switch before s
        cur = (cpu_switches[i][3], cpu_switches[i][4], cpu_switches[i][6]) if i >= 0 else ("?", None, None)
        t = s
        j = i + 1
        while j < len(cpu_switches) and cpu_switches[j][0] < e:
            by[(cur[0], cur[1])] += cpu_switches[j][0] - t
            t = cpu_switches[j][0]
            cur = (cpu_switches[j][3], cpu_switches[j][4], cpu_switches[j][6])
            j += 1
        by[(cur[0], cur[1])] += e - t
        info["by"] = [(c, p, ns) for (c, p), ns in by.most_common()]
    return {"launcher_prio": launcher_prio, "runnable": Intervals(runnable), "sleeps": Intervals(sleeps),
            "irq": Intervals(irqs), "softirq": Intervals(softs), "wakings": sorted(wakings)}


def classify_host(kv, a, b, launcher_prio, excess_ns):
    """Explain a host-side delay in window [a, b] on the launcher's CPU."""
    if kv is None:
        return "no_trace", {}
    rq = kv["runnable"].overlapping(a, b)
    run_ns = sum(min(e, b) - max(s, a) for s, e, _ in rq)
    irq_ns = kv["irq"].overlap_ns(a, b) + kv["softirq"].overlap_ns(a, b)
    detail = {"runnable_wait_us": run_ns / 1e3, "irq_us": irq_ns / 1e3}
    if run_ns > 0.3 * excess_ns and run_ns >= irq_ns:
        by = Counter()
        for s, e, info in rq:
            for comm, prio, ns in info.get("by", []):
                by[(comm, prio)] += ns
        (comm, prio), _ = by.most_common(1)[0]
        detail["ran_instead"] = comm
        detail["ran_instead_prio"] = prio
        lp = launcher_prio if launcher_prio is not None else 120
        if prio is None:
            return "invisible", detail
        if lp < 100:  # launcher is RT (ftrace prio = 99 - rtprio)
            if prio < lp:
                return "preempt_rt", detail
            return "rt_throttle", detail  # RT launcher runnable while lower-priority tasks or idle ran
        return "cpu_contention", detail
    if irq_ns > 0.3 * excess_ns:
        names = Counter(info["name"] for _, _, info in kv["irq"].overlapping(a, b) + kv["softirq"].overlapping(a, b))
        detail["irq_names"] = dict(names.most_common(3))
        return "irq", detail
    return "invisible", detail


def analyze(run, recs, kview, probe, faults, deadline_us):
    host_of = two_point(run)
    mode = run.get("mode", "cpu")
    spin = float(run.get("spin_us", 0)) * 1e3
    deadline = (deadline_us if deadline_us is not None else float(run.get("deadline_us", 500))) * 1e3
    lp = kview["launcher_prio"] if kview else None
    probe_iv = None
    if probe:
        probe_iv = Intervals([(host_of(a), host_of(b), {}) for a, b in probe["gaps"]])
    rows = []
    for r in recs:
        T = r["t_target"]
        row = {"slot": r["slot"], "t_target": T, "skipped": bool(r["flags"] & 1), "flags": r["flags"]}
        if r["flags"] & 1 or not r["g0"] or not r["g1"]:
            row["late"] = True
            rows.append(row)
            continue
        start, end = host_of(r["g0"]), host_of(r["g1"])
        row["latency_us"] = (end - T) / 1e3
        row["exec_us"] = (end - start) / 1e3
        if mode == "cpu" and r.get("t_launch"):
            row["wake_us"] = (r["t_wake"] - (T - spin)) / 1e3
            row["host_us"] = (r["t_launch"] - T) / 1e3
            row["call_us"] = (r["t_return"] - r["t_launch"]) / 1e3
            row["queue_us"] = (start - r["t_return"]) / 1e3
        else:
            row["queue_us"] = (start - T) / 1e3
        row["late"] = (end - T) > deadline
        row["_win"] = (T - spin, start, end, r.get("t_wake"), r.get("t_launch"), r.get("t_return"))
        rows.append(row)
    comps = [c for c in ("host_us", "call_us", "queue_us", "exec_us") if any(c in x for x in rows)]
    base = {c: median([x[c] for x in rows if not x["late"] and c in x]) for c in comps}
    prev_end = None
    last_root = None
    for i, row in enumerate(rows):
        if row["skipped"]:
            row["cause"] = f"carryover:{last_root or 'unknown'}"
            continue
        a_spin, start, end, tw, tl, tr = row["_win"]
        if not row["late"]:
            prev_end = end
            continue
        excess = {c: row[c] - base[c] for c in comps if c in row}
        comp = max(excess, key=lambda c: excess[c])
        ex_ns = max(excess[comp], 0.0) * 1e3
        detail = {"component": comp, "excess_us": excess[comp]}
        if comp in ("host_us", "call_us"):
            a, b = (a_spin, tl) if comp == "host_us" else (tl, tr)
            cause, d = classify_host(kview, a, b, lp, ex_ns)
            if comp == "call_us" and cause == "invisible":
                cause = "driver_call"
            if comp == "host_us" and cause == "invisible" and kview is not None:
                # the wake-up itself late with nothing competing: timer path
                j = bisect.bisect_left(kview["wakings"], a_spin)
                if j < len(kview["wakings"]) and kview["wakings"][j] - a_spin > 0.5 * ex_ns:
                    cause = "timer_late"
            detail.update(d)
        else:
            a, b = ((tr if tr else row["t_target"]), start) if comp == "queue_us" else (start, end)
            gap = probe_iv.overlap_ns(a, b) if probe_iv else 0
            if gap > 0.3 * ex_ns and gap > 0:
                cause = "gpu_timeslice"
                detail["probe_gap_us"] = gap / 1e3
            elif comp == "queue_us" and prev_end is not None and prev_end > a:
                cause = "previous_overrun"
            else:
                cause = "gpu_slow" if comp == "exec_us" else "gpu_queue"
        row["cause"] = cause
        row["detail"] = detail
        last_root = cause
        prev_end = end
    # ground truth
    for row in rows:
        row.pop("_win", None)
        if faults and row.get("late"):
            T = row["t_target"]
            hit = [f for f in faults if f[0] < T + 2 * deadline and f[1] > T - 2 * spin]
            row["fault"] = hit[0][2] if hit else "none"
    return rows, base


def summarize(rows, faults, base):
    late = [r for r in rows if r.get("late")]
    causes = Counter(r.get("cause", "?").split(":")[-1] if r.get("cause", "").startswith("carryover") else r.get("cause", "?")
                     for r in late)
    direct = Counter(r.get("cause", "?") for r in late if not r.get("skipped"))
    out = {"slots": len(rows), "late": len(late), "skipped": sum(1 for r in rows if r.get("skipped")),
           "late_by_root_cause": dict(causes.most_common()), "late_slots_direct_cause": dict(direct.most_common()),
           "component_medians_on_time_us": base}
    if faults:
        conf = defaultdict(Counter)
        for r in late:
            c = r.get("cause", "?")
            c = c.split(":", 1)[1] if c.startswith("carryover:") else c
            conf[r.get("fault", "none")][c] += 1
        out["confusion"] = {k: dict(v) for k, v in conf.items()}
        per = {}
        for ftype, expect in FAULT_EXPECT.items():
            if ftype not in conf:
                continue
            tot = sum(conf[ftype].values())
            hit = sum(n for c, n in conf[ftype].items() if c in expect)
            pred = sum(conf[f][c] for f in conf for c in expect)
            per[ftype] = {"late_slots": tot, "correct": hit, "recall": hit / tot if tot else None,
                          "precision": hit / pred if pred else None}
        out["per_fault"] = per
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--host-raw")
    ap.add_argument("--trace")
    ap.add_argument("--probe")
    ap.add_argument("--faults")
    ap.add_argument("--deadline-us", type=float)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    run = json.loads(Path(a.run).read_text())
    recs = load_records(a.raw, a.host_raw)
    kview = None
    if a.trace:
        evs = slottrace.load_events(a.trace)
        tid, cpu = int(run["launcher_tid"]), int(run.get("launcher_cpu", -1))
        cpus = Counter(r.get("cpu") for r in recs if r.get("cpu", -1) >= 0)
        if cpus:
            cpu = cpus.most_common(1)[0][0]
        kview = build_kernel_view(evs, tid, cpu)
    probe = load_probe(a.probe)
    faults = load_faults(a.faults)
    rows, base = analyze(run, recs, kview, probe, faults, a.deadline_us)
    summ = summarize(rows, faults, base)
    summ["inputs"] = {k: getattr(a, k) for k in ("run", "raw", "host_raw", "trace", "probe", "faults")}
    summ["launcher_prio_ftrace"] = kview["launcher_prio"] if kview else None
    if a.trace:
        meta_p = Path(a.trace).with_suffix(".meta.json")
        if meta_p.exists():
            m = json.loads(meta_p.read_text())
            summ["trace_meta"] = {k: m.get(k) for k in ("events_lost", "resolution", "cpu_time_delta_ms", "kernel")}
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    keys = ["slot", "t_target", "skipped", "late", "latency_us", "wake_us", "host_us", "call_us", "queue_us", "exec_us",
            "cause", "fault", "detail"]
    with open(od / "slots.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = dict(r)
            if "detail" in rr:
                rr["detail"] = json.dumps(rr["detail"], default=str)
            w.writerow(rr)
    (od / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps({k: summ[k] for k in ("slots", "late", "skipped", "late_by_root_cause")}, default=str))
    if "per_fault" in summ:
        print(json.dumps(summ["per_fault"]))
        print(json.dumps(summ["confusion"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
