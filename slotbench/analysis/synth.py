"""Synthetic, format-exact run tree for demos and tests: python3 -m analysis.synth OUT_DIR [--slots N].

Writes OUT_DIR/<config>/<cell>/ exactly as the harness would (slots.bin, meta.json, calib_pre.csv,
calib_post.csv, adversary.json, telemetry.csv, env.txt, run.json, status) plus SOLO_<W>_r<rep> baselines.
The numbers are invented: latency = launch cost + queue delay + GPU execution + completion observation,
~200 us at idle, with duty- and mechanism-dependent heavy tails. The driver's overrun rule (skip to the
first boundary after t1) and a GPU clock with a rate error are simulated, so skipped slots and
queue_delay recovery through the clock fits are exercised. NOT measurement data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import sbio, stats

PERIOD_NS = 500_000
DEADLINE_NS = 500_000
SPIN_NS = 50_000
GPU_RATE = 23e-6          # GPU clock runs 23 ppm fast relative to host
GPU_OFFSET = 7_000_000_000_123

# Tail probability at D=100 per workload, magnitude range (ns) of a tail event, adversary units/s solo.
WORKLOADS = {"sgemm": (0.02, (60e3, 11e6), 85.0), "llm": (0.01, (40e3, 400e3), 38.0),
             "vision": (0.006, (20e3, 250e3), 140.0), "idle": (0.0, (0, 0), 0.0)}
# Tail-probability multiplier and adversary throughput factor per mechanism.
MECHS = {"M0": (1.0, 1.0), "M1": (0.25, 0.97), "M2": (0.02, 0.78), "M3": (0.5, 0.9), "M4": (0.3, 0.97),
         "M5": (0.25, 1.05), "M6": (0.0, 0.5)}


def gpu_of(h, t_ref):
    return (np.uint64(GPU_OFFSET) + ((np.asarray(h, np.int64) - t_ref) * (1 + GPU_RATE)).astype(np.int64)
            .astype(np.uint64))


def simulate_slots(rng, n, mech, workload, duty, t_start):
    """Records (RECORD_DTYPE) plus the true per-slot queue delay in ns."""
    p_tail, (lo, hi), _ = WORKLOADS[workload]
    p = p_tail * duty / 100.0 * MECHS[mech][0]
    streams = mech == "M4"
    launch = rng.lognormal(np.log(45e3 if streams else 4e3), 0.08, n)
    q = 5e3 + rng.exponential(1.5e3, n)
    tail = rng.random(n) < p
    if tail.any():  # Pareto-ish magnitude clipped to the workload's range
        mag = lo * (1 + rng.pareto(1.3, tail.sum()))
        q[tail] += np.minimum(mag, hi)
    exec_ = 190e3 * (1 + 0.12 * duty / 100.0 * (workload != "idle")) * rng.lognormal(0, 0.015, n)
    obs = 1e3 + rng.exponential(0.8e3, n)
    overshoot = rng.lognormal(np.log(3e3), 0.4, n)
    rec = np.zeros(n, dtype=sbio.RECORD_DTYPE)
    k = 0
    prev_t1 = t_start - PERIOD_NS
    for i in range(n):
        t_sched = t_start + k * PERIOD_NS
        t_wake = max(prev_t1, t_sched - SPIN_NS) + int(overshoot[i])
        t0 = max(t_sched, t_wake) + 150
        t_l = t0 + int(launch[i])
        g0h = t_l + int(q[i])
        t1 = g0h + int(exec_[i]) + int(obs[i])
        rec[i] = (k, t_sched, t_wake, t0, t_l, t1, 0, 0)
        prev_t1 = t1
        k = max(k + 1, (t1 - t_start) // PERIOD_NS + 1)  # overrun: first boundary after t1
    g0h = rec["t_launched"] + q.astype(np.int64)
    rec["g0"] = gpu_of(g0h, t_start)
    rec["g1"] = gpu_of(g0h + exec_.astype(np.int64), t_start)
    true_qd = (g0h - rec["t0"]).astype(np.float64)
    return rec, true_qd


def calib(rng, n, t_begin, t_start):
    t0 = t_begin + np.cumsum(rng.integers(20_000, 40_000, n)).astype(np.int64)
    w = (900 + rng.exponential(300, n) + (rng.random(n) < 0.05) * rng.exponential(20_000, n)).astype(np.int64)
    t1 = t0 + w
    g = gpu_of(t0 + (rng.random(n) * w).astype(np.int64), t_start)
    return t0, t1, g


def write_csv(path, header, rows, sep=","):
    with open(path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(sep.join(str(x) for x in r) + "\n")


def telemetry(path, rng, seconds, duty, sm_mhz=1500, reasons=0):
    rows = []
    for s in range(int(seconds)):
        busy = duty > 0 or s % 7 == 0
        util = int(min(100, duty + rng.integers(0, 5))) if duty else int(rng.integers(20, 45))
        r = reasons if reasons else (0x0 if busy else 0x1)
        rows.append((f"2026/09/30 12:{s // 60:02d}:{s % 60:02d}.{rng.integers(0, 999):03d}", 55 + duty // 10,
                     f"{sm_mhz} MHz", "7501 MHz", f"{60 + duty * 1.1:.2f} W", f"{util} %", f"0x{r:016x}"))
    write_csv(path, "timestamp, temperature.gpu, clocks.current.sm [MHz], clocks.current.memory [MHz], "
                    "power.draw [W], utilization.gpu [%], clocks_event_reasons.active",
              rows, sep=", ")


def adversary_json(workload, duty, mech, seconds, rng, prio="default"):
    base = WORKLOADS[workload][2]
    ups = base * duty / 100.0 * MECHS.get(mech, (0, 1.0))[1] * rng.normal(1, 0.01) if duty else 0.0
    d = {"workload": workload, "duty": duty, "period_ms": 100.0, "prio": prio, "seconds_total": seconds,
         "seconds_active": seconds * duty / 100.0, "units": int(ups * seconds), "units_per_s": ups,
         "gpu": "Synthetic RTX 3060", "pid": 4242, "start_time": "2026-09-30T12:00:00Z",
         "end_time": "2026-09-30T12:10:00Z"}
    if workload == "sgemm":
        d["tflops"] = ups * 2 * 4096 ** 3 / 1e12
    elif workload == "llm":
        d["gb_per_s"] = ups * 2.0
    elif workload == "vision":
        d["fps"] = ups
    return d


def make_run(run_dir, mech, workload, duty, rep=0, n_slots=20000, seed=0, *, ring_overflows=0,
             stamp_mismatches=0, crash=False, drop_stamps=0, throttle_reasons=0, extra_late=0,
             not_supported=None, lock_clocks=True, calib_samples=2000):
    """One cell directory. Fault knobs: ring_overflows/stamp_mismatches go into meta counts; crash writes
    n_records=0 and no meta.json; drop_stamps zeroes g0/g1 in that many records; extra_late forces that
    many recorded slots past the deadline; throttle_reasons sets the telemetry reason bitmask."""
    os.makedirs(run_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    t_start = int(1_000_000_000_000 + rng.integers(0, 10**12))
    rec, _ = simulate_slots(rng, n_slots, mech, workload, duty, t_start)
    if extra_late:
        idx = rng.choice(n_slots - 1, extra_late, replace=False)
        rec["t1"][idx] = rec["t0"][idx] + DEADLINE_NS + 10_000
    if drop_stamps:
        idx = rng.choice(n_slots, drop_stamps, replace=False)
        rec["g0"][idx] = 0
        rec["g1"][idx] = 0
    sbio.write_slots(os.path.join(run_dir, "slots.bin"), rec, PERIOD_NS, DEADLINE_NS, t_start,
                     n_records=0 if crash else None)
    pre = calib(rng, calib_samples, t_start - 5_000_000_000, t_start)
    post = calib(rng, calib_samples, int(rec["t1"][-1]) + 1_000_000, t_start)
    for name, (a, b, g) in (("calib_pre.csv", pre), ("calib_post.csv", post)):
        write_csv(os.path.join(run_dir, name), "t0_ns,t1_ns,gpu_ns", zip(a.tolist(), b.tolist(), g.tolist()))
    skipped = stats.slot_gaps(rec["slot"])["skipped"]
    prio = "default" if mech == "M0" else "high"
    meta = {
        "config": {"slots": n_slots, "warmup": 2000, "period_us": PERIOD_NS / 1e3, "deadline_us": DEADLINE_NS / 1e3,
                   "prio": prio, "mode": "streams" if mech == "M4" else "graph", "wait": "event",
                   "spin_us": SPIN_NS / 1e3, "core": 4, "collector_core": 5, "fifo": 99, "gpu": 0,
                   "calib_samples": calib_samples, "label": "synthetic"},
        "gpu": {"name": "Synthetic RTX 3060", "uuid": "GPU-00000000-0000-0000-0000-000000000000", "sm_count": 28,
                "cc": "8.6"},
        "counts": {"recorded": n_slots, "skipped_boundaries": skipped, "stamp_mismatches": stamp_mismatches,
                   "ring_overflows": ring_overflows},
        "clock_fit_pre": stats.fit_clock(*pre), "clock_fit_post": stats.fit_clock(*post),
        "stream_priority": {"range": [-5, 0], "used": -5 if prio == "high" else 0},
        "exit_reason": "done", "synthetic": True,
        "start_time": "2026-09-30T12:00:00Z", "end_time": "2026-09-30T12:00:10Z",
    }
    if not crash:
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
    seconds = n_slots * PERIOD_NS / 1e9 + 5
    telemetry(os.path.join(run_dir, "telemetry.csv"), rng, seconds, duty, reasons=throttle_reasons)
    adv_prio = {"M0": "default", "M1": "low", "M4": "low", "M5": "low"}.get(mech, "default")
    if workload != "idle":
        with open(os.path.join(run_dir, "adversary.json"), "w") as f:
            json.dump(adversary_json(workload, duty, mech, seconds, rng, adv_prio), f, indent=1)
    runj = {"mechanism": mech, "workload": workload, "duty": duty, "rep": rep, "synthetic": True,
            "commands": [f"bin/slot_driver --out {run_dir} --slots {n_slots} --prio {prio}"],
            "applied": {"lock_clocks": lock_clocks, "gc_mhz": 1500 if lock_clocks else None,
                        "mps": mech == "M2", "timeslice": mech == "M3"},
            "not_supported": list(not_supported or [])}
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(runj, f, indent=1)
    with open(os.path.join(run_dir, "env.txt"), "w") as f:
        f.write("synthetic environment\n")
    with open(os.path.join(run_dir, "status"), "w") as f:
        f.write("invalid:ring_overflows\n" if ring_overflows else ("invalid:crashed\n" if crash else "ok\n"))


def make_solo(run_dir, workload, rep=0, seed=0, seconds=60.0):
    os.makedirs(run_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    with open(os.path.join(run_dir, "adversary.json"), "w") as f:
        json.dump(adversary_json(workload, 100, "SOLO", seconds, rng), f, indent=1)
    telemetry(os.path.join(run_dir, "telemetry.csv"), rng, 10, 100)
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump({"mechanism": "SOLO", "workload": workload, "duty": 100, "rep": rep, "synthetic": True}, f)
    with open(os.path.join(run_dir, "status"), "w") as f:
        f.write("ok\n")


def make_tree(out_dir, config="synth", mechanisms=("M0", "M1", "M2"), workloads=("sgemm", "llm"),
              duties=(0, 50, 100), reps=1, n_slots=20000, seed=1, extra_unsupported=True):
    """Matrix like run_matrix.py expands it: SOLO_<W>, then per mechanism one <M>_idle_d0 cell and
    <M>_<W>_d<D> for D > 0. With extra_unsupported, adds an M3_idle_d0 cell whose run.json records the
    time-slice policy as NOT SUPPORTED (a partial mechanism, as on GeForce cards)."""
    root = os.path.join(out_dir, config)
    s = seed
    for w in workloads:
        make_solo(os.path.join(root, f"SOLO_{w}_r0"), w, 0, s)
        s += 1
    for m in mechanisms:
        for rep in range(reps):
            cells = [("idle", 0)] if 0 in duties else []
            cells += [(w, d) for w in workloads for d in duties if d > 0]
            for w, d in cells:
                make_run(os.path.join(root, f"{m}_{w}_d{d}_r{rep}"), m, w, d, rep, n_slots, s)
                s += 1
    if extra_unsupported:
        make_run(os.path.join(root, "M3_idle_d0_r0"), "M3", "idle", 0, 0, n_slots, s,
                 not_supported=["NOT SUPPORTED: nvidia-smi compute-policy --set-timeslice=1"])
    return root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write a synthetic slotbench run tree (fake data).")
    ap.add_argument("out_dir")
    ap.add_argument("--config", default="synth")
    ap.add_argument("--slots", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mechanisms", default="M0,M1,M2")
    ap.add_argument("--workloads", default="sgemm,llm")
    ap.add_argument("--duties", default="0,50,100")
    a = ap.parse_args(argv)
    root = make_tree(a.out_dir, a.config, a.mechanisms.split(","), a.workloads.split(","),
                     tuple(int(x) for x in a.duties.split(",")), a.reps, a.slots, a.seed)
    print(f"wrote synthetic run tree {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
