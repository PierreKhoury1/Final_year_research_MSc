#!/usr/bin/env python3
"""slottrace fault injector: inject host-side timing faults on the launcher's CPU on a schedule and log each fault
window (CLOCK_MONOTONIC_RAW ns) as ground truth for analyze.py --faults.

  faults.py --cpu 2 --plan cfs_hog:1.5,rt_hog:1.5,rt_throttle:2.5 [--gap-s 1.0] [--start-delay-s 1.0]
            [--rt-hog-prio 60] [--burst-ms 2] [--burst-every-ms 40] --log faults.jsonl

Fault types (all pinned to --cpu, each runs for its duration, separated by --gap-s quiet time):
  cfs_hog      a SCHED_OTHER busy loop (competes with a SCHED_OTHER launcher; no effect on a SCHED_FIFO one)
  rt_hog       SCHED_FIFO --rt-hog-prio busy bursts of --burst-ms every --burst-every-ms (preempts a lower-prio launcher)
  rt_throttle  a SCHED_FIFO priority-1 busy loop for the whole window: with the default RT bandwidth limit
               (sched_rt_runtime_us 950000 of 1000000) the CPU's RT tasks, the launcher included, are throttled
               for the rest of every period once RT time exceeds the limit
  irq_storm    UDP loopback flood from a process pinned to --cpu (softirq NET_RX/NET_TX load on that CPU)
Each window is logged as {"type", "t_start", "t_end", ...} when it ends. Needs CAP_SYS_NICE for SCHED_FIFO.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time


def now() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def busy_until(t_end: int) -> None:
    while now() < t_end:
        pass


def worker(kind: str, cpu: int, t_end: int, args) -> None:
    os.sched_setaffinity(0, {cpu})
    if kind == "cfs_hog":
        busy_until(t_end)
    elif kind == "rt_hog":
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(args.rt_hog_prio))
        burst, every = int(args.burst_ms * 1e6), int(args.burst_every_ms * 1e6)
        t = now()
        while t < t_end:
            busy_until(min(t + burst, t_end))
            t += every
            delta = t - now()
            if delta > 0:
                time.sleep(delta / 1e9)
    elif kind == "rt_throttle":
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(1))
        busy_until(t_end)
    elif kind == "irq_storm":
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(("127.0.0.1", 0))
        rx.setblocking(False)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dst = rx.getsockname()
        payload = b"x" * 1400
        while now() < t_end:
            for _ in range(64):
                tx.sendto(payload, dst)
            try:
                while True:
                    rx.recv(2048)
            except BlockingIOError:
                pass
    else:
        raise SystemExit(f"unknown fault {kind}")
    os._exit(0)


def run_fault(kind: str, cpu: int, dur_s: float, args) -> dict:
    t_start = now()
    t_end = t_start + int(dur_s * 1e9)
    pid = os.fork()
    if pid == 0:
        try:
            worker(kind, cpu, t_end, args)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"fault {kind}: {e}\n")
            os._exit(3)
    _, status = os.waitpid(pid, 0)
    return {"type": kind, "t_start": t_start, "t_end": now(), "cpu": cpu, "exit_status": status,
            "params": {"rt_hog_prio": args.rt_hog_prio, "burst_ms": args.burst_ms,
                       "burst_every_ms": args.burst_every_ms} if kind == "rt_hog" else {}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", type=int, required=True)
    ap.add_argument("--plan", required=True, help="comma list of type:seconds")
    ap.add_argument("--gap-s", type=float, default=1.0)
    ap.add_argument("--start-delay-s", type=float, default=1.0)
    ap.add_argument("--rt-hog-prio", type=int, default=60)
    ap.add_argument("--burst-ms", type=float, default=2.0)
    ap.add_argument("--burst-every-ms", type=float, default=40.0)
    ap.add_argument("--log", required=True)
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    plan = []
    for item in a.plan.split(","):
        k, _, d = item.partition(":")
        plan.append((k.strip(), float(d or 1.0)))
    time.sleep(a.start_delay_s)
    with open(a.log, "w") as log:
        for i, (kind, dur) in enumerate(plan):
            rec = run_fault(kind, a.cpu, dur, a)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"fault {kind}: {(rec['t_end'] - rec['t_start']) / 1e9:.2f} s on cpu {a.cpu} (status {rec['exit_status']})")
            if i < len(plan) - 1:
                time.sleep(a.gap_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
