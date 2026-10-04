# slottrace: why was this 5G slot late?

A slot-level tracer for GPU-based 5G physical layers. It puts the slot's own timestamps, the host kernel's scheduler
and interrupt events, and GPU residency on one timeline, and attributes every late slot to a cause.

| Layer | What it records | Where |
|---|---|---|
| Slot | target, wake, launch call enter/return, CPU (host); slot start/end (GPU `%globaltimer`) | `--host-raw` and `--raw` from `bin/lockstep_driver` (and `bin/mock_slots`) |
| Kernel | `sched_switch`, `sched_waking/wakeup`, `sched_migrate_task`, hard/soft IRQ entry/exit, `hrtimer_expire_entry` on the launcher's CPU | `trace/slottrace.py` (ftrace instance, `trace_clock mono_raw`, `trace-cmd extract`, ns timestamps) |
| GPU residency | intervals in which a resident probe thread did not run (context switched out by another process) | `--probe 1 --probe-out F` on `bin/lockstep_driver` (`trace/probe.cuh`) |
| Clock | host↔GPU two-point mapping from the run's pre/post clock fits; per-CPU steal/IRQ time over the capture | driver JSON; `trace.meta.json` |
| Ground truth | injected fault windows | `trace/faults.py` |

All host timestamps (slot records and kernel events) are `CLOCK_MONOTONIC_RAW`, so they line up without conversion;
GPU times are mapped onto that axis with the run's clock fits (±~2 µs on A100).

## Use

```sh
make bin/lockstep_driver                      # or bin/mock_slots on a machine without a GPU
sudo python3 trace/slottrace.py run --cpus 2 --out run/trace -- \
    bin/lockstep_driver --mode cpu --core 2 --fifo 50 --slots 20000 --deadline-us 300 \
      --out run/run.json --raw run/run.bin --host-raw run/run.host.bin --probe 1 --probe-out run/run.probe.bin
python3 trace/analyze.py --run run/run.json --raw run/run.bin --host-raw run/run.host.bin \
    --trace run/trace.txt --probe run/run.probe.bin --out-dir run/analysis
# with injected faults (ground truth):
python3 trace/faults.py --cpu 2 --plan cfs_hog:2,rt_hog:2 --gap-s 1.5 --log run/faults.jsonl &
...  then add --faults run/faults.jsonl to analyze.py
```

`analysis/slots.csv` has one row per slot (components, cause, evidence); `analysis/summary.json` has late slots by root
cause and, with `--faults`, the confusion matrix and per-fault recall/precision. The kernel layer needs root (tracefs)
and, for real-time cases, `CAP_SYS_NICE`; vast.ai containers allow neither, so GPU runs need a VM or bare-metal host.

## Attribution

A slot is late if it was skipped or completed more than the deadline after its boundary. Its delay is split into
wake (sleep overshoot), host (launch call entered late), call (launch call duration), queue (call return to GPU start)
and exec (GPU execution); the component with the largest excess over its on-time median is refined with the evidence
in that window: `preempt_rt`, `rt_throttle`, `cpu_contention`, `irq`, `timer_late`, `invisible` (nothing in the guest
kernel explains it), `driver_call`, `gpu_timeslice`, `previous_overrun`, `gpu_slow`, `gpu_queue`. Skipped boundaries
inherit the root cause of the slot that blocked them (`carryover:<cause>`). See the docstring of `trace/analyze.py`.

## Validation so far (2026-10-04, host layer only, `bin/mock_slots` on a 4-vCPU Firecracker VM, kernel 6.18)

The mock runs the driver's CPU-mode loop with modelled GPU times; faults are injected on the launcher's CPU and scored
against their logged windows. Numbers are late slots inside each fault window and how they were attributed.

| Launcher | Injected fault | Late slots | Attributed | Recall (precision) |
|---|---|---:|---|---|
| SCHED_OTHER | CPU hog (`cfs_hog`) | 1 370 | 1 349 `cpu_contention` | 98.5 % |
| SCHED_FIFO 50 | higher-priority RT bursts (`rt_hog`, FIFO 60, 2 ms every 40 ms) | 264 | 248 `preempt_rt` | 93.9 % (100 %) |
| SCHED_FIFO 50 | RT throttling attempt (`rt_throttle`, FIFO 1 busy loop) | 218 | 1 `rt_throttle`, 154 `invisible`, 63 `timer_late` | 0.5 % |
| SCHED_OTHER | UDP loopback flood (`irq_storm`) | 1 608 | 1 598 `cpu_contention` | — (see below) |
| none | baseline at SCHED_FIFO | ~6–16 per second | `invisible` / `timer_late` | — |

What these show, and what they do not:
- Scheduler-visible causes are identified reliably: CPU contention and preemption by a higher-priority RT task.
- **RT throttling is not identified yet.** The kernel logged `sched: RT throttling activated` during the window, but in
  the late slots the launcher was never switched out: it woke on time, was switched in, and then made no progress for
  hundreds of µs while the guest kernel showed it running. The delay happened outside the guest's view (hypervisor
  steal of the busy vCPU is the most likely explanation; the VM's steal counter rose by ~1 % under a 100 % busy loop).
  Kernel 6.12+ also replaced classic RT throttling with a fair-server mechanism. The injector therefore did not
  produce the classic throttling the forum system (RHEL 9, kernel 5.14) has; this must be re-tested on a pre-6.12
  kernel on bare metal before any throttling claim.
- The IRQ injector is not a clean IRQ fault: its own sender process runs on the launcher's CPU, so `cpu_contention`
  is the correct answer for most of its late slots. A clean test needs device-interrupt affinity on bare metal.
- On a VM, `invisible` is common and correct at the guest level: it means "outside the guest OS". It cannot say which
  host-side mechanism took the time.
- The GPU layer (residency probe, launch call, GPU queue/exec) is implemented in the driver but not yet validated on a
  GPU; it needs a GPU host with root for the kernel layer.

## Files

`slottrace.h` record layouts · `probe.cuh` residency probe · `mock_slots.cpp` CPU-only launcher · `slottrace.py`
ftrace capture and parser · `analyze.py` merge and attribution · `faults.py` fault injector · `tests/` unit tests
(`python3 -m pytest trace/tests`).
