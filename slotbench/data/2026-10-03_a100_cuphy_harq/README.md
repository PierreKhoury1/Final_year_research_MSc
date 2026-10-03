# NVIDIA cuPHY PUSCH on an A100 at a 5G completion budget: CPU-launched vs GPU-launched slot

**Question.** Does letting the GPU itself trigger NVIDIA's PUSCH slot graph on its own clock (the resident
"slot executive", `slotbench/cuphy/`) reduce the fraction of slots whose decode would be *late for HARQ*, compared
with the ordinary host launch? Earlier campaigns scored slots against the full 500 µs period; this one scores them
against tight completion budgets (150–500 µs after the slot boundary) so that the launcher's start jitter and the
host's stalls are actually inside the budget.

**Answer on this host (12 randomised triplets, 36 trials, 180 000 measured boundaries, all 36 trials passing
decoded-payload, TB CRC, CB CRC and raw-record checks):** at every budget from 200 µs to 500 µs the GPU-launched
slot misses fewer deadlines than the CPU-launched slot in 12 of 12 triplets, by 0.6–0.8 percentage points at
250–500 µs and 5.8 points at 200 µs. The GPU launcher skipped no boundary and never stalled; the CPU launcher
skipped ~28 boundaries per 5 000 and stalled (>1 ms) 2–12 times per trial, up to 13.5 ms.

## Miss rate versus completion budget (pooled over 60 000 boundaries per launcher)

A miss is a boundary that was skipped (the previous slot was still running) or whose decode completed more than
the budget after the boundary. Start error and completion use the two-point host/GPU clock mapping of each trial
(fit bounds 1.7–2.3 µs; timer tick 1.024 µs).

| Launcher | skipped | stalls >1 ms | 200 µs | 250 µs | 300 µs | 400 µs | 500 µs | p50 / p99 / p99.9 / max completion (µs) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CPU (ordinary host launch) | 329 | 354 | 6.04 % | 0.97 % | 0.85 % | 0.62 % | 0.60 % | 195.8 / 206.1 / 385.4 / 13 477 |
| CPU + GPU keep-alive | 345 | 381 | 1.02 % | 0.92 % | 0.82 % | 0.66 % | 0.64 % | 156.4 / 166.6 / 372.1 / 14 396 |
| **GPU (slot executive)** | **0** | **0** | **0.22 %** | **0.13 %** | **0.04 %** | **0.00 %** | **0.00 %** | 140.1 / 142.1 / 269.7 / 339 |

Paired per-triplet differences, mean over 12 triplets [95 % whole-triplet bootstrap] (`analysis/deadline_pairs.txt`):

| Budget | GPU − CPU | GPU − CPU+keep-alive |
|---|---:|---:|
| 200 µs | −5.82 pp [−8.71, −3.32], lower in 12/12 | −0.80 pp [−1.09, −0.51], 12/12 |
| 250 µs | −0.83 pp [−1.11, −0.58], 12/12 | −0.79 pp [−1.04, −0.53], 11/12 |
| 300 µs | −0.80 pp [−0.97, −0.65], 12/12 | −0.78 pp [−0.97, −0.57], 11/12 |
| 400 µs | −0.62 pp [−0.66, −0.59], 12/12 | −0.66 pp [−0.83, −0.50], 11/12 |
| 500 µs | −0.60 pp [−0.63, −0.56], 12/12 | −0.64 pp [−0.81, −0.49], 11/12 |

Reading the two CPU rows: with the plain host launch the A100 idles at 1095 MHz (every sampled clock in every CPU
trial), so the PHY takes 174 µs and almost every slot is late for a 200 µs budget; the keep-alive variant holds the
GPU at 1410 MHz (PHY 135 µs) and removes that effect, but **not** the host's skips and stalls, which are the ~0.6 %
floor shared by both CPU rows at any budget. The GPU launcher has neither problem: p99 start error 4.9 µs
(CPU 31–39 µs on this host), no skips, no stalls.

What the GPU launcher's residual misses are: not launch timing. Its p99.9 start error is 5 µs in every trial;
the 131 misses at 200 µs (and 26 at 300 µs) are slots whose *execution* took 230–330 µs instead of 136 µs
(`analysis/deadline_sweep/trials.csv`, columns per trial). The same execution excursions appear in the CPU+keep-alive
trials (max PHY interval 265–345 µs), so they are a property of this host/GPU (clocks were not lockable; sampled
clocks ranged 1095–1410 MHz within trials) rather than of the launcher. Above 300 µs the GPU launcher's miss rate
is exactly zero over 60 000 boundaries.

## In 5G terms

A PUSCH slot that completes after the HARQ budget is a lost transport block for that UE (here one TB of 152 code
blocks at MCS 27, 100 MHz, four layers) and a retransmission several milliseconds later. On this host, with the
GPU dedicated to the cell, the ordinary CPU launch path loses about 0.6 % of slots to host-side skips and stalls
at any budget, and 1 % at a 250 µs budget; the GPU-timed launch loses 0.13 % at 250 µs and none at ≥400 µs. The
companion campaign on a quieter 16-CPU host (`../2026-10-02_a100_cuphy_repeated/`) showed the same ordering at a
lower level: CPU 0.07–0.11 % lost at 160–200 µs, GPU 0. The improvement is in the tail, not the median: the
GPU launcher removes the host's multi-millisecond stalls and the boundary skips they cause, and tightens the slot
start from tens of microseconds to ~5 µs; it does not make the PHY itself faster.

## Setup

- Rental 53947901, NVIDIA A100-SXM4-40GB (400 W limit, clocks not lockable: `nvidia-smi -lgc` denied, recorded in
  `out/host-controls/clock_control.json`), driver 595.91.07, CUDA 13.3, Ubuntu 24.04; host 2 × AMD EPYC 7K62
  (192 logical CPUs). PHY worker pinned to CPU 1, telemetry on CPU 3, background on the rest; `SCHED_OTHER` (the
  container denies real-time scheduling) — a normal-priority CPU baseline.
- NVIDIA Aerial `aerial-cuda-accelerated-ran` commit `4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c` with the lockstep
  adapter patch (`out/applied-adapter.patch`, hashes in `out/adapter_sha256.txt`); slotbench commit `6fe2a90`.
- TC 7304 vector regenerated on the instance with the pinned recipe (MATLAB Runtime R2026a, `aerial_mcore`):
  273 PRB, 100 MHz at 30 kHz SCS, 4 layers / 4 Rx, 256-QAM, MCS 27, one TB with 152 CBs. The generator is not
  bit-reproducible (random payload): this vector's SHA-256 is `f8c3e068…79eb7` (`out/test_vector_sha256.txt`,
  schema in `out/test_vector_schema.txt`), different bytes from the earlier campaigns' vector, same configuration.
- Campaign: `cloud/onstart_cuphy_lockstep.sh` with `SB_CUPHY_CAMPAIGN=harq` → `cloud/control_cuphy_activity.sh`,
  seed 20261003, 12 triplets (each of the six variant orders twice, shuffled), 1 000 warm-up + 5 000 measured
  boundaries per trial at a 500 µs period, run-time accounting deadline 200 µs, idle GPU (no adversary). Correctness
  gates (64 slots) before the measured trials; payload and CRC checks before and after every trial.
- Raw records: `trials/*.bin` (64 B per measured boundary, format in `../../cuphy/README.md`), per-trial JSON,
  100 ms GPU telemetry (`*.telemetry.continuous.csv`), scheduler/affinity evidence, PUSCH logs, `experiment.json`
  and `schedule.tsv`. Logs of the build, vector generation and the campaign are under `logs/`.
- Cost: about $0.20 for this rental (0.48 h at $0.415/h) plus $0.12 for two aborted launches (a controller bug and
  the vector-hash guard, both fixed in `cloud/`). Instance destroyed and confirmed absent by the controller.

## Reproduce the analysis

```sh
python3 slotbench/analysis/cuphy_deadline_sweep.py slotbench/data/2026-10-03_a100_cuphy_harq/trials \
    --output-dir slotbench/data/2026-10-03_a100_cuphy_harq/analysis/deadline_sweep
python3 slotbench/analysis/cuphy_deadline_pairs.py slotbench/data/2026-10-03_a100_cuphy_harq/trials \
    --output slotbench/data/2026-10-03_a100_cuphy_harq/analysis/deadline_pairs.json
python3 slotbench/analysis/cuphy_idle_control.py slotbench/data/2026-10-03_a100_cuphy_harq/trials \
    --output-dir slotbench/data/2026-10-03_a100_cuphy_harq/analysis/control --plots
```

`analysis/control/` holds the activity-control analysis (per-variant summaries, paired contrasts with bootstrap
intervals, sampled clocks and power, `idle_activity_control.png`): GPU − CPU p99 start −33.8 µs [−45.4, −24.5],
GPU − CPU+keep-alive −27.9 µs [−36.8, −21.2], GPU − CPU+keep-alive p50 PHY +1.0 µs (same clocks).

## Limits

Fixed-vector replay of the PHY on an idle GPU: no fronthaul ingress, no per-slot descriptor changes, no MAC or
HARQ state progression, outputs not delivered anywhere; intermediate decoded outputs are not checked per slot.
One host, one GPU, one vector; the CPU baseline is normal-priority (real-time scheduling denied) and this host's CPU
path is noticeably noisier than the earlier 16-CPU host's. Clocks were not locked; the keep-alive is an activity
control, not a clock lock. Bootstrap intervals resample whole triplets and ignore calibration uncertainty.
