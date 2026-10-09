# tickbound findings, 8-9 October 2026 (A100, H100, 8x A100 on one host)

tickbound measures GPU work from inside the kernel: each warp reads the SM cycle counter (`clock64`) and the GPU timer
(`%globaltimer`) at checkpoints, and each run syncs the GPU timer to the host clock (`CLOCK_MONOTONIC_RAW`) before
and after the work, so every checkpoint lands on the host time axis **with a stated error bound**. The SASS of the
binary that measured is checked so that the checkpoints sit where the source says. The tool itself lives in a
separate repository (not yet in git); this repository holds the data, figures, run scripts and this write-up.

Everything below was measured on rented vast.ai machines (8 Oct: $1.07, 9 Oct: $1.59).

| date | hardware | toolkit | what ran | data |
|---|---|---|---|---|
| 8 Oct | 1x A100-SXM4-40GB | CUDA 12.6 | ktrace profile, FFMA x blocks sweep, instr / co-tenant / MPS | `tickbound_data/2026-10-08_a100_cuda12.6` |
| 8 Oct | 1x H100 80GB HBM3 | CUDA 12.2 and 12.6 | same | `tickbound_data/2026-10-08_h100_cuda12.2`, `..._h100_cuda12.6` |
| 9 Oct | 8x A100-PCIE-40GB, one host | CUDA 12.6 and 12.2 | 8-GPU ktrace on one timeline, reorder E1-E7, 30 hand-edited SASS kernels | `tickbound_data/2026-10-09_a100x8_pcie_cuda12.6+12.2` |
| 9 Oct | 1x H100 80GB HBM3 (another host) | CUDA 12.6 and 12.2 | reorder E1-E7, ktrace profile (timer census) | `tickbound_data/2026-10-09_h100_cuda12.6+12.2` |

Raw traces (`*.bin`), timeline viewer pages (`*.trace.json`, `timeline.html`) and uncompressed SASS listings are kept
locally and not committed (size); every analysis result, report, log and gzipped SASS listing is here.

## 1. Eight GPUs on one timeline, instruction-phase level

![8 GPUs on one host clock](figures_2026-10-09/multi_gpu_instruction_timeline.png)

Eight A100s in one host, one process per GPU, each with its own clock sync before and after. All processes started
together and launched the same ktrace kernel (108 blocks x 256 threads, a 256-FFMA chain between checkpoints) on a
shared 2 ms grid of the host clock. The figure shows one grid instant: every block's warp 0, phase by phase (two
dependent loads, barrier, FFMA chain, barrier, store + GPU fence, atomic ticket, flag store), all eight GPUs on the
same host axis, each with its strict placement bound.

Over all 40 grid instants at which all eight GPUs launched:

- the eight host processes made their launch calls within **30 ns** of each other (median; max 41 ns);
- the kernels started 7.6-8.3 µs after the launch call (per-GPU median), and the kernel starts of the eight GPUs
  spread **0.78 µs** (median; range 0.51-2.70 µs);
- each GPU's placement bound on the host axis is **±0.85-1.20 µs** (strict clock bound 0.79-1.03 µs plus the
  checkpoint's own placement, ≤ 0.17 µs);
- so the start order of two GPUs is proved only when they are more than **1.7-2.4 µs** apart: that held for **11 of
  1120** GPU pairs (in 2 of the 40 instants, where one GPU lagged);
- ticket-order checks inside the kernels: 0 violations; clock-fit widening for the 108-block launches: at most 2.6 ns.

What this shows: per-GPU instruction-phase timelines of several GPUs can be put on one host clock with a stated,
checked error on every point; cross-GPU ordering through the host clock is limited to gaps above ~2 µs.

Figure data: `figures_2026-10-09/multi_gpu_instruction_timeline.json`, all instants: `multi_gpu_all_instants.json`.
Method note: the start barrier and the shared launch grid were a run-time patch to the probe for this run
(`tickbound_runs/2026-10-09/patches/gputrace_start_barrier_launch_grid.diff`).

## 2. How exact is each timestamp: strict bounds

The clock sync gives a set of feasible (rate, offset) lines between GPU timer and host clock. The bound reported by
the fit so far (the "chord") was the half-width of the feasible offsets at the fitted rate. The **strict** bound is
the half-extent of the projection of the whole feasible set at the event's time, taken here as its maximum over the
gap between the two sync windows (where the measured work happens). The strict bound is the one to quote.

| data | runs | reported (chord) ± ns | strict ± ns | strict / chord |
|---|---|---|---|---|
| 2026-10-08_a100_cuda12.6 instruction / co-tenant / MPS | 4 | 527-540 | 532-1399 | 1.00-2.59 |
| 2026-10-08_a100_cuda12.6 kernel timing (ktrace) | 7 | 731-835 | 814-860 | 1.00-1.13 |
| 2026-10-08_h100_cuda12.2 instruction / co-tenant / MPS | 4 | 638-717 | 664-820 | 1.00-1.29 |
| 2026-10-08_h100_cuda12.2 kernel timing (ktrace) | 7 | 645-714 | 680-716 | 1.00-1.06 |
| 2026-10-08_h100_cuda12.6 kernel timing (ktrace) | 7 | 641-715 | 682-732 | 1.00-1.09 |
| 2026-10-09_a100x8_pcie_cuda12.6+12.2 8-GPU kernel timing | 8 | 749-787 | 791-1029 | 1.04-1.35 |
| 2026-10-09_a100x8_pcie_cuda12.6+12.2 reorder E1-E6 | 13 | 564-858 | 742-990 | 1.00-1.72 |
| 2026-10-09_h100_cuda12.6+12.2 kernel timing (ktrace) | 3 | 542-560 | 552-599 | 1.00-1.08 |
| 2026-10-09_h100_cuda12.6+12.2 reorder E1-E6 | 12 | 545-652 | 570-712 | 1.01-1.29 |

All runs: `tickbound_data/strict_bounds_2026-10-09.json`. For kernel-timing runs the strict bound is 1.00-1.35x the
reported one (A100 ±0.79-1.03 µs, H100 ±0.55-0.73 µs). Runs with a long or lopsided gap between the sync windows
(the 8 Oct instr / MPS runs, the 9 Oct A100 load-use runs) widen up to 2.6x (worst ±1.40 µs).

Inside one GPU, between SMs, a checkpoint is placed far more tightly, per block, from the SM's own cycle counter:
on H100 (64 ns timer steps) ±17-21 ns median and ±18-26 ns for the worst block in kernels up to ~10 µs (±31-55 /
±50-96 ns in ~29 µs kernels); on A100 (1024 ns timer steps) ±13-246 ns median and ±32-374 ns for the worst block,
depending on the run (per-run values in each `*.analysis.json`).

## 3. Kernel shapes

Same kernel, three grid shapes (1 block, 1 block per SM, "4 per SM") x four FFMA chain lengths, plus the 9 Oct 8-GPU
run (1 block vs 1 per SM, 40 launches each per GPU). Block size was 256 threads in every run (never varied), and no
memory-heavy kernel shape was tried.

| GPU (toolkit) | shape | FFMA | launches | launch -> first warp µs | block-start spread µs | kernel span µs | cycles / FFMA | last exit -> sync return µs | clock-fit widening ns |
|---|---|---|---|---|---|---|---|---|---|
| A100-SXM4 (12.6) | 1 block | 64 | 5 | 5.30 | 0.00 | 1.60 | 4.06 | 6.26 | 0.0 |
| A100-SXM4 (12.6) | 1 per SM (108) | 64 | 5 | 5.22 | 0.48 | 2.70 | 4.05 | 7.20 | 0.0 |
| A100-SXM4 (12.6) | 4 per SM (432) | 64 | 5 | 5.12 | 7.43 | 13.00 | 4.36 | 7.19 | 8.0 |
| A100-SXM4 (12.6) | 1 block | 256 | 5 | 5.49 | 0.00 | 2.40 | 4.01 | 6.30 | 0.0 |
| A100-SXM4 (12.6) | 1 per SM (108) | 256 | 5 | 5.12 | 0.59 | 3.55 | 4.01 | 7.22 | 0.0 |
| A100-SXM4 (12.6) | 4 per SM (432) | 256 | 5 | 5.32 | 7.64 | 14.07 | 5.75 | 7.37 | 21.0 |
| A100-SXM4 (12.6) | 1 block | 1024 | 5 | 5.28 | 0.00 | 5.17 | 4.00 | 6.28 | 0.0 |
| A100-SXM4 (12.6) | 1 per SM (108) | 1024 | 5 | 5.24 | 0.46 | 6.34 | 4.00 | 7.06 | 0.0 |
| A100-SXM4 (12.6) | 4 per SM (432) | 1024 | 5 | 5.27 | 13.44 | 20.48 | 6.21 | 7.44 | 33.5 |
| A100-SXM4 (12.6) | 1 block | 4096 | 5 | 6.64 | 0.00 | 16.46 | 4.00 | 6.70 | 0.0 |
| A100-SXM4 (12.6) | 1 per SM (108) | 4096 | 5 | 6.80 | 0.56 | 17.61 | 4.00 | 8.04 | 10.6 |
| A100-SXM4 (12.6) | 4 per SM (432) | 4096 | 5 | 6.82 | 47.05 | 64.50 | 6.06 | 7.46 | 57.2 |
| H100 (12.6) | 1 block | 64 | 5 | 5.05 | 0.00 | 1.48 | 4.11 | 4.67 | 0.0 |
| H100 (12.6) | 1 per SM (132) | 64 | 5 | 5.07 | 0.21 | 1.94 | 4.12 | 4.99 | 0.0 |
| H100 (12.6) | 4 per SM (528) | 64 | 5 | 4.78 | 5.59 | 8.74 | 4.44 | 4.84 | 0.0 |
| H100 (12.6) | 1 block | 256 | 5 | 5.52 | 0.00 | 1.88 | 4.03 | 4.66 | 0.0 |
| H100 (12.6) | 1 per SM (132) | 256 | 5 | 5.26 | 0.20 | 2.30 | 4.03 | 4.96 | 0.0 |
| H100 (12.6) | 4 per SM (528) | 256 | 5 | 4.93 | 5.59 | 9.43 | 4.12 | 4.86 | 0.0 |
| H100 (12.6) | 1 block | 1024 | 5 | 5.24 | 0.00 | 3.42 | 4.01 | 4.66 | 0.0 |
| H100 (12.6) | 1 per SM (132) | 1024 | 5 | 5.14 | 0.20 | 3.89 | 4.01 | 4.93 | 0.0 |
| H100 (12.6) | 4 per SM (528) | 1024 | 5 | 4.74 | 6.94 | 12.10 | 4.07 | 4.86 | 5.1 |
| H100 (12.6) | 1 block | 4096 | 5 | 5.30 | 0.00 | 9.64 | 4.00 | 4.68 | 0.0 |
| H100 (12.6) | 1 per SM (132) | 4096 | 5 | 4.98 | 0.24 | 10.14 | 4.00 | 4.91 | 6.2 |
| H100 (12.6) | 4 per SM (528) | 4096 | 5 | 4.79 | 18.02 | 28.64 | 4.02 | 4.75 | 45.7 |
| 8x A100-PCIe (12.6), median of 8 GPUs | 1 block | 256 | 40 per GPU | 7.85 | 0.00 | 3.77 | 4.01 | 8.17 | 0.0 (max) |
| 8x A100-PCIe (12.6), median of 8 GPUs | 1 per SM (108) | 256 | 40 per GPU | 7.98 | 0.66 | 4.79 | 4.01 | 9.31 | 0.0 (max) |

- Launch to first warp does not depend on the shape: A100 5.1-6.8 µs, H100 4.6-5.5 µs. After 50 ms idle it rises to
  9.8 µs (A100) and 24-32 µs (H100). On the 9 Oct PCIe host (8 processes launching together) it was 7.6-8.3 µs.
- A full wave starts within 0.4-0.6 µs on A100 (108 blocks) and 0.2-0.3 µs on H100 (132 blocks).
- "4 blocks per SM" never had four blocks resident: from the trace, at most **3 blocks were resident per SM** at any
  time, on every SM of both GPUs; the fourth ran as a tail wave, which is why those spans are ~4x the 1-per-SM spans.
- A dependent FFMA costs 4.0 cycles alone; with 3 resident blocks the A100 slows to 5.8-6.2 cycles (its FP32 pipe
  saturates), the H100 stays at 4.0-4.5.
- Last warp exit to the host's sync return: 6.3-8.0 µs (A100), 4.6-5.0 µs (H100), for every shape (9 Oct PCIe host:
  8.2-9.3 µs).
- Kernels under ~10 µs need no clock-fit widening (the SMs' cycle-counter lines agree exactly); 28-64 µs kernels
  need 45-64 ns, so their within-GPU placement is empirical rather than hard.

## 4. Instruction-order experiments (E1-E6) and two compilers (E7)

Each experiment times variants that hold the same instructions in a different order; a variant counts only where
the reorder SASS gate shows the intended order in the binary that measured. Same source, two toolkits (CUDA 12.6
and 12.2) on A100 (sm_80) and H100 (sm_90).

| | A100 12.6 | A100 12.2 | H100 12.6 | H100 12.2 |
|---|---|---|---|---|
| gate (295 builds) | 286 pass, 9 fail | 278 pass, 17 fail | 295 pass | 287 pass, 8 fail |

Gate failures are compiler choices, and the analysis drops those builds:
- **CUDA 12.2 emits `BAR.SYNC`, 12.6 emits `BAR.SYNC.DEFER_BLOCKING`** for the same `__syncthreads()` (616 of 616
  barriers in the binary, on sm_80 and sm_90, both disassemblers agree): the barrier experiment is not measured for 12.2.
- On sm_80 (both toolkits), with two loads and 96 or more FFMA, ptxas moves the loaded value into a uniform register
  (`R2UR`) in the middle of the chain, so the intended overlap does not exist: 9 builds dropped.
- On sm_90, one PTX fence is `MEMBAR.ALL.CTA` directly followed by the scoped `MEMBAR` (then `ERRBAR`, `CGAERRBAR`,
  `CCTL.IVALL`). The gate first refused this; it now accepts the adjacent pair as one fence
  (`tickbound_runs/2026-10-09/patches/sass_reorder_sm90_fence_pair.diff`, with a regression test), and the H100 data
  was re-analysed with it.

Measured (cycles unless stated; CUDA 12.6; full one-line results per run in `tickbound_data/summary_2026-10-09.json`):

- **E1 load to use:** latency L1 / L2 / DRAM: A100 50 / 353 / 614, H100 49 / 370 / 771. Independent FFMA hide a load
  up to the predicted break-even N* = (L - d) / 4 (A100 L2: 73 measured vs 81 predicted; H100 L2: 84.8 vs 84.7);
  the largest saving from hoisting a DRAM load: 595 (A100), 634 (H100). Open: on H100 the two estimates of L
  disagree (L2 370 vs 428, DRAM 771 vs 692 cycles), and the A100 L2 saving (198 at most) is below the model's
  L - d = 323.
- **E2 write-after-read:** reusing a just-stored register instead of a fresh one costs 0-10 cycles in most cases;
  STG.128 with 32 warps: +8 (A100), +46 (H100); one A100 case (STG.32, 32 warps) measured -28, not yet explained.
- **E3 barrier:** M register-only FFMA placed after `__syncthreads()` (`BAR.SYNC.DEFER_BLOCKING`) cost +0 cycles:
  they run while the warp waits (the clock read after the barrier issued during the wait in 100% of samples). Work
  that touches shared memory after the barrier costs about its full length (+296 at M = 64, +1066 at M = 256,
  against 4M = 256 / 1024). Moving the late warp's M FFMA past the barrier releases it earlier by about 4M (A100
  +254 / +1025, H100 +224 / +1030 at M = 64 / 256). Same on A100 and H100.
- **E4 fence cost:** GPU scope / system scope / acq_rel: A100 410 / 2277 / 413, H100 682 / 1252 / 684. Giving the
  preceding store up to 1024 FFMA to drain before the fence saves ≤ 8 cycles: the fence's cost does not shrink.
- **E5 FFMA throughput:** cycles per FFMA per warp follow max(4 / ILP, p x warps per sub-partition) with p = 2 on
  A100 and p = 1 on H100 (A100 reaches 15.8 at 32 warps, H100 8.0).
- **E6 dependent chains:** measured cycles per dependent link: 4 for the integer / FP32 ALU ops tested (as ptxas
  encodes them), 8 for IABS, 6 (A100) / 8 (H100) for DADD and DFMA, and 13 (A100) / 12 (H100) for IMAD.WIDE, where
  ptxas encodes 10 / 8: the hardware adds 3-4 cycles beyond the encoding. Independent FP32 ops issue every ~2 cycles
  on A100 and every cycle on H100; DADD / DFMA every 4.0 (A100) and 2.2 (H100) cycles.
- **E7 (12.2 vs 12.6):** apart from the barrier form above, the two toolkits measure the same within a few cycles.
  One open point: the A100 acq_rel fence measured 413 (12.6) vs 565 (12.2) with identical fence SASS, so it is not a
  compiler effect; it needs a repeat.

## 5. Hand-edited SASS: does the legality checker match the hardware?

Thirty edits to real sm_80 SASS (CuAssembler), each run 200 times on an A100 against the unedited kernel's output:
legal moves of a load ("slide", k slots), deliberate violations (moved past its consumer, cleared scoreboard wait,
stall counts below the latency), and probes of rules the checker applies conservatively.

| edit | group | checker | output vs reference (200 launches) | cycles p50 |
|---|---|---|---|---|
| `slide_k00` | slide | legal | correct (0/200 wrong) | 267 |
| `slide_k04` | slide | legal | correct (0/200 wrong) | 266 |
| `slide_k08` | slide | legal | correct (0/200 wrong) | 268 |
| `slide_k12` | slide | legal | correct (0/200 wrong) | 284 |
| `slide_k16` | slide | legal | correct (0/200 wrong) | 300 |
| `slide_k20` | slide | legal | correct (0/200 wrong) | 316 |
| `slide_k24` | slide | legal | correct (0/200 wrong) | 332 |
| `slide_k28` | slide | legal | correct (0/200 wrong) | 348 |
| `slide_k32` | slide | legal | correct (0/200 wrong) | 365 |
| `slide_k36` | slide | legal | correct (0/200 wrong) | 380 |
| `slide_k40` | slide | legal | correct (0/200 wrong) | 396 |
| `slide_k44` | slide | legal | correct (0/200 wrong) | 412 |
| `slide_k48` | slide | legal | correct (0/200 wrong) | 428 |
| `slide_k52` | slide | legal | correct (0/200 wrong) | 443 |
| `slide_k56` | slide | legal | correct (0/200 wrong) | 459 |
| `slide_k60` | slide | legal | correct (0/200 wrong) | 475 |
| `slide_k64` | slide | legal | correct (0/200 wrong) | 491 |
| `viol_load_past_consumer` | violation | illegal | WRONG (200/200 wrong) | 266 |
| `viol_consumer_above_load` | violation | illegal | WRONG (200/200 wrong) | 266 |
| `viol_wait_cleared` | violation | illegal | WRONG (1/200 wrong) | 267 |
| `viol_stall_ffma_3` | violation | illegal | WRONG (200/200 wrong) | 103 |
| `viol_stall_ffma_2` | violation | illegal | WRONG (200/200 wrong) | 71 |
| `viol_stall_ffma_1` | violation | illegal | WRONG (200/200 wrong) | 71 |
| `viol_stall_iadd3_3` | violation | illegal | WRONG (8/200 wrong) | 113 |
| `viol_stall_iadd3_2` | violation | illegal | WRONG (8/200 wrong) | 82 |
| `viol_stall_iadd3_1` | violation | illegal | WRONG (8/200 wrong) | 82 |
| `probe_mem_consumer_ffma_4` | probe | illegal | correct (0/200 wrong) | 132 |
| `probe_mem_consumer_iadd3_4` | probe | illegal | correct (0/200 wrong) | 143 |
| `viol_ldgsts_swap` | violation | illegal | correct (0/200 wrong) | 337 |
| `bar_ffma_across` | barrier | illegal | correct (0/200 wrong) | 766 |

- The checker agreed with the hardware on **26 of 30** edits and called **0** broken edit legal.
- The four disagreements are all conservative refusals that ran correctly: a 4-cycle ALU-to-store distance (ptxas
  never uses less than 5; 4 was enough here, twice), a swap of two `LDGSTS` with disjoint targets, and an FFMA moved
  across a barrier with no reader before it.
- Some broken edits fail rarely: a cleared scoreboard wait gave a wrong result in 1 of 200 launches, stall counts
  below the IADD3 latency in 8 of 200 (FFMA: 200 of 200). A short test would call them safe.
- Legal slides of a load: cycles stay flat while the load hides under the FFMA chain, then rise by 4 cycles per slot.

**What sections 4 and 5 mean for optimisation.** The rules and the latency-hiding law hold on hardware, and the
checker is safe (0 broken edits approved). No reordering made a kernel faster than ptxas's own schedule: in the
kernels tested, ptxas had already issued each load as early as possible, so the legal moves (sliding it later) could
only be equal or slower, and E1's large savings are against a deliberately bad order. Fences cannot be hidden (E4),
and a full FP32 pipe leaves no latency to hide (E5). Two levers have a measured effect but no whole-kernel gain yet:
moving the late warp's register-only work past a barrier (E3) and a fresh register instead of reusing one a store
has not read (E2, up to 46 cycles). The open question is whether checker-approved moves speed up real kernels where
ptxas leaves latency exposed.

## 6. H100 `%globaltimer` steps are not uniform

The GPU timer does not always advance in 64 ns steps on H100: at the sync edges it stepped by 64, 96 and occasionally
128 ns, on a second, different H100 host as well (A100: always 1024 ns).

| GPU, host | runs | %globaltimer step sizes seen at sync edges (count) |
|---|---|---|
| 2026-10-08_a100_cuda12.6 | 11 | 1024 ns x693 (100.0%) |
| 2026-10-08_h100_cuda12.2 | 11 | 64 ns x668 (96.4%), 96 ns x25 (3.6%) |
| 2026-10-08_h100_cuda12.6 | 7 | 64 ns x424 (96.1%), 96 ns x16 (3.6%), 128 ns x1 (0.2%) |
| 2026-10-09_a100x8_pcie_cuda12.6+12.2 | 21 | 1024 ns x1323 (100.0%) |
| 2026-10-09_h100_cuda12.6+12.2 | 15 | 64 ns x890 (94.2%), 96 ns x54 (5.7%), 128 ns x1 (0.1%) |

Any error model for GPU-side timestamps on H100 (including a software clock-correlation fallback) should take the
96 / 128 ns steps into account.

## 7. Relation to TempoTrace

TempoTrace (Elbakoury and Sharma, arXiv:2609.23301, an unreviewed technical white paper) co-designs PTP time
synchronisation with distributed tracing so that events across a GPU cluster are ordered correctly, and builds a
diagnosis layer on top: spans, a causal graph with its critical path, and a rule + XGBoost engine that names the root
cause (compute saturation, KV-cache misses, scheduler preemption, network stalls, ...). GPU events are timestamped at
the network card (GPUDirect RDMA doorbell, PTP-disciplined clock, TAI timescale). By the paper's own provenance
notes, its GPU-to-host figures (0.056 µs residual sigma, 0.218 µs bound) are design targets with the hardware
experiments pending, the large-cluster tables are illustrative, and the diagnosis accuracy comes from a synthetic
corpus. Its own ablation finds that hardware GPU timestamping (instead of software clock correlation) improves
diagnosis only for operations shorter than 20 µs; without PTP (NTP only) diagnosis degrades at every scale.

Where this work sits:
- it is a different layer: no spans or root-cause engine, but measured GPU timing on real A100 / H100 hardware, with
  a strict bound on every timestamp, at instruction-phase level inside kernels and across 8 GPUs, on commodity
  hardware (no special NIC, no PTP);
- the strict per-GPU bound (±0.55-1.03 µs on the host clock) is below TempoTrace's 2.1 µs software baseline and
  2.5-5x wider than its 0.218 µs hardware target; it is one host's clock (`CLOCK_MONOTONIC_RAW`), not TAI;
- two results bear directly on its design: the H100 timer's non-uniform steps (section 6), and the measured limit
  of ordering GPUs through the host clock (~2 µs, section 1), in the sub-20 µs regime where its ablation finds
  hardware timestamps matter.

## 8. Limits and corrections

- One host per run; host clock, not TAI; no external reference clock: the checks are internal (ticket order inside
  the kernel, flag write before the host sees it, feasibility of the clock fit).
- 8 Oct samples are small (5 launches per shape); block size fixed at 256 threads; no memory-heavy kernel shapes.
- The 8 Oct write-up (`tickbound findings and rules 2026-10-08 (A100 + H100) v2.docx`) and figures quote the
  reported (chord) host-clock bounds; the strict values are in section 2 (1.00-1.13x for those kernel-timing runs).
  Its description of TempoTrace should follow section 7 (GPU timestamps taken at the NIC, TAI timescale, the 0.218 µs
  bound is a statistical design target).
- Open: the A100 acq_rel fence difference (section 4), a multi-node or PTP-referenced run.

## Files

- `FINDINGS.md`: this write-up.
- `figures_2026-10-08/`, `figures_2026-10-09/`: figures (PNG + SVG) with their data.
- `tickbound_data/<date>_<gpu>_<toolkit>/`: per run `.json` (run metadata), `.analysis.json`, `.log`, `report.md`,
  SASS gate results, gzipped SASS listings; `summary_2026-10-09.json`, `strict_bounds_2026-10-09.json`.
- `tickbound_runs/2026-10-09/`: the scripts that ran on the rented machines and locally (`onbox_*.sh`, `babysit.sh`,
  `backstop.sh`, `multi_fig.py`, `analyze_9oct.py`, `strict_all.py`) and the two patches.
