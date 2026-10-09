# tickbound findings, 8-9 October 2026 (A100, H100, 8x A100 on one host)

tickbound measures GPU work from inside the kernel. Each warp reads the SM cycle counter (`clock64`) and the GPU timer
(`%globaltimer`) at checkpoints. Each run syncs the GPU timer to the host clock (`CLOCK_MONOTONIC_RAW`) before and
after the work, so every checkpoint lands on the host time axis **with a stated error bound**. The bound is hard under
assumptions that are named where they matter. The SASS of the binary that measured is checked, so that the
checkpoints sit where the source says.

The tool itself lives in a separate repository (not yet published). This repository holds the data, figures, run
scripts and this write-up.

**Revised on 10 October 2026 after an independent audit.** Several readings of the first version were wrong or
overstated; they are corrected in place, and section 8 lists every correction.

Everything below was measured on rented vast.ai machines (8 Oct: $1.07, 9 Oct: $1.59).

| date | hardware | toolkit | what ran | data |
|---|---|---|---|---|
| 8 Oct | 1x A100-SXM4-40GB | CUDA 12.6 | ktrace profile, FFMA x blocks sweep, instr / co-tenant / MPS | `tickbound_data/2026-10-08_a100_cuda12.6` |
| 8 Oct | 1x H100 80GB HBM3 | CUDA 12.2 and 12.6 | same | `tickbound_data/2026-10-08_h100_cuda12.2`, `..._h100_cuda12.6` |
| 9 Oct | 8x A100-PCIE-40GB, one host | CUDA 12.6 and 12.2 | 8-GPU ktrace on one timeline, reorder E1-E7, 30 hand-edited SASS kernels | `tickbound_data/2026-10-09_a100x8_pcie_cuda12.6+12.2` |
| 9 Oct | 1x H100 80GB HBM3 (another host) | CUDA 12.6 and 12.2 | reorder E1-E7, ktrace profile (timer census) | `tickbound_data/2026-10-09_h100_cuda12.6+12.2` |

Some files are kept locally and not committed, because of their size:
- raw traces (`*.bin`);
- timeline viewer pages (`*.trace.json`, `timeline.html`);
- uncompressed SASS listings.

Every analysis result, report, log and gzipped SASS listing is here.

## 1. Eight GPUs on one host clock

![8 GPUs on one host clock](figures_2026-10-09/multi_gpu_instruction_timeline.png)

**Setup.**
- Eight A100s in one host, one process per GPU, each process pinned to cores on its GPU's own CPU socket.
- Each GPU had its own clock sync before and after.
- All processes started together and launched the same ktrace kernel on a shared 2 ms grid of the host clock: 108
  blocks x 256 threads, with a 256-FFMA chain between checkpoints.
- The figure shows the middle one of the 40 shared instants. For every block it draws warp 0, phase by phase: two
  dependent loads, barrier, FFMA chain, barrier, store + GPU fence, atomic ticket, flag store. Each GPU is drawn with
  its placement bound.
- The SM clock ran at about 0.76 GHz in this run.

**Over all 40 instants at which all eight GPUs launched:**
- **Launch calls.** The eight host processes made their launch calls within **30 ns** of each other (median; max
  41 ns). Each process read its own socket's `CLOCK_MONOTONIC_RAW`.
- **Launch to first entry.** The per-GPU medians were 7.6-8.3 µs. Each carries ±0.85-1.2 µs, so the difference
  between GPUs is not resolved. Individual launches took 7.1-10.2 µs. Differences between launches on one GPU are
  resolved to about ±0.2 µs.
- **Start spread.**
  - The estimated kernel starts of the eight GPUs spread by **0.78 µs** (median; range 0.51-2.70 µs).
  - At a single instant, this is smaller than the cross-GPU bound. A spread of at least 0.61 µs is proved in only 2
    instants (4 with exact at-instant intervals).
  - Each GPU's clock error is the same for all its launches, though. For every GPU pair, the start offset changed
    between instants by more than the stamp errors allow. So the starts were not simultaneous, but the size of each
    instant's spread is not resolved.
- **Per-GPU bound.** Each GPU's placement bound on the host axis is **±0.85-1.20 µs**. This is the strict clock bound
  of 0.79-1.03 µs plus the checkpoint's own placement (≤ 0.19 µs). The clock bound is its maximum over the 16 s sync
  gap; at the instants themselves it is 0.75-0.93 µs.
- **Provable order.** Under the conservative rule |Δ| > sum of both GPUs' bounds, the start order of two GPUs at one
  instant was provable only for gaps above 1.75-2.3 µs. That held for **11 of 1,120** GPU pairs, in 2 instants. With
  exact at-instant intervals the threshold is about 1.66 µs, and 20 pairs are proved, in 4 instants.
- **Checks.** The in-kernel ticket-order checks found 0 violations. Clock-fit widening for the 108-block launches was
  at most 2.6 ns.
- **What is not checked.** The checks run per GPU. Nothing in this run tests the alignment between GPUs, or the
  assumption of one host clock across the two CPU sockets. Five of the 11 proofs are cross-socket pairs; the smallest
  cross-socket margin is 42 ns. In every instant, the launch reads from both sockets share one 10-ns phase, which is
  consistent with one synchronised TSC.

**What this shows.**
- **What works.** Per-GPU phase timelines of eight GPUs can share one host clock, each with a stated per-GPU bound.
  The bound is hard if the GPU/host rate is constant between the syncs and the host clock is one clock across sockets.
- **What compares across GPUs.** Phase durations, and phase positions relative to each GPU's own kernel start, compare
  exactly across GPUs.
- **What does not compare yet.** Absolute positions on two GPUs can be ordered only when they are 1.7-2.3 µs apart or
  more. That is longer than every phase (0.05-1.35 µs at 0.76 GHz), so absolute positions of phases cannot yet be
  compared across GPUs.
- **Where the threshold comes from.** It comes from this sync method on this A100-PCIe host, not from a fundamental
  limit of the host clock. Syncing next to the launch burst would lower it.

**Data.**
- `figures_2026-10-09/multi_gpu_instruction_timeline.json` is the figure's data.
- `figures_2026-10-09/multi_gpu_all_instants.json` holds every instant, launch and pair. It was regenerated on 10 Oct
  by `tickbound_runs/2026-10-09/recompute_multi.py` from the raw traces.
- `multi_gpu_order_rules.json` compares the ordering rules. The first committed version of the all-instants file used
  the chord bound, which gives 15 pairs and is not a hard bound at the instants.
- Method note: the start barrier and the shared launch grid were a run-time patch to the probe
  (`tickbound_runs/2026-10-09/patches/gputrace_start_barrier_launch_grid.diff`). It is not yet merged into the tool.

## 2. How exact is each timestamp: strict bounds

**How the strict bound is defined.**
- The clock sync gives a set of feasible (rate, offset) lines between the GPU timer and the host clock.
- The bound the fit reported at first (the "chord") was the half-width of the feasible offsets at the fitted rate.
  That is not a worst case.
- The **strict** bound is the largest deviation, over the gap between the two sync windows, of any feasible line from
  the fitted estimate, taken at conservative end points. An independent exact recomputation reproduces every value in
  this table.

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

All runs are in `tickbound_data/strict_bounds_2026-10-09.json`.

**When the strict bound is hard.**
- It is hard only if the GPU/host clock rate stays constant between the two syncs. The fit cannot detect a rate change
  inside the gap: it only checks that the two sync windows have a common solution, and all 65 runs do.
- **Short gaps.** Kernel-timing, sweep and most reorder runs have their syncs at most 0.26 s apart. There, any
  plausible drift adds ≤ 0.33 ns, so ±0.55-1.03 µs is effectively hard.
- **8-GPU runs.** Their events sit within 0.17 s of the second sync (16 s gap), so the effect is ≤ ~50 ns.
- **Long gaps.** The 8 Oct instruction / co-tenant / MPS runs had 7-32 s between syncs, and their events spread across
  the gap. Consecutive 8 Oct runs show rate intervals 0.15-0.85 ppm apart. So their values (up to ±1.40 µs) are
  conditional, and at that drift the mid-gap error could reach several µs.

**Inside one GPU, between SMs.** Each checkpoint is placed from its block's fitted cycle-to-time line.
- **A100.** The timer steps every 1,024 ns. Placement is ±29-285 ns (median per run), worst block ±374 ns: far finer
  than the timer step.
- **H100.** The timer advances every 32 ns (section 6), but the analysis used a 64 ns tick. At the true 32 ns step the
  straight-line model fails for most multi-block launches, because the SMs' cycle rates differ by about 1%. The H100
  placements are therefore empirical, not proven: ±12-23 ns (median), worst block ±28 ns, in kernels ≤ 9.6 µs, and
  ±32-43 ns at 29 µs.
- **Assumption.** All of this assumes every SM reads the same `%globaltimer`. The ticket-order check tests that only
  above 182-226 ns (H100) and 251-481 ns (A100).
- **Differences.** A ± here is per checkpoint. A difference between two SMs carries both half-widths.

## 3. Kernel shapes

Same kernel, three grid shapes (1 block, 1 block per SM, "4 per SM") x four FFMA chain lengths, plus the 9 Oct 8-GPU
run (1 block vs 1 per SM, 40 launches each per GPU). All of these kernel-timing runs used 256-thread blocks, and no
memory-heavy kernel shape was tried.

The kernel is our own instrumented probe:
- 12 stamps per warp;
- a done-flag and a system-scope fence from the last block;
- 768 bytes of records per warp after exit;
- 80 registers per thread.

No uninstrumented kernel was timed as a control. A100 and H100 also differ in host CPU, driver and SM clock (about 1.1
vs 2.0 GHz).

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
| 8x A100-PCIe (12.6), median of 8 GPUs | 1 block | 256 | 40 per GPU | 7.85 | 0.00 | 3.77 | 4.01 | 8.17 | 0.0 |
| 8x A100-PCIe (12.6), median of 8 GPUs | 1 per SM (108) | 256 | 40 per GPU | 7.98 | 0.66 | 4.79 | 4.01 | 9.31 | 0.0 (max 2.6) |

**Launch to first warp.**
- Warm medians of 4-5 launches: A100 5.1-6.8 µs, H100 4.6-5.6 µs. On A100 the same 256-FFMA kernel gave 5.1 µs in one
  process and 6.4 µs in another. 1.5-3.6 µs of this is the host's launch call.
- **After 50 ms of host sleep.** It rose to 9.8 µs (A100) and 24-32 µs (H100), but the extra time is spent inside the
  host's launch API call. Launch return to first warp did not grow. Host sleep and GPU idle were not separated.
- **First launch of the probe kernel in a process.** This came after a warm-up kernel and the clock-sync kernel. It
  took 24-39 µs (A100) and 40-60 µs (H100) for kernels up to 16 KB of code, and 95-121 µs for the 64 KB kernel. Most
  of it is inside the launch call. There is one sample per process.

**Wave start.** A full wave starts within 0.4-0.6 µs on A100 (108 blocks) and 0.2-0.3 µs on H100 (132 blocks).

**Residency.**
- At most 3 blocks were resident per SM, because the probe uses 80 registers per thread. The fourth block ran as a tail
  wave.
- These crowding results describe this register-limited probe, not a hardware limit.
- The 4-per-SM spans are 2.8-4.8x the 1-per-SM spans: a tail wave (about 2x) plus contention among the three resident
  blocks.

**FFMA under contention.**
- With 3 blocks resident, A100 blocks take turns on the FP32 pipe. On each SM the first, second and third resident block
  run at 4, 8 and 12 cycles per FFMA, and the tail-wave block at about 4. The median is 5.8-6.2 and the mean about 7.
- This matches the A100's 16 FP32 lanes per scheduler (E5).
- On H100, 86-90% of warps stay at about 4.

**Completion.**
- Last warp exit to the host's sync return took 6.3-8.0 µs (A100) and 4.6-5.0 µs (H100); on the 9 Oct PCIe host it
  was 8.2-9.3 µs.
- The last block stores its done-flag before its system-scope fence. The fence costs 3,056 / 3,374 cycles (2.85 /
  1.74 µs on the 8 Oct hosts) and is counted in the kernel time.
- A CPU polling that flag sees it about 6.5 / 4.4 µs before `cudaStreamSynchronize` returns. But the kernel has not
  finished at that point.

**Clock-fit widening.**
- At the analysed ticks, kernels up to about 6 µs needed at most 7 ns of widening, and 28-64 µs kernels needed
  23-100 ns.
- At the H100's true 32 ns step, every multi-block H100 launch needs widening (1.4-29 ns).
- Placement is empirical wherever widening is non-zero.

**Code size.** Warm straight-line code up to 64 KB runs at the 4-cycle FFMA latency with one block per SM. The first
execution of the 64 KB kernel costs about +405 cycles (A100) and +103 (H100). Throughput-bound code was not tested.

**Latency table (8 Oct, dependent chains; A100 CUDA 12.6, H100 CUDA 12.2).**
- FFMA, FADD and IMAD: 4 cycles.
- Shared-memory load: 23.
- Warp shuffle: 26.
- L1 hit: 39.
- L2 load (1 MB set, `.cg`): 282 / 283.
- DRAM misses (128 MB set): about 541 / 683. The first version said 461 / 550, which mixed L2 hits into the 128 MB
  chase.

## 4. Instruction-order experiments (E1-E6) and two compilers (E7)

Each experiment times variants that hold the same bracketed instructions in a different order. A variant counts only
where the reorder SASS gate shows the intended order in the binary that measured. The exceptions:
- E1 variants read different addresses;
- E2 variants differ in their stores after the bracket;
- in E3's late_sink variant the early warps do no work.

The gate checks only the bracket. The same source was built with two toolkits (CUDA 12.6 and 12.2), for A100 (sm_80)
and H100 (sm_90). Each run is one launch per configuration, on one SM. The A100 experiments ran concurrently on six
boards of the 8-GPU host.

| | A100 12.6 | A100 12.2 | H100 12.6 | H100 12.2 |
|---|---|---|---|---|
| gate (295 builds) | 286 pass, 9 fail | 278 pass, 17 fail | 295 pass | 287 pass, 8 fail |

Gate failures are compiler choices, and the analysis drops those builds:
- **Barrier form.** CUDA 12.2 emits `BAR.SYNC` and 12.6 emits `BAR.SYNC.DEFER_BLOCKING` for the same `__syncthreads()`.
  This is so in all 616 barriers of the H100 binaries, also when disassembled with cuobjdump 12.6. The barrier
  experiment is therefore measured with 12.6 only.
- **R2UR.** On sm_80 (both toolkits), with two loads and 96 or more FFMA, ptxas moves the loaded value into a uniform
  register (`R2UR`) in the middle of the chain. The intended overlap then does not exist, so 9 builds were dropped.
- **sm_90 fence.** On sm_90, one PTX fence becomes `MEMBAR.ALL.CTA` directly followed by the scoped `MEMBAR` (then
  `ERRBAR`, `CGAERRBAR`, `CCTL.IVALL`). The gate first refused this. A patch makes it accept the adjacent pair as one
  fence (`tickbound_runs/2026-10-09/patches/sass_reorder_sm90_fence_pair.diff`, with a regression test), and the H100
  data was re-analysed with it. Without the patch the H100 gate gives 268 and 260 passes.

Measured (cycles unless stated; CUDA 12.6; the full one-line results per run are in
`tickbound_data/summary_2026-10-09.json`):

**E1, load to use (corrected 10 Oct).**
- **Latencies are bimodal.** On both GPUs a load's latency falls into one of two classes:

  | | L2 near / far | DRAM near / far |
  |---|---|---|
  | A100 | 220 / 374 | 493 / 645 |
  | H100 | 280 / 472 | 572 / 774 |

  L1 is about 49-50 on both.
- **What went wrong.** Each (variant, N) group is one launch reading its own 64 lines of a fixed random chase, so the
  group medians compare different line mixes. The following were line-mix artefacts:
  - the first version's L2 and DRAM latencies (353 / 370 / 614 / 771);
  - the 73-vs-81 and 84.8-vs-84.7 comparisons;
  - the "inconsistencies" listed as open;
  - the savings of 595 / 634 (634 exceeds the possible maximum, 4N = 512).
- **Per latency class, the law holds.** The lower envelope of overlap(N) follows max(L, 4N + d) + c to the cycle.
  Each class's break-even is within 1.5 FFMA of (L - d) / 4 for L2 (A100 47.5 vs 47.4 and 85.7 vs 85.9; H100 61.6 vs
  62.1 and 108.8 vs 110.1), and within 2 FFMA for DRAM.
- **Savings.** Hoisting a DRAM load ahead of enough independent work (about 190 FFMA or more) saves about L - d: A100
  about 463 / 615, H100 about 542 / 744 cycles for the two classes.
- **Next.** Clean group numbers need a paired redesign that runs both orders on the same lines.

**E2, write-after-read.**
- At 1 warp, reusing a register that a store has not yet read costs 0-11 cycles, measured to about ±6 cycles.
- The 32-warp differences (-44 to +46.5) are confounded by the variants' different stores after the bracket. A no-WAR
  H100 pair also gives -24.5 / -30. So these are not WAR costs.

**E3, barrier.**
- **Register-only work after the barrier.** With `BAR.SYNC.DEFER_BLOCKING`, register-only FFMA placed after the
  barrier cost nothing while the warp would otherwise wait. This was measured with waits of about 4,100 cycles and up
  to 1,024 cycles of work; the clock read after the barrier issued during the wait in every sample. On the
  last-arriving warp the same work costs its full length (+263 / +1,043 on A100).
- **Shared-memory work after the barrier** costs its full length: +296-298 at M = 64 and +1,066 at M = 256.
- **Plain BAR.SYNC.** With CUDA 12.2's plain `BAR.SYNC` (gate-dropped, H100), the same register-only work costs its
  full length (+273 / +1,041), and the clock read never issues during the wait. So the zero cost is due to
  DEFER_BLOCKING.
- **Moving the late warp's work past the barrier.** Moving the late warp's M FFMA past the barrier releases the other
  warps about 4M earlier (+254 / +1,025 on A100, +224 / +1,030 on H100). But the block does not finish sooner: +8 /
  +18 cycles at p50 on A100 and +50 / +28 on H100. The first version presented this as a lever; it is not one.

**E4, fence.** The test ran one thread with one 4-byte store and only register-only work around the fence.
- **Cost.** A fence stalls its warp for a fixed time:

  | | GPU scope | system scope | acq_rel |
  |---|---|---|---|
  | A100 | 410 | 2,277 | 411-413 |
  | H100 | 682 | 1,252 | 684 |

- **Drain time does not help.** Placing up to 1,024 FFMA before the fence, 4,110 issue cycles between the store and
  the fence, changes its cost by only -6.5 to +8 cycles.
- **Work after it does not overlap.** Register-only work after the fence in the same warp does not overlap it either.
- **Not tested:** hiding by other warps, by other outstanding memory operations, or for a fence with no preceding
  store.

**E5, FFMA throughput.**
- Cycles per FFMA per warp follow max(4 / ILP, p x k), with k = warps per sub-partition.
- p = 2 on A100 and 1 on H100. A free fit gives 1.9-2.1 and 1.00-1.02. The A100 reaches 15.8 at 32 warps.
- With dependent chains (ILP 1), the FP32 pipe fills at 2 warps per scheduler on A100 and at 4 on H100.

**E6, dependent chains.** Cycles per dependent link:
- 4 for the integer and FP32 ALU ops tested. This reproduces ptxas's encoding by construction.
- 8 for an IABS link (with its LOP3 helper).
- 6 / 8 for DADD and DFMA.
- 13 (A100) / 12 (H100) for an IMAD.WIDE.U32 + IADD3 link, where the encoded stalls sum to 10 / 8. The extra 3-4
  cycles are not localised to IMAD.WIDE, whose own stall is 5 / 3.

Independent FP32 ops issue every ~2 cycles on A100 and every cycle on H100. DADD and DFMA issue every 4.0 (A100) and
2.2 (H100) cycles.

**E7, CUDA 12.2 vs 12.6.**
- Apart from the barrier form, there is no toolkit effect beyond run-to-run spread: E1 DRAM about ±50 cycles, E2 at 32
  warps about ±20, E5 and E6 within 0.07.
- The E1 L2 agreement is a replay of the same lines on the same GPU, not an independent replication.
- The A100 acq_rel difference (565 vs 413) was a baseline artefact. Mode-matched, CUDA 12.2 gives 412, so this open
  item is closed.

## 5. Hand-edited SASS: does the legality checker match the hardware?

Thirty edits were made to real sm_80 SASS (CuAssembler). Each ran 200 times on an A100 against the unedited kernel's
output. They are:
- legal moves of a load ("slide", k slots);
- deliberate violations: a load moved past its consumer, a cleared scoreboard wait, stall counts below the latency;
- probes of rules the checker applies conservatively.

How the comparison works:
- The reference is ptxas 12.9's own cubin, and slide_k00 is that same order.
- Each edit changes only its own kernel.
- Every launch gets the same initialised input.
- The output is zeroed before each launch and compared bit for bit with the reference.

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
| `viol_wait_cleared` | violation | illegal | WRONG (1/200 wrong: the first, cold launch) | 267 |
| `viol_stall_ffma_3` | violation | illegal | WRONG (200/200 wrong) | 103 |
| `viol_stall_ffma_2` | violation | illegal | WRONG (200/200 wrong) | 71 |
| `viol_stall_ffma_1` | violation | illegal | WRONG (200/200 wrong) | 71 |
| `viol_stall_iadd3_3` | violation | illegal | WRONG (8/200 wrong: first visit to each placement) | 113 |
| `viol_stall_iadd3_2` | violation | illegal | WRONG (8/200 wrong: first visit to each placement) | 82 |
| `viol_stall_iadd3_1` | violation | illegal | WRONG (8/200 wrong: first visit to each placement) | 82 |
| `probe_mem_consumer_ffma_4` | probe | illegal | correct (0/200 wrong) | 132 |
| `probe_mem_consumer_iadd3_4` | probe | illegal | correct (0/200 wrong) | 143 |
| `viol_ldgsts_swap` | violation | illegal | correct (0/200 wrong) | 337 |
| `bar_ffma_across` | barrier | illegal | correct (0/200 wrong) | 766 |

**Counts.** The checker's verdict matched the outcome for **26 of 30** edits.
- **Violations.** It refused all 13 deliberate violations. Nine of them broke the output and four did not. The four
  are where the checker is stricter than needed:
  - a 4-cycle ALU-to-store distance, twice (ptxas never uses less than 5);
  - a swap of two `LDGSTS` with disjoint targets;
  - an FFMA moved across a barrier.
- **Approvals.** It approved 17 slides of one load, one of them identical to the unedited kernel, and all 17 ran
  correctly.
- **Scope.**
  - Five small single-block test kernels, on one A100, with one fixed input.
  - The approvals were not tested on edits close to a hazard.
  - Several rules were never violated: scoreboard setup, read-scoreboard WAR, real memory aliasing, and fences,
    branches, labels and EXIT.

**Partial failures (corrected 10 Oct).**
- **Stale registers.** Below the minimum stall, every dependent link read a stale register in every launch; cycles
  were identical across all launches.
  - FFMA's output was wrong in 200 of 200 launches.
  - IADD3's output was right in 192 of 200 only because the stale register still held the value the same kernel had
    left on that SM placement, with the same input. The 8 wrong launches are the first visit to each of the 8
    placements.
- **Cold load.** The cleared scoreboard wait failed only on the first, cold launch: the load then returned about 110
  cycles after its consumer, against about 30 cycles before it when warm. It passed the 199 warm repeats.
- **What this means for testing.** A test that warms up and repeats one input would call these edits safe. A cold
  first launch, a register scrub, or a fresh input on each launch catches them.
- **Correction.** The first version called these rare random races that a short test would miss. That reading was
  backwards.

**Slides.** Cycles stay flat at 266-268 up to about 7.6 slots, then rise 3.98 cycles per slot. ptxas had placed this
load first, so the slides could only be equal or slower. They show what the compiler's placement is worth, not a gain
over it.

**What sections 4 and 5 mean for optimisation.**

Measured and solid:
- Per load, latency hiding follows max(L, 4N + d).
- A fence stalls its own warp for a fixed time that register-only work around it does not hide.
- Register-only work after a DEFER_BLOCKING barrier is free while the warp waits.
- With dependent chains, the FP32 pipe fills at 2 (A100) or 4 (H100) warps per scheduler.
- On small test kernels, the checker refused every violation tried and approved only edits that ran correctly.

Not shown: any reordering that makes a kernel faster than ptxas.
- Tier B could not show one by design: it only slid one load later, in a kernel that issues that load first.
- The barrier move made the block slightly slower.
- E2's fresh-register effect is within ±6 cycles at 1 warp.

Whether checker-approved moves speed up real kernels where ptxas leaves latency exposed is the open question.

## 6. H100 `%globaltimer`: a 32 ns step (corrected 10 Oct)

On both H100 hosts, `%globaltimer` advances in 32 ns increments.
- **Evidence.** In-kernel reads about 7 ns apart differed by exactly 32 ns 936,649 times, and by 64 ns only 10,870
  times. Its values split evenly between the two residues mod 64.
- **Published figures agree.**
  - Fusco et al. 2024 measured 32 ns on GH200 (arXiv:2408.11556).
  - NVIDIA's CUTLASS IKET documentation states a 32 ns granularity on SM90 and later.
- **A100.** 1,024 ns.

**Why the census shows larger steps.** The census below comes from the sync loop. The loop re-reads the timer less often
than every 32 ns, so it reports 32 ns x (1 + the updates it missed). That gives 64, 96 and 128 ns.

**Correction.** The first version read these as non-uniform timer steps; that was wrong. The analysis also used a 64 ns
tick on H100, which makes its H100 placement windows too wide (section 2).

| GPU, host | runs | steps seen by the sync loop (not the timer's update period), count |
|---|---|---|
| 2026-10-08_a100_cuda12.6 | 11 | 1024 ns x693 (100.0%) |
| 2026-10-08_h100_cuda12.2 | 11 | 64 ns x668 (96.4%), 96 ns x25 (3.6%) |
| 2026-10-08_h100_cuda12.6 | 7 | 64 ns x424 (96.1%), 96 ns x16 (3.6%), 128 ns x1 (0.2%) |
| 2026-10-09_a100x8_pcie_cuda12.6+12.2 | 21 | 1024 ns x1323 (100.0%) |
| 2026-10-09_h100_cuda12.6+12.2 | 15 | 64 ns x890 (94.2%), 96 ns x54 (5.7%), 128 ns x1 (0.1%) |

An error model for H100 GPU timestamps should use the 32 ns step plus the measured per-read uncertainty.

## 7. Relation to TempoTrace

TempoTrace (Elbakoury and Sharma, arXiv:2609.23301) is an unreviewed technical white paper.
- **What it does.** It co-designs PTP time synchronisation with distributed tracing, so that events across a GPU
  cluster are ordered correctly.
- **Diagnosis layer.** On top it builds spans, a causal graph with its critical path, and a rule + XGBoost engine that
  names the root cause (compute saturation, KV-cache misses, scheduler preemption, network stalls, ...).
- **GPU timestamps.** GPU events are timestamped at the network card: a GPUDirect RDMA doorbell, a PTP-disciplined
  clock, on the TAI timescale.
- **What is measured.** By the paper's own provenance notes:
  - its GPU-to-host figures (0.056 µs residual sigma, 0.218 µs bound) are design targets, with the hardware experiments
    pending;
  - the large-cluster tables are illustrative;
  - the diagnosis accuracy comes from a synthetic corpus.
- **Its ablation.** Hardware GPU timestamping, instead of software clock correlation, improves diagnosis only for
  operations shorter than 20 µs. Without PTP (NTP only), diagnosis degrades at every scale.

Where this work sits:
- **A different layer.** It has no spans and no root-cause engine. It measures GPU timing on real A100 and H100
  hardware, with a stated bound on every timestamp, inside kernels and on 8 GPUs of one host, on commodity hardware (no
  special NIC, no PTP). Those bounds are hard under the assumptions named in sections 1 and 2.
- **The numbers.** Our strict per-GPU bound for kernel timing (±0.55-1.03 µs on the host clock) is below TempoTrace's
  2.1 µs software baseline and 2.5-5x wider than its 0.218 µs hardware target. It is one host's clock
  (`CLOCK_MONOTONIC_RAW`), not TAI.
- **What bears on its design.** Two results bear on it, in the sub-20 µs regime where its ablation finds hardware
  timestamps matter:
  - the H100 timer's 32 ns step, which confirms published figures;
  - the order-provability threshold of host-clock sync on this A100-PCIe host: about 1.7-2.3 µs, or about 1.66 µs with
    exact intervals. This is comparable to its 2.1 µs software estimate.

## 8. Limits and corrections

**Limits.**
- **Clocks.** Each run uses one host's clock, not TAI, and there is no external reference clock. The checks are
  internal and per GPU: ticket order inside the kernel, the flag written before the host sees it, and whether the
  clock fit has a solution.
- **Rate assumption.** The strict host bound is hard only if the GPU/host rate is constant between the syncs. The
  long-gap instruction / co-tenant / MPS runs are conditional (section 2).
- **Across SMs.** Placement across SMs assumes all SMs read one `%globaltimer`, which is checked only above about
  0.18-0.48 µs. The H100 values across SMs are empirical.
- **What was measured.**
  - Our own probes and microbenchmarks only, no production kernels.
  - The kernel-timing probe is instrumented (80 registers), and no uninstrumented control was run.
  - The 8 Oct samples are small (5 launches per shape).
  - Kernel-timing runs use 256-thread blocks. E2 uses 32 and 1,024 threads, E5 32-1,024, and the Tier B edits one
    32-thread warp (256 threads for the barrier kernel).
- **Machines.** These are rented containers. Comparisons between A100 and H100 are confounded by host CPU, driver and
  SM clock; the 8-GPU run's SM clock was 0.76 GHz.
- **The 8 Oct write-up.** The Word document `tickbound findings and rules 2026-10-08 (A100 + H100) v2.docx` is
  superseded by this file.

**Corrections of 10 Oct** (after an independent audit that recomputed the results from the raw data):
1. **H100 timer.** It has a 32 ns step, not 64/96/128 ns (section 6). The H100 placement across SMs is empirical
   (section 2).
2. **E1.** Its latencies, comparisons and savings were line-mix artefacts. They are restated per latency class
   (section 4).
3. **E3.** The "lever" (moving the slowest warp's work past a barrier) does not make the block faster.
4. **Tier B.** The partial failures come from a cold first launch and stale registers, not rare races (section 5).
5. **System fence.** The 3,056 / 3,374-cycle fence runs after the done-flag store, not before it. A polling CPU sees
   the flag before the kernel has finished (section 3).
6. **DRAM latency.** It is about 541 / 683 cycles, not 461 / 550 (section 3).
7. **Idle penalty.** It sits inside the host's launch call. The 3-resident-block limit comes from the probe's
   registers (section 3).
8. **8-GPU timeline.**
   - The start spread lies within the bounds at one instant.
   - The cross-GPU ordering threshold belongs to this sync method.
   - The checks are per GPU.
   - The per-instant data file is regenerated with strict bounds (section 1).
9. **A100 acq_rel.** The open item is closed: 412 cycles with both toolkits (section 4).

## Files

- `FINDINGS.md`: this write-up.
- `figures_2026-10-08/` and `figures_2026-10-09/`: figures (PNG + SVG) with their data.
- `tickbound_data/<date>_<gpu>_<toolkit>/`: per run, the `.json` run metadata, `.analysis.json`, `.log` and SASS gate
  results; `report.md` per characterize directory; gzipped probe SASS per machine. Also `summary_2026-10-09.json` and
  `strict_bounds_2026-10-09.json`.
- `tickbound_runs/2026-10-09/`: the scripts that ran on the rented machines and locally, and the two patches:
  - on-box: `onbox_*.sh`, `babysit.sh`, `backstop.sh`;
  - analysis: `multi_fig.py`, `analyze_9oct.py`, `strict_all.py`, `recompute_multi.py`, `make_findings.py`.

  They need the tickbound source (not yet published) and the raw traces (kept locally).
