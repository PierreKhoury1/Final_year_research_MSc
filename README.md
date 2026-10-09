# Full-kernel GPU profiling with hardware timers

MSc Advanced Computing research, King's College London · Pierre Khoury · status 10 October 2026

This repository holds the research behind **tickbound**, a profiler that timestamps work *inside* GPU kernels with
the GPU's own hardware counters. It puts every checkpoint, from every SM of every GPU in a machine, on one host clock
with a stated error bound. The bound is hard under assumptions this README names. With it we measured NVIDIA A100 and
H100 GPUs (single GPUs, and eight A100s in one host), and tested which instruction reorderings are legal and which
save time.

The full tables, data paths and limits are in **[FINDINGS.md](FINDINGS.md)**. All GPU measurements ran on rented
vast.ai machines on 8 and 9 October 2026 ($2.66 in total).

**This version was revised on 10 October 2026 after an independent audit**, which recomputed the results from the raw
data. Where the first version was wrong or overstated, the text below is corrected; FINDINGS section 8 lists every
correction.

## Summary

1. **Host time is not kernel time.** On the A100, 17.3 µs pass from the launch call to `cudaStreamSynchronize`
   returning, around 3.5 µs of warp execution. This was measured on our instrumented probe kernel; no uninstrumented
   control was run yet.
2. **We can place every part of a kernel on one clock, with stated bounds.**
   - **One SM.** Inside one SM, timing is exact to the cycle.
   - **Host clock.** It is ±0.55–1.03 µs for kernel timing. This is hard if the GPU/host clock rate is constant
     between the syncs, which holds in practice when the syncs are at most 0.26 s apart.
   - **Across SMs.** ±29–285 ns on A100 (median per run). On H100, ±12–23 ns, but empirical, not proven.
   - **Eight GPUs.** Eight GPUs share one host clock with ±0.85–1.2 µs per GPU. Their start order can be proved only
     when they are more than ~1.7–2.3 µs apart.
3. **Reordering: no speed-up over the compiler is shown.**
   - **Latency hiding works.** For each load, hiding work under the load's latency follows the textbook model to
     within 1.5 FFMA.
   - **The compiler already did it.** In our test kernel the compiler had already put the load first.
   - **Fences.** A fence stalls its warp for a fixed time that the work around it does not hide.
   - **The checker.** Our legality checker refused every deliberate violation it was given and approved only edits
     that ran correctly. But this was on small test kernels only.
4. **What looks new** (section 6): a hard bound on every in-kernel event, mapped to the host clock across several
   GPUs; a fence's cost does not shrink when its store has time to drain; and with CUDA 12.6's barrier, register-only
   work runs free while the warp waits. Per-warp in-kernel tracing itself already exists in other tools.

## 1. Why we are doing this

GPUs now run work with deadlines and work spread over many GPUs at once:

- **GPU-based 5G/6G base stations.** NVIDIA Aerial runs the radio physical layer on GPUs, and each slot's
  processing must finish on time.
- **Multi-GPU training and inference.** Every collective step waits for the slowest GPU.
- **Distributed tracing across GPU clusters** (for example TempoTrace). It needs GPU timestamps it can trust in order
  to put events in the right order.

To find out why something is late, you have to see when each part of the work actually ran: on which GPU, on which
SM, in which part of the kernel, and how certain that time is. Today's tools each see part of that:

| Tool | What it sees | What it does not give |
|---|---|---|
| Nsight Systems / CUPTI | kernel start and end on a timeline; sampled counters and PC samples inside kernels | per-warp events inside a kernel |
| Nsight Compute | per-kernel metrics, sampled stall reasons, sampled metric timelines | when each warp did what |
| Ervin Tasnadi's nanobenchmarking | exact cycle cost of one instruction in an isolated snippet | other SMs, other GPUs, the host clock |
| Intra-kernel tracers (Neutrino, Triton Proton / KPerfIR, NVIDIA IKET, TVM CudaProfiler, Xtrace) | per-warp events of every SM on `%globaltimer` | a stated error bound; mapping to the host clock; alignment across GPUs |
| TempoTrace (arXiv:2609.23301) | cluster-wide event order from PTP-disciplined NIC timestamps | inside kernels; its GPU-side precision is a design target, not yet measured |

The idea of this project is **full-kernel profiling with hardware timers, with an error bound on every point**. Every
warp of every SM of every GPU stamps its own progress with the GPU's counters. The stamps then land on one host clock,
each with a stated worst-case error.

The second question builds on the first: with timing that exact, can we learn the rules of instruction reordering?
And can moving arithmetic into the time a memory load takes make a kernel faster than the compiler makes it?

## 2. Starting point: Ervin Tasnadi's nanobenchmarking

Ervin Tasnadi's post "Nanobenchmarking: cycle accurate benchmarking of CUDA kernels" (Core Velocity Lab,
3 December 2025) times single instructions to the cycle:

- He reads the SM cycle counter (`CS2R Rx, SR_CLOCKLO`) before and after a small code snippet.
- He then edits the compiled SASS with CuAssembler so that the closing read waits until the measured instruction
  has finished.

This gives exact cycle counts for an isolated snippet on one SM. Each SM's counter starts at its own value, so the
method cannot compare times between SMs.

tickbound keeps the exact cycle counter and adds the layers his method does not have:

| | Tasnadi | tickbound |
|---|---|---|
| Inside one SM | exact to the cycle | exact to the cycle. Shared-memory load: 23 cycles on A100 and H100 (his post: 22–23 on Turing) |
| Across SMs | not comparable | Each checkpoint reads timer · cycle · timer, and each block's cycle counter is fitted onto the GPU-wide timer. Result: A100 ±29–285 ns (median per run); H100 ±12–23 ns, empirical |
| GPU to host | – | Clock sync before and after every run: ±0.55–1.03 µs for kernel timing, hard under a constant clock rate between the syncs |
| Several GPUs | – | 8 A100s on one host clock, ±0.85–1.2 µs per GPU |
| What is timed | one snippet, with the SASS edited by hand | every warp of a whole kernel. A SASS check confirms that each timed step in the binary that ran holds only the instructions its name says |
| Reordering | – | Experiments E1–E7 compile the same instructions in different orders. 30 SASS edits made with CuAssembler are judged by a legality checker and then run on the GPU |

## 3. How it works

- **Three clocks.**
  - `clock64` counts cycles exactly on each SM, but each SM's count starts from a different value.
  - `%globaltimer` is one nanosecond timer per GPU that moves in steps: 1,024 ns on A100 and 32 ns on H100.
  - The host clock, `CLOCK_MONOTONIC_RAW`, is where the CPU stamps launches, polls and syncs.
- **Every checkpoint is a bracket.** It reads the timer, then the cycle counter, then the timer again, so the cycle
  value is known to lie inside a window of the timer. Each timed phase ends with an instruction that uses the phase's
  result. This makes the closing read wait for the work, as in Tasnadi's method, but done from source code.
- **One line per block.** A block's checkpoints give many timer windows on one SM counter, and a straight
  cycle-to-time line is fitted through them.
  - **A100.** The timer steps only every 1,024 ns. The windows fall at different points between steps, like the marks
    on a vernier scale, so the line is placed far more finely than one step.
  - **H100.** The step is 32 ns and the fitted band is about one step wide, so the fit adds no resolution. At that
    step the straight-line model also fails for most multi-block launches, so H100 placement across SMs is empirical.
- **Host sync.**
  - Before and after each run, a GPU thread and a host thread exchange timestamps through shared host memory. The
    exchanges are timed around the moments the GPU timer steps to its next value.
  - The **strict bound** is the worst case over every clock offset and drift rate that agrees with both syncs.
  - It is hard if the rate is constant in between. The fit cannot detect a rate change inside the gap.
- **SASS gate.** The binary that measured is disassembled. Each timed step is checked to contain exactly what its
  name says, for example 1 FADD + N FFMA and nothing else. Builds that fail are dropped. This caught CUDA 12.2
  inserting extra waits into timed steps.
- **Built-in checks.** On every run, three things are checked against the placed times, per GPU: barrier releases,
  the order of a grid-wide atomic ticket, and when the CPU saw the done-flag. There were 0 violations. These checks
  confirm the one-timer assumption across SMs only above ~0.2–0.5 µs. Nothing yet checks alignment between GPUs.

## 4. What we found

### 4.1 Host time is not kernel time

![Where the time goes between CPU and GPU](figures_2026-10-08/where_time_goes.png)

These are medians of 4–5 warm launches of our instrumented probe kernel, one host per GPU type (8 Oct):

| | A100 | H100 |
|---|---|---|
| Launch call to `cudaStreamSynchronize` returning | 17.3 µs | 12.6 µs |
| of which warps ran | 3.5 µs | 2.3 µs |
| system fence after the done-flag store (inside the kernel time) | 2.85 µs | 1.74 µs |
| Launch to first warp, warm | 5.1–6.8 µs | 4.6–5.6 µs |
| Launch to first warp, after 50 ms of host sleep | 9.8 µs | 24–32 µs |
| First launch of the kernel in a process | 24–39 µs | 40–60 µs |

- **Where the extra time goes.**
  - The idle penalty and most of the first-launch cost are spent inside the host's launch API call. The GPU does not
    start later once the call returns.
  - Host sleep and GPU idle were not separated.
  - The first-launch values are one sample per process.
- **Polling the done-flag.**
  - The last block stores its done-flag *before* its system fence. A CPU polling that flag sees it 6.5 µs (A100) or
    4.4 µs (H100) before `cudaStreamSynchronize` returns, but the kernel has not finished at that point.
  - With the fence placed before the flag, the lead would be about 3.7 / 2.7 µs. That is an estimate, not a
    measurement.
- **Confounds.** A100 and H100 differ in host CPU, driver and SM clock. Each split point carries the run's clock bound
  (±0.8–0.9 µs on A100, ±0.7 µs on H100).

### 4.2 Inside a kernel

![Where a warp's cycles go](figures_2026-10-08/corrected_2026-10-09/kernel_phases.png)

**Phase costs.** These are cycles per warp, with one block of 256 threads per SM (CUDA 12.6). Every phase except the
atomic ticket is checked in SASS.

| Phase | A100 | H100 |
|---|---|---|
| 256 dependent FFMA | ~1,030 | ~1,030 |
| store + GPU-scope fence | 628 | 1,534 |
| system-scope fence after the done-flag store | 3,056 | 3,374 |

On the 9 Oct H100 host the system fence cost 2,156 cycles. Fence costs depend on the host, the driver and the store
target.

**Latency per instruction**, in cycles (8 Oct dependent chains):

| Instruction | A100 | H100 |
|---|---|---|
| FFMA, FADD, IMAD | 4 | 4 |
| shared-memory load | 23 | 23 |
| warp shuffle | 26 | 26 |
| L1 hit | 39 | 39 |
| L2 load (1 MB set) | 282 | 283 |
| DRAM load, misses only (128 MB set) | ~541 | ~683 |

Load latency per line is bimodal on the 9 Oct machines:

| | L2, near / far | DRAM, near / far |
|---|---|---|
| A100 | 220 / 374 | 493 / 645 |
| H100 | 280 / 472 | 572 / 774 |

**Crowding.**
- **Residency.** At most 3 blocks of 256 threads were resident per SM. This limit comes from the probe's own 80
  registers per thread, not from the hardware.
- **A100 FP32 pipe.** With 3 resident blocks, A100 blocks take turns on the FP32 pipe: the first, second and third
  block on each SM run at 4, 8 and 12 cycles per FFMA. This matches the A100's 16 FP32 lanes per scheduler. On H100,
  86–90% of warps stay at 4.
- **Code size.** Warm straight-line code up to 64 KB runs at full speed with one block per SM. Its first execution
  costs a few hundred cycles more.

### 4.3 Eight GPUs on one host clock

![8 GPUs on one host clock](figures_2026-10-09/multi_gpu_instruction_timeline.png)

**Setup.**
- Eight A100s in one host, one process per GPU, each with its own clock sync.
- All processes launched the same kernel (108 blocks × 256 threads) on a shared 2 ms grid of the host clock.
- The figure draws every block's warp 0 phase by phase on one host axis, each GPU with its bound.
- The SM clock ran at about 0.76 GHz in this run.

**Results, over 40 shared launch instants.**
- **Launch calls.** The eight launch calls were made within **30 ns** of each other (median).
- **Start spread.** The estimated kernel starts spread by **0.78 µs** (median). At any single instant this is smaller
  than the error bars, so the size of the spread is not resolved. But for every GPU pair the start offset changed
  between instants by more than the stamp errors allow, so the starts were not simultaneous.
- **Per-GPU bound.** Each GPU's position on the host axis is known to ±0.85–1.2 µs.
- **Provable order.** The start order of two GPUs at one instant is provable only for gaps above **~1.7–2.3 µs**
  (about 1.66 µs with exact intervals). That held for 11 of 1,120 pairs.

**What this means.**
- **What compares exactly.** Phase durations, and positions relative to each GPU's own kernel start, compare exactly
  across GPUs.
- **What does not compare yet.** Absolute positions on two GPUs can be ordered only when they are more than ~1.7–2.3 µs
  apart. That is longer than every phase (0.05–1.35 µs in this run), so a phase-by-phase cross-GPU comparison is not
  yet possible.
- **Where the threshold comes from.** It comes from this sync method on this host (the syncs were 16 s apart), not
  from the host clock itself.
- **Not checked.** The checks are per GPU. Alignment between GPUs, and one host clock across the two CPU sockets, are
  not tested.

### 4.4 Instruction reordering: the rules, tested on hardware

GPU machine code (SASS) carries its own timing. The compiler writes a stall count and scoreboard waits in front of
every instruction, and the hardware does not re-check dependencies. Moving an instruction without updating these
codes silently gives wrong results. We encoded five rules in a legality checker, and timed or violated them on the GPU.

**Rule 1: a load stalls only at its first use.**
- What the hardware does: independent work placed between a load and its use runs while the load is in flight.
- Measured (A100 / H100, CUDA 12.6), for each load:
  - Time follows max(L, 4N + d). The break-even is within 1.5 FFMA of (L − d)/4 for each latency class.
  - Hoisting a DRAM load ahead of enough work saves about its latency minus d: about 460–615 cycles on A100 and
    540–745 on H100.
  - In hand-edited SASS, the time stays flat while the load is hidden, then rises 4 cycles per slot.
- Correction: the first version's group-median numbers were line-mix artefacts.

**Rule 2: a store reads its registers late.**
- What the hardware does: overwriting a register that a store has not read yet makes the next instruction wait.
- Measured: at 1 warp, 0–11 cycles, measured to about ±6. The 32-warp numbers are confounded by other differences
  between the variants.

**Rule 3: results need their stall cycles.**
- What the hardware does: a dependent instruction issued too early reads the old register value.
- Measured:
  - Below the minimum stall, every link read a stale register in every launch.
  - FFMA results were wrong in 200 of 200 runs.
  - IADD3 results were right in 192 of 200 runs only because the register still held the value the same kernel had
    left there, with the same input.

**Rule 4: a barrier blocks memory, not arithmetic.**
- What the hardware does: after `BAR.SYNC.DEFER_BLOCKING` (CUDA 12.6), a warp keeps issuing instructions until its
  first memory instruction.
- Measured:
  - Register-only arithmetic after the barrier costs nothing while the warp would otherwise wait. Work that touches
    shared memory costs its full length.
  - With CUDA 12.2's plain `BAR.SYNC`, register-only work costs its full length too.
  - Moving the slowest warp's arithmetic past the barrier does **not** make the block finish sooner.

**Rule 5: fences and memory order.**
- What the hardware does: no instruction moves across a fence, and async copies keep their order.
- Measured, in one thread with one 4-byte store:
  - A fence stalls its warp for a fixed 410 / 682 cycles (GPU scope) and 2,277 / 1,252 (system scope).
  - Up to 1,024 FFMA of drain time before the fence change this by at most 8 cycles.
  - Hiding by other warps was not tested.

**Checker against hardware.** Thirty edits to real A100 SASS were each run 200 times.
- **Agreement.** The checker's verdict matched the hardware on 26 of 30.
- **Violations.** It refused all 13 deliberate violations. Nine broke the output. Four ran correctly, so there the
  checker was stricter than needed.
- **Approvals.** It approved 17 slides of one load, and all of them ran correctly.
- **Scope.** Five small test kernels, one A100, one fixed input. The approvals have not been tested on edits close to
  a hazard.
- **Testing lesson.** Every broken edit that sometimes passed failed on its first, cold launch. A test that warms up
  and repeats one input would miss them. A cold launch, a register scrub, or a fresh input on each launch catches them.

**Throughput limit (E5).** Cycles per FFMA follow max(4 / ILP, p × warps per scheduler), with p = 2 on A100 and
p = 1 on H100. With dependent chains, the FP32 pipe fills at 2 warps per scheduler on A100 and 4 on H100.

## 5. Did we learn how to make kernels faster by reordering?

**The rules, in part. A faster kernel, no.**

What the data show:

1. **The rules hold where we tested them.** On five small test kernels, the checker refused every violation we tried
   and approved only edits that ran correctly. Several rules were never violated: fences, branches, real memory
   aliasing, and the store-register hazard.
2. **Latency hiding follows the model for each load:** free until the independent work covers the load's latency,
   then 4 cycles per FFMA.
3. **Where hiding cannot help.**
   - A fence stalls its own warp for a fixed time that the work around it does not hide.
   - Once a few warps compete, the FP32 pipe is full: at 2 warps per scheduler on A100 and 4 on H100.
4. **One free slot.** Register-only work placed after a CUDA 12.6 barrier runs while the warp waits.

What the data do not show is **any kernel made faster than the compiler made it**:

- **Tier B could not show a gain, by design.** It only slid one load later, in a kernel that already issues the load
  first.
- **E1's big savings** are measured against an order we forced to be bad.
- **The barrier move** made the block slightly slower.
- **A fresh register** saved nothing measurable at 1 warp.
- **CuAsmRL** reports speed-ups from reordering ptxas's SASS for some kernels. We have not reproduced that.

So the claim "we found how to optimise kernels by hiding execution under load time" is **not supported**.

What is supported:
- **A measurement method.** We can measure what a reordering does, with stated error bars.
- **The rules.** We know the rules that keep a reordering correct, tested on small kernels.
- **The costs.** We measured what each kind of move costs.

The experiment that decides the optimisation question comes first in section 8.

## 6. What is new, and what confirms known results

**Not found in the work we checked.** We checked NVIDIA's documentation and forums, Tasnadi, CuAssembler, CuAsmRL,
maxas, the microbenchmark papers on Volta, Turing, Ampere and Hopper, the intra-kernel tracers above, and TempoTrace.
A full literature review is still to do.

1. **A hard error bound on every in-kernel event, across GPUs.**
   - Each event comes from timer · cycle · timer brackets and is mapped onto the host clock with a strict bound, for
     several GPUs on one host clock.
   - Other intra-kernel tracers place per-warp events on `%globaltimer` without a stated bound. NVIDIA IKET states
     that multi-GPU traces are not aligned.
   - The bound is hard for kernel timing under the named assumptions.
2. **A fence's cost does not shrink when the store before it has time to drain.** It changed by at most 8 cycles with
   up to 1,024 FFMA in between, in one thread with one store.
3. **On the barrier.**
   - With `BAR.SYNC.DEFER_BLOCKING`, register-only work after `__syncthreads()` costs nothing while the warp waits.
     With CUDA 12.2's plain `BAR.SYNC`, it costs its full length.
   - NVIDIA does not document these semantics.
   - Moving such work across the barrier is correct on hardware (200 of 200 runs, one kernel).

**Confirms known results:**
- **The H100 timer.** `%globaltimer` steps every 32 ns (Fusco et al. 2024; NVIDIA IKET documentation). Our data
  confirm it. The first version of this README misread our sync loop's 64/96/128 ns gaps as timer steps.
- **Latency hiding** with independent work (Volkov, GTC 2010).
- **The control-code rules:**
  - stall counts and scoreboards (maxas; Jia et al. 2018; CuAssembler; CuAsmRL);
  - stall violations fail non-deterministically (maxas);
  - IMAD.WIDE takes extra issue cycles (NVIDIA forum, 2023).
- **The FP32 pipe width:** 16 lanes per scheduler on A100, 32 on H100.
- **Launch and completion overhead** of several µs per kernel.
- **Cross-GPU ordering.** Ordering through host-clock sync works only above ~2 µs. That is comparable to
  TempoTrace's 2.1 µs estimate for software correlation.

## 7. Limits

- **Clocks.** Each run uses one host and its host clock, not TAI, with no external reference clock. All checks are
  internal and per GPU.
- **The rate assumption.** Bounds are hard if the GPU/host rate is constant between the two syncs. That holds in
  practice for syncs up to 0.26 s apart. It is not established for the 8 Oct instruction / co-tenant / MPS runs, whose
  syncs were 7–32 s apart.
- **Across SMs.** Placement across SMs assumes all SMs read one `%globaltimer`, which is checked only above
  ~0.2–0.5 µs. H100 placement across SMs is empirical.
- **Kernels and machines.**
  - The kernels are our own probes and microbenchmarks, not production kernels.
  - The kernel-timing probe carries its own instrumentation, and there is no uninstrumented control.
  - The 8 Oct samples are small.
  - The machines are rented containers, and A100 and H100 ran on different hosts and drivers.
- **Open items.**
  - Re-analyse H100 placement across SMs at the 32 ns step.
  - Redo E1 with both orders on the same lines.
  - Repeat the reorder and Tier B runs with cold and warm launches and fresh inputs.

## 8. Next steps

1. **Reorder real kernels.** Apply moves the checker approves to kernels where ptxas leaves latency exposed (for
   example CuAsmRL's benchmark kernels, CUTLASS or cuPHY), and time them against ptxas's own schedule. This decides
   the optimisation question.
2. **Cheap reruns, under $2 in total.**
   - E1 with both orders on the same lines.
   - Tier B with per-launch records, a register scrub, fresh inputs, and cold and warm launches.
   - A run of the same kernel without our timing code, as a control.
   - A direct SM-to-SM timer-skew test.
   - Clock syncs placed next to the launch burst.
3. **Re-analysis, no GPU.** Use the H100's 32 ns step, compute bounds at the event times, and give exact per-pair
   verdicts across GPUs.
4. **Stamps anywhere, automatically.** Insert timer · cycle · timer reads into any compiled kernel, using NVBit for
   closed-source binaries. Position this against Xtrace and NVIDIA IKET.
5. **A common reference clock across hosts** (PTP hardware timestamps, as TempoTrace proposes), to order events below
   ~2 µs and across machines.

## 9. Repository

| Path | Contents |
|---|---|
| [FINDINGS.md](FINDINGS.md) | full results, all tables, limits and the list of corrections |
| `figures_2026-10-08/` | 8 Oct figures (A100, H100). `corrected_2026-10-09/` holds the corrected versions |
| `figures_2026-10-09/` | the 8-GPU timeline (PNG, SVG and its data), per-instant results and the ordering rules |
| `tickbound_data/` | per run: metadata, analysis, log and SASS check results; reports and gzipped SASS per run set. Also strict bounds and the summary for all runs |
| `tickbound_runs/2026-10-09/` | the scripts that ran on the rented machines and locally, and two patches |
| `tickbound findings and rules 2026-10-08 (A100 + H100) v2.docx` | the 8 Oct write-up, replaced by FINDINGS |
| `code/`, `data/`, `demo/`, `gpu_run/`, `storyboard/`, `GPU_Clock_Findings.pptx` | earlier feasibility work (September 2026), from before tickbound |

The raw traces are kept out of git because of their size. tickbound's source code is in a separate repository and is
not yet published, so the scripts here cannot yet be run from this repository alone.

## 10. References

- E. Tasnadi, "Nanobenchmarking: cycle accurate benchmarking of CUDA kernels", Core Velocity Lab, 3 Dec 2025:
  https://corevelocitylab.com/posts/nanobenchmarking-cycle-accurate-benchmarking-of-cuda-kernels/
- TempoTrace, Elbakoury and Sharma, arXiv:2609.23301 (unreviewed white paper; its GPU timestamp figures are design
  targets): https://arxiv.org/abs/2609.23301
- L. Fusco et al., "Understanding Data Movement in Tightly Coupled Heterogeneous Systems: A Case Study with the Grace
  Hopper Superchip", 2024, arXiv:2408.11556 (H100 `%globaltimer` resolution 32 ns): https://arxiv.org/abs/2408.11556
- Intra-kernel tracers: Neutrino (OSDI 2025); Triton Proton intra-kernel profiling (2025) and KPerfIR (OSDI 2025,
  arXiv:2505.21661); NVIDIA IKET (CUTLASS documentation); TVM CudaProfiler; Xtrace (arXiv:2609.28769, 2026)
- CuAssembler: https://github.com/cloudcores/CuAssembler
- G. He and E. Yoneki, "CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning", CGO 2025,
  arXiv:2501.08071: https://arxiv.org/abs/2501.08071
- Z. Jia, M. Maggioni, B. Staiger, D. P. Scarpazza, "Dissecting the NVIDIA Volta GPU Architecture via
  Microbenchmarking", arXiv:1804.06826: https://arxiv.org/abs/1804.06826
- V. Volkov, "Better Performance at Lower Occupancy", GTC 2010.
- NVIDIA PTX ISA, special registers `%globaltimer` and `%clock64`: https://docs.nvidia.com/cuda/parallel-thread-execution/
- NVIDIA Aerial CUDA-Accelerated RAN: https://github.com/NVIDIA/aerial-cuda-accelerated-ran
