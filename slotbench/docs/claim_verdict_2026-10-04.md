# Claim verdict: does the GPU-timed cuPHY work support a better 5G architecture, or 6G? (2026-10-04)

Produced by an adversarial verification workflow: four independent lenses (raw-data re-audit, production Aerial baseline in the local `aerial-cuda-accelerated-ran` clone, 5G NR timing budgets, 6G requirements and prior art), then an arbiter who re-checked every disagreement against the raw records, the Aerial source and primary documents. All four lenses: 5G **partially supported**, 6G **argued extrapolation**.

## One-line answer

The data supports a narrow 5G claim: a GPU-resident slot executive is a sturdier launch path for a GPU-based 5G L1 when the host has no real-time tuning. It does not measure anything about 6G. 6G can only be argued as an extrapolation, and the shared-GPU (AI-RAN) results weaken that argument rather than support it.

## Defensible claim (thesis wording)

We replayed NVIDIA's unmodified cuPHY PUSCH full-slot graph (TC7304: 273 PRB, 4 layers, 256-QAM, 30 kHz SCS) on A100 GPUs whose host CPUs had normal-priority (non-real-time) scheduling. Moving the per-slot trigger from a host thread to a resident, GPU-timed executive using device graph launch removed host-induced slot loss: 0 skipped boundaries and 0 stalls over 1 ms in 120,000 idle boundaries (95% upper bound on the skip rate about 2.5e-5), whereas the host-launched path had 0.6-14 ms stalls in 28 of 42 idle CPU-launched trials (primary host-1 campaign: none). The executive also cut p99 slot-start error from 12-31 us to about 5 us, independent of GPU clock state, and every trial passed CRC checks before and after replay. This is evidence for taking per-slot host submission out of the PUSCH critical path. It is not a comparison against a real-time-tuned production Aerial host, and it gives no protection against a co-located GPU workload.

## Measured 5G results (re-verified)

- **Correctness: GPU self-launch of the real cuPHY PUSCH graph decodes correctly.** 90/90 measured trials of 5,000 slots each (36 contention+idle on host 1, 18 activity-control on host 1, 36 on host 2) passed payload, TB CRC and CB CRC checks before and after each replay, with 0 launch errors and 0 timeouts. Gates and smokes also pass. Checks are at the endpoints only, not per slot. _(source: slotbench/data/2026-10-02_a100_cuphy_repeated/out/*.json, activity_control/out/*.json; slotbench/data/2026-10-03_a100_cuphy_harq/trials/*.json (data-audit lens recomputation))_
- **Idle p99 slot-start error, clock-matched comparison (GPU minus CPU+keepalive).** Host 1, median of per-trial p99: CPU 13.4 us, CPU+keepalive 12.2 us, GPU 4.7-5.1 us. Paired GPU minus keepalive: -7.6 us [-8.5, -7.1], lower in 6/6. Host 2: CPU 31.3 us (per-trial 21-87), CPU+keepalive 29.0 us (20-73), GPU 5.0 us. Paired GPU minus keepalive: -27.9 us [-36.9, -21.2], lower in 12/12. _(source: activity_control/out and harq/trials raw .bin; harq/analysis/control/control_variant_summary.csv)_
- **Mechanism of the start-error gap (verified by the arbiter): the host wakes up on time; the gap is host-submission-to-GPU-start latency.** CPU launch lateness against the boundary: p50 0.04-0.1 us, p99 at most 0.17 us. CPU launch-to-PHY-start: p50 10.3 us (host 1 AC), 11.7 us (host 1 primary), 21.0 us (host 2); cudaGraphLaunch call p50 5.4-16.3 us. GPU device-launch-to-start: a constant 4.1 us. _(source: arbiter script /tmp/claude-0/-home-user/c22b0607-07f1-5108-a24d-4caf6e85f472/scratchpad/verdict/arb.py over the same .bin files)_
- **Host stalls and skips, idle.** GPU: 0 skips and 0 completions over 1 ms in 120,000 boundaries (30k host 1 primary + 30k host 1 AC + 60k host 2); worst completion 151 us (host 1) and 339 us (host 2). CPU host 1 primary: 0 skips in 30,000. CPU host 1 AC: 18 skips (CPU) and 18 skips (keepalive) in 5/12 trials, pre-submission delays 0.6-3.4 ms. Host 2: 329 skips (CPU) and 345 skips (keepalive) in 23/24 trials, typically one 7.5-14 ms stall early in each 2.5 s trial plus about one 1-2 ms stall later. Bursts reached 19-28 consecutive slots. Max completion 13.5 ms (CPU) and 14.4 ms (keepalive). Some host-2 stall time was spent inside cudaGraphLaunch (100-555 us, once 1.38 ms). _(source: raw .bin flags (data-audit audit.py/stallpos.py; 5g-budgets runs.py))_
- **Completion-budget misses (budget measured from the slot boundary).** Host 1 AC, pooled over 30k: CPU+keepalive 0.110% at 160 us, 0.073% at 200-500 us; plain CPU 100% at 175 us or less (clock-limited at 1095 MHz), 0.113% at 200 us, 0.077-0.080% at 250-500 us; GPU 0.007% at 150 us, 0 at 160 us and above. Host 1 primary: plain CPU 0 at 250 us and above. Host 2, pooled over 60k: CPU 6.04% / 0.97% / 0.60% at 200 / 250 / 500 us; keepalive 1.02% / 0.92% / 0.64%; GPU 0.22% / 0.13% / 0 (0 from 400 us). Paired on host 2: GPU better than plain CPU in 12/12 at every budget from 200 to 500 us; better than keepalive in 12/12 at 200 us and 11/12 at 250-500 us. _(source: harq/analysis/deadline_sweep/summary.txt, deadline_pairs.txt; activity_control raw records)_
- **Contention: GPU launch does not protect the slot from a co-located GPU workload.** Separate-process SGEMM: CPU misses 100.00%, GPU 59.32%, paired -40.7 pp [-40.8, -40.6], 6/6. MPS: CPU 52.21%, GPU 51.24%, paired -0.88 pp [-1.33, -0.45]. p99 start about 2.47 ms in both modes under SGEMM. _(source: slotbench/data/2026-10-02_a100_cuphy_repeated/out (6 pairs, 500 us deadline))_

## Relative to production NVIDIA Aerial

The baseline must be stated carefully, because production Aerial does not launch at the boundary.

How production Aerial launches PUSCH (verified in the local clone):
- Pinned SCHED_FIFO-95 workers launch the order kernel and PUCCH at T0-500 us and the PUSCH task at T0-400 us (cuPHY-CP/cuphydriver/include/constant.hpp:127,130,132).
- The PUSCH graph waits on the GPU for the order kernel's completion event (task_function_ul_aggr.cpp:302). With sub-slot work, it is started from the GPU by symbolWaitKernel plus deviceGraphLaunchKernel (cuPHY/src/cuphy/pusch_start_kernels.cu).
- pusch_deviceGraphLaunchEn: 1 is set in 21 of 23 cuphycontroller yamls.
- Clocks are locked with nvidia-smi -lgc, and hosts use isolcpus/nohz_full.

So the following are NOT improvements over production:
- (a) The 8-28 us p99 start-error gain. The arbiter verified that our CPU thread woke on time (p99 lateness at most 0.2 us). The gap is host-submission-to-GPU-start latency of 10-21 us, which production hides by enqueueing 400-500 us early behind a GPU event.
- (b) GPU self-launch itself. Production already uses device graph launch.
- (c) GPU clock effects. Production locks clocks.
- (d) Contention. The executive does not help there.

What the executive does change relative to production:
- It removes the remaining per-slot CPU submission from the PUSCH start path, and with it production's need for the host to stay within its 400-500 us enqueue lead.
- In production a host stall longer than that lead delays the slot. If waitOrderLaunched exceeds 1 ms, the code takes error_next, then EXIT_L1, which stops the L1 with no recovery (task_function_ul_aggr.cpp:290, 518).
- Our data shows such stalls (0.6-14 ms, some inside cudaGraphLaunch) on non-real-time hosts, and the executive was immune to them.

So the gain over production is robustness of the launch step where production's real-time host guarantees cannot be provided: containers, cloud, shared or AI-co-located servers. This is argued from our measurements, not measured against production, for three reasons:
- We never ran a real-time-tuned, pre-enqueued CPU arm, so the stall rate on a correctly configured Aerial host is unknown.
- The executive is time-gated, while production is data-gated. A deployable version would need to combine timer gating with symbol-arrival gating.
- Per-slot setup, C-plane, FAPI and CRC.indication delivery still run on the CPU.

## 6G

6G is expected to keep slot-based, GPU-software RAN designs on shared AI-and-RAN infrastructure (vendor and AI-RAN Alliance direction, not a 3GPP or ITU requirement). Our 5G measurements suggest that on such hosts the per-slot host launch path, not the GPU, caused the lost slots in the idle runs. That makes a GPU-resident slot trigger a candidate design element for future GPU-based RANs. We did not test 6G numerologies, shorter slots, sensing, reliability at HRLLC levels, or isolation from co-located AI workloads, and under GPU contention the executive alone did not keep slots on time (59% misses without isolation, 51% under MPS).

Status: **argued extrapolation**.

## Must not say

- "Our findings support 6G", "enable 6G" or "are 6G-ready". Nothing 6G was measured, and there is no normative 6G timing requirement yet (Rel-20 is a study, normative work is Rel-21).
- "A better 5G architecture" without qualification, or "better than NVIDIA Aerial". The CPU baseline was a SCHED_OTHER host launching at the boundary, not production's real-time, pre-enqueued, data-gated launch.
- "First GPU-launched / CPU-free L1" or "GPU self-launch is novel". Aerial ships device graph launch for PUSCH (pusch_deviceGraphLaunchEn: 1; deviceGraphLaunchKernel), and DOCA GPUNetIO, GPU Persistent Graphs (ICPE 2025) and Blink are prior art for the mechanism.
- "Removes the CPU from the slot loop". Setup, C-plane, FAPI and output delivery still need the CPU, and fixed descriptors hide setup cost.
- "5 us vs 13-39 us start precision matters for 3GPP timing." Air-interface timing (3 us cell phase sync, TAE) is set by the RU and PTP. The gap is 1-3% of NVIDIA's own 1.25-1.5 ms PUSCH budgets, and production hides it.
- "CPU launch loses 0.6% of slots" as a long-run rate. It comes mostly from one early multi-millisecond stall per 2.5 s trial on host 2, of unidentified cause.
- "Lost transport blocks." A late PUSCH costs a HARQ retransmission and that slot's UL capacity, not the data, except where 10-14 ms bursts might use up all HARQ attempts.
- "Zero misses, so the executive meets URLLC/HRLLC (1e-5 to 1e-7)." 0 in 120,000 only bounds the rate at about 2.5e-5.
- "Solves AI-RAN GPU sharing" or "GPU launch protects the RAN from AI neighbours". The executive still misses 59% (no isolation) and about 51% (MPS).
- "GPU-held time for 6G synchronisation/ISAC." %globaltimer has a 1.024 us tick and the host/GPU mapping has 1.7-2.3 us bounds; it is not an air-interface time source.
- "The 6% CPU loss at 200 us shows CPU launch is too slow." It is a GPU clock artefact (1095 vs 1410 MHz) that does not occur with production's locked clocks.
- "Scales to 120 kHz / 125 us slots." PHY takes 135-175 us, and the skip-on-overlap design would drop about every other slot without pipelining.

## Corrections to earlier claims

- C1 — confirmed, needs a wording fix: "All 90 measured trials passed payload, TB CRC and CB CRC checks before and after each 5,000-slot replay, with 0 launch errors or timeouts. Per-slot outputs were not individually checked, and skipped boundaries were not decoded."
- C2 — corrected. '31-39 us' combines the median (31.3) and the mean (38.7) of per-trial p99; report the median 31.3 us (per-trial range 21-87). Host 1 CPU is 13.4 us, CPU+keepalive 12.2, GPU 4.7-5.1; host 2 CPU+keepalive 29.0, GPU 5.0. Add the CI for host 2: -27.9 us [-36.9, -21.2]. Add the mechanism: the host woke within 0.2 us at p99, so the gap is submission-to-start latency, which production Aerial hides by enqueueing 400-500 us early. Do not claim this gain against production.
- C3 — corrected. The stall range is 0.6-14.4 ms (13.5 is CPU-only; keepalive reached 14.4). The delay is not purely before submission: host-2 stalls spent 100-555 us, and once 1.38 ms, inside cudaGraphLaunch. State that the host-1 primary campaign had 0 CPU stalls or skips in 30,000 boundaries; stalls were in 5/12 host-1 AC trials and 23/24 host-2 trials. GPU: 0 skips and 0 over 1 ms in 120,000 idle boundaries (verified by three lenses), with the 2.5e-5 upper bound. In the HARQ README, the 'stalls >1 ms' column (354/381) is skips plus executions over 1 ms; actual executions over 1 ms are 25 (CPU) and 36 (keepalive).
- C4 — corrected. Host 1: '0.07-0.11% at 160-500 us' holds for CPU+keepalive only; plain CPU is 100% at 175 us or less (clock-limited) and 0.08-0.11% at 200-500 us; these losses come from 2-3 of 6 trials, and the primary campaign lost 0 at 250 us and above. Host 2: CPU 6.0% / 0.97% / 0.60% at 200 / 250 / 500 us; GPU 0.22% / 0.13% / 0. '12/12 at every budget' holds only against plain CPU; against the clock-matched keepalive it is 12/12 at 200 us and 11/12 at 250-500 us. Present only the keepalive comparison as the production-relevant one. Call budgets 'boundary-referenced completion budgets', not HARQ deadlines: 3GPP sets no gNB decode deadline, and NVIDIA's own PUSCH budget is 1.25-1.5 ms.
- C5 — confirmed: 59.32% vs 100.00% (paired -40.7 pp, 6/6); MPS 51.24% vs 52.21% (-0.88 pp [-1.33, -0.45]). Add that GPU p99 start was +1.2 us worse under MPS, and that both outcomes amount to cell outages: this is an isolation problem, not a launch-mode win.
- C6 — reword. No lens re-verified it. The 2026-10-01 dataset measures per-iteration GPU time (setup+run, about 2.5-2.65 ms with SGEMM/LLM neighbours; about 0.4 ms LLM / 0.57 ms SGEMM mean under MPS), not boundary-anchored start delay. The direct start-delay evidence is the lockstep proc_sgemm p99 of about 2.47 ms in both modes. Say 'co-located GPU work adds about 2.5 ms per slot via time-slicing; stream priority and green contexts did not prevent it in our tests.'
- C7 — corrected. Not 'better 5G architecture' or '6G'. Say: 'evidence for removing per-slot host submission from the GPU L1 critical path, demonstrated on non-real-time hosts; 6G relevance is argued, not measured.'
- Additional correction (production lens): production enqueues PUSCH at T0-400 us, not T0-500 us; the order kernel and PUCCH are at T0-500 us (constant.hpp:127,130,132).

## Biggest threat and fix

Threat: the CPU baseline is a strawman relative to production Aerial in two ways.
- It runs SCHED_OTHER on non-isolated, unlocked-clock containers.
- It calls cudaGraphLaunch at the boundary, whereas production pre-enqueues 400-500 us early on SCHED_FIFO-95 isolated cores, gates on the GPU, and locks clocks.

An examiner can say the start-error gain disappears under production's early enqueue (our own decomposition confirms this). They can also say the stalls may disappear on a real-time host, which would remove the deadline benefit.

Fix:
1. Add a production-faithful CPU arm. It should enqueue the frozen graph at T0-400 us from a SCHED_FIFO thread pinned to an isolated core (if the host allows it; otherwise rent bare metal), gated on the GPU by a stream wait on an event set by a small %globaltimer wait kernel at T0. Lock clocks where possible, or use the keepalive.
2. Run it against the executive for at least 3e5 boundaries per arm, both idle and with CPU stress (stress-ng on the other cores plus a telemetry and I/O load).
3. Report start error, launch-to-T0 slack, the rate of stalls over 400 us and over 1 ms (production's EXIT_L1 condition), and misses.

Interpreting the outcome:
- If the executive matches this arm on a quiet real-time host and wins under CPU stress or without isolation, the claim becomes 'removes the host real-time dependency', measured against a production-style launch.
- If it does not win, reframe the contribution as enabling GPU L1 on non-real-time and cloud hosts only.

If the rerun is impossible, state the limitation explicitly and make the thesis claim conditional on 'hosts without real-time isolation'.

## The experiment that would turn the 6G argument into evidence

No experiment can measure '6G' until 6G timing requirements exist. What can be done is to measure the conditions the 6G argument depends on: shorter slots, AI work sharing the GPU, and a production-grade CPU comparator.

Proposed experiment: a paired campaign of {production-faithful RT pre-enqueue CPU arm, GPU executive} at a 250 us period, set with SB_CUPHY_LOCKSTEP_PERIOD_US=250.
- Vector: a 60 kHz-SCS PUSCH vector, either an NR-legal 135-PRB 100 MHz case or a wider-bandwidth case approximating the upper-FR1 6GR candidates. Optionally add a 125 us run with cross-slot pipelining.
- Neighbour: a co-located AI inference process isolated by MIG or green contexts with SM reservation, so the comparison is isolation plus launch mode rather than an unprotected neighbour.
- Scale: at least 3e6 boundaries per arm (about 12.5 min at 250 us), reported with rule-of-three bounds against boundary-referenced and NVIDIA-style budgets.

What a positive result would and would not support:
- If the executive keeps zero or near-zero misses while the real-time pre-enqueued CPU arm does not, under short slots and an isolated AI neighbour, the statement 'GPU-resident slot triggering suits slot-based, shared-GPU RANs of the kind proposed for 6G' becomes measured evidence.
- It would still be NR numerology evidence, framed as 6G-relevant, not a 6G result.

## Disagreements between lenses and how they were resolved

- Materiality of the p99 start-error gain. The data-audit lens counted it as an architectural improvement; the production and 5G-budgets lenses said production hides it and it is immaterial to 3GPP timing. Resolved by the arbiter's own decomposition of the raw records (arb.py): CPU wakeup lateness at p99 is at most 0.17 us, and the gap is submission-to-start latency of 10-21 us at p50. Production enqueues 400-500 us early behind a GPU event (constant.hpp; task_function_ul_aggr.cpp:302), so the gain is real against a boundary-launching host but must not be claimed against production. The production lens wins.
- '31-39 us' host-2 CPU p99. The 5G-budgets lens confirmed it; the data-audit lens showed it combines the median (31.3) and the mean (38.7) of per-trial p99. Data-audit recomputed from raw data, so report median 31.3 us (range 21-87).
- Count of configs enabling device graph launch: the production lens said 21/23, the 6G lens 'about 10'. Arbiter grep of the clone: 21 of 23 cuphycontroller_*.yaml files set pusch_deviceGraphLaunchEn: 1.
- Production PUSCH enqueue time. The production lens said T0-500 us; constant.hpp shows order kernel and PUCCH at T0-500 us and PUSCH at T0-400 us. The difference is minor and the conclusion is unchanged.
- Consequence of a long host stall in production. Only the production lens claimed EXIT_L1; the arbiter verified that task_function_ul_aggr.cpp:290 (waitOrderLaunched with a 1 ms limit) leads to error_next and then EXIT_L1 at line 518, with no pipeline recovery.
- 120,000 idle GPU boundaries. The 5G-budgets lens re-read 90k and took 30k from the README; the data-audit and 6G lenses re-read all 120k with 0 skips. Confirmed.
- Max stall: 13.5 ms (claim) vs 14.4 ms (5G-budgets). Both are correct: 13.5 is CPU and 14.4 is CPU+keepalive. Report 'up to 14.4 ms'.
- 12/12 triplets. True against plain CPU at every budget; against keepalive it is 11/12 at 250-500 us (data-audit and 5G-budgets agree). Use the keepalive figure as the headline.
- 6G framing. The 6G lens offered a 'candidate building block' sentence that used the RAN1#122 30 kHz bridge; the 5G-budgets lens said 'no evidence about 6G'. Resolution: allow a clearly labelled argued sentence, but drop the RAN1 numerology bridge as the support, because it rests on a secondary summary and the agreements are still evolving; at most cite it with meeting number as context. Also state the contention counter-evidence in the same sentence.
- C6. No lens verified it. The data-audit lens noted the 2026-10-01 dataset measures per-iteration GPU time, not start delay. Reworded accordingly rather than confirmed.

## Lens reports (summary)

### data-audit

5G: partially_supported — The data supports one narrow architectural claim. On a dedicated, idle A100, letting the GPU time and launch NVIDIA's real cuPHY PUSCH slot graph itself has two effects. (a) It cuts slot-start p99 from about 13 us (host 1) or about 31 us (host 2) to about 5 us, and the gain survives matched SM clocks: -7.6 us host 1, -27.9 us host 2, lower in 18 of 18 triplets. (b) It removes the dependence on the host thread, whose 0.6-14 ms pre-submission stalls cause boundary skips in the CPU-launched path. No GPU-launched idle boundary was skipped or stalled. That is enough to say "GPU-timed slot launch is a better L1 launch architecture than a normal-priority host launch for a GPU-inline 5G PHY". It is not enough to say "a better 5G architecture" in general, for six reasons. 1. The CPU baseline was SCHED_OTHER with no real-time priority, no isolcpus and no clock lock, whereas production Aerial uses real-time-tuned cores. 2. Where the host was quiet (host 1, primary campaign), the CPU path had zero skips, zero stalls and zero misses at budgets of 250 us or more. On host 1 at budgets of 250 us or more the GPU advantage is 0-14 slots per 5,000, with ties in half or more of the pairs. 3. On host 2 almost all CPU misses at 250 us and above come from one systematic 7.5-14 ms stall early in each 2.5 s trial plus about one 1-2 ms stall later. The cause is unidentified and possibly the harness or driver, so "loses 0.6% of slots" cannot be read as a long-run rate. 4. The setup is fixed-vector replay: no fronthaul, PTP, MAC/HARQ, per-slot descriptors or output delivery, and correctness is checked only before and after each series. 5. The GPU's own execution varied up to 334 us (versus a 136 us median), causing 0.13% misses at 250 us on host 2. 6. Under GPU sharing the architecture does not help much: 59% misses against a separate-process SGEMM, and about 51% under MPS, only 0.9 pp better than CPU launch.

6G: argued_extrapolation — Nothing in the data is 6G-specific. Every trial is a 5G NR 30 kHz SCS, 500 us-slot, 273-PRB PUSCH. There is no higher numerology, sub-slot or mini-slot processing, AI-native receiver, ISAC or new waveform. Any 6G link can only be argued. For example: if slots shrink (125 us at 120 kHz) or completion budgets tighten, a 5 us versus 13-31 us p99 start error and immunity to host stalls would matter more. That is a hypothesis this data has not tested. The one 6G-adjacent theme the data does touch is AI-RAN, meaning AI workloads sharing the GPU with the RAN. There the measured evidence is negative: a co-located SGEMM still makes GPU-launched slots miss 59% (no isolation) or about 51% (MPS) of boundaries, and earlier studies show about 2.5 ms time-slicing delays that green contexts do not prevent. "Our findings support 6G" is therefore not a measured result. At most it is a motivation or future-work extrapolation, and for shared-GPU AI-RAN the data argues against it.

Threats:
- The CPU baseline is not production-grade. It runs under SCHED_OTHER at nice 0 with no SCHED_FIFO/RR, no isolcpus or nohz_full, and no clock lock. The GPU-launch advantage over an RT-tuned host, which is how Aerial is deployed, is untested.
- Host-2 CPU stalls are systematic rather than random: one 7.5-14 ms stall in measured slots 58-698 of every CPU and keepalive trial (23/24), plus about one 1-2 ms stall roughly 2 s later. The cause is unidentified and could be the harness, the telemetry sampler or the driver; the launch-call times of 100-555 us suggest driver-side contention. Each trial is only 2.5 s, so per-trial miss rates such as 0.6% cannot be extrapolated to long-run loss rates.
- Results depend on the host. On the quiet host-1 primary campaign the CPU path had zero skips and zero misses at budgets of 250 us or more, so the GPU deadline benefit there rests on 5 of 12 activity-control CPU/keepalive trials.
- The pooled 'stalls >1 ms' metric in cuphy_deadline_sweep.py adds skips to executions above 1 ms (354/381 versus actual 25/36), and the HARQ README labels it as stalls. Per-trial stall counts in the README mix the CPU and keepalive variants.
- Several summary numbers mix variants or statistics. '31-39 us' combines the median and the mean of per-trial p99. Host 1's '0.07-0.11% at 160-500 us' holds for CPU+keepalive but not for plain CPU at 160-175 us. '12/12 at every budget' holds against plain CPU only; against keepalive it is 11/12 at 250-500 us.
- The plain-CPU comparison is confounded by GPU clock state: SM clock 1095 vs 1410 MHz, PHY 174-179 vs 136-139 us. Only the keepalive comparison is clock-matched, and the keepalive itself changes residency and occupancy.
- The GPU path has its own tail. Execution excursions reached 334 us against a 136 us median on host 2, with clocks unlocked, giving 0.13% misses at 250 us. Zero events in 120,000 idle boundaries only bounds the rate at about 2.5e-5.
- Correctness is checked at the endpoints only: payload and CRC before and after each series, not per slot. The setup is fixed-vector replay with no fronthaul, PTP-disciplined timing, per-slot descriptor changes, MAC/HARQ progression or output delivery, which are the parts of a real L1 pipeline that would need host and GPU interaction each slot.
- The evidence is n=1 GPU model and one vector per host (two hosts, a few hours of rental, 30-60 s of measured runtime per variant). The bootstrap intervals ignore within-trial and calibration uncertainty; fit epsilon is about 1.7-2.3 us, comparable to some of the differences.
- Under any GPU sharing the GPU-launched executive does not deliver deadlines: 59% misses against a separate-process SGEMM and 51% under MPS. Any AI-RAN or co-location framing is contradicted, not supported.

### production-aerial-baseline

5G: partially_supported — Our "cpu" mode calls cudaGraphLaunch at the slot boundary. Production Aerial does not work that way, so the baseline captures only one component of production: the per-slot host enqueue.

How production schedules uplink:
- L2A tick: the tick thread runs at SCHED_FIFO 99 on a pinned core and fires slot_advance=3 slots (1.5 ms) before T0. Sources: l2_adapter_config_F08.yaml:21,32-35 and nv_tick_generator.cpp:335.
- Enqueue ahead of T0: cuphydriver works out T0 (cuphydriver_api.cpp:1435-1456). Pinned UL workers (SCHED_FIFO 95) then enqueue the order kernel at T0-500 us and the PUCCH/PUSCH task at T0-500 us (constant.hpp:127,130; cuphydriver_api.cpp:537-582; worker.cpp:354-409; task.cpp:330).
- GPU decides when compute starts, gated on data:
  - Without sub-slot work, the PUSCH graph waits on the stream for the order kernel's completion event (task_function_ul_aggr.cpp:302).
  - With early-HARQ or front-loaded DMRS, a 1-thread symbolWaitKernel polls pSymbolRxStatus, which the order kernel fills from DOCA GPUNetIO NIC receive. A deviceGraphLaunchKernel then calls cudaGraphLaunch(...FireAndForget) (pusch_start_kernels.cu:42-119; pusch_rx.cpp:9771-9850; order_cuda_kernels.cu:422,2061).
- Device graph launch is already in production: pusch_deviceGraphLaunchEn=1 in 21 of 23 shipped cuphycontroller yamls.

So production already does what our GPU path does, a GPU-side device-graph launch. The difference is that production gates it on data arrival, not on a timer, and has the CPU enqueue it with 500 us or more of slack.

What this means for the evidence:
1. The roughly 8-27 us p99 start-error advantage does not carry over to production. Our CPU host wakeups were accurate (launch lateness p50 0.0 us, p99 at most 5.4 us). The 11-22 us p50 start error is host submission-to-GPU-start latency. Production hides that latency by enqueueing early, and its GPU-gated device-graph launch should carry the same ~4-5 us latency as our executive.
2. The GPU-clock confound does not apply to production. NVIDIA locks GPU clocks at maximum (nvidia-smi -lgc in cubb_scripts/install/install_services.sh:386-410 and the install guides), so only the cpu_keepalive comparison is relevant.
3. Stall robustness is the one real gain, and it is limited. Our CPU stalls were 0.6-14 ms. On host 2 there were 24-37 submissions per 60k more than 500 us late and 19-22 more than 1.5 ms late, plus 329-345 skipped boundaries. A stall like that would also blow production's 500 us lead and could trip waitOrderLaunched(1 ms), which leads to EXIT_L1 (task_function_ul_aggr.cpp:290,518). NVIDIA even ships a CPU-stall injection hook (cuphyoam.cpp:374; worker.cpp:399), which shows it treats this as a real risk. But these stalls happened on containers without SCHED_FIFO, isolcpus, nohz_full or idle=poll, all of which production requires. We have no evidence that such stalls occur at that rate on a production-configured host.
4. The executive replaces only the launch step. Per-slot CPU work stays: pusch->setup() for descriptors, C-plane DPDK TX, FAPI/nvIPC, and the UL3 completion poll and CRC.indication at about T0+1.0 ms with 2-10 ms waits.

So for 5G the supportable claim is narrow. A resident GPU time-triggered launcher removes per-slot host submission, and its exposure to OS stalls, from the PUSCH start path. That matters mainly for deployments that cannot provide production RT isolation, such as shared or cloud or AI-co-located hosts. It is not shown to be better than production Aerial on a properly configured host.

6G: argued_extrapolation — Nothing we measured is 6G-specific. The study used 5G NR TC7304, a fixed replayed vector, no fronthaul, no MAC/HARQ progression and no per-slot setup. The 6G link has to be argued: AI-RAN and cloud-native RAN may run on shared GPUs and non-RT hosts, where the CPU timing guarantees Aerial depends on (isolcpus, SCHED_FIFO 95/99, sched_rt_runtime_us=-1, locked clocks) are unavailable. A GPU-resident time-triggered executive would remove one CPU timing dependency there. That case is plausible, but it was not measured under 6G numerologies, shorter slots, real symbol ingress or a full control plane.

The comparison with production also cuts against novelty. Production already uses GPU-side waiting (symbolWaitKernel) and device graph launch, so our contribution is a resident, time-gated executive with no per-slot CPU enqueue. It is not GPU self-launch as such.

Threats:
- Strawman risk (main threat). The 'cpu' baseline launches at T0. Production Aerial enqueues the order kernel and PUSCH at T0-500 us, L2 ticks 1.5 ms ahead (slot_advance=3), and compute start is gated on the GPU by data arrival (order-kernel completion event, or symbolWaitKernel polling pSymbolRxStatus followed by deviceGraphLaunchKernel). The CPU start-error penalty we measured (host launch-to-GPU-start latency, ~11-22 us p50) is therefore hidden in production.
- GPU self-launch is not new relative to production. NVIDIA already ships device graph launch for PUSCH (pusch_deviceGraphLaunchEn=1). The novel element is a resident executive gated on %globaltimer instead of a per-slot, CPU-enqueued, data-gated wait kernel.
- Non-RT CPU environment. The containers ran SCHED_OTHER with no isolcpus, nohz_full or idle=poll. Production requires all of these plus SCHED_FIFO 95/99 and sched_rt_runtime_us=-1. The multi-ms CPU stalls behind C3/C4 may be largely an artifact of this environment, and we never measured the production configuration.
- GPU clock confound. Production locks GPU clocks at maximum (nvidia-smi -lgc), so the plain 'cpu' rows, with the A100 idling at 1095 MHz, have no production counterpart. Only cpu_keepalive is a fair comparison.
- Scope of replacement. The executive removes only the launch call. Per-slot CPU work stays in production's critical path: pusch->setup() descriptor computation, DPDK C-plane TX, FAPI/nvIPC handling, and UL3 completion polling with CRC.indication. Frozen descriptors exclude setup cost in both modes, so 'CPU removed from the slot loop' is not demonstrated.
- Timing reference differs. Our deadlines and budgets start at the slot boundary. Production's start at last fronthaul symbol arrival (T0+~500 us+Ta4, Ta4_max 331 us) and end at CRC.indication delivery to L2 through a CPU thread.
- Stall consequence differs. In production a CPU stall over 1 ms between the order-kernel task and the PUSCH task makes waitOrderLaunched fail and then EXIT_L1 (a crash, not one missed slot). Stalls under ~500 us are absorbed. Our skip-the-slot accounting therefore models neither.

### 5g-budgets

5G: partially_supported — Only a narrow 5G claim holds. The slot-start precision gap (about 5 us for GPU vs 13-39 us for CPU at p99) does not touch any 3GPP or O-RAN requirement. Air-interface timing (cell phase sync of 3 us, i.e. +-1.5 us per cell, and TAE of 65 ns to 3 us) is enforced by the radio unit and PTP/GNSS. The O-DU's software start time is separated from the air interface by fronthaul buffering. Aerial's own configs allow UL packets to arrive anywhere in Ta4 = 50-331 us, a window about 281 us wide.

NVIDIA's own PUSCH capacity budgets are 1250-1500 us (2000 us for mMIMO), and 650-755 us for early-HARQ sub-slot work (testBenches/perf/cubb_gpu_test_config.yaml, check_cell_capacity.py). Against those budgets a p99 difference of 8-34 us is 1-3% of the budget, so it does not matter.

What does matter for 5G is the tail. The CPU-launched path, on a host set up as SCHED_OTHER in a container with no isolcpus, had multi-millisecond stalls. On host 2 the 329 CPU skips came in 30 bursts, 16 of them 19-26 consecutive slots long (about 9.5-13 ms of uplink outage). That is roughly one outage every 2 s of replay. The GPU-launched path had none in 90,000 idle boundaries that I re-read from the raw records (120,000 including host-1 out/ per its README). Outages that long could use up the HARQ retransmissions and are material.

However, NVIDIA's deployment guide requires a low-latency kernel, isolcpus/nohz_full, SCHED_FIFO priority 95 and locked GPU clocks (nvidia-smi -lgc). Vendor vRAN reference configs expect cyclictest latency of about 10 us or less. So the CPU baseline is not a deployed-5G baseline, and the 6% miss rate at 200 us on host 2 comes from GPU clocks the rental would not let us lock.

NVIDIA's uplink is also already GPU-triggered. cuPHY's symbolWaitKernel polls %globaltimer plus per-symbol RX status, and deviceGraphLaunchKernel then calls cudaGraphLaunch(..., cudaStreamGraphFireAndForget) from the device (cuPHY/src/cuphy/pusch_start_kernels.cu). The order kernel polls the NIC. So "GPU-side triggering" is not a new 5G architecture.

The fair wording: "On hosts without real-time scheduling (e.g., containerised or shared cloud GPUs), letting the GPU trigger NVIDIA's cuPHY PUSCH slot graph on its own timer removed the host-induced boundary skips (0.06% host 1, 0.55% host 2) and multi-ms stalls that a normal-priority CPU launcher showed, and tightened p99 slot-start error from 13-39 us to about 5 us, in fixed-vector single-cell replay. This is evidence for keeping the host out of the per-slot critical path, a direction NVIDIA's data-driven uplink already takes. It is not a demonstration of compliance with any 3GPP timing requirement or a URLLC-grade guarantee."

6G: argued_extrapolation — Nothing measured is 6G. All data is 5G NR FR1 numerology mu=1 (30 kHz, 500 us slots), one cell, fixed-vector PUSCH replay. 6G is still in the Rel-20 study phase, and the first normative specs are expected in Rel-21 (stage-3 freeze around Dec 2028, ASN.1 around Mar 2029), so there is no 6G timing requirement to measure against.

The only route to a 6G argument is an extrapolation: if 6G uses shorter slots and tighter processing budgets, and keeps AI and RAN on shared GPUs, a host-free trigger matters more. Example: at 120 kHz (125 us slots, 8.9 us symbols) the CPU's 31-39 us p99 start error would be 25-31% of a slot (3.5-4.4 symbols), against about 4% for GPU self-launch. That example is untested arithmetic. The measured PHY took 135-175 us, which is longer than a 125 us slot, so the skip-on-overlap design would not even work there without pipelining.

The contention results (C5/C6) argue against an "AI-native 6G on shared GPUs" framing as much as for it. Under a separate-process SGEMM neighbour, GPU launch still missed 59% of boundaries, and under MPS both modes missed about 51-52%. At most: "motivates/informs 6G AI-RAN design", stated as an argument, not as a finding.

Threats:
- The CPU baseline is not a deployed 5G baseline: it runs SCHED_OTHER in a container with no isolcpus/nohz_full, low-latency kernel or locked GPU clocks. NVIDIA Aerial and vendor vRAN configs require all of these. The multi-ms stalls (cause unidentified) may disappear on an RT-tuned host, which would take away most of the deadline-loss benefit.
- Clock-timed launch at the slot boundary is not how a deployed uplink starts. Real PUSCH starts when packets arrive (Ta4 window 50-331 us in Aerial configs, order kernel + symbolWaitKernel + device graph launch). So 'slot-start error' has no direct counterpart in deployed 5G, and NVIDIA already has the GPU-triggered path.
- The 200-500 us 'HARQ budgets' are not 3GPP gNB requirements. 3GPP gives N1/N2 only for the UE, and NVIDIA's own PUSCH capacity budgets are 1250-1500 us (650-755 us early-HARQ). Calling results 'late for HARQ' at 200 us overstates the stakes for eMBB.
- A late or skipped PUSCH is a retransmission, not a lost TB: NR UL HARQ is asynchronous, the UE keeps the data, and RLC ARQ sits above. The 'lost transport block' wording in the host-2 README overstates the effect, except where 10-14 ms skip bursts could use up the HARQ attempts.
- Losses are whole-slot and bursty: on host 2, 16 bursts of 19-26 consecutive slots. Mean percentages (0.6%) hide that the operational effect is periodic about 10-13 ms uplink outages.
- Statistical reach: 0 misses in 60k-120k boundaries bounds the GPU skip rate only to about 2.5e-5 to 5e-5 at 95%. That cannot support URLLC (1e-5) or 99.999% claims; reaching that would need at least about 300k zero-miss slots, ideally hours of runtime as in Concordia's 8 h tests.
- Workload scope: one cell, one fixed 273-PRB 4-layer MCS-27 TB (eMBB-like, not URLLC mini-slot), no fronthaul, no MAC/HARQ progression, no DL/PDSCH (where T1a windows bind tighter, with NVIDIA PDSCH budgets of 300-375 us), and no multi-cell or CPU-core-per-cell measurement. Claims about FR2, TDD turnaround, massive MIMO or core savings are not measured.
- The host-2 6% miss at 200 us comes from unlockable rental GPU clocks (1095 vs 1410 MHz). It should not be presented as a CPU-launch property, because Aerial locks clocks (nvidia-smi -lgc).
- Host-to-host variability is large (CPU skip rate 0.06% host 1 vs 0.55% host 2), so the size of the CPU-path loss is a property of the host, not of the launch architecture.

### 6g-and-prior-art

5G: partially_supported — The data supports a narrower claim than "a better 5G architecture". Setup: NVIDIA's real cuPHY PUSCH graph (TC7304) on an A100, run on a normal-priority (SCHED_OTHER) container host. Moving the slot trigger onto the GPU there removed the boundary skips and multi-millisecond stalls caused by the host. I checked the raw flags: 24 idle GPU trials, 120,000 boundaries, zero skips. It also cut p99 start error to about 5 us and lowered HARQ-budget misses in 12/12 triplets on host 2. So this is evidence for a better launch path for a GPU L1, not a better 5G architecture. Four limits:
(a) The CPU baseline is not what production uses. Aerial runs its workers at SCHED_FIFO priority 95 (cuphycontroller_F08.yaml workers_sched_priority: 95) on isolcpus/nohz_full lowlatency kernels (cubb_scripts/install/install_aerial_kernel.sh). The 0.6% loss floor on host 2 comes from host stalls and may largely be a container artefact.
(b) NVIDIA already ships device graph launch for PUSCH in production. pusch_deviceGraphLaunchEn: 1 is set in about 10 cuphycontroller configs and documented since Aerial 23-4. There, a GPU symbolWaitKernel (with %globaltimer timeouts) gates deviceGraphLaunchKernel. So the GPU already decides when the L1 starts, triggered by data arrival rather than a clock.
(c) Under the most relevant shared-GPU case the launch mechanism does not protect the RAN. With an SGEMM neighbour the GPU path still misses 59% of slots, and p99 start is about 2.48 ms in both modes. Under MPS both paths miss about 51-52%.
(d) The runs replay a fixed vector: no fronthaul ingress, no per-slot setup, no HARQ progression. Correctness is checked only at the endpoints.

6G: argued_extrapolation — Nothing in the 6G requirement documents is measured by this work, and nothing in them asks for GPU-held time or GPU-triggered L1.
- ITU-R M.2160 gives research targets only: latency 0.1-1 ms, reliability 1-1e-5 to 1-1e-7, positioning 1-10 cm, plus usage scenarios such as ISAC and AI-and-Communication. It says nothing about how a RAN is implemented.
- 3GPP TR 38.914 (v1.0.0, RP-261565, approved 14 June 2026) covers scenarios and KPIs. Public summaries give no compute or implementation requirements.
- "Removing the CPU from the critical path" is a vendor design direction, not a 6G requirement. It appears in NVIDIA DOCA GPUNetIO and in AI-RAN platform marketing (NVIDIA/Nokia ARC-Pro: "5G-Advanced to 6G through software upgrades").

The one real bridge: RAN1#122 agreed that 6GR supports at least 30 kHz SCS for TDD below 6 GHz and keeps the NR slot-based numerology. So the 500 us, 30 kHz slot tested here is also a valid 6GR sub-6 GHz TDD numerology.

Not tested:
- the higher 6G SCS and bandwidths: 60 kHz (250 us slot) or 120 kHz (125 us), and at least 200 MHz in upper FR1/FR3
- symbol-level or sub-slot processing (early HARQ and sub-slot were disabled)
- ISAC, cell-free/distributed MIMO, AI-native workloads
- reliability at HRLLC levels. Zero misses in 60,000 boundaries at budgets of 400 us or more bounds the miss probability only to about 5e-5 (rule of three, 95%). That is above the 1e-5 to 1e-7 range.

The 6G link is therefore an argument built from 5G measurements, not measured evidence.

Threats:
- The CPU baseline is weaker than production. The containers ran SCHED_OTHER with no isolcpus. NVIDIA Aerial runs its workers at SCHED_FIFO priority 95 on isolcpus/nohz_full lowlatency kernels (aerial clone: cuphycontroller_F08.yaml workers_sched_priority: 95; cubb_scripts/install/install_aerial_kernel.sh). The 0.6-13.5 ms host stalls behind the CPU loss floor may not occur on an RT-tuned host.
- Strawman trigger: production cuPHY does not launch at the boundary. L2 runs about 3 slots ahead (testMAC SLOT_ADVANCE = 3), and the GPU waits on symbol arrival (symbolWaitKernel then deviceGraphLaunchKernel, pusch_deviceGraphLaunchEn: 1). A fairer CPU arm would pre-enqueue early and gate on the GPU, which would absorb CPU jitter of tens of microseconds.
- Prior art overlaps the mechanism: Aerial device graph launch, DOCA GPUNetIO persistent kernels and Accurate Send Scheduling, ICPE 2025 GPU Persistent Graphs, Blink (arXiv 2604.07609). Only the time-triggered cuPHY application and the measurement method are new.
- No 6G parameter was measured: SCS 60/120 kHz, bandwidth of at least 200 MHz, sub-slot or symbol-level processing (disabled in the adapter), ISAC, distributed/cell-free MIMO, AI-native workloads.
- Statistical power is far below 6G reliability targets. 0/60,000 gives a 95% upper bound of about 5e-5. ITU-R M.2160 HRLLC targets 1-1e-5 to 1-1e-7, which needs at least 3e5 to 3e7 miss-free slots.
- The most 6G-specific scenario, a shared AI-RAN GPU, is where the GPU launch helps least: SGEMM 59% misses, MPS about 51% in both modes. Isolation (MPS, green contexts, MIG), not the launch path, is the binding constraint, and prior work (YinYangRAN, Weaver, arXiv 2512.06493) already covers that space.
- Only two A100 rentals, with unlocked clocks (1095-1410 MHz excursions). The keep-alive is an activity control, not a pure clock intervention. Results may not transfer to GH200, Grace-Blackwell ARC-Pro or RTX parts.
- Fixed-vector replay with endpoint-only CRC: no fronthaul ingress, per-slot descriptor setup, HARQ state or output delivery. Production per-slot setup still needs CPU or L2 involvement, so 'removing the CPU' is only partial.
- The 6G documents are moving targets. TR 38.914 v1.0.0 was approved in June 2026 and the RAN1 6GR agreements are still FFS for many numerologies. Cite meeting and version numbers.

## Sources

- /home/user/Final_year_research_MSc/slotbench/data/2026-10-02_a100_cuphy_repeated/out/*.bin, *.json, README.md
- /home/user/Final_year_research_MSc/slotbench/data/2026-10-02_a100_cuphy_repeated/activity_control/out/*.bin, *.json
- /home/user/Final_year_research_MSc/slotbench/data/2026-10-03_a100_cuphy_harq/trials/*.bin, *.json, README.md, analysis/deadline_sweep/summary.txt, analysis/deadline_pairs.txt, analysis/control/control_variant_summary.csv
- /home/user/Final_year_research_MSc/slotbench/data/2026-10-01_a100_cuphy/README.md
- /home/user/Final_year_research_MSc/slotbench/cuphy/README.md (record format, endpoint-only correctness)
- /tmp/claude-0/-home-user/c22b0607-07f1-5108-a24d-4caf6e85f472/scratchpad/verdict/arb.py (arbiter: launch lateness vs submission-to-start decomposition)
- /tmp/claude-0/-home-user/c22b0607-07f1-5108-a24d-4caf6e85f472/scratchpad/verdict/audit.py, sweep.py, stallpos.py, runs.py, lead.py, rows.json (verifier scripts)
- aerial/cuPHY-CP/cuphydriver/include/constant.hpp:124-139 (UL task offsets: order and PUCCH at T0-500 us, PUSCH at T0-400 us)
- aerial/cuPHY-CP/cuphydriver/src/uplink/task_function_ul_aggr.cpp:286-303 (waitOrderLaunched with a 1 ms limit; PUSCH waits on the order-kernel completion event), 513-518 (error_next leads to EXIT_L1, no recovery)
- aerial/cuPHY/src/cuphy/pusch_start_kernels.cu (symbolWaitKernel, deviceGraphLaunchKernel with cudaStreamGraphFireAndForget)
- aerial/cuPHY-CP/cuphycontroller/config/cuphycontroller_*.yaml (pusch_deviceGraphLaunchEn: 1 in 21/23; workers_sched_priority: 95; Ta4 50-331 us)
- aerial/cubb_scripts/install/install_services.sh (nvidia-smi -lgc), install_aerial_kernel.sh (isolcpus/nohz_full)
- https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/install_guide/installing_tools_gh.html
- https://docs.nvidia.com/aerial/archive/aerial-sdk/23-4/text/oam_guide/features.html
- https://networking-docs.nvidia.com/doca/archive/3-2-0/DOCA+GPUNetIO/
- https://dl.acm.org/doi/10.1145/3676151.3719359 (GPU Persistent Graphs, ICPE 2025)
- https://arxiv.org/abs/2604.07609 (Blink)
- 3GPP TS 38.214, TR 38.913, TS 38.133 (gNB decode deadlines are not specified; URLLC 1e-5; 3 us phase sync)
- https://www.itu.int/dms_pubrec/itu-r/rec/m/R-REC-M.2160-0-202311-I!!PDF-E.pdf (IMT-2030 targets are research targets, with no implementation requirements)
- https://www.3gpp.org/news-events/3gpp-news/6g-38914 (TR 38.914 v1.0.0)
- https://xfoukas.github.io/files/concordia_sigcomm21.pdf (vRAN 99.999% deadline context)
