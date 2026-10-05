# Direction search (2026-10-05): candidates and independent vetting

Six candidates survived selection from 15; each was checked by a prior-art hunter and a feasibility/impact critic.

| Candidate | Novelty | Feasibility/impact | Why |
|---|---|---|---|
| Measure the stale-payload race in speculative all-reduce (SiFAR, arXiv 2607.08973, MICRO'26) under natural NVLink skew | viable | viable | No published measurement; integer payloads give exact ground truth; authors concede the race and rest on an untested "monotonicity" assumption; their own data shows 10-32% mis-speculation. Ceiling limited: deployed stacks (NCCL LL, FlashInfer) use race-free designs. |
| CUPTI/Kineto timestamp error grows with idle gap x GPU crystal ppm | weak | viable (reframed) | Measured numbers would be new; qualitative behaviour already acknowledged. Good measurement chapter, not a breakthrough. |
| Device-launched work escaping green-context SM partitions | weak | viable | Already shown publicly on NVIDIA forums (Aug 2026); NVIDIA docs disclaim it. |
| URLLC latency floor of cuPHY PUSCH + fused receiver | weak | viable | Floor already published 2025-26; fused receiver is standard megakernel technique. |
| SM partitions do not isolate memory/power (cuPHY vs AI) | weak | (vet failed: usage limit) | Memory half published (Beaver, Oct 2026); power coupling published outside RAN. |
| GPU device-side noise census | weak | weak | Mostly covered or predictably null; but see the lead below. |

## Free lead found in existing data (verified 2026-10-05)
data/2026-10-03_a100_cuphy_harq/trials, GPU-launcher PUSCH, 12 trials x 5000 slots: slots with exec > 180 us
(median 136 us) fall on one slot parity within a trial in 147 of 150 cases (1 ms period at a 500 us slot), adding
~100-190 us. Same parity lock in the CPU+keep-alive launcher. Cause unknown (cuPHY buffer alternation, a 1 kHz
driver/firmware source, or an SM-wide pause). Discriminating test: vary the slot period (400/600/750/1000 us) with a
%globaltimer gap-detector warp on a spare SM.
