# Time-aware GPU gating vs static MPS sharing: NVIDIA cuPHY PUSCH (TC7304) on one A100 SXM4 (vast.ai, Alberta)

Runs: `2026-10-05_a100_cuphy_gating/` (time-slicing and MPS-50%, n = 512/768/1024) and
`2026-10-05_a100_cuphy_gating_mps_shares/` (MPS tenant at 50/25/10% of threads, n = 512/768). 4 repeats x 5000 slots
per arm, 500 us period, CPU launcher pinned, tenant = FP32 SGEMM chunks (a stand-in for AI work, not a model).
Gate: tenant issues a chunk only in [S_k + 229 us, S_{k+1} - 30 us - est]; 229 us = p99.9 of on-time cuPHY completion + 30.
Misses below are GPU-caused only (slots launched > 50 us late by the host, ~0-1%, are excluded); AI compute = 2n^3 x GEMMs/s.

| Arm (MPS share, chunk) | AI TFLOP/s | GPU misses @160 us | @175 us | @200 us |
|---|---:|---:|---:|---:|
| MPS 50%, n=768, **gated** | 3.41 | **0.06%** | **0.00%** | **0.00%** |
| MPS 50%, n=512, **gated** | 2.15 | 0.01% | 0.00% | 0.00% |
| MPS 25%, n=512, ungated | 3.14 | 65.6% | 3.1% | 0.00% |
| MPS 25%, n=768, ungated | 3.85 | 74.8% | 48.7% | 18.9% |
| MPS 10%, n=512, ungated | 1.50 | 50.5% | 19.7% | 0.01% |
| MPS 50%, n=512, ungated | 5.19 | 99.98% | 96.8% | 8.7% |
| MPS 50%, n=768, ungated | 7.28 | 100% | 99.95% | 84.3% |

Findings (this host only):
- At a 200 us completion budget a well-chosen static share (MPS 25%, small chunks, no gate) already meets every
  deadline with 3.14 TFLOP/s of AI; the gate's best (3.41) is only ~9% more.
- At 175 us and 160 us only gated arms meet the budget; every ungated MPS setting tried misses 3-100%.
- Under process time-slicing the gate does not help: a context switch costs a few hundred us (a 69 us chunk took
  ~373 us inside a gap), so chunks overrun into the next slot (13% / 50% misses at 500 us for n = 768 / 1024, gated and not).
- MPS n=1024 chunks never fit the gap (the gate correctly stays closed: 0 AI).
- These budgets (160-200 us) are far tighter than NVIDIA's production PUSCH budget (about 1.25-1.5 ms); at relaxed
  budgets plain MPS sharing suffices. cuPHY-alone runs leave the GPU idle and down-clocked (176 us exec vs ~138 us
  with a tenant), so compare gated vs ungated arms, not against "alone", at tight budgets.
Not covered: MIG / green contexts, a real AI model, other hosts.
