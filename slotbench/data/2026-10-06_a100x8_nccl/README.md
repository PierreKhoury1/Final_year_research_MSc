# Eight A100s on one bounded time axis: per-GPU timers and NCCL all-reduce (vast.ai, 2026-10-06)

Host: vast.ai verified, Quebec, 8× A100-SXM4-40GB, driver 595.71, NCCL 2.23.4, 192 vCPU on two sockets.
**No NVLink and no GPU-to-GPU P2P in this container**: `nvidia-smi topo` shows GPUs 0–3 on socket 0 and 4–7 on
socket 1 (NODE within a socket, SYS across), and NCCL routed every ring hop through host shared memory
(`via SHM/direct/direct`). Bandwidth numbers here are therefore host-memory numbers, far below NVLink.
Method as in `../2026-10-06_a100x2_nccl` (gated stamps, per-GPU tick-edge sync before and after). Cost $0.72.

## Every GPU has its own clock (`gpus`)
| GPU | timer reading relative to GPU 0 at the same host instant | drift vs GPU 0 | bound |
|---|---|---|---|
| 1 | −27.535 s | +8.38 ppm | ±0.74 µs |
| 2 | −17.297 s | +5.03 ppm | ±0.82 µs |
| 3 | −13.567 s | +4.06 ppm | ±0.75 µs |
| 4 | −19.057 s | +5.71 ppm | ±0.93 µs |
| 5 | −22.309 s | +7.11 ppm | ±0.87 µs |
| 6 | −22.447 s | +6.39 ppm | ±0.88 µs |
| 7 | −34.654 s | +10.38 ppm | ±0.85 µs |

Offsets of tens of seconds and drift up to 10 µs per second: raw `%globaltimer` values from different GPUs of one
server cannot be compared at all, and a one-off alignment is wrong by 1 µs after 0.1 s.

## All-reduce across eight GPUs (`nccl`, 200 per size)
- **Start**: released by one host store, all eight GPUs begin within −1.4…+0.2 µs of GPU 0 at every size, inside the
  ±1.65–1.75 µs skew bounds.
- **End, 8 B**: GPUs finish in ring order, 1.1, 2.2, 3.9, 4.6, 5.2, 5.9, 8.4 µs after GPU 0 (GPUs 1–7): the last
  hop of the ring is visible at ~1 µs per GPU, above the bound from GPU 2 on.
- **End, 4 KB**: all within −1.1…−0.4 µs of GPU 0, a tie within the bound.
- **End, ≥ 64 KB**: GPUs on socket 0 (1–3) finish before GPU 0, GPUs on socket 1 (4–7) after it; at 16 MB the spread
  is −2.6 … +1.4 ms and the per-GPU span ranges 26.5–30.5 ms. The cross-socket hop dominates.
- **Host**: enqueueing one 8-GPU all-reduce takes 46–54 µs of host time (2 GPUs: 12–16 µs).
- **Causality**: 0 violations in 1 200 collectives × 56 GPU pairs.
- 8 B all-reduce span 30.7–38.9 µs per GPU; bus bandwidth peaks at 1.5 GB/s (1 MB) because every hop goes through
  host memory.
