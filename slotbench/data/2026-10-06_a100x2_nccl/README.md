# NCCL all-reduce on two A100s, every GPU on one bounded time axis (vast.ai, 2026-10-06)

Host: vast.ai verified, California, 2× A100-SXM4-40GB, driver 580.178, NCCL 2.23.4. **The two GPUs were connected by
PCIe through the host bridge (`nvidia-smi topo`: NODE), not NVLink**; NCCL used `P2P/direct pointer`, ring 0→1→0.
These are PCIe numbers, not what NVLink-connected A100s would give.

Method (`gputrace/gputrace_nccl.cu`): each GPU gets its own tick-edge clock sync before and after (bounds 0.79 / 0.88 µs);
on each GPU's stream a one-block stamp kernel runs just before and just after every `ncclAllReduce`. In `run2_gated`
a gate kernel on every stream holds the stamps until the host releases all gates with one store to mapped memory
*after* `ncclGroupEnd()` returns, so stamp → all-reduce → stamp run back to back. `run1_ungated` and
`run2_gated/nccl_nogate` omit the gate. 200 collectives per size. `run1_ungated/gpus.*`: the per-GPU timer mapping.
Analysis: `python3 analysis/gputrace.py run2_gated/nccl`; viewer: `python3 analysis/gputrace_export.py run2_gated/nccl`.

## Results (run2, gated; p50 unless stated)

| size | span on GPU 0 / GPU 1 | start skew GPU1−GPU0 (p50 / p99) | end skew (p50 / p99) | bus bw |
|---|---|---|---|---|
| 8 B | 13.3 / 14.3 µs | −0.30 / 1.75 µs | +0.72 / 1.75 µs | — |
| 4 KB | 15.4 / 15.4 µs | −0.32 / 1.73 µs | −0.31 / 0.72 µs | 0.3 GB/s |
| 64 KB | 35.8 / 36.9 µs | −0.33 / 1.72 µs | +0.70 / 2.76 µs | 1.8 GB/s |
| 1 MB | 225 / 225 µs | −0.37 / 1.70 µs | −0.37 / 1.69 µs | 4.7 GB/s |
| 16 MB | 3.08 / 3.06 ms | 0.01 / 1.16 µs | **−14.3** / 10.2 µs | 5.5 GB/s |
| 128 MB | 24.28 / 24.27 ms | −0.21 / 1.14 µs | **−14.9** / 11.6 µs | 5.5 GB/s |

Skew bound (sum of the two GPU bounds): ±1.67 µs. Span = all-reduce + two inter-kernel gaps (≈ 2 µs each on A100).

- **Both GPUs start within the bound of each other when released together**, and for messages up to 1 MB they also
  finish within the bound (|end skew| p50 ≤ 0.72 µs). From 16 MB, GPU 1 finishes ≈ 14 µs before GPU 0, well outside
  the ±1.67 µs bound: a real asymmetry of the ring's last step, not measurement error.
- **Causality**: no GPU finished before the other had started, in 2 000 collectives (gated and ungated); 0 violations.
- **The host's enqueue dominates small collectives**: `ncclGroupStart … ncclGroupEnd` for two GPUs takes 12–16 µs of host
  time. Without the gate the measured 8 B span is 23.6 µs instead of 13.3 µs, and the start skew is 3.6 µs instead of
  −0.3 µs (the host enqueues the GPUs one after the other). Measured with the same tool, same host, same hour.
- A host store to mapped memory is seen by a polling gate and turned into a running stamp kernel in 5.2–5.9 µs.
- **Per-GPU timers**: on this host GPU 1's `%globaltimer` reads +446.187 ms relative to GPU 0 (±1.7 µs) and the
  two drift apart at 1.0 ppm (`run1_ungated/gpus`); on the other 2×A100 host it was −1.742 s and +0.63 ppm.

Cost of both runs: ≈ $0.14.
