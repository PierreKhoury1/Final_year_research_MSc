# gputrace on a 2×A100 SXM4 host (vast.ai, 2026-10-06, commit c7ab0ab)

Second A100 host (verified, US, 48 vCPU, driver 580.173, two A100-SXM4-40GB). The full campaign ran on GPU 0 and
the `gpus` strategy clock-synced both GPUs from one host thread, 3 rounds each, interleaved. Only the analysis
JSONs, metadata and logs were recovered (the raw archive lost one of 26 parts to ssh-proxy lines interleaved in
the container log); the `gpus` result below is computed from 9 000 up and ~8 800 down edge samples per GPU.

## The two GPUs' hardware timers

| | GPU 0 | GPU 1 |
|---|---|---|
| edge bound (host ↔ this GPU's `%globaltimer`) | 0.77 µs | 0.83 µs |
| rate vs host | −3.24 ppm | −2.61 ppm |
| `%globaltimer` reading at the same host instant, relative to GPU 0 | 0 | **−1 742 487.04 µs ± 1.6 µs** |
| rate relative to GPU 0 | 0 | **+0.63 ppm** |

`%globaltimer` is a per-GPU counter: the two cards in one host differ by 1.74 s and drift apart at 0.63 µs per
second. Timestamps taken on two GPUs cannot be compared without a per-GPU mapping to a common clock, and that
mapping has to be refreshed about once a second to hold ±1 µs. Both mappings came out of plain PCIe traffic from
one host thread with no NIC timestamping, each with a hard bound.

## Replication of the single-GPU results on this host (p50)

Launch → first instruction 5.9 µs (graph 5.0 µs; first A100 host: 4.2 / 4.1 µs). Consecutive kernels 2.0 µs.
Kernel end → host: flag 1.1 µs, event 3.2 µs, sync 3.2 µs. Two streams: B waits A's block length − 100 µs
(1.9046 ms at 2 ms blocks), priority or not. Time-slicing: quantum 2.089 ms (us) / 2.431 ms (them), hog on 108/108
SMs; under MPS at 50 %: 5 gaps of 0.41 ms over the window, hog on 54/54 SMs. Copies: 8 B 8.5 / 8.7 µs; GPU read
of host memory 1.1 µs.
