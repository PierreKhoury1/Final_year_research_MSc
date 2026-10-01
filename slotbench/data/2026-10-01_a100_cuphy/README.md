# NVIDIA Aerial cuPHY PUSCH on a rented A100 SXM4 40GB (2026-10-01)

cuPHY built from github.com/NVIDIA/aerial-cuda-accelerated-ran (main) with cloud/onstart_cuphy.sh; test vector
TC 7304 (273 PRB = 100 MHz at 30 kHz SCS, 4 layers, 4 Rx, 256-QAM MCS 27, 152 code blocks) generated with
the MATLAB Runtime; `cuphy_ex_pusch_rx_multi_pipe -m 1 -r 3000` (CUDA graphs, 1 ms delay kernel between
iterations). Neighbour = slotbench `bin/adversary` at 100% duty in a separate process (proxies, not real models).
gc = cuPHY in a 54-SM green context; mps = MPS daemon, neighbour capped at CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50.

GPU time per slot (CUDA events, setup + run, us; the single ~52 ms maximum in every case is the first
iteration and is excluded here). Block error rate 0 in every case.

| case | mean | p90 | p99 | neighbour units/s |
|---|---|---|---|---|
| alone | 301 | 294 | 311 | - |
| alone, green context | 381 | 370 | 375 | - |
| alone, MPS | 312 | 305 | 314 | - |
| LLM proxy, no isolation | 2599 | 2605 | 2612 | 349 tok/s |
| LLM proxy, green context | 2650 | 2655 | 2662 | 315 |
| LLM proxy, MPS | 395 | 406 | 418 | 350 |
| sgemm, no isolation | 2531 | 2631 | 2634 | 109 GEMM/s |
| sgemm, green context | 2580 | 2730 | 2733 | 98 |
| sgemm, MPS | 573 | 872 | 1272 | 64 |

Files: out/<case>.pusch.txt (full cuPHY report), out/<case>.timing.json (summary + histogram per phase),
out/<case>.adversary.json, logs/ (build, test-vector generation), instance.log.
