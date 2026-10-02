# Actual NVIDIA cuPHY PUSCH lockstep replay — A100, 2026-10-02

**Both CPU and GPU launch modes executed NVIDIA Aerial's actual cuPHY PUSCH receiver.** All six cases passed decoded-payload, TB CRC, and CB CRC checks before and after replay, with zero launch errors, stale/missing completion records, or timeouts. Deadline misses under contention remain substantial.

This validates a fixed-vector PHY replay experiment. It does not establish operation of a complete live Aerial base station: radio/fronthaul ingress, changing descriptors, per-slot setup, MAC, HARQ progression, and output delivery are outside the measured interval. Intermediate decoded outputs are not individually checked.

## Configuration and observed results

- NVIDIA A100-SXM4-40GB, 108 SMs; driver 595.58.03, CUDA runtime/compiler 13.3; Ubuntu 24.04.
- NVIDIA upstream commit [`4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c`](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/commit/4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c), slotbench commit `a235559`.
- Official generated TC7304: 273 PRBs, 100 MHz at 30 kHz SCS, four layers/four receive antennas, 256-QAM, MCS 27, one transport block containing 152 code blocks.
- One configured full-slot 7.2a PHY graph, fresh RV0 SCH data, no UCI; early HARQ, early SCH decoding and work cancellation disabled. CPU/GPU modes use the same graph configuration, explicit node priority -2, target schedule, and completion-based skip policy.
- Each case: 100 warmup boundaries, then 1,000 measured boundaries at 500 µs spacing and a 500 µs completion deadline. The SGEMM adversary runs in a separate process at 100% duty; MPS cases cap the adversary at 50% active threads.

Timing percentiles below include executed measurements only. The miss denominator includes **all 1,000 measured boundaries**, including skipped ones. A skipped boundary is covered by the previous PHY execution and receives no new launch.

| Condition | Launcher | Executed | Skipped | Deadline misses | p99 start error, µs | p99 PHY interval, µs | p99 completion from target, µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Idle | CPU | 1000 | 0 | 0.0% | 14.006 | 185.344 | 197.557 |
| Idle | GPU | 1000 | 0 | 0.0% | 5.069 | 144.384 | 148.208 |
| Separate SGEMM | CPU | 209 | 791 | 100.0% | 2474.305 | 163.840 | 2636.097 |
| Separate SGEMM | GPU | 521 | 479 | 59.5% | 2473.534 | 2696.192 | 2699.640 |
| SGEMM with MPS | CPU | 737 | 263 | 47.9% | 16.034 | 1539.072 | 1551.317 |
| SGEMM with MPS | GPU | 724 | 276 | 50.0% | 14.225 | 1356.800 | 1363.270 |

The idle GPU launch path has lower observed p99 start error. Separate-process contention improves in the GPU case but still misses most deadlines. MPS provides no measured deadline-miss advantage for GPU launching in this run. These are single short runs on one host, in fixed order; they are not confidence intervals or evidence of a general production latency benefit. GPU clocks were not locked, so the different PHY execution intervals cannot be attributed solely to launch overhead. The container denied the example's request for CPU real-time scheduling; this is a normal-priority CPU baseline. The example also logged a missing optional nvlog configuration and continued with its fallback logging.

The global timer tick was 1.024 µs. Clock-fit bounds were approximately 1.8–2.3 µs; start errors use centered pre/post calibration. The small p99 start-error differences under contention are within or comparable to these measurement limits. Instrumentation brackets all original graph roots/leaves and includes marker scheduling overhead. With the restricted work-cancellation configuration, the full-slot DAG contains no enabled nested device-graph launches; this matters for interpreting the end marker.

## Evidence and reproduction

- [Independent audit](audit.json): recomputed raw timing distributions/miss counts, matching source hashes and CPU/GPU settings, CRC log checks, adversary overlap, and MPS membership. No audit errors were found. Its dataset name refers to the original collected directory, now stored as `out/` here.
- [out/](out/): full PHY logs, per-case JSON, 64-byte raw timestamp records, adversary logs/timelines, MPS server logs, run manifests, source/vector hashes, and the exact applied NVIDIA patch.
- [logs/](logs/): successful reconfiguration/incremental build and runtime/MPS logs.
- [bootstrap/](bootstrap/): original dependency/vector generation logs, initial configuration failure, successful full build, binary hash/link dependencies, and the scripts used for the parallel repair build/vector generation. The initial missing `CUDA::cudadevrt` CMake target was fixed by using CMake's automatic device-runtime linkage. CUDA 13 dependency API signatures were also fixed before the successful full build.
- [mechanism/](mechanism/): separately labeled synthetic branched-DAG launcher smoke. Both modes passed 64 measured slots and payload/execution-counter checks, including poisoning the baseline buffers. These files are not cuPHY results.
- [Adapter usage](../../cuphy/README.md), [bootstrap](../../cloud/onstart_cuphy_lockstep.sh), [incremental runner](../../cloud/debug_cuphy_lockstep.sh), and [raw checker](../../cloud/cuphy_lockstep_check.py).

The TC7304 PUSCH input is 64,214,888 bytes, SHA256 `85796206a068087c3e2e03bafb7a22118fa7f27ce87f94f00afcf1ec4ac106ca`. Its schema is preserved in `bootstrap/out/test_vector_schema.txt`, and [parameter values extracted from the verified HDF5](out/test_vector_parameters.json) confirm the workload. The input's early-HARQ flag is overridden to zero by the experimental adapter. The input was downloaded and hash-verified locally outside Git at `C:\Users\pierr\Desktop\Codex\cuphy_test_vectors\TVnr_7304_PUSCH_gNB_CUPHY_s0p0.h5`; regeneration is available through the pinned recipe.

The complete downloaded evidence archive had SHA256 `db09a491c52184876c907f8574debe3f44719b09d14a5aaf596ec902606baf8c`; its extracted files are preserved here. Two rentals were used: instance 53896943 failed to start its container and was destroyed; instance 53898272 completed the tests and was destroyed after collection. Both were verified absent. Estimated combined rental cost was about $0.23, excluding any separately billed transfer charges.
