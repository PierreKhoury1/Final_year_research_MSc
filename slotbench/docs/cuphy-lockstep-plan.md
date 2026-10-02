# cuPHY lockstep integration notes

Updated 2026-10-02. The experimental adapter is implemented. Its CUDA helpers compile with CUDA 12.6 for SM80, and patch application against the pinned revision passes. A full upstream build and actual cuPHY GPU validation are still pending. Synthetic results and the earlier A100 cuPHY run do not validate this adapter.

Upstream is NVIDIA `aerial-cuda-accelerated-ran`, pinned to [`4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c`](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/commit/4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c), committed 2026-09-15. See [usage and output details](../cuphy/README.md).

## Observed upstream integration points

- The standalone target [`cuphy_ex_pusch_rx_multi_pipe`](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c/cuPHY/examples/pusch_rx_multi_pipe/CMakeLists.txt#L17) links the example, common datasets, `cuphy_channels`, `cuphy_hdf5`, and `cuphy`.
- [Dataset defaults](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c/cuPHY/examples/common/datasets.cpp#L996) disable device graph launch. The adapter overrides this only in the opted-in PUSCH example before constructing the pipeline.
- `PuschRx` [constructs and instantiates](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c/cuPHY/src/cuphy_channels/pusch_rx.cpp#L1339) the full-slot graph. [Setup](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c/cuPHY/src/cuphy_channels/pusch_rx.cpp#L7382) updates executable parameters and enable state, then uploads the graph. The adapter borrows this configured executable; reinstantiating the source graph would omit executable updates.
- The original [example loop](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c/cuPHY/examples/pusch_rx_multi_pipe/pusch_rx_test.cpp#L647) inserts synchronization, delays, and event bookkeeping. The new path runs one ordinary setup/decode baseline, then enters replay before that loop.
- The original [run path](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c/cuPHY/src/cuphy_channels/pusch_rx.cpp#L9832) includes readiness and output handling. NVIDIA's [`deviceGraphLaunchKernel`](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c/cuPHY/src/cuphy/pusch_start_kernels.cu#L95) already launches a supplied graph after a symbol-readiness check. This experiment replays prepared input and does not replace that production protocol.

## Implemented artifacts

| Repository artifact | Purpose |
| --- | --- |
| [Upstream patch](../cuphy/patches/cuphy-lockstep.patch) | Patches six files: `pusch_rx.hpp/.cpp`, `cuphy_api.h`, `pusch_rx_test.cpp`, and channel/example CMake files. Adds a checked borrowed-graph getter and optional benchmark path. |
| [Executive](../cuphy/cuphy_lockstep.cu), [header](../cuphy/cuphy_lockstep.h) | CPU/GPU scheduling, calibration, records, summaries, and validation callbacks. Compiled into the example with separable CUDA compilation and `cudadevrt`. |
| [DAG stamps](../cuphy/cuphy_lockstep_stamps.cu), [header](../cuphy/cuphy_lockstep_stamps.h) | Adds markers before every original root and after every original leaf before instantiation. Compiled into `cuphy_channels`. |
| [Cloud bootstrap](../cloud/onstart_cuphy_lockstep.sh) | Pins upstream, checks/applies the patch, copies helper files, builds, generates TC 7304, and runs guarded CPU/GPU cases. |
| [Collection checker](../cloud/cuphy_lockstep_check.py) | Checks endpoint correctness flags, raw-row counts, ordered timestamps, and recomputed deadline misses. |

`SB_CUPHY_LOCKSTEP_MODE=cpu` or `gpu` enables the path; with the variable absent, the original example runs. The adapter requires one pipeline/vector/transmission, full 7.2a processing, and fresh SCH data (`ndi=1`, `rv=0`, no UCI). It rejects green contexts, subcontexts, neural receiver configuration, delayed nodes, partial processing, and incompatible full-slot modes. It disables early HARQ, early SCH decoding, and work cancellation; the getter also rejects front-loaded DMRS sub-slot processing.

Both modes reuse the configured graph, input, descriptors, buffers, priority, target schedule, and skip rule. Targets at or before the previous PHY end stamp are skipped. CPU mode launches and polls stream completion. GPU mode launches one PHY graph fire-and-forget and tail-launches its executive. The next generation consumes completed stamps before reusing the executable. Sequence increments and ordered, nonzero stamps detect stale or missing completions.

Kernel node priorities, including instrumentation, are explicitly matched to the example stream and honored with `UseNodePriority`; the executive uses captured stream priority. NVIDIA documents [node priority behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html) separately from launch-stream priority. This avoids relying on undocumented device-launch priority inheritance.

CUDA requires uploaded device graphs, restricts node types, and prohibits overlapping device launches of one executable. Tail launch waits for generated child work. One child per generation also stays below the fire-and-forget limit. These rules motivate the single-handle design. [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

## Validation and interpretation

The helper checks payload and all TB/CB CRCs after the ordinary baseline and after the entire replay series, using the normal output-copy path and [`EvalDataset::computeNumCbErrors`](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c/cuPHY/examples/common/datasets.cpp#L2551). These are **endpoint-only correctness checks**. Intermediate outputs are overwritten and are not individually validated.

Setup, descriptor updates, calibration, and output copies are outside replay timing. Markers bracket the full-slot DAG and include marker scheduling overhead; they do not timestamp the first and last signal-processing instruction. Pre/post calibration estimates host/GPU clock mapping. GPU-target precision is reported separately. The mapped-memory calibration handshake retains slotbench's platform assumptions and is not a general portable CPU/GPU atomic protocol.

The cloud recipe targets A100 SM80, CUDA 13.3, and Ubuntu 24.04. Isolated CPU/GPU cases must pass before process and MPS contention tests. Full build, decoded-data/CRC, freshness, accounting, and contention results remain necessary before performance claims. The [previous A100 run](../data/2026-10-01_a100_cuphy/README.md) exercised an earlier experiment.

This experiment concerns repeated ready input. Production integration must separately preserve changing slot descriptors, symbol arrival, per-slot setup, HARQ progression, and output delivery, and validate individual slot outputs and deadlines.