# CPU–GPU Timing and Lockstep Execution (MSc Advanced Computing, KCL)

Early feasibility work for an MSc research project: how precisely can a CPU and a GPU be kept in step, first in *time* (do their clocks agree?) and then in *execution* (can the GPU start work at an exact chosen moment?), with 5G/6G GPU base stations as the motivating use case.

Status: exploratory. Everything so far ran on a laptop iGPU (AMD Ryzen AI 7 350, Radeon 860M, OpenCL). The NVIDIA version is written but not yet run.

## Findings so far (laptop, AMD Radeon 860M)

| Finding | Result |
|---|---|
| OpenCL `clGetDeviceAndHostTimer` on this driver | Returns the host clock twice; the GPU clock is never read |
| GPU real-time counter | 99.81 MHz, read via `s_sendmsg_rtn` (msg 131) on RDNA 3.5 |
| CPU TSC : GPU counter | Exactly 20 : 1 (within 0.6 ppb): one shared clock source on this APU |
| Best CPU↔GPU bracket (native ping-pong over fine-grained SVM) | 1.16 µs; typical 16–200 µs; worst tens of ms |
| Offset bound after keeping the fastest round trips | ≤ 0.84 µs within ~1 s of sampling, in every load condition |
| Start GPU work at a chosen instant T (`demo/lockstep.py`) | Normal launch: ~90 µs late, worst 1.1 ms. GPU self-timed on its own clock: within the ±0.7 µs clock bound, idle and under load |

Caveat on the last row: the self-timed kernel only stamped the time and did no real work, so it did not compete with the load for compute.

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Clock experiments: `probe*.py`, `exp.py` (Python/OpenCL v1), `pingpong.c` (native C ping-pong, build with `py -m ziglang cc -O2 -target x86_64-windows-gnu pingpong.c -o pingpong.exe -lwinmm`), `ana2.py`, `improve.py` (min-filter calibration and time-to-sync), `pingpong_cuda.cu` (NVIDIA port), `build.js` (slide deck generator) |
| `data/` | Raw measurements (`pp_default.bin`, `pp_tuned.bin`, `data.npz`) and `summary.json` |
| `demo/` | `lockstep.py` (four ways to start GPU work at time T) with `lockstep_results.json`; `phy.py` (unfinished GPU 5G receiver chain, paused) |
| `gpu_run/` | NVIDIA A100 test, not yet run: `lockstep.cu` (normal launch, CUDA Graph, persistent block, GPU self-timed via `%globaltimer`), `load_torch.py` (competing AI load), `run.sh` |
| `storyboard/` | Experiment visual for an alternative idea (distributed MIMO holdover): diagram page plus AI-generated illustrations |
| `GPU_Clock_Findings.pptx` | 13-slide summary of the laptop clock findings |

## Open questions

1. On NVIDIA GPUs, how large is kernel-launch timing jitter, and how much does a co-located AI job inflate it?
2. Does GPU self-timed execution keep its precision when it does real work under contention?
3. How much slot-budget safety margin do GPU 5G stacks (e.g. NVIDIA Aerial) actually reserve for that jitter?

## References

- TempoTrace, arXiv:2609.23301 (unreviewed white paper; GPU timestamp claims are design targets): https://arxiv.org/abs/2609.23301
- NVIDIA Aerial CUDA-Accelerated RAN: https://github.com/NVIDIA/aerial-cuda-accelerated-ran
- REEF, microsecond-scale GPU preemption (OSDI 2022): https://ipads.se.sjtu.edu.cn/_media/publications/reef-osdi22.pdf
- RTGPU, real-time GPU scheduling: https://arxiv.org/pdf/2101.10463
