# CPU–GPU Timing and Lockstep Execution (MSc Advanced Computing, KCL)

Early feasibility work for an MSc research project: how precisely can a CPU and a GPU be kept in step, first in *time* (do their clocks agree?) and then in *execution* (can the GPU start work at an exact chosen moment?), with 5G/6G GPU base stations as the motivating use case.

Status: exploratory. The findings in this README ran on a laptop iGPU (AMD Ryzen AI 7 350, Radeon 860M, OpenCL); the NVIDIA lockstep version in `gpu_run/` is written but not yet run. NVIDIA timing measurements with tickbound (A100, H100, and eight A100s on one host timeline, 8-9 Oct 2026) are in [FINDINGS.md](FINDINGS.md).

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

## Why a rented GPU (vast.ai)

The laptop results come from an AMD integrated GPU, whose clock is locked to the CPU and which shares memory with it. The research question is about NVIDIA GPUs, because that is what GPU-based 5G stacks (NVIDIA Aerial) run on. A discrete NVIDIA card differs in the ways that matter here: it has its own crystal (so its clock can drift from the CPU's), sits across PCIe, exposes `%globaltimer`, and supports CUDA Graphs, which is the launch mechanism Aerial's cuPHY uses every slot.

The test is rented rather than bought:

| Option | Cost | Why / why not |
|---|---|---|
| Rent an A100 on vast.ai | about $0.70–1.50 per hour; this test needs ~15 minutes | Compute capability 8.0 is what NVIDIA's cuPHY is built and tested for. Hourly billing makes a first answer cost a few dollars |
| Buy a DGX Spark | about £4,900 | Aerial-capable, but far too much before the question is known to be worth pursuing |
| Buy a used A100 | from about $14,700 | Worse value than renting or a Spark |
| KCL research computing | likely free | To check after enrolment; the preferred home for the full study |

The first run is a go/no-go check. If launch jitter on NVIDIA is already small, the lockstep idea is dropped for a few dollars. If it is large, or grows when an AI job shares the GPU, it becomes the thesis direction.

Caveats of rented machines, to be recorded with every result:
- They run in containers on shared hosts, so timing can include virtualisation effects. The host GPU model, driver and machine ID are logged in `gpu_info.txt`.
- NVIDIA's full Aerial container may need system privileges a rented container does not grant. `gpu_run/` therefore depends only on CUDA; Aerial's cuPHY pipeline is an optional second step.

To run: rent an A100 with a PyTorch "devel" image (it includes `nvcc`), copy `gpu_run/` to the machine, then `bash run.sh`. It writes `results.jsonl` and prints a table. Destroy the instance afterwards, because a stopped instance still bills for storage.

## Open questions

1. On NVIDIA GPUs, how large is kernel-launch timing jitter, and how much does a co-located AI job inflate it?
2. Does GPU self-timed execution keep its precision when it does real work under contention?
3. How much slot-budget safety margin do GPU 5G stacks (e.g. NVIDIA Aerial) actually reserve for that jitter?

## References

- TempoTrace, arXiv:2609.23301 (unreviewed white paper; GPU timestamp claims are design targets): https://arxiv.org/abs/2609.23301
- NVIDIA Aerial CUDA-Accelerated RAN: https://github.com/NVIDIA/aerial-cuda-accelerated-ran
- REEF, microsecond-scale GPU preemption (OSDI 2022): https://ipads.se.sjtu.edu.cn/_media/publications/reef-osdi22.pdf
- RTGPU, real-time GPU scheduling: https://arxiv.org/pdf/2101.10463
