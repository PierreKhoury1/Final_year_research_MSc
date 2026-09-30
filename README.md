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
| `gpu_run/` | NVIDIA test, not yet run: `lockstep.cu` (normal launch, CUDA Graph, persistent block, GPU self-timed via `%globaltimer`), `load_torch.py` (competing AI load), `run.sh` (three conditions plus a short clock ping-pong), `onstart.sh` (vast.ai on-start wrapper), `vast_run.py` (rent, run, collect, destroy through the vast API) |
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

### Running the NVIDIA test

Decision rule, written down before the run so the result means something either way:

| Result of the separate-process load condition | Reading |
|---|---|
| p99 start lateness of a normal or CUDA Graph launch under 50 µs (a tenth of a 500 µs slot) | Launch jitter is too small to build a thesis on. Drop the lockstep direction. |
| p99 in the hundreds of µs, or millisecond tails from context time-slicing, and the self-timed block stays within its clock bound | The mechanism is worth a proper study: a slot-shaped pipeline, minutes not seconds, with stream priorities and MPS as baselines, deadline misses per million slots as the metric. |
| Self-timed block also loses precision under the other process's load | The resident-kernel idea does not survive time-slicing. The thesis becomes a characterisation, not a mechanism. |

Three ways to run it, cheapest first:

1. **Driver script (no SSH).** `pip install vastai`, put the API key in `VAST_API_KEY`, then `python3 gpu_run/vast_run.py --search` to see offers and `python3 gpu_run/vast_run.py` to rent the cheapest fit, run, collect and destroy. Results land in `gpu_run/vast_results/<instance>/`.
2. **By hand on vast.ai.** Rent an A100 with a PyTorch `devel` image (it has `nvcc`) and paste this as the on-start command: `bash -c "curl -fsSL https://raw.githubusercontent.com/PierreKhoury1/Final_year_research_MSc/claude/modest-ride-bc8yxa/gpu_run/onstart.sh | bash"`. Read the results from the instance log; they sit between `LOCKSTEP_RESULTS_BEGIN` and `LOCKSTEP_RESULTS_END`.
3. **From a shell on any CUDA machine.** Clone the repo and `bash gpu_run/run.sh`. It writes `results_<utc>.tgz` next to itself.

Destroy the instance afterwards, because a stopped instance still bills for storage. The run takes about four minutes plus image load.

## Open questions

1. On NVIDIA GPUs, how large is kernel-launch timing jitter, and how much does a co-located AI job inflate it?
2. Does GPU self-timed execution keep its precision when it does real work under contention?
3. How much slot-budget safety margin do GPU 5G stacks (e.g. NVIDIA Aerial) actually reserve for that jitter?

## References

- TempoTrace, arXiv:2609.23301 (unreviewed white paper; GPU timestamp claims are design targets): https://arxiv.org/abs/2609.23301
- NVIDIA Aerial CUDA-Accelerated RAN: https://github.com/NVIDIA/aerial-cuda-accelerated-ran
- REEF, microsecond-scale GPU preemption (OSDI 2022): https://ipads.se.sjtu.edu.cn/_media/publications/reef-osdi22.pdf
- RTGPU, real-time GPU scheduling: https://arxiv.org/pdf/2101.10463
