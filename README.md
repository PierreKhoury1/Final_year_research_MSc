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

What the run is for, and what we expect, written down before running it. A literature check (30 Sep 2026) changed the purpose: the size of NVIDIA launch jitter is already known, and NVIDIA Aerial never asks the GPU to start at an exact instant (see *What is already known* below). The run now validates the measurement tool and gives the first numbers nobody has published.

| Measurement | Prediction | If the prediction fails |
|---|---|---|
| `globaltimer` update step | ~1 µs on an A100 (pre-Hopper); 32 ns only if a profiler has reconfigured it | A 32 ns step means the timer was left reconfigured by an earlier tool on that host; note it, the self-timed floor changes |
| Normal and CUDA Graph launch, idle | median 5–15 µs late, p99 under ~30 µs on a rented box | Much worse points to host noise (no isolated cores, no `SCHED_FIFO`): check `pinned` and `sched_fifo` in the output before blaming the GPU |
| Self-timed block, idle and same-process load | within the clock bound, quantised to the timer step | Loses precision with same-process load on a high-priority stream: real result, record it |
| Every method, separate-process PyTorch load | millisecond stalls in all four methods, self-timed included, from 1–2 ms context time-slicing | If the resident block is *not* preempted, that contradicts the published scheduling model and is worth a second run |
| Clock ping-pong | brackets of a few µs over PCIe; a measurable rate offset between `globaltimer` and the CPU TSC, since a discrete GPU has its own crystal | A rate offset of exactly zero would mean the driver disciplines the timer, which NVIDIA says it does not |
| Certified clock bound | a few µs after ~3000 brackets, from the tightest 5 % | This is the first such number on NVIDIA in public; whatever it is, it goes in the write-up |

Three ways to run it, cheapest first:

1. **Driver script (no SSH).** `pip install vastai`, put the API key in `VAST_API_KEY`, then `python3 gpu_run/vast_run.py --search` to see offers and `python3 gpu_run/vast_run.py` to rent the cheapest fit, run, collect and destroy. Results land in `gpu_run/vast_results/<instance>/`.
2. **By hand on vast.ai.** Rent an A100 with a PyTorch `devel` image (it has `nvcc`) and paste this as the on-start command: `bash -c "curl -fsSL https://raw.githubusercontent.com/PierreKhoury1/Final_year_research_MSc/claude/modest-ride-bc8yxa/gpu_run/onstart.sh | bash"`. Read the results from the instance log; they sit between `LOCKSTEP_RESULTS_BEGIN` and `LOCKSTEP_RESULTS_END`.
3. **From a shell on any CUDA machine.** Clone the repo and `bash gpu_run/run.sh`. It writes `results_<utc>.tgz` next to itself.

Destroy the instance afterwards, because a stopped instance still bills for storage. The run takes about four minutes plus image load.

## What is already known (checked 30 Sep 2026)

- NVIDIA Aerial does not need the GPU to start at an exact instant. The ConnectX NIC's PTP hardware clock owns time; downlink packets carry a transmit timestamp and the NIC emits them at that instant, so GPU work only has to finish before the send time. Uplink already uses a GPU-resident "order kernel", launched ahead of the slot, that waits on `%globaltimer` while polling the NIC. Budgets are hundreds of µs to ~1.2 ms; the L2 tick runs 1.5 ms ahead of air time with a 100 µs jitter tolerance. This answers the old open question 3: ~90 µs of launch lateness is absorbed by slack.
- Spinning a resident kernel on `%globaltimer` until a target time was published in 2019 (tt-gpu, OSPERT'19, Jetson TX2), which also found the timer advances only every 1 µs on pre-Hopper GPUs.
- Bracketing a device clock read between two host reads and reporting the bracket as the bound is the model behind Vulkan's calibrated-timestamps extension (2018) and Intel's Xe kernel driver.
- Launch-to-running latency on an A100 is measured: 0–99th percentile in roughly 6–10 µs on clean Linux, unchanged by MPS or MIG (Bakita & Anderson, ECRTS 2025). Millisecond tails come from the launching CPU thread being descheduled, not from the GPU.
- Separate processes' contexts are time-sliced at ~1–2 ms and a resident kernel is preempted when its slice ends; only MPS, MIG or green contexts avoid it. Deployments co-locate AI with RAN through MIG partitions.
- The OCXO + GPS + FPGA "time referee" card on the slides exists as the open-hardware OCP Time Card (about £1,200, mainline Linux driver). PTM-capable NICs give host time to 30–50 ns. Designing a board would be redundant.
- The AMD `clGetDeviceAndHostTimer` behaviour is by design in ROCm CLR's source; Mesa's rusticl takes the mirror shortcut. Unreported, and a conformance-test proposal rather than a research result.

## Open questions

1. A certified CPU↔GPU clock mapping for NVIDIA GPUs: CUPTI and Nsight interpolate and publish no error bound, and CUDA has no calibrated-timestamp API. What bound does min-filtered bracketing reach on a discrete GPU, how does it hold under load, and how does the GPU crystal drift against a PTM-traceable host reference over hours and temperature?
2. A "cyclictest for GPUs": no tool measures how late GPU work starts against an absolute deadline with a certified clock bound, across launch methods and isolation modes (time-slicing, MPS, MIG, green contexts).
3. Deadline-miss behaviour of a slot-shaped GPU pipeline with a co-located AI job, per isolation mode, in misses per million slots. NVIDIA publishes throughput only.

Candidate thesis framings, sharing the tool from question 2: (A) the clock-metrology thesis built on question 1, with a bought PTM reference rather than a custom board; (B) the systems thesis built on question 3. "Start GPU work at exact instant T" stays as one launch mode inside the benchmark, not as the headline.

## References

- NVIDIA Aerial CUDA-Accelerated RAN, source: https://github.com/NVIDIA/aerial-cuda-accelerated-ran (uplink order kernel `cuPHY-CP/cuphydriver/src/uplink/order_cuda_kernels.cu`; NIC wait-on-time transmit `cuPHY-CP/aerial-fh-driver/lib/gpu_comm_doca.cu`; slot tick generator `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_tick_generator.cpp`)
- Kreiliger, Matějka, Sojka, Hanzálek, time-triggered GPU execution on `%globaltimer` (tt-gpu), OSPERT 2019: https://ospert19.tudos.org/ospert19-proceedings.pdf and https://github.com/CTU-IIG/tt-gpu
- Bakita & Anderson, NVIDIA launch latency and partitioning measurements, ECRTS 2025: https://www.cs.unc.edu/~jbakita/ecrts25.pdf; RTAS 2024: https://www.cs.unc.edu/~jbakita/rtas24.pdf
- Vulkan VK_EXT_calibrated_timestamps proposal (the bracket-as-bound model): https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_calibrated_timestamps.adoc
- NVIDIA on `%globaltimer` resolution and update rate: https://forums.developer.nvidia.com/t/questions-about-globaltimer-functionality-accessing-and-configuring/304268
- NVIDIA libcudacxx time library note that the GPU clock is not synchronised with the host: https://github.com/NVIDIA/cccl/blob/main/docs/libcudacxx/standard_api/time_library.rst
- OCP Time Appliances Project, Time Card: https://github.com/Time-Appliances-Project/Time-Card
- ROCm CLR `clGetDeviceAndHostTimer` (both outputs from the host clock): https://github.com/ROCm/clr/blob/develop/opencl/amdocl/cl_execute.cpp
- Holohub green-context benchmark (launch-to-start under a competing kernel): https://nvidia-holoscan.github.io/holohub/benchmarks/green_context_benchmarking/
- ConnectX accurate send scheduling characterised: https://arxiv.org/abs/2607.11305
- TempoTrace, arXiv:2609.23301 (unreviewed white paper; GPU timestamp claims are design targets): https://arxiv.org/abs/2609.23301
- REEF, microsecond-scale GPU preemption (OSDI 2022): https://ipads.se.sjtu.edu.cn/_media/publications/reef-osdi22.pdf
- RTGPU, real-time GPU scheduling: https://arxiv.org/pdf/2101.10463
