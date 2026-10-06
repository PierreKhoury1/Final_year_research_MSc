# gputrace: the GPU scheduling path, every latency, hardware timers, one axis

A tracer that measures where time goes between a host asking a GPU to do something and the host learning it is
done, with the GPU's own hardware timers (`%globaltimer`, `clock64`, `%smid`) and the host's `CLOCK_MONOTONIC_RAW`,
on one time axis with a **hard bound** on the host↔GPU mapping (the tick-edge method of `tools/pcieclock.cu`,
run before and after every measurement). No NIC timestamping, no vendor profiler, no driver changes: it runs in a
rented container.

## Angles

| Strategy | Latency measured | How |
|---|---|---|
| `launch` | launch call → first instruction of the block; call duration; kernel end → `cudaStreamSynchronize` returns | 1-block kernels, by idle gap before the launch (power state), queue depth (`--depth`, gives the inter-kernel gap) and CUDA graph vs stream launch |
| `notify` | kernel end → host observes it | three observers interleaved: mapped-memory flag written by the kernel and polled by the host, `cudaEventQuery` polling, `cudaStreamSynchronize` |
| `dispatch` | one kernel's blocks: which SM, in which order, how fast (blocks/µs), how many concurrent per SM, waves, kernel span vs ideal | spinning blocks of fixed GPU-time duration; sweep over block counts, threads and shared memory |
| `concurrency` | second stream's launch → its first block starts while the first kernel occupies the GPU; co-residency on SMs; whether running blocks were preempted | two streams, optional priorities; every block records the largest gap in its own timer reads (a gap ≫ one read = it was not running) |
| `clocks` | SM clock frequency over time: idle → loaded → idle, ramp time | a resident thread samples (`%globaltimer`, `clock64`) every 100 µs while a load kernel runs on the other SMs |
| `copy` | H2D/D2H `cudaMemcpyAsync` call and completion latency by size; PCIe read latency as the GPU sees it | host events around copies; one GPU thread does dependent 64-bit loads from mapped host memory, each timed with `clock64` |
| `timeslice` | with another process on the GPU: intervals our resident thread did not run (its quantum, their quantum, switch rate); which SMs the other process got (MPS partition) | the binary forks itself as a `hog` that traces its own blocks |

Every record is 64 bytes: block start/end in `%globaltimer` ns and `clock64` cycles, SM id, kernel id, block index, a
strategy tag, the largest timer gap seen while spinning. Host events are 32 bytes (launch enter/return, sync
enter/return, flag/event seen, copy enter/return/done, idle end, marks). Formats: `gputrace.h`; the analysis reads
them with the same layouts (`analysis/gputrace.py`, tested on synthetic runs in `analysis/tests/test_gputrace.py`).

## View

`python3 analysis/gputrace_export.py PREFIX [--max-kernels N] [--start-ms X --window-ms Y]` writes
`PREFIX.trace.json` in the Chrome trace-event format; open it in https://ui.perfetto.dev. Host calls, every block on
its SM's track per GPU, the resident thread's not-running intervals, the other process of a timeslice run, and NCCL
collectives per GPU, all on the host clock, with each GPU's bound in its process name.

## Command-line tool

`gputrace/gputrace` (Python, standard library + numpy; needs `nvcc` and `nvidia-smi`) characterises the GPU it
finds from one command:

```sh
gputrace/gputrace characterize --profile quick          # ~6 min: launch, notify, dispatch, concurrency, copy, memory, instr, timeslice
gputrace/gputrace characterize --profile full           # every angle incl. idle gaps, graphs, clocks, co-tenants (~25 min)
gputrace/gputrace characterize --profile sharing        # memory + instruction table alone / same-SM co-tenant / other process / MPS
gputrace/gputrace characterize --profile instr          # the instruction table only (plus its MPS variant when MPS is available)
gputrace/gputrace run --strategy launch --iters 2000 --out run/launch       # one strategy, raw binary args
gputrace/gputrace analyze run/launch [--json FILE]      # one run: one line + JSON
gputrace/gputrace export run/launch                     # Chrome/Perfetto trace of one run
gputrace/gputrace viewer out.html run/launch:A100 ...   # the interactive timeline with these runs embedded
gputrace/gputrace sass [BINARY]                         # verify the instruction brackets in the compiled SASS
```

`characterize` detects the GPU, compiles the probes for its architecture (`bin/gputrace_smXX`, rebuilt when the
sources change), runs the profile with a tick-edge clock sync before and after every run, analyses each run as it
finishes, verifies that every instruction bracket holds exactly N target opcodes between its clock reads, exports
every run to the trace format, builds `timeline.html`, and writes `report.md` (one line per run, the instruction
table alone / co-tenant / MPS, the memory hierarchy table) and `summary.json` (every analysis). MPS variants
(`--mps auto|on|off`) are added for the runs the profile contains when an MPS daemon can be started.
`cloud/onstart_cli.sh` runs the same command on a rented GPU (`cloud/vast.py run --script onstart_cli.sh`,
`SB_GT_PROFILE=...`).

## Run

```sh
make bin/gputrace                                    # or: nvcc -O2 -std=c++17 -arch=sm_80 -Icommon -Igputrace -o bin/gputrace gputrace/gputrace.cu -lpthread
bin/gputrace --strategy launch --iters 2000 --out run/launch --core 2 --clock-core 3
python3 analysis/gputrace.py run/launch              # one line + JSON; --json FILE to save
cloud/onstart_gputrace.sh                            # the whole campaign on a rented GPU (cloud/vast.py run --script onstart_gputrace.sh)
```

The campaign (`cloud/onstart_gputrace.sh`) runs: launch (plain, after 100 µs / 2 ms / 50 ms idle, depth 8, graph),
notify, dispatch (256 threads; 64 threads; 48 KB shared memory), concurrency (same priority; priorities),
clocks, copy, timeslice. About 5 minutes of GPU time.

## What the clock bound means

`pcieclock` showed (data/2026-10-05_a100_pcieclock, 3 hosts, A100 and H100) that `%globaltimer` can be placed on
the host clock to within ±0.57–0.79 µs with no symmetry assumption, versus ±0.86–1.32 µs for the usual
round-trip method. Every host-vs-GPU number gputrace reports (launch → block start, kernel end → host saw it)
carries that bound; GPU-vs-GPU numbers (block durations, dispatch spans, gaps, quanta) are in `%globaltimer` ns and
need no mapping, limited only by the timer tick (1024 ns on A100, 64 ns on H100; measured per run).

## Prior art, and what is different

Reverse-engineering NVIDIA scheduling with per-block `%globaltimer`/`%smid` records is established: Otterness et
al.'s `cuda_scheduling_examiner` and Amert et al., "GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed"
(RTSS 2017); Bakita and Anderson, "Hardware Compute Partitioning on NVIDIA GPUs" (RTAS 2023, `libsmctrl`) and
"Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management" (RTAS 2024). NVIDIA's CUPTI/Nsight give
kernel start/end on a host timeline using a driver-internal conversion whose error is not stated. TempoTrace
(arXiv 2609.23301) proposes NIC-timestamped GPU spans; its GPU numbers are design targets, not measurements.

gputrace differs in: (1) the host↔GPU mapping has a measured hard bound, so host-side latencies (launch, notify)
are results, not estimates; (2) it covers the host-visible angles (launch, notify, copy) and the GPU-only angles
(dispatch, concurrency, clocks, time-slicing) in one tool with one record format; (3) it runs on datacenter GPUs
(A100/H100) in containers, including MPS and process time-slicing; (4) it is built to extend to several GPUs
(per-GPU runs from one host thread give the offset between their `%globaltimer`s with a bound) and then to
several hosts via PTP on the host side.

## Status

2026-10-06: campaigns on six hosts (RTX 3060, A100 ×3 hosts, H100 PCIe, 2×A100, 8×A100); every host↔GPU latency
carries a feasible hard bound (153 of 153 runs, 0.3–0.9 µs). Findings across GPUs: `docs/gputrace_findings_2026-10-06.md`;
per-run detail in `data/2026-10-06_*_gputrace/`. Headlines: `%globaltimer` is per GPU (two A100s differ by 1.74 s
and drift 0.63 µs/s); a priority stream never preempts running blocks (wait = co-tenant's block length, 3 GPUs);
process time-slicing preempts mid-block with a 2.09 ms quantum and ≈ 170 µs per switch; MPS removes it; launch →
first instruction 2.9–5.9 µs, consecutive kernels 1–2 µs, kernel end → host 0.9–1.1 µs by mapped flag; the
instruction table (SASS-verified) alone / same-SM co-tenant / MPS on A100 and RTX 3060 (`data/*_instr*`).
Next: PTP across hosts, NCCL spans, MIG, cuPHY slots on this timeline.
