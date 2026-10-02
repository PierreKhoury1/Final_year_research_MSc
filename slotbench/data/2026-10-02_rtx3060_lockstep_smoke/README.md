# RTX 3060 lockstep mechanism smoke, 2026-10-02

The GPU executive ran successfully in all four tested sharing conditions, including MPS. It reduced
start jitter in this short experiment, but did **not** consistently reduce deadline misses. This is
a mechanism check using an incomplete PHY workload, not a cuPHY result or a real-time guarantee.

## Setup and validation

- Vast instance `53892791`, verified offer `42699572`, Romania; RTX 3060, 28 SMs, 12 GB;
  Core i7-3770 host; driver 595.71.05, CUDA runtime 12.6, Linux 6.8.0-124-generic.
- Driver implementation: commit `fc03611d942cc50c2b18685a2421d1911526adc1` on
  `codex/continue-lockstep`. Later commits on the branch changed the local controller/docs only.
  The onstart script did not record its cloned HEAD, so an exact remote checkout SHA is unavailable.
- Eight sequential cases, CPU before GPU in each condition. Each has 500 warm-up slots and 2,000
  measured boundaries at 500 us spacing/deadline: only **one second of measured boundaries per case**.
- Explicit `no_cublas` workload in every case: FFT 4096, 14 symbols, 612 subcarriers, 4 antennas/layers,
  256-QAM, 22 LDPC codewords, 3 iterations. Nine kernel nodes. Stages S3-S7 are omitted.
- High stream/node priority (-5); CPU core 1 affinity succeeded. FIFO scheduling and memory locking
  were denied. GPU clocks were not locked. `%globaltimer` advanced in 1.024 us steps.
- Separate-process neighbour: FP32 4096-square SGEMM at 100% duty. Under MPS, the neighbour's active
  thread percentage was capped at 50%; the slot process was uncapped. In-process load is the driver's
  low-priority FMA kernel loop.
- Independent audit matched all 16,000 raw records to JSON counts and timing summaries. No launch
  errors, timeouts, or missing records/stamps; every CPU/GPU workload signature matched.
- Pre/post calibration epsilon estimates ranged from 0.779 to 0.9035 us; two-point rate corrections
  ranged from about 22.35 to 26.52 ppm. These estimates are not worst-case timing guarantees.

## Observations

Start-error percentiles below are conditional on **executed** slots. Deadline misses include skipped
boundaries, so the last two columns are the complete accounting, each out of 2,000 requested slots.

| Condition | CPU p99 start error (us) | GPU p99 start error (us) | CPU misses | GPU misses |
|---|---:|---:|---:|---:|
| Idle | 26.4 | 4.1 | 0 (0%) | 0 (0%) |
| In-process FMA load | 165.0 | 56.5 | 1,584 (79.20%) | 1,699 (84.95%) |
| Separate SGEMM process | 3,113.9 | 2,277.4 | 1,993 (99.65%) | 1,090 (54.50%) |
| SGEMM under MPS | 21.8 | 5.2 | 0 (0%) | 0 (0%) |

The idle GPU median start error was 3.6 us and the observed maximum was 4.1 us. Under MPS these were
3.8 and 5.2 us, versus CPU maximum 29.5 us. Those are observed values from this small sample.

Under in-process load, median execution alone was about 772 us for CPU launch and 768 us for GPU
launch, already beyond the 500 us budget. GPU launch skipped fewer boundaries (971 versus 999), but
more executed slots finished late (728 versus 585), producing the worse total miss count. Reduced
launch jitter does not solve this execution-time bottleneck; one sequential pair cannot establish
that GPU launch intrinsically worsens deadlines.

Without MPS, GPU launch substantially improved the observed miss count, but still missed 54.5%.
Its execution-time p99 was 2,464.8 us, versus 138.2 us for the CPU case; millisecond-scale delays
remain even when median start error is small. Do not describe the GPU executive as isolation.
Neighbour throughput also needs to be included in follow-up comparisons: whole-session SGEMM
throughput was 53.69 units/s for CPU launch and 51.78 for GPU launch, but these averages include
startup/calibration and do not isolate the one-second measurement interval.

## Artifacts and next step

`out/` contains per-case JSON, all raw 64-byte records, driver/adversary logs and neighbour timelines.
`logs/matrix.log` contains the original table. `instance.log` and `collect.json` preserve transport
evidence; both result blocks passed SHA-256 verification. `audit_lockstep.py` is a standalone,
read-only audit; rerun with `python audit_lockstep.py out`. Its output is saved in `audit.txt`.

The next useful experiment is repeated, balanced CPU/GPU comparisons with longer measurement windows
and measured neighbour throughput, especially under MPS where both short runs met the deadline.
Arrange raw-file transfer before increasing the matrix size beyond the Vast log API limit. The
[cuPHY adapter plan](../../docs/cuphy-lockstep-plan.md) is documented but not implemented; the
standalone example needs explicit device-launch enablement and a restricted graph interface.

The first rental, `53891717`, exposed no container logs after about six minutes and was destroyed
(estimated $0.005). The successful rental took about 5.6 minutes (estimated $0.007). Both instance
IDs were subsequently queried and reported not found. **Total estimated spend: about $0.012**;
Vast's billing page is authoritative. No instance was left running.
