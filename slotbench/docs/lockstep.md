# CPU versus GPU slot launch

`driver/lockstep_driver.cu` compares a host-timed CUDA graph launch with a resident GPU executive
that waits for the same host-derived target on `%globaltimer`. Both modes capture the same slot
pipeline and final completion kernel. Overlapping slots are not allowed; boundaries observed while
the previous slot is busy are counted as skipped deadline misses.

## Matched workloads

`--slot-variant full` is the default. If CUDA cannot instantiate that graph for device launch, GPU
mode fails explicitly. It never silently removes cuBLAS stages. For a timing mechanism experiment,
select `--slot-variant no_cublas` for **both** modes; this omits stages S3-S7 and is not a complete
PHY receiver. The matrix forwards its selected variant to every case and rejects mismatches.

```bash
cd slotbench
make SM=86
make test
python3 -m pytest -q analysis/tests scripts/tests cloud/tests
bash scripts/lockstep_matrix.sh --out results/lockstep-smoke --slots 2000 \
  --slot-variant no_cublas --workloads sgemm
```

The matrix includes idle, in-process load, separate-process load, and separate-process MPS load.
An unsupported MPS/device-launch combination is a failed case, not evidence of poor timing. The
matrix exits nonzero if any case fails, and preserves the completed cases and logs.

## Executive and failure handling

Each executive generation uses a pool of 100 distinct uploaded executable graphs, at most once
per handle. A self tail-launch waits for all child work before the next generation reuses the
pool. This avoids both the per-generation fire-and-forget limit and reuse of a handle whose final
kernel has published completion but whose graph has not yet retired. The completion sequence in
device memory uses release/acquire atomics. Both CPU and GPU readiness gates originate in the final
completion kernel.

The executive is captured/uploaded before target timestamps are chosen. Its absolute deadline is
passed in device state. Wall guards exit without further blocking device copies or destructors if
the slot stream remains incomplete. Missing slot stamps count as failed measurements rather than
disappearing from the denominator. A valid summary requires every requested measurement to be
accounted for and a valid post-run clock mapping. The in-process load selects the requested GPU,
checks CUDA errors, and records completed work during the run.

CUDA constraints: [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

## Interpreting results

- `launch_precision_us`: slot start minus the target supplied to the launcher.
- `start_error_us`: slot start minus the target corrected by pre/post clock anchors.
- `target_pred_error_us`: difference between those target estimates.
- `exec_us`: time between slot start and end stamps.
- `latency_from_target_us`: slot end minus the corrected target; values beyond `deadline_us`
  count as misses. Skips, launch errors, and timeouts also count as misses.

The raw format remains 64-byte `LsRec` entries, excluding warm-up. Generation-first slots have
flag 4 so tail-launch overhead can be inspected separately. The added JSON fields report pool size
and in-process completed work/errors.

Mapped host notifications and calibration use the existing aligned volatile memory protocol on
x86/Linux with NVIDIA mapped host memory; they are not a portable C++ atomic protocol. Two-point
clock correction assumes approximately linear drift and does not remove asymmetric PCIe latency.
Separate compilation of the executive does not guarantee that an MPS runtime module failure leaves
CPU mode available. These properties require hardware validation.

## Cloud smoke run

Push the branch first; the rented machine clones it. Supply `VAST_API_KEY` only through the local
controller's environment. For example:

```bash
SB_SLOTS=2000 SB_MATRIX_ARGS='--slot-variant no_cublas --workloads sgemm' \
  python3 cloud/vast.py run --gpu 'RTX 3060' --branch codex/continue-lockstep \
  --script onstart_lockstep.sh --runtype args --cuda 12.6 --max-dph 0.10 \
  --max-cost 1 --max-hours 0.75 --max-load-min 10 --poll-s 20 --error-grace-s 60
```

Keep smoke runs small: the Vast log API returns at most 20,000 lines, and this onstart script emits
compressed raw records through that log. Long matrices need a separate raw-file transfer path before
they can be collected reliably. `run` collects results and destroys its instance in a `finally` block.

Local validation on 2026-10-02: CUDA 12.6.3 build for SM86 passed, 267 host assertions passed,
111 Python tests passed, and 37 malformed CLI cases were rejected before CUDA initialization.
The Linux test environment used an unprivileged sudo shim for existing mocked GPU setup tests.
Compilation alone does not establish runtime correctness or a timing advantage.

The first [RTX 3060 hardware smoke and raw audit](../data/2026-10-02_rtx3060_lockstep_smoke/README.md)
completed all eight cases on 2026-10-02. GPU launch reduced start jitter in this short sample, but
deadline misses were not consistently lower. Both rented instances were destroyed.
