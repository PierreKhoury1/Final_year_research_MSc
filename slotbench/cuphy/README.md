# Experimental cuPHY lockstep adapter

Periodically replays the actual NVIDIA cuPHY PUSCH full-slot graph with CPU or GPU scheduling. The adapter targets upstream commit `4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c`. CUDA helpers compile locally; full upstream build and GPU validation are pending as of 2026-10-02. See [integration notes and source evidence](../docs/cuphy-lockstep-plan.md).

## Build and run

The [cloud bootstrap](../cloud/onstart_cuphy_lockstep.sh) contains the complete A100 SM80 / CUDA 13.3 / Ubuntu 24.04 recipe. It checks/applies [the patch](patches/cuphy-lockstep.patch), copies helper sources and common headers, builds `cuphy_ex_pusch_rx_multi_pipe`, and generates TC 7304. This helper is not a standalone executable.

After building, run each mode in a separate process with identical input/settings:

```bash
W=/workspace/cuphy-lockstep
PUSCH="$W/build/cuPHY/examples/pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe"
TV="$W/tv/GPU_test_input/TVnr_7304_PUSCH_gNB_CUPHY_s0p0.h5"
mkdir -p "$W/manual"

for mode in cpu gpu; do
    timeout --signal=TERM --kill-after=20 180 env \
        SB_CUPHY_LOCKSTEP_MODE="$mode" \
        SB_CUPHY_LOCKSTEP_SLOTS=1000 SB_CUPHY_LOCKSTEP_WARMUP=100 \
        SB_CUPHY_LOCKSTEP_PERIOD_US=500 SB_CUPHY_LOCKSTEP_DEADLINE_US=500 \
        SB_CUPHY_LOCKSTEP_OUT="$W/manual/$mode.json" \
        SB_CUPHY_LOCKSTEP_RAW="$W/manual/$mode.bin" \
        "$PUSCH" -i "$TV" -m 1 -r 1 || break
    python3 "$W/sb/slotbench/cloud/cuphy_lockstep_check.py" \
        "$W/manual/$mode.json" "$W/manual/$mode.bin" \
        --mode "$mode" --slots 1000 --warmup 100 \
        --period-us 500 --deadline-us 500 || break
done
```

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `SB_CUPHY_LOCKSTEP_MODE` | Unset | `cpu` or `gpu`; unset runs the ordinary example. |
| `SB_CUPHY_LOCKSTEP_SLOTS` | `1000` | Measured target boundaries. |
| `SB_CUPHY_LOCKSTEP_WARMUP` | `100` | Initial boundaries omitted from statistics/raw output. |
| `SB_CUPHY_LOCKSTEP_PERIOD_US` | `500` | Target spacing in microseconds. |
| `SB_CUPHY_LOCKSTEP_DEADLINE_US` | `500` | Completion deadline relative to target. |
| `SB_CUPHY_LOCKSTEP_OUT` | `cuphy_lockstep.json` | Summary path. |
| `SB_CUPHY_LOCKSTEP_RAW` | `cuphy_lockstep.bin` | Binary record path. |
| `SB_CUPHY_LOCKSTEP_LABEL` | Empty | Optional label. |

The bootstrap also runs SGEMM contention in separate processes and under MPS, records source/vector hashes, and applies timeouts. Its `*.run.json` metadata identifies workload and isolation; the helper's `load: external` field alone does not.

## Method and limits

One pipeline/vector/transmission is supported, with fresh SCH data, full 7.2a processing, and no UCI. Early HARQ, early SCH decoding, and work cancellation are disabled. Green contexts, subcontexts, partial processing, neural configuration, delayed nodes, and incompatible sub-slot modes are excluded.

An ordinary setup/decode prepares the graph and baseline. Replay freezes input, descriptors, and buffers. CPU mode launches from the host; GPU mode launches one PHY graph and then tail-launches its executive, waiting for completion before reuse. Both modes skip boundaries covered by the previous PHY execution and use matching kernel priorities. Markers surround all original DAG roots/leaves; a sequence counter checks freshness.

Payload comparisons and TB/CB CRC checks occur before and after the series. **Intermediate outputs are not checked.** Timing excludes setup, output copies, and calibration. The experiment does not model real symbol ingress, changing descriptors, HARQ progression, or production output delivery.

`ok: true` means correctness and execution/accounting checks passed; it does not mean every deadline was met. Skipped boundaries and recorded completions after the deadline both count as misses. Inspect `miss_rate`, `recorded`, `skipped`, and timing distributions together.

## Raw records

Each measured boundary has one 64-byte record, parsed as Python `struct.Struct("<Qq6Q")` on target x86 Linux:

```text
slot, host_target_ns, gpu_target_ns, launch_ns, launch_return_ns,
phy_start_ns, phy_end_ns, flags
```

Slot numbering includes omitted warmup boundaries. Flag bits: `1` skipped, `2` launch error, `4` GPU launch attempted, `8` missing/invalid/timed-out record. Host launch timestamps are mapped into GPU time. GPU-target precision uses the supplied target; corrected start error and completion latency use pre/post clock anchors. Failure JSON may lack a complete raw file; the checker rejects incomplete results.