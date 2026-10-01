# slotbench

Does a GPU finish every 5G uplink slot inside its 500 us deadline while an AI workload shares it, and which
GPU sharing setting gets misses to zero at the lowest cost to the AI workload? Measured on a consumer card
(RTX 3060) and datacenter cards (A100/H100).

`DESIGN.md` is the contract: workload, flags, file formats, metric definitions. Each component has its own
README.

## What one run measures

Every 500 us the driver launches one slot of GPU signal processing (FFT, channel estimation, MMSE
equalisation, 256-QAM demodulation, 5G NR LDPC decoding) and records when it started and finished, on the
host clock and on the GPU's own clock. A million slots per run. Meanwhile a second process (the
"adversary") loads the same GPU with matrix multiplies, an LLM-like token loop or a vision-like burst of
small kernels, at 0-100% duty. Outputs per run: miss rate with a confidence interval, p50 ... p99.99 and
max slot time, where the delay came from (CPU wake-up, launch, queueing behind the adversary, execution),
and how much adversary throughput was given up.

## Components

| Path | What |
|---|---|
| `phy/` | the slot workload: kernels, cuFFT/cuBLAS stages, LDPC decoder (5G NR base graphs from 38.212) with a bit-exact host reference, CUDA graph capture, GPU selftest and stage profile |
| `driver/slot_driver.cu` | the timed 500 us loop, clock calibration, collector thread, `slots.bin` + `meta.json` |
| `adversary/` | co-located workloads W1 sgemm, W2 LLM proxy, W3 vision proxy, duty cycling; optional real llama.cpp / YOLO wrappers |
| `clock/` | CPU<->GPU clock calibration over hours, CPU<->NIC PTP clock offset |
| `scripts/`, `configs/` | OS/RT setup, GPU clock locking, one-run procedure, resumable matrix runner |
| `analysis/` | statistics, validation checks, figures (CCDF, heatmap, Pareto), report |
| `cloud/` | vast.ai controller: rent, build, selftest, tune, run a config, stream results back, destroy |

## Quick start

```bash
make                      # compile (no GPU needed to compile); make SM=86 for an RTX 3060 only
make test                 # host tests: LDPC, demod, ring buffer, clock fit
./bin/slot_driver --selftest            # on a GPU: decoder vs host reference, stage profile
./bin/slot_driver --tune-us 250         # size the slot to ~250 us idle; prints the flags to use
python3 scripts/run_matrix.py configs/quick.toml        # a 6-run smoke matrix
python3 -m analysis.report runs/quick --out report/quick
```

Home lab: follow `scripts/README.md` (RT kernel, isolated cores, locked clocks, then `configs/home_lab.toml`).
Rented GPU: `cloud/README.md` (`python3 cloud/vast.py run --gpu "RTX 3060" --config cloud.toml`).

## Status and findings so far

First runs on a rented RTX 3060 (vast.ai, container, no RT kernel, clocks not lockable), 2026-09-30/10-01:

- The selftest caught a host-to-device copy that was not ordered with the decoder's stream; fixed. The GPU
  decoder now agrees with the host reference on 100% of bits.
- A full 100 MHz, 4-layer, 256-QAM slot does not fit the 500 us budget on an RTX 3060 with this decoder
  (about 1.4 ms idle; the LDPC decoder is ~75% of it). `--tune-us` therefore narrows the carrier: a 20 MHz
  carrier with 4-5 decoder iterations runs in about 225-255 us idle.
- With a back-to-back 4096^3 sgemm neighbour, the slot time roughly doubled and stream priority made no
  difference (early, pre-fix workload; to be repeated).
