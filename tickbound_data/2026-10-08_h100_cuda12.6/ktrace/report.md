# tickbound characterisation: NVIDIA H100 80GB HBM3 (82fbfcd49b63, 2026-10-08 02:04)

**ktrace SASS check: PASS (4 k_ktrace instantiation(s), 28 exact checks on the binary that measured)** (`ktrace_sass.json`)

GPU 0: NVIDIA H100 80GB HBM3, sm_90, driver 535.309.01, persistence Enabled; 1 GPU(s) on the host; profile `ktrace`; MPS not used; 1.8 min of runs; tickbound 0.1.0.

Host<->GPU clock bound (tick-edge sync before and after every run): 654-715 ns, feasible in 3 of 3 runs.

## Results, one line per run

- `ktrace`: ktrace ±654 ns (edge): B=132: load 386 | bar1 wait 38/255 rel 29 | compute 1032 | store+fence 1534 | atomic 498 cy; span 2.3 us, block bound 21/25 ns p50/max, SM 1.945-2.006 GHz, rate incons 0 ns, ticket 0+0/660 viol (floor 214 ns), launch->first 5.2 us warm, flag->seen 0.72 us, flag store 2 cy, writer fence 3374 cy, 4.03 cy/FFMA || B=528: load 610 | bar1 wait 172/2298 rel 66 | compute 1058 | store+fence 1458 | atomic 568 cy; span 9.0 us, block bound 20/27 ns p50/max, SM 1.960-1.983 GHz, rate incons 0 ns, ticket 0+0/2640 viol (floor 217 ns), launch->first 5.1 us warm, flag->seen 0.75 us, flag store 0 cy, writer fence 3608 cy, 4.13 cy/FFMA
- `ktrace_long`: ktrace ±713 ns (edge): B=528: load 576 | bar1 wait 160/9657 rel 52 | compute 16816 | store+fence 1352 | atomic 492 cy; span 28.6 us, block bound 41/91 ns p50/max, SM 1.963-1.967 GHz, rate incons 70 ns, ticket 0+0/1584 viol (floor 200 ns), launch->first 5.6 us warm, flag->seen 0.72 us, flag store 0 cy, writer fence 3240 cy, 4.11 cy/FFMA
- `ktrace_idle`: ktrace ±715 ns (edge): B=132: load 386 | bar1 wait 36/258 rel 29 | compute 1032 | store+fence 1500 | atomic 472 cy; span 2.3 us, block bound 21/26 ns p50/max, SM 1.943-2.005 GHz, rate incons 0 ns, ticket 0+0/660 viol (floor 218 ns), launch->first 24.4 us warm, flag->seen 0.73 us, flag store 4 cy, writer fence 3358 cy, 4.03 cy/FFMA

SASS check of the instruction brackets: 84/91 kernels have exactly N target opcodes between the clock reads; not verified: RED.

## Files

- `summary.json`: every run's analysis
- `timeline.html`: interactive timeline (open in a browser)
- `*.trace.json`: per-run Chrome trace format (https://ui.perfetto.dev)
- `ktrace_sass.json`: the exact ktrace SASS checks of the probe binary (`tickbound sass --phases`)
- `*.gpu.bin`, `*.host.bin`, `*.json`: raw records
