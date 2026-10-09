# tickbound characterisation: NVIDIA H100 80GB HBM3 (aed87a1b0dd5, 2026-10-09 18:05)

**ktrace SASS check: PASS (4 k_ktrace instantiation(s), 56 exact checks on the binary that measured; 16 scoreboard wait(s) in checkpoint brackets, 8 on a cycle read (listed in the ktrace_sass JSON))** (`ktrace_sass.json`)

GPU 0: NVIDIA H100 80GB HBM3, sm_90, driver 595.71.05, persistence Enabled; 1 GPU(s) on the host; profile `ktrace`; MPS not used; 2.6 min of runs; tickbound 0.1.0.

Host<->GPU clock bound (tick-edge sync before and after every run): 542-560 ns, feasible in 3 of 3 runs.

## Results, one line per run

- `ktrace`: ktrace ±542 ns (edge): B=132: load 401 | bar1 wait 40/221 rel 28 | compute 1032 | store+fence 1561 | atomic 534 cy; span 2.3 us, block bound 20/25 ns p50/max, SM 1.966-2.011 GHz, rate incons 0 ns, ticket 0+0/660 viol (floor 221 ns), launch->first 4.5 us warm, flag->seen 0.59 us, flag store 2 cy, writer fence 2156 cy, 4.03 cy/FFMA || B=528: load 633 | bar1 wait 183/2310 rel 64 | compute 1056 | store+fence 1509 | atomic 592 cy; span 9.4 us, block bound 12/22 ns p50/max, SM 1.977-1.983 GHz, rate incons 3 ns, ticket 0+0/2640 viol (floor 214 ns), launch->first 4.4 us warm, flag->seen 0.64 us, flag store 1 cy, writer fence 2266 cy, 4.12 cy/FFMA
- `ktrace_long`: ktrace ±556 ns (edge): B=528: load 618 | bar1 wait 175/16215 rel 53 | compute 16835 | store+fence 1368 | atomic 478 cy; span 28.3 us, block bound 32/121 ns p50/max, SM 1.982-1.986 GHz, rate incons 100 ns, ticket 0+0/1584 viol (floor 226 ns), launch->first 5.0 us warm, flag->seen 0.60 us, flag store -1 cy, writer fence 1717 cy, 4.05 cy/FFMA
- `ktrace_idle`: ktrace ±560 ns (edge): B=132: load 394 | bar1 wait 42/260 rel 29 | compute 1032 | store+fence 1531 | atomic 484 cy; span 2.3 us, block bound 21/25 ns p50/max, SM 1.974-2.034 GHz, rate incons 0 ns, ticket 0+0/660 viol (floor 218 ns), launch->first 31.0 us warm, flag->seen 0.60 us, flag store 2 cy, writer fence 1896 cy, 4.03 cy/FFMA

SASS check of the instruction brackets: 84/91 kernels have exactly N target opcodes between the clock reads; not verified: RED.

## Files

- `summary.json`: every run's analysis
- `timeline.html`: interactive timeline (open in a browser)
- `*.trace.json`: per-run Chrome trace format (https://ui.perfetto.dev)
- `ktrace_sass.json`: the exact ktrace SASS checks of the probe binary (`tickbound sass --phases`)
- `*.gpu.bin`, `*.host.bin`, `*.json`: raw records
