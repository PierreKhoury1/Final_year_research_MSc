# tickbound characterisation: NVIDIA A100-SXM4-40GB (5d89b2e2e8cf, 2026-10-08 01:50)

**ktrace SASS check: PASS (4 k_ktrace instantiation(s), 28 exact checks on the binary that measured)** (`ktrace_sass.json`)

GPU 0: NVIDIA A100-SXM4-40GB, sm_80, driver 580.173.02, persistence Enabled; 1 GPU(s) on the host; profile `ktrace`; MPS not used; 2.8 min of runs; tickbound 0.1.0.

Host<->GPU clock bound (tick-edge sync before and after every run): 760-835 ns, feasible in 3 of 3 runs.

## Results, one line per run

- `ktrace`: ktrace ±835 ns (edge): B=108: load 338 | bar1 wait 40/193 rel 30 | compute 1027 | store+fence 628 | atomic 403 cy; span 3.5 us, block bound 77/269 ns p50/max, SM 1.073-1.097 GHz, rate incons 0 ns, ticket 0+0/540 viol (floor 337 ns), launch->first 6.4 us warm, flag->seen 0.93 us, flag store -2 cy, writer fence 3070 cy, 4.01 cy/FFMA || B=432: load 543 | bar1 wait 88/2092 rel 56 | compute 1501 | store+fence 947 | atomic 433 cy; span 14.3 us, block bound 40/279 ns p50/max, SM 1.078-1.080 GHz, rate incons 24 ns, ticket 0+0/2160 viol (floor 321 ns), launch->first 6.2 us warm, flag->seen 0.92 us, flag store -1 cy, writer fence 3233 cy, 5.86 cy/FFMA
- `ktrace_long`: ktrace ±760 ns (edge): B=432: load 455 | bar1 wait 69/16784 rel 47 | compute 24801 | store+fence 701 | atomic 408 cy; span 62.0 us, block bound 100/293 ns p50/max, SM 1.139-1.142 GHz, rate incons 69 ns, ticket 0+0/1296 viol (floor 333 ns), launch->first 6.6 us warm, flag->seen 0.97 us, flag store -1 cy, writer fence 3437 cy, 6.05 cy/FFMA
- `ktrace_idle`: ktrace ±782 ns (edge): B=108: load 340 | bar1 wait 41/195 rel 29 | compute 1027 | store+fence 733 | atomic 401 cy; span 3.4 us, block bound 141/334 ns p50/max, SM 1.068-1.095 GHz, rate incons 0 ns, ticket 0+0/540 viol (floor 420 ns), launch->first 9.8 us warm, flag->seen 0.88 us, flag store -2 cy, writer fence 3068 cy, 4.01 cy/FFMA

SASS check of the instruction brackets: 91/91 kernels have exactly N target opcodes between the clock reads.

## Files

- `summary.json`: every run's analysis
- `timeline.html`: interactive timeline (open in a browser)
- `*.trace.json`: per-run Chrome trace format (https://ui.perfetto.dev)
- `ktrace_sass.json`: the exact ktrace SASS checks of the probe binary (`tickbound sass --phases`)
- `*.gpu.bin`, `*.host.bin`, `*.json`: raw records
