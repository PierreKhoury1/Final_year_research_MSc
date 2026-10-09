# tickbound characterisation: NVIDIA H100 80GB HBM3 (82fbfcd49b63, 2026-10-08 01:48)

**ktrace SASS check: FAIL (8 of 28 checks): this build breaks the layout-2 promises (exact phases, no spill); the ktrace phase notes do not hold for it** (`ktrace_sass.json`)

- FAIL k_ktrace<64>: load: 2 LDG + 1 STS + address arithmetic (LDG=2 STS=1 not allowed: ['02c0 DEPBAR.LE SB0, 0x0'])
- FAIL k_ktrace<64>: flag -> exit: one STG (+ ISETP/PLOP3) only (STG=1 ISETP/PLOP3=0 other={'DEPBAR.LE': 1})
- FAIL k_ktrace<256>: load: 2 LDG + 1 STS + address arithmetic (LDG=2 STS=1 not allowed: ['02c0 DEPBAR.LE SB0, 0x0'])
- FAIL k_ktrace<256>: flag -> exit: one STG (+ ISETP/PLOP3) only (STG=1 ISETP/PLOP3=0 other={'DEPBAR.LE': 1})
- FAIL k_ktrace<1024>: load: 2 LDG + 1 STS + address arithmetic (LDG=2 STS=1 not allowed: ['02c0 DEPBAR.LE SB0, 0x0'])
- FAIL k_ktrace<1024>: flag -> exit: one STG (+ ISETP/PLOP3) only (STG=1 ISETP/PLOP3=0 other={'DEPBAR.LE': 1})
- FAIL k_ktrace<4096>: load: 2 LDG + 1 STS + address arithmetic (LDG=2 STS=1 not allowed: ['02c0 DEPBAR.LE SB0, 0x0'])
- FAIL k_ktrace<4096>: flag -> exit: one STG (+ ISETP/PLOP3) only (STG=1 ISETP/PLOP3=0 other={'DEPBAR.LE': 1})

GPU 0: NVIDIA H100 80GB HBM3, sm_90, driver 535.309.01, persistence Enabled; 1 GPU(s) on the host; profile `ktrace`; MPS not used; 1.8 min of runs; tickbound 0.1.0.

Host<->GPU clock bound (tick-edge sync before and after every run): 653-714 ns, feasible in 3 of 3 runs.

## Results, one line per run

- `ktrace`: ktrace ±654 ns (edge): B=132: load 382 | bar1 wait 40/292 rel 21 | compute 1026 | store+fence 1496 | atomic 501 cy; span 2.3 us, block bound 21/26 ns p50/max, SM 1.948-2.002 GHz, rate incons 0 ns, ticket 0+0/660 viol (floor 223 ns), launch->first 5.1 us warm, flag->seen 0.73 us, flag store 8 cy, writer fence 3166 cy, 4.01 cy/FFMA || B=528: load 588 | bar1 wait 144/2349 rel 50 | compute 1070 | store+fence 1354 | atomic 594 cy; span 9.3 us, block bound 18/26 ns p50/max, SM 1.961-1.979 GHz, rate incons 0 ns, ticket 0+0/2640 viol (floor 221 ns), launch->first 5.0 us warm, flag->seen 0.76 us, flag store 16 cy, writer fence 3466 cy, 4.18 cy/FFMA
- `ktrace_long`: ktrace ±653 ns (edge): B=528: load 540 | bar1 wait 150/16450 rel 40 | compute 16824 | store+fence 1312 | atomic 508 cy; span 28.7 us, block bound 43/96 ns p50/max, SM 1.962-1.966 GHz, rate incons 76 ns, ticket 0+0/1584 viol (floor 212 ns), launch->first 5.2 us warm, flag->seen 0.74 us, flag store 16 cy, writer fence 3268 cy, 4.11 cy/FFMA
- `ktrace_idle`: ktrace ±714 ns (edge): B=132: load 384 | bar1 wait 44/304 rel 21 | compute 1026 | store+fence 1470 | atomic 478 cy; span 2.3 us, block bound 21/26 ns p50/max, SM 1.946-2.001 GHz, rate incons 0 ns, ticket 0+0/660 viol (floor 223 ns), launch->first 31.7 us warm, flag->seen 0.73 us, flag store 8 cy, writer fence 3172 cy, 4.01 cy/FFMA

SASS check of the instruction brackets: 84/91 kernels have exactly N target opcodes between the clock reads; not verified: RED.

## Files

- `summary.json`: every run's analysis
- `timeline.html`: interactive timeline (open in a browser)
- `*.trace.json`: per-run Chrome trace format (https://ui.perfetto.dev)
- `ktrace_sass.json`: the exact ktrace SASS checks of the probe binary (`tickbound sass --phases`)
- `*.gpu.bin`, `*.host.bin`, `*.json`: raw records
