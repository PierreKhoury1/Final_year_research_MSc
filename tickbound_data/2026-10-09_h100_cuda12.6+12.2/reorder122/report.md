# tickbound characterisation: NVIDIA H100 80GB HBM3 (aed87a1b0dd5, 2026-10-09 17:59)

**reorder SASS check: FAIL (35 of 295 instantiation(s)): their instruction order is not the intended one; the analysis drops them; 1 E2 reuse build(s) without the WAR wait (not a WAR measurement)** (`reorder_sass.json`)

- FAIL bar pre M=64: the barrier is BAR.SYNC.DEFER_BLOCKING (barrier forms ['BAR.SYNC'])
- FAIL bar pre M=256: the barrier is BAR.SYNC.DEFER_BLOCKING (barrier forms ['BAR.SYNC'])
- FAIL bar post M=64: the barrier is BAR.SYNC.DEFER_BLOCKING (barrier forms ['BAR.SYNC'])
- FAIL bar post M=256: the barrier is BAR.SYNC.DEFER_BLOCKING (barrier forms ['BAR.SYNC'])
- FAIL bar post_mem M=64: the barrier is BAR.SYNC.DEFER_BLOCKING (barrier forms ['BAR.SYNC'])
- FAIL bar post_mem M=256: the barrier is BAR.SYNC.DEFER_BLOCKING (barrier forms ['BAR.SYNC'])
- FAIL bar late_sink M=64: the barrier is BAR.SYNC.DEFER_BLOCKING (barrier forms ['BAR.SYNC'])
- FAIL bar late_sink M=256: the barrier is BAR.SYNC.DEFER_BLOCKING (barrier forms ['BAR.SYNC'])
- FAIL fence after sc_gpu M=0: one sc_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0210 MEMBAR.ALL.CTA', '0220 MEMBAR.SC.GPU'] + ['0230 ERRBAR', '0240 CGAERRBAR', '0250 CCTL.IVALL'])
- FAIL fence after sc_gpu M=64: one sc_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0210 MEMBAR.ALL.CTA', '0220 MEMBAR.SC.GPU'] + ['0230 ERRBAR', '0240 CGAERRBAR', '0250 CCTL.IVALL'])
- FAIL fence after sc_gpu M=256: one sc_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0210 MEMBAR.ALL.CTA', '0220 MEMBAR.SC.GPU'] + ['0230 ERRBAR', '0240 CGAERRBAR', '0250 CCTL.IVALL'])
- FAIL fence after sc_gpu M=1024: one sc_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0210 MEMBAR.ALL.CTA', '0220 MEMBAR.SC.GPU'] + ['0230 ERRBAR', '0240 CGAERRBAR', '0250 CCTL.IVALL'])
- FAIL fence after sc_sys M=0: one sc_sys fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0210 MEMBAR.ALL.CTA', '0220 MEMBAR.SC.SYS'] + ['0230 ERRBAR', '0240 CGAERRBAR', '0250 CCTL.IVALL'])
- FAIL fence after sc_sys M=64: one sc_sys fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0210 MEMBAR.ALL.CTA', '0220 MEMBAR.SC.SYS'] + ['0230 ERRBAR', '0240 CGAERRBAR', '0250 CCTL.IVALL'])
- FAIL fence after sc_sys M=256: one sc_sys fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0210 MEMBAR.ALL.CTA', '0220 MEMBAR.SC.SYS'] + ['0230 ERRBAR', '0240 CGAERRBAR', '0250 CCTL.IVALL'])
- FAIL fence after sc_sys M=1024: one sc_sys fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0210 MEMBAR.ALL.CTA', '0220 MEMBAR.SC.SYS'] + ['0230 ERRBAR', '0240 CGAERRBAR', '0250 CCTL.IVALL'])
- FAIL fence after acqrel_gpu M=0: one acqrel_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0210 MEMBAR.ALL.CTA', '0220 MEMBAR.ALL.GPU'] + ['0230 ERRBAR', '0240 CGAERRBAR', '0250 CCTL.IVALL'])
- FAIL fence after acqrel_gpu M=64: one acqrel_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0210 MEMBAR.ALL.CTA', '0220 MEMBAR.ALL.GPU'] + ['0230 ERRBAR', '0240 CGAERRBAR', '0250 CCTL.IVALL'])
- FAIL fence after acqrel_gpu M=256: one acqrel_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0210 MEMBAR.ALL.CTA', '0220 MEMBAR.ALL.GPU'] + ['0230 ERRBAR', '0240 CGAERRBAR', '0250 CCTL.IVALL'])
- FAIL fence after acqrel_gpu M=1024: one acqrel_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0210 MEMBAR.ALL.CTA', '0220 MEMBAR.ALL.GPU'] + ['0230 ERRBAR', '0240 CGAERRBAR', '0250 CCTL.IVALL'])

GPU 0: NVIDIA H100 80GB HBM3, sm_90, driver 595.71.05, persistence Enabled; 1 GPU(s) on the host; profile `reorder`; MPS not used; 0.2 min of runs; tickbound 0.1.0.

Host<->GPU clock bound (tick-edge sync before and after every run): 545-652 ns, feasible in 6 of 6 runs.

## Results, one line per run

- `reorder_loaduse`: reorder ±549 ns (edge): gate FAIL | load_use L1 L 43 d 33 N* 2.5 ((L-d)/4 2.6) max saving 18@N=16, L2 L 376 d 33 N* 85.9 ((L-d)/4 85.8) max saving 390@N=96, DRAM L 762 d 33 N* 163.9 ((L-d)/4 182.4) max saving 645@N=128 | 3 inconsistencies
- `reorder_war`: reorder ±570 ns (edge): gate FAIL | war reuse-fresh (1/32 warps) STS.32 -6/+5 (D 14), STS.128 -4/+1 (D 14), STG.32 -6/-0 (D 14), STG.128 +5/+30 (D 14) cy, WAR waits 8/8 | war reuse_clock-fresh_clock (1/32 warps) STS.32 +3/-4 (D 9), STS.128 +6/+7 (D 9), STG.32 +0/+1 (D 74), STG.128 -1/-30 cy, WAR waits 6/8
- `reorder_bar`: reorder ±553 ns (edge): gate FAIL, 8 dropped | bar M=64: post-pre -, post_mem-pre -, sink - (4M 256, post clock in the wait 0%); M=256: post-pre -, post_mem-pre -, sink - (4M 1024, post clock in the wait 0%)
- `reorder_fence`: reorder ±627 ns (edge): gate FAIL, 27 dropped | fence sc_gpu -: saving  @M=; sc_sys -: saving  @M=; acqrel_gpu -: saving  @M= | 3 inconsistencies
- `reorder_pipe`: reorder ±545 ns (edge): gate FAIL | pipe cy/FFMA ILP1 4.01..7.97 (W 1..32, exp 4.0..7.8, k 7.8), ILP2 2.02..8.01 (W 1..32, exp 2.0..7.9, k 7.9), ILP4 1.02..2.14 (W 1..32, exp 1.0..2.6, k 2.6)
- `reorder_stall`: reorder ±652 ns (edge): gate FAIL | stall dep_consumed 16 ops slope [encoded per link, op stall, CuAsmRL]: IADD3 4.0[4,4,4] LOP3 4.0[4,4,-] SHF 4.0[4,4,-] LEA 4.0[4,4,4] SEL 4.0[4,4,4] IMNMX 4.0[4,4,4] IABS 8.0[8,4,4] HADD2 4.0[4,4,4] HFMA2 4.0[4,4,-] IMAD_WIDE 12.0[8,3,5] FMUL 4.0[4,4,-] FADD 4.0[4,4,4] FFMA 4.0[4,4,-] IMAD 4.0[4,4,4] DADD 8.0[8,8,-] DFMA 8.0[8,8,-]; indep IADD3 2.0 LOP3 2.0 SHF 2.0 LEA 2.0 SEL 2.0 IMNMX 2.0 IABS 4.0 HADD2 2.0 HFMA2 2.0 IMAD_WIDE 7.5 FMUL 1.0 FADD 1.0 FFMA 1.0 IMAD 2.0 DADD 2.2 DFMA 2.2

SASS check of the instruction brackets: 84/91 kernels have exactly N target opcodes between the clock reads; not verified: RED.

## Files

- `summary.json`: every run's analysis
- `timeline.html`: interactive timeline (open in a browser)
- `*.trace.json`: per-run Chrome trace format (https://ui.perfetto.dev)
- `reorder_sass.json`: the instruction-order checks of the reorder kernels in the probe binary (`tickbound sass --reorder`)
- `*.gpu.bin`, `*.host.bin`, `*.json`: raw records
