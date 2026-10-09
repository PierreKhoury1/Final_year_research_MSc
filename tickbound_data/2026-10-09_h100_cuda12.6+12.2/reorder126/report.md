# tickbound characterisation: NVIDIA H100 80GB HBM3 (aed87a1b0dd5, 2026-10-09 17:58)

**reorder SASS check: FAIL (27 of 295 instantiation(s)): their instruction order is not the intended one; the analysis drops them; 1 E2 reuse build(s) without the WAR wait (not a WAR measurement); 9 with a wait on a scoreboard set before the bracket (reorder_sass.json pre_waits)** (`reorder_sass.json`)

- FAIL fence after sc_gpu M=0: one sc_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['01f0 MEMBAR.ALL.CTA', '0200 MEMBAR.SC.GPU'] + ['0210 ERRBAR', '0220 CGAERRBAR', '0230 CCTL.IVALL'])
- FAIL fence after sc_gpu M=64: one sc_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['01f0 MEMBAR.ALL.CTA', '0200 MEMBAR.SC.GPU'] + ['0210 ERRBAR', '0220 CGAERRBAR', '0230 CCTL.IVALL'])
- FAIL fence after sc_gpu M=256: one sc_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['01f0 MEMBAR.ALL.CTA', '0200 MEMBAR.SC.GPU'] + ['0210 ERRBAR', '0220 CGAERRBAR', '0230 CCTL.IVALL'])
- FAIL fence after sc_gpu M=1024: one sc_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['01f0 MEMBAR.ALL.CTA', '0200 MEMBAR.SC.GPU'] + ['0210 ERRBAR', '0220 CGAERRBAR', '0230 CCTL.IVALL'])
- FAIL fence after sc_sys M=0: one sc_sys fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['01f0 MEMBAR.ALL.CTA', '0200 MEMBAR.SC.SYS'] + ['0210 ERRBAR', '0220 CGAERRBAR', '0230 CCTL.IVALL'])
- FAIL fence after sc_sys M=64: one sc_sys fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['01f0 MEMBAR.ALL.CTA', '0200 MEMBAR.SC.SYS'] + ['0210 ERRBAR', '0220 CGAERRBAR', '0230 CCTL.IVALL'])
- FAIL fence after sc_sys M=256: one sc_sys fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['01f0 MEMBAR.ALL.CTA', '0200 MEMBAR.SC.SYS'] + ['0210 ERRBAR', '0220 CGAERRBAR', '0230 CCTL.IVALL'])
- FAIL fence after sc_sys M=1024: one sc_sys fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['01f0 MEMBAR.ALL.CTA', '0200 MEMBAR.SC.SYS'] + ['0210 ERRBAR', '0220 CGAERRBAR', '0230 CCTL.IVALL'])
- FAIL fence after acqrel_gpu M=0: one acqrel_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['01f0 MEMBAR.ALL.CTA', '0200 MEMBAR.ALL.GPU'] + ['0210 ERRBAR', '0220 CGAERRBAR', '0230 CCTL.IVALL'])
- FAIL fence after acqrel_gpu M=64: one acqrel_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['01f0 MEMBAR.ALL.CTA', '0200 MEMBAR.ALL.GPU'] + ['0210 ERRBAR', '0220 CGAERRBAR', '0230 CCTL.IVALL'])
- FAIL fence after acqrel_gpu M=256: one acqrel_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['01f0 MEMBAR.ALL.CTA', '0200 MEMBAR.ALL.GPU'] + ['0210 ERRBAR', '0220 CGAERRBAR', '0230 CCTL.IVALL'])
- FAIL fence after acqrel_gpu M=1024: one acqrel_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['01f0 MEMBAR.ALL.CTA', '0200 MEMBAR.ALL.GPU'] + ['0210 ERRBAR', '0220 CGAERRBAR', '0230 CCTL.IVALL'])
- FAIL fence before sc_gpu M=0: one sc_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0220 MEMBAR.ALL.CTA', '0230 MEMBAR.SC.GPU'] + ['0240 ERRBAR', '0250 CGAERRBAR', '0260 CCTL.IVALL'])
- FAIL fence before sc_gpu M=64: one sc_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0620 MEMBAR.ALL.CTA', '0630 MEMBAR.SC.GPU'] + ['0640 ERRBAR', '0650 CGAERRBAR', '0660 CCTL.IVALL'])
- FAIL fence before sc_gpu M=256: one sc_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['1220 MEMBAR.ALL.CTA', '1230 MEMBAR.SC.GPU'] + ['1240 ERRBAR', '1250 CGAERRBAR', '1260 CCTL.IVALL'])
- FAIL fence before sc_gpu M=1024: one sc_gpu fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['4220 MEMBAR.ALL.CTA', '4230 MEMBAR.SC.GPU'] + ['4240 ERRBAR', '4250 CGAERRBAR', '4260 CCTL.IVALL'])
- FAIL fence before sc_sys M=0: one sc_sys fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0220 MEMBAR.ALL.CTA', '0230 MEMBAR.SC.SYS'] + ['0240 ERRBAR', '0250 CGAERRBAR', '0260 CCTL.IVALL'])
- FAIL fence before sc_sys M=64: one sc_sys fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['0620 MEMBAR.ALL.CTA', '0630 MEMBAR.SC.SYS'] + ['0640 ERRBAR', '0650 CGAERRBAR', '0660 CCTL.IVALL'])
- FAIL fence before sc_sys M=256: one sc_sys fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['1220 MEMBAR.ALL.CTA', '1230 MEMBAR.SC.SYS'] + ['1240 ERRBAR', '1250 CGAERRBAR', '1260 CCTL.IVALL'])
- FAIL fence before sc_sys M=1024: one sc_sys fence (MEMBAR + ERRBAR / CGAERRBAR / CCTL) (fence ['4220 MEMBAR.ALL.CTA', '4230 MEMBAR.SC.SYS'] + ['4240 ERRBAR', '4250 CGAERRBAR', '4260 CCTL.IVALL'])

GPU 0: NVIDIA H100 80GB HBM3, sm_90, driver 595.71.05, persistence Enabled; 1 GPU(s) on the host; profile `reorder`; MPS not used; 0.2 min of runs; tickbound 0.1.0.

Host<->GPU clock bound (tick-edge sync before and after every run): 559-641 ns, feasible in 6 of 6 runs.

## Results, one line per run

- `reorder_loaduse`: reorder ±579 ns (edge): gate FAIL | load_use L1 L 49 d 33 N* 3.9 ((L-d)/4 4.1) max saving 19@N=32, L2 L 370 d 32 N* 84.8 ((L-d)/4 84.7) max saving 396@N=96, DRAM L 771 d 30 N* 164.1 ((L-d)/4 185.2) max saving 634@N=128 | 3 inconsistencies
- `reorder_war`: reorder ±586 ns (edge): gate FAIL | war reuse-fresh (1/32 warps) STS.32 +0/+1 (D 14), STS.128 +2/+8 (D 14), STG.32 +0/+0 (D 14), STG.128 +6/+46 (D 14) cy, WAR waits 8/8 | war reuse_clock-fresh_clock (1/32 warps) STS.32 +1/+3 (D 9), STS.128 +6/+18 (D 9), STG.32 +0/+1 (D 74), STG.128 +0/-24 cy, WAR waits 6/8
- `reorder_bar`: reorder ±559 ns (edge): gate FAIL | bar M=64: post-pre +0, post_mem-pre +296, sink +224 (4M 256, post clock in the wait 100%); M=256: post-pre +0, post_mem-pre +1066, sink +1030 (4M 1024, post clock in the wait 100%)
- `reorder_fence`: reorder ±625 ns (edge): gate FAIL, 27 dropped | fence sc_gpu -: saving  @M=; sc_sys -: saving  @M=; acqrel_gpu -: saving  @M= | 3 inconsistencies
- `reorder_pipe`: reorder ±571 ns (edge): gate FAIL | pipe cy/FFMA ILP1 4.01..7.96 (W 1..32, exp 4.0..7.8, k 7.8), ILP2 2.02..8.02 (W 1..32, exp 2.0..7.9, k 7.9), ILP4 1.02..2.21 (W 1..32, exp 1.0..3.0, k 3.0)
- `reorder_stall`: reorder ±641 ns (edge): gate FAIL | stall dep_consumed 16 ops slope [encoded per link, op stall, CuAsmRL]: IADD3 4.0[4,4,4] LOP3 4.0[4,4,-] SHF 4.0[4,4,-] LEA 4.0[4,4,4] SEL 4.0[4,4,4] IMNMX 4.0[4,4,4] IABS 8.0[8,4,4] HADD2 4.0[4,4,4] HFMA2 4.0[4,4,-] IMAD_WIDE 12.0[8,3,5] FMUL 4.0[4,4,-] FADD 4.0[4,4,4] FFMA 4.0[4,4,-] IMAD 4.0[4,4,4] DADD 8.0[8,8,-] DFMA 8.0[8,8,-]; indep IADD3 2.0 LOP3 2.0 SHF 2.0 LEA 2.0 SEL 2.0 IMNMX 2.0 IABS 4.0 HADD2 2.0 HFMA2 2.0 IMAD_WIDE 7.5 FMUL 1.0 FADD 1.0 FFMA 1.0 IMAD 2.0 DADD 2.2 DFMA 2.2

SASS check of the instruction brackets: 84/91 kernels have exactly N target opcodes between the clock reads; not verified: RED.

## Files

- `summary.json`: every run's analysis
- `timeline.html`: interactive timeline (open in a browser)
- `*.trace.json`: per-run Chrome trace format (https://ui.perfetto.dev)
- `reorder_sass.json`: the instruction-order checks of the reorder kernels in the probe binary (`tickbound sass --reorder`)
- `*.gpu.bin`, `*.host.bin`, `*.json`: raw records
