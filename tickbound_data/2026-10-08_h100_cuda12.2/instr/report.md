# tickbound characterisation: NVIDIA H100 80GB HBM3 (82fbfcd49b63, 2026-10-08 01:56)

GPU 0: NVIDIA H100 80GB HBM3, sm_90, driver 535.309.01, persistence Enabled; 1 GPU(s) on the host; profile `instr`; MPS used; 0.9 min of runs; tickbound 0.1.0.

Host<->GPU clock bound (tick-edge sync before and after every run): 638-717 ns, feasible in 3 of 3 runs.

## Results, one line per run

- `instr`: instr(cotenant=0): LDG.ca@16K 39cy (ovh 290) | LDG.ca@1M 310cy (ovh 263) | LDG.ca@24M 327cy (ovh 182) | LDG.ca@128M 558cy (ovh 189) | LDG.cg@16K 283cy (ovh 286) | LDG.cg@1M 283cy (ovh 289) | LDG.cg@24M 289cy (ovh 275) | LDG.cg@128M 550cy (ovh 218) | LDG.cs@16K 39cy (ovh 290) | LDG.cs@1M 285cy (ovh 283) | LDG.cs@24M 300cy (ovh 241) | LDG.cs@128M 566cy (ovh 145) | LDG.nc@16K 39cy (ovh 290) | LDG.nc@1M 283cy (ovh 287) | LDG.nc@24M 294cy (ovh 264) | LDG.nc@128M 548cy (ovh 264) | LDS 23cy (ovh 286) | FADD 4cy (ovh 286) | FFMA 4cy (ovh 286) | IMAD 4cy (ovh 285) | SHFL 26cy (ovh 291) | ATOM(ret) 483cy (ovh 284) | RED+fence 8cy (ovh 978) | STG+fence 8cy (ovh 973) | BAR 27cy (ovh 145)
- `instr_cotenant`: instr(cotenant=1): LDG.ca@16K 976cy (ovh 395) | LDG.ca@1M 974cy (ovh 454) | LDG.ca@24M 972cy (ovh 426) | LDG.ca@128M 978cy (ovh 397) | LDG.cg@16K 975cy (ovh 440) | LDG.cg@1M 978cy (ovh 441) | LDG.cg@24M 970cy (ovh 468) | LDG.cg@128M 969cy (ovh 440) | LDG.cs@16K 966cy (ovh 468) | LDG.cs@1M 977cy (ovh 410) | LDG.cs@24M 978cy (ovh 390) | LDG.cs@128M 977cy (ovh 390) | LDG.nc@16K 972cy (ovh 440) | LDG.nc@1M 972cy (ovh 449) | LDG.nc@24M 971cy (ovh 429) | LDG.nc@128M 977cy (ovh 422) | LDS 23cy (ovh 442) | FADD 4cy (ovh 436) | FFMA 4cy (ovh 432) | IMAD 4cy (ovh 428) | SHFL 26cy (ovh 435) | ATOM(ret) 529cy (ovh 323) | RED+fence 8cy (ovh 1184) | STG+fence 8cy (ovh 1180) | BAR 26cy (ovh 244)
- `instr_mps50`: instr(cotenant=2): LDG.ca@16K 39cy (ovh 284) | LDG.ca@1M 847cy (ovh 216) | LDG.ca@24M 850cy (ovh 266) | LDG.ca@128M 859cy (ovh 238) | LDG.cg@16K 424cy (ovh 626) | LDG.cg@1M 852cy (ovh 270) | LDG.cg@24M 854cy (ovh 272) | LDG.cg@128M 855cy (ovh 248) | LDG.cs@16K 39cy (ovh 283) | LDG.cs@1M 853cy (ovh 250) | LDG.cs@24M 852cy (ovh 286) | LDG.cs@128M 843cy (ovh 298) | LDG.nc@16K 39cy (ovh 283) | LDG.nc@1M 853cy (ovh 295) | LDG.nc@24M 852cy (ovh 267) | LDG.nc@128M 855cy (ovh 252) | LDS 23cy (ovh 279) | FADD 4cy (ovh 278) | FFMA 4cy (ovh 294) | IMAD 4cy (ovh 278) | SHFL 26cy (ovh 284) | ATOM(ret) 320cy (ovh 200) | RED+fence 8cy (ovh 966) | STG+fence 8cy (ovh 961) | BAR 27cy (ovh 139)

## Instruction table (cycles per instruction = slope over chain length)

| instruction | ws | alone | co-tenant | MPS 50 % |
|---|---|---|---|---|
| LDG.ca | 16 KB | 38.7 | 976.0 | 38.6 |
| LDG.ca | 1 MB | 309.9 | 973.6 | 846.6 |
| LDG.ca | 24 MB | 326.6 | 972.2 | 850.2 |
| LDG.ca | 128 MB | 558.1 | 977.5 | 858.8 |
| LDG.cg | 16 KB | 282.9 | 975.4 | 424.2 |
| LDG.cg | 1 MB | 282.7 | 977.6 | 852.2 |
| LDG.cg | 24 MB | 288.9 | 970.3 | 853.8 |
| LDG.cg | 128 MB | 549.7 | 968.7 | 855.5 |
| LDG.cs | 16 KB | 38.7 | 966.2 | 38.7 |
| LDG.cs | 1 MB | 284.7 | 977.4 | 853.4 |
| LDG.cs | 24 MB | 299.6 | 978.0 | 851.6 |
| LDG.cs | 128 MB | 565.9 | 977.4 | 842.7 |
| LDG.nc | 16 KB | 38.7 | 971.9 | 38.7 |
| LDG.nc | 1 MB | 283.2 | 971.8 | 853.0 |
| LDG.nc | 24 MB | 294.0 | 971.4 | 852.1 |
| LDG.nc | 128 MB | 547.9 | 976.8 | 854.7 |
| LDS |  | 23.0 | 23.2 | 23.0 |
| FADD |  | 4.0 | 4.0 | 4.1 |
| FFMA |  | 4.0 | 4.1 | 3.8 |
| IMAD |  | 4.0 | 4.2 | 4.0 |
| SHFL |  | 26.0 | 26.3 | 26.0 |
| ATOM(ret) |  | 483.2 | 529.1 | 319.9 |
| RED+fence |  | 7.9 | 8.1 | 7.9 |
| STG+fence |  | 8.0 | 8.2 | 8.0 |
| BAR |  | 26.6 | 25.6 | 26.6 |

Note: rows whose working set exceeds L2 mix L2 hits and DRAM misses (each chain re-walks lines the previous chain left in L2); use the memory run for DRAM latency.

SASS check of the instruction brackets: 84/91 kernels have exactly N target opcodes between the clock reads; not verified: RED.

## Files

- `summary.json`: every run's analysis
- `timeline.html`: interactive timeline (open in a browser)
- `*.trace.json`: per-run Chrome trace format (https://ui.perfetto.dev)
- `*.gpu.bin`, `*.host.bin`, `*.json`: raw records
