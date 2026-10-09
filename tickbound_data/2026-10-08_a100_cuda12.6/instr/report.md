# tickbound characterisation: NVIDIA A100-SXM4-40GB (5d89b2e2e8cf, 2026-10-08 01:58)

GPU 0: NVIDIA A100-SXM4-40GB, sm_80, driver 580.173.02, persistence Enabled; 1 GPU(s) on the host; profile `instr`; MPS used; 1.0 min of runs; tickbound 0.1.0.

Host<->GPU clock bound (tick-edge sync before and after every run): 527-535 ns, feasible in 3 of 3 runs.

## Results, one line per run

- `instr`: instr(cotenant=0): LDG.ca@16K 39cy (ovh 211) | LDG.ca@1M 236cy (ovh 200) | LDG.ca@24M 252cy (ovh 84) | LDG.ca@128M 471cy (ovh 161) | LDG.cg@16K 287cy (ovh 190) | LDG.cg@1M 282cy (ovh 195) | LDG.cg@24M 286cy (ovh 189) | LDG.cg@128M 461cy (ovh 219) | LDG.cs@16K 39cy (ovh 211) | LDG.cs@1M 214cy (ovh 208) | LDG.cs@24M 355cy (ovh 38) | LDG.cs@128M 475cy (ovh 69) | LDG.nc@16K 39cy (ovh 211) | LDG.nc@1M 213cy (ovh 211) | LDG.nc@24M 245cy (ovh 173) | LDG.nc@128M 301cy (ovh 12) | LDS 23cy (ovh 209) | FADD 4cy (ovh 208) | FFMA 4cy (ovh 208) | IMAD 4cy (ovh 208) | SHFL 26cy (ovh 213) | ATOM(ret) 242cy (ovh 155) | RED+fence 8cy (ovh 562) | STG+fence 8cy (ovh 505) | BAR 36cy (ovh 58)
- `instr_cotenant`: instr(cotenant=1): LDG.ca@16K 956cy (ovh 1052) | LDG.ca@1M 953cy (ovh 1087) | LDG.ca@24M 955cy (ovh 1076) | LDG.ca@128M 946cy (ovh 1124) | LDG.cg@16K 950cy (ovh 1065) | LDG.cg@1M 942cy (ovh 1080) | LDG.cg@24M 950cy (ovh 1068) | LDG.cs@16K 950cy (ovh 1099) | LDG.cs@1M 946cy (ovh 1125) | LDG.cs@24M 950cy (ovh 1084) | LDG.cs@128M 955cy (ovh 1013) | LDG.nc@16K 957cy (ovh 1033) | LDG.nc@1M 949cy (ovh 1063) | LDG.nc@24M 956cy (ovh 1065) | LDG.nc@128M 955cy (ovh 1017) | LDS 22cy (ovh 1068) | FADD 4cy (ovh 1060) | FFMA 4cy (ovh 1052) | IMAD 4cy (ovh 1064) | SHFL 27cy (ovh 1065) | ATOM(ret) 847cy (ovh 1069) | RED+fence 8cy (ovh 1547) | STG+fence 8cy (ovh 1537) | BAR 34cy (ovh 849)
- `instr_mps50`: instr(cotenant=2): LDG.ca@16K 38cy (ovh 394) | LDG.ca@1M 711cy (ovh 397) | LDG.ca@24M 732cy (ovh -652) | LDG.ca@128M 722cy (ovh 382) | LDG.cg@16K 266cy (ovh 805) | LDG.cg@1M 701cy (ovh 187) | LDG.cg@24M 718cy (ovh -475) | LDG.cg@128M 712cy (ovh 385) | LDG.cs@16K 39cy (ovh 376) | LDG.cs@1M 698cy (ovh 437) | LDG.cs@24M 709cy (ovh 403) | LDG.cs@128M 719cy (ovh 380) | LDG.nc@16K 39cy (ovh 375) | LDG.nc@1M 710cy (ovh 377) | LDG.nc@24M 712cy (ovh 376) | LDG.nc@128M 720cy (ovh 375) | LDS 23cy (ovh 373) | FADD 4cy (ovh 372) | FFMA 4cy (ovh 372) | IMAD 3cy (ovh 336) | SHFL 26cy (ovh 223) | ATOM(ret) 247cy (ovh 207) | RED+fence 8cy (ovh 578) | STG+fence 8cy (ovh 521) | BAR 36cy (ovh 65)

## Instruction table (cycles per instruction = slope over chain length)

| instruction | ws | alone | co-tenant | MPS 50 % |
|---|---|---|---|---|
| LDG.ca | 16 KB | 38.9 | 955.9 | 38.0 |
| LDG.ca | 1 MB | 236.4 | 952.7 | 710.6 |
| LDG.ca | 24 MB | 251.5 | 954.7 | 731.8 |
| LDG.ca | 128 MB | 470.9 | 946.4 | 722.4 |
| LDG.cg | 16 KB | 287.4 | 950.0 | 266.3 |
| LDG.cg | 1 MB | 281.7 | 941.7 | 700.5 |
| LDG.cg | 24 MB | 285.6 | 949.7 | 717.9 |
| LDG.cg | 128 MB | 460.7 | - | 712.2 |
| LDG.cs | 16 KB | 38.9 | 949.7 | 38.7 |
| LDG.cs | 1 MB | 214.1 | 946.2 | 697.7 |
| LDG.cs | 24 MB | 355.1 | 949.6 | 709.3 |
| LDG.cs | 128 MB | 475.0 | 955.0 | 719.4 |
| LDG.nc | 16 KB | 38.9 | 957.5 | 38.8 |
| LDG.nc | 1 MB | 213.1 | 948.7 | 709.9 |
| LDG.nc | 24 MB | 244.5 | 956.5 | 712.0 |
| LDG.nc | 128 MB | 300.6 | 955.3 | 720.4 |
| LDS |  | 23.0 | 22.4 | 23.0 |
| FADD |  | 4.0 | 3.9 | 4.0 |
| FFMA |  | 4.0 | 3.9 | 4.0 |
| IMAD |  | 4.0 | 3.7 | 2.8 |
| SHFL |  | 26.0 | 26.5 | 26.0 |
| ATOM(ret) |  | 242.2 | 846.8 | 246.6 |
| RED+fence |  | 8.1 | 8.4 | 8.1 |
| STG+fence |  | 7.7 | 7.7 | 7.7 |
| BAR |  | 36.1 | 33.9 | 36.0 |

Note: rows whose working set exceeds L2 mix L2 hits and DRAM misses (each chain re-walks lines the previous chain left in L2); use the memory run for DRAM latency.

SASS check of the instruction brackets: 91/91 kernels have exactly N target opcodes between the clock reads.

## Files

- `summary.json`: every run's analysis
- `timeline.html`: interactive timeline (open in a browser)
- `*.trace.json`: per-run Chrome trace format (https://ui.perfetto.dev)
- `*.gpu.bin`, `*.host.bin`, `*.json`: raw records
