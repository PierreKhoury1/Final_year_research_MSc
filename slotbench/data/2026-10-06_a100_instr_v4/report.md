# gputrace characterisation: NVIDIA A100-SXM4-40GB (312ac2d1eb0f, 2026-10-06 20:47)

GPU 0: NVIDIA A100-SXM4-40GB, sm_80, driver 570.133.20, persistence Enabled; 1 GPU(s) on the host; profile `instr`; MPS used; 1.0 min of runs.

Host<->GPU clock bound (tick-edge sync before and after every run): 675-908 ns, feasible in 3 of 3 runs.

## Results, one line per run

- `instr`: instr(cotenant=0): LDG.ca@16K 39cy (ovh 217) | LDG.ca@1M 235cy (ovh 200) | LDG.ca@24M 367cy (ovh 132) | LDG.ca@128M 479cy (ovh 94) | LDG.cg@16K 285cy (ovh 190) | LDG.cg@1M 279cy (ovh 205) | LDG.cg@24M 371cy (ovh 217) | LDG.cg@128M 515cy (ovh 122) | LDG.cs@16K 39cy (ovh 217) | LDG.cs@1M 211cy (ovh 216) | LDG.cs@24M 373cy (ovh 174) | LDG.cs@128M 474cy (ovh 167) | LDG.nc@16K 39cy (ovh 217) | LDG.nc@1M 211cy (ovh 216) | LDG.nc@24M 359cy (ovh 183) | LDG.nc@128M 475cy (ovh 115) | LDS 23cy (ovh 215) | FADD 4cy (ovh 214) | FFMA 4cy (ovh 214) | IMAD 4cy (ovh 214) | SHFL 26cy (ovh 219) | ATOM(ret) 240cy (ovh 219) | RED+fence 8cy (ovh 564) | STG+fence 8cy (ovh 505) | BAR 36cy (ovh 63)
- `instr_cotenant`: instr(cotenant=1): LDG.ca@16K 951cy (ovh 1067) | LDG.ca@1M 950cy (ovh 1079) | LDG.ca@24M 953cy (ovh 1080) | LDG.ca@128M 960cy (ovh 1020) | LDG.cg@16K 949cy (ovh 1057) | LDG.cg@1M 942cy (ovh 1067) | LDG.cg@24M 951cy (ovh 1086) | LDG.cg@128M 949cy (ovh 1076) | LDG.cs@16K 954cy (ovh 1043) | LDG.cs@1M 953cy (ovh 1032) | LDG.cs@24M 948cy (ovh 1085) | LDG.cs@128M 956cy (ovh 1082) | LDG.nc@16K 950cy (ovh 1093) | LDG.nc@1M 945cy (ovh 1057) | LDG.nc@24M 952cy (ovh 1078) | LDG.nc@128M 957cy (ovh 1075) | LDS 23cy (ovh 1074) | FADD 4cy (ovh 1070) | FFMA 4cy (ovh 1071) | IMAD 4cy (ovh 1059) | SHFL 26cy (ovh 1073) | ATOM(ret) 846cy (ovh 1051) | RED+fence 8cy (ovh 1547) | STG+fence 8cy (ovh 1548) | BAR 34cy (ovh 849)
- `instr_mps50`: instr(cotenant=2): LDG.ca@16K 40cy (ovh 238) | LDG.ca@1M 237cy (ovh 325) | LDG.ca@24M 714cy (ovh 198) | LDG.ca@128M 720cy (ovh 221) | LDG.cg@16K 281cy (ovh 542) | LDG.cg@1M 706cy (ovh 212) | LDG.cg@24M 698cy (ovh 373) | LDG.cg@128M 717cy (ovh 187) | LDG.cs@16K 39cy (ovh 223) | LDG.cs@1M 282cy (ovh 184) | LDG.cs@24M 474cy (ovh 324) | LDG.cs@128M 578cy (ovh 918) | LDG.nc@16K 39cy (ovh 371) | LDG.nc@1M 706cy (ovh 358) | LDG.nc@24M 708cy (ovh 331) | LDG.nc@128M 724cy (ovh 210) | LDS 23cy (ovh 220) | FADD 4cy (ovh 218) | FFMA 4cy (ovh 347) | IMAD 4cy (ovh 369) | SHFL 26cy (ovh 375) | ATOM(ret) 420cy (ovh 305) | RED+fence 8cy (ovh 793) | STG+fence 8cy (ovh 794) | BAR 35cy (ovh 174)

## Instruction table (cycles per instruction = slope over chain length; alone / same-SM co-tenant / MPS 50 %)

| instruction | ws | alone | co-tenant | MPS 50 % |
|---|---|---|---|---|
| LDG.ca | 16 KB | 38.8 | 951.4 | 39.7 |
| LDG.ca | 1 MB | 235.2 | 949.9 | 236.8 |
| LDG.ca | 24 MB | 367.5 | 953.4 | 713.8 |
| LDG.ca | 128 MB | 478.7 | 960.4 | 719.8 |
| LDG.cg | 16 KB | 285.5 | 949.4 | 280.9 |
| LDG.cg | 1 MB | 279.5 | 942.5 | 705.7 |
| LDG.cg | 24 MB | 371.4 | 951.3 | 697.7 |
| LDG.cg | 128 MB | 515.4 | 949.4 | 716.5 |
| LDG.cs | 16 KB | 38.9 | 954.0 | 38.8 |
| LDG.cs | 1 MB | 211.2 | 952.7 | 282.2 |
| LDG.cs | 24 MB | 373.0 | 948.5 | 473.5 |
| LDG.cs | 128 MB | 473.7 | 955.9 | 578.2 |
| LDG.nc | 16 KB | 38.9 | 950.0 | 38.9 |
| LDG.nc | 1 MB | 210.8 | 945.4 | 706.5 |
| LDG.nc | 24 MB | 359.2 | 952.4 | 707.7 |
| LDG.nc | 128 MB | 474.6 | 957.4 | 724.4 |
| LDS |  | 23.0 | 22.9 | 23.0 |
| FADD |  | 4.0 | 3.9 | 4.0 |
| FFMA |  | 4.0 | 3.6 | 4.2 |
| IMAD |  | 4.0 | 3.9 | 4.0 |
| SHFL |  | 26.0 | 26.4 | 26.0 |
| ATOM(ret) |  | 240.1 | 846.1 | 419.7 |
| RED+fence |  | 8.1 | 8.2 | 8.1 |
| STG+fence |  | 7.7 | 7.8 | 7.8 |
| BAR |  | 36.0 | 33.9 | 34.9 |

SASS verification of the instruction brackets: 91/91 kernels have exactly N target opcodes between the clock reads.

## Files

- `summary.json`: every run's analysis
- `timeline.html`: interactive timeline (open in a browser)
- `*.trace.json`: per-run Chrome trace format (https://ui.perfetto.dev)
- `*.gpu.bin`, `*.host.bin`, `*.json`: raw records
