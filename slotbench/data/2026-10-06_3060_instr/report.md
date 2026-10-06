# gputrace characterisation: NVIDIA GeForce RTX 3060 (91a15f2d2610, 2026-10-06 20:46)

GPU 0: NVIDIA GeForce RTX 3060, sm_86, driver 580.173.02, persistence Enabled; 1 GPU(s) on the host; profile `instr`; MPS used; 0.9 min of runs.

Host<->GPU clock bound (tick-edge sync before and after every run): 497-645 ns, feasible in 3 of 3 runs.

## Results, one line per run

- `instr`: instr(cotenant=0): LDG.ca@16K 39cy (ovh 213) | LDG.ca@1M 242cy (ovh 109) | LDG.ca@24M 410cy (ovh 126) | LDG.ca@128M 443cy (ovh 188) | LDG.cg@16K 214cy (ovh 212) | LDG.cg@1M 213cy (ovh 211) | LDG.cg@24M 430cy (ovh 199) | LDG.cg@128M 440cy (ovh 211) | LDG.cs@16K 39cy (ovh 213) | LDG.cs@1M 226cy (ovh 154) | LDG.cs@24M 514cy (ovh 135) | LDG.cs@128M 522cy (ovh 137) | LDG.nc@16K 39cy (ovh 213) | LDG.nc@1M 213cy (ovh 212) | LDG.nc@24M 433cy (ovh 214) | LDG.nc@128M 439cy (ovh 190) | LDS 23cy (ovh 212) | FADD 4cy (ovh 211) | FFMA 4cy (ovh 211) | IMAD 4cy (ovh 210) | SHFL 26cy (ovh 221) | ATOM(ret) 242cy (ovh 205) | RED+fence 8cy (ovh 499) | STG+fence 8cy (ovh 499) | BAR 37cy (ovh 58)
- `instr_cotenant`: instr(cotenant=1): LDG.ca@16K 1306cy (ovh 187) | LDG.ca@1M 1286cy (ovh 117) | LDG.ca@24M 1281cy (ovh 75) | LDG.ca@128M 1285cy (ovh 126) | LDG.cg@16K 1304cy (ovh 222) | LDG.cg@1M 1278cy (ovh 62) | LDG.cs@16K 1310cy (ovh 152) | LDG.cs@1M 1285cy (ovh 20) | LDG.cs@24M 1287cy (ovh 165) | LDG.cs@128M 1280cy (ovh -16) | LDG.nc@16K 1302cy (ovh 189) | LDG.nc@1M 1268cy (ovh 152) | LDG.nc@24M 1281cy (ovh 95) | LDG.nc@128M 1287cy (ovh 44) | LDS 23cy (ovh 213) | FADD 4cy (ovh 212) | FFMA 4cy (ovh 212) | IMAD 4cy (ovh 212) | SHFL 26cy (ovh 221) | ATOM(ret) 321cy (ovh -45) | RED+fence 8cy (ovh 535) | STG+fence 8cy (ovh 536) | BAR 37cy (ovh 59)
- `instr_mps50`: instr(cotenant=2): LDG.ca@16K 39cy (ovh 217) | LDG.ca@1M 723cy (ovh 161) | LDG.ca@24M 577cy (ovh 483) | LDG.ca@128M 725cy (ovh 204) | LDG.cg@16K 725cy (ovh 181) | LDG.cg@1M 724cy (ovh 168) | LDG.cg@24M 715cy (ovh 102) | LDG.cg@128M 740cy (ovh 189) | LDG.cs@16K 39cy (ovh 230) | LDG.cs@1M 734cy (ovh 190) | LDG.cs@24M 720cy (ovh 211) | LDG.cs@128M 728cy (ovh 197) | LDG.nc@16K 39cy (ovh 216) | LDG.nc@1M 520cy (ovh 1049) | LDG.nc@24M 733cy (ovh 194) | LDG.nc@128M 728cy (ovh 193) | LDS 23cy (ovh 214) | FADD 4cy (ovh 214) | FFMA 4cy (ovh 214) | IMAD 4cy (ovh 213) | SHFL 26cy (ovh 222) | ATOM(ret) 271cy (ovh 66) | RED+fence 8cy (ovh 500) | STG+fence 8cy (ovh 498) | BAR 37cy (ovh 67)

## Instruction table (cycles per instruction = slope over chain length; alone / same-SM co-tenant / MPS 50 %)

| instruction | ws | alone | co-tenant | MPS 50 % |
|---|---|---|---|---|
| LDG.ca | 16 KB | 38.9 | 1305.9 | 38.8 |
| LDG.ca | 1 MB | 242.2 | 1286.3 | 722.8 |
| LDG.ca | 24 MB | 409.5 | 1281.2 | 577.0 |
| LDG.ca | 128 MB | 443.3 | 1285.0 | 724.6 |
| LDG.cg | 16 KB | 213.8 | 1303.6 | 725.3 |
| LDG.cg | 1 MB | 213.4 | 1278.4 | 723.7 |
| LDG.cg | 24 MB | 429.9 | - | 714.6 |
| LDG.cg | 128 MB | 439.6 | - | 740.1 |
| LDG.cs | 16 KB | 38.9 | 1309.9 | 38.8 |
| LDG.cs | 1 MB | 226.3 | 1285.1 | 734.1 |
| LDG.cs | 24 MB | 513.6 | 1287.5 | 720.1 |
| LDG.cs | 128 MB | 521.7 | 1280.1 | 728.1 |
| LDG.nc | 16 KB | 38.9 | 1302.3 | 38.9 |
| LDG.nc | 1 MB | 213.5 | 1267.9 | 519.8 |
| LDG.nc | 24 MB | 432.7 | 1281.0 | 733.5 |
| LDG.nc | 128 MB | 439.2 | 1287.0 | 728.1 |
| LDS |  | 23.0 | 23.4 | 23.0 |
| FADD |  | 4.0 | 4.0 | 4.0 |
| FFMA |  | 4.0 | 4.0 | 4.0 |
| IMAD |  | 4.0 | 4.0 | 4.0 |
| SHFL |  | 25.9 | 26.2 | 25.9 |
| ATOM(ret) |  | 241.9 | 320.7 | 271.2 |
| RED+fence |  | 8.1 | 8.3 | 8.1 |
| STG+fence |  | 7.7 | 7.9 | 7.8 |
| BAR |  | 37.0 | 37.0 | 37.0 |

SASS verification of the instruction brackets: 91/91 kernels have exactly N target opcodes between the clock reads.

## Files

- `summary.json`: every run's analysis
- `timeline.html`: interactive timeline (open in a browser)
- `*.trace.json`: per-run Chrome trace format (https://ui.perfetto.dev)
- `*.gpu.bin`, `*.host.bin`, `*.json`: raw records
