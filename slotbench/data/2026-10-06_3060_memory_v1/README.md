# Memory hierarchy by pointer chase, RTX 3060 (vast.ai, 2026-10-06, commit 869d870) - v1

Same method as `../2026-10-06_a100_memory_v1` (idle run only). Cost $0.008.

| tier | working set | .ca | .cg | .cs |
|---|---|---|---|---|
| L1 | ≤ 64 KB | 41 cy | 214–215 cy | 41 cy |
| L2 | 192 KB – 2 MB | 215 cy (112 ns) | 215 cy | 223–326 cy |
| DRAM | ≥ 4 MB | 511–521 cy (256–272 ns) | 513–521 cy | 513–521 cy |

Knees at 128 KB (L1 128 KB per SM) and 4 MB (L2 3 MB). Compared with the BSc measurement on an RTX 3060 Ti
(39 / 329 / 533 cycles, single loads bracketed by CS2R): L1 and DRAM agree; the L2 figure differs (215 vs 329) and
needs a check (different die GA106 vs GA104, or bracket overhead per load vs a 64-load batch). SM clock ~1.92 GHz;
the apparent 2.56 GHz at ≤ 64 KB is the 1024 ns tick quantising a ~1.4 µs batch, not a real clock.
