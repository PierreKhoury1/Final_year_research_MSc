# Memory hierarchy by pointer chase, A100 SXM4 (vast.ai, 2026-10-06, commit 869d870) - v1

One warp, one load per hop over a random cyclic permutation of 128-byte lines; per batch of 64 hops one record
of (clock64, %globaltimer); 17 working sets from 4 KB to 128 MB, modifiers .ca/.cg/.cs, 3 reps. Raw records were
dropped by the transport (53 MB); `memory*.analysis.json` carries the curves. Cost $0.05.

| tier | working set | .ca | .cg | .cs |
|---|---|---|---|---|
| L1 | ≤ 128 KB | 41 cy (32 ns) | 284–287 cy | 41 cy |
| L2 near | 256 KB – 16 MB | 211 cy (144 ns) | 282–285 cy (208 ns) | 161–279 cy |
| L2 far half | 24–32 MB | 424 cy (304 ns) | 282–321 cy | 439–460 cy |
| DRAM | ≥ 48 MB | 565–577 cy (400 ns) | 566–570 cy | 573–577 cy |

Knees at 192 KB (L1 is 192 KB per SM on A100) and 24 MB (the 40 MB L2 is two halves; crossing ~20 MB reaches the
far one at twice the latency). `.cg` never uses L1 (flat ~284 cy); `.cs` (evict-first) is penalised between 256 KB and
16 MB. SM clock during the sweep 1.28–1.41 GHz.

**The co-tenant variant of this run is invalid**: the 120 s streaming kernel (one 256-thread block per SM) was
launched first, and the chase kernel on another stream did not start until it finished (host events: launched at
0.10 s, returned at 120.08 s), so `memory_cotenant.analysis.json` equals the idle run. See `coexist*` in the next
run: a kernel from another stream appears to be placed only when a block of the running kernel retires.
At the L1 tier a 64-hop batch lasts ~2 µs, two 1024 ns ticks, so the ns column there is quantised; cycles are exact.
