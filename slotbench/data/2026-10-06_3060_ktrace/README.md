# Kernel timeline, RTX 3060 (2026-10-06, commit 64b1a31, corrected build)

As `../2026-10-06_a100_ktrace` on a rented RTX 3060 (28 SMs, 1024 ns tick; B = 28 and 112; `ktrace_phases.json`
from the sm_86 build). Cost $0.01. This host's `%globaltimer` reads 1.79e18 ns: the analysis rebases every timer
value in integer arithmetic (float64 would quantise differences to 256 ns).

Per warp, cycles (as the A100 README): two dependent loads 512 (1 block per SM) / 821 (4 per SM); first barrier
wait 56 (p99 162) / 158 (p99 1 220), release 34 / 64; 256 FFMA 1 237 / 1 488; store + fence 569 / 559; warp 0's
atomic 282 / 310. Launch to first warp 3.7–3.9 µs warm (31 µs on the run's first); last exit to sync return
1.5 µs; flag write to the host seeing it at most ~0.5 µs, not resolved below the run's ±0.33 µs host bound.
Block bound (largest half-width over a block's stamps, median over blocks): 162 ns (B = 28, 3 µs launches with
few tick edges) / 78 ns (B = 112), worst block 445 / 261 ns; common-rate interval 1.80–1.82 GHz (B = 112,
median), 1.60–1.88 GHz (B = 28); rate inconsistency 0 ns (`ktrace`), 15–47 ns (`ktrace_long`, 28 µs). 0 early
releases, 0 ticket-order violations on every launch; ticket-check floor 180–210 ns (B = 112), up to 640 ns (B =
28). The earlier README's "timer guard up to 82 ns" and "warps' %globaltimer readings disagree by ~80 ns" were an
artefact of a ±5 % rate clip in the fit and are retracted. The first build's deferred-barrier artefact is
documented in `../2026-10-06_a100_ktrace_v1_deferred`.
