# Kernel timeline, RTX 3060 (2026-10-06, commit 64b1a31, corrected build)

As `../2026-10-06_a100_ktrace` on a rented RTX 3060 (28 SMs, 1024 ns tick; B = 28 and 112; `ktrace_phases.json`
from the sm_86 build). Cost $0.01. This host's `%globaltimer` reads 1.79e18 ns: the analysis rebases every timer
value in integer arithmetic (float64 would quantise differences to 256 ns).

Per warp, cycles: checkpoint 14; two dependent loads 512 (1 block per SM) / 821 (4 per SM); first barrier wait
24 (p99 155) / 65 (p99 874), release 35 / 54; 256 FFMA 1 237 / 1 488; store + fence 569 / 559; ticket 33. Launch
to first warp 4.0 µs; flag write to host 1.2 µs; last exit to sync return 1.9 µs. Block lines: half-width 178 ns
median, 362 ns max; timer guard up to 82 ns; 0 early releases, 0 ticket-order violations on every launch. The
`%globaltimer` readings of different warps on one SM disagree by up to ~80 ns (the guard); the first build's
deferred-barrier artefact is documented in `../2026-10-06_a100_ktrace_v1_deferred`.
