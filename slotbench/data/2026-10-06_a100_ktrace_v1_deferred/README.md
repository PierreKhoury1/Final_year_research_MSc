# A100 kernel timeline, first build: the deferred-barrier artefact (2026-10-06, commit 3c34d41)

The first `ktrace` build stamped the barrier release immediately after `__syncthreads`. On this A100 (as on the
RTX 3060 run before it) every warp's release stamp sits 13–17 cycles after its own arrival stamp whatever the other
warps did, and 67–80 % of the warps are "released" before the last warp of their block arrived: `__syncthreads`
compiles to `BAR.SYNC.DEFER_BLOCKING`, and the warp keeps issuing non-memory instructions (the clock read
included) until it reaches a memory instruction. Kept as the evidence for that finding; the corrected build
(release stamps after a shared load that waits for the barrier, cycle reads bracketed by two timer reads) is in
`../2026-10-06_a100_ktrace` and `../2026-10-06_3060_ktrace`. `ktrace.gpu.bin.gz` holds the 237 600 stamps
(gunzip before `analysis/gputrace.py`); the analysis of this build's records with the current code is not
meaningful for the barrier phases.
