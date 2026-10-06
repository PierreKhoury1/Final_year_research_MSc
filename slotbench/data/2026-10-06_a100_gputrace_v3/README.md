# gputrace on an A100 SXM4 40 GB, campaign v3 (vast.ai, 2026-10-06, commit 7c67286)

Same host and GPU as `../2026-10-06_a100_gputrace` (New York, driver 580.173, 108 SMs, tick 1024 ns), run after
the tracer's wait loop was changed to yield (`__nanosleep`) between timer reads. Clock bound 0.52–0.85 µs, edge
method feasible in all 35 runs. Hog raw records (up to 215k blocks per run) are summarised in the analysis JSON only.
This campaign supersedes v2's dispatch and concurrency gap numbers; the launch, notify, copy and time-slice numbers
agree with v2 within noise.

## What changed and what it showed

**The wait loop is now inert**: with the yielding wait, 0 of 23 225 blocks had a timer gap above 5 µs at 8 blocks/SM
and every kernel span was within 0.5 % of waves × block length. The busy modes, kept as experiments, reproduce v2:
`dispatch_busy` (stall-free clock64 spin) p50 gap 202 µs at 8 blocks/SM; `dispatch_timer` (a stalling timer read
each iteration) p50 40 µs, max 150 µs. Warp-issue starvation of stall-free warps, not the timer read path.

**Two streams, yielding blocks**: B's first block still starts exactly when A's wave ends (wait = A block length −
100 µs: 0.103 / 0.404 / 1.904 / 9.904 ms), priority or not, although A's warps spend their time in `__nanosleep`.
The block scheduler does not place B's blocks on SMs whose slots A holds, however idle those warps are.

**Two processes under MPS** (`timeslice_mps50`, `timeslice_mps100`): 11 gaps of 316–348 µs p50 (max 466 µs) over the
7.5 s hog window, versus 1 707 gaps of 2.43 ms without MPS; our thread not running 0.0 % vs 37 %. The hog's 5 ms
blocks were never interrupted (max gap 2 µs vs 2.47 ms without MPS) and at 50 % it ran on exactly 54 of 108 SMs.
Launch → first instruction under MPS: 4.2 µs, unchanged.

**Bursts of 8 kernels**: 2.0 µs from one kernel's end to the next one's start, the same for 8 stream launches and
for one graph of 8 kernels; first kernel of a burst 4.2 µs (stream) / 4.0 µs (graph) after the call.

Other numbers (p50): launch → start 4.2 µs, graph 4.1 µs; notify flag 1.0 µs, event 3.0 µs, sync 3.0 µs; 8 B copy
7.2 / 7.3 µs; GPU read of host memory 1.0 µs; time-slice quantum 2.089 ms (us) / 2.431 ms (them), 228 switches/s.
