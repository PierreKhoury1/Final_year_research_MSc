# gputrace characterisation: NVIDIA GeForce RTX 3060 (c955f6158c67, 2026-10-06 23:23)

GPU 0: NVIDIA GeForce RTX 3060, sm_86, driver 580.173.02, persistence Enabled; 1 GPU(s) on the host; profile `ktrace`; MPS not used; 0.2 min of runs.

Host<->GPU clock bound (tick-edge sync before and after every run): 329-338 ns, feasible in 3 of 3 runs.

## Results, one line per run

- `ktrace`: ktrace ±329 ns (edge): B=28: load 512 | bar1 wait 24/155 lat 29 | compute 1237 | store+fence 569 | ticket 33 cy; span 3.3 us, starts spread 0.3 us, SM lines ±211 ns, ticket order 16/140 viol, launch->first 3862 ns, flag write->seen 1110 ns || B=112: load 821 | bar1 wait 65/874 lat 37 | compute 1488 | store+fence 559 | ticket 35 cy; span 8.4 us, starts spread 4.1 us, SM lines ±74 ns, ticket order 50/560 viol, launch->first 3760 ns, flag write->seen 536 ns
- `ktrace_long`: ktrace ±333 ns (edge): B=112: load 735 | bar1 wait 130/939 lat 38 | compute 22470 | store+fence 556 | ticket 35 cy; span 28.4 us, starts spread 14.1 us, SM lines ±14 ns, ticket order 187/336 viol, launch->first 4505 ns, flag write->seen 1208 ns
- `ktrace_idle`: ktrace ±338 ns (edge): B=28: load 510 | bar1 wait 24/157 lat 29 | compute 1237 | store+fence 569 | ticket 33 cy; span 3.1 us, starts spread 0.0 us, SM lines ±199 ns, ticket order 24/140 viol, launch->first 38147 ns, flag write->seen 1268 ns

SASS verification of the instruction brackets: 91/91 kernels have exactly N target opcodes between the clock reads.

## Files

- `summary.json`: every run's analysis
- `timeline.html`: interactive timeline (open in a browser)
- `*.trace.json`: per-run Chrome trace format (https://ui.perfetto.dev)
- `*.gpu.bin`, `*.host.bin`, `*.json`: raw records
