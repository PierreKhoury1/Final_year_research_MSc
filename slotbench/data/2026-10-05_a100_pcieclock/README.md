# GPU %globaltimer <-> host clock over PCIe: classic brackets vs tick-edge two-way bounds (A100 SXM4, vast.ai, NY)

Tool: tools/pcieclock.cu (commit 1e4f4db); analysis: analysis/pcieclock.py. 5 runs x (20k classic, 20k up, 20k down).
Timer edge step measured in-kernel: 1024 ns.

| Run | classic eps (slotbench clock_fit method) | tick-edge hard bound (strict, all samples) | violations |
|---|---:|---:|---:|
| 1 | 1146 ns | 743 ns | 0 |
| 2 | 1158 ns | 594 ns | 0 |
| 3 | 1183 ns | 572 ns | 0 |
| 4 | 1252 ns | 574 ns | 0 |
| 5 | 1194 ns | 576 ns | 0 |

- The tick-edge bound needs no assumption that PCIe delays are symmetric; it is the sum of the smallest observed
  GPU->host and host->GPU latencies, halved. Per-quarter windows: 573-809 ns.
- The classic midpoint estimate sits 53-605 ns (always positive) away from the tick-edge midpoint: the classic
  method's built-in symmetry assumption is biased on this host, though it stays inside its own eps.
- Earlier attempts (not used): pcieclock_a100_1/2 had a torn value/sequence message and an unordered load
  (the SM read %globaltimer before the host-clock load returned), which produced infeasible or impossibly
  tight (6-56 ns) bounds. Both fixed; the fixes are in the tool's history.
- One host, one GPU, idle. Not yet tested: other hosts/GPUs, GPU under load, longer windows, H100.
