# clock/: CPU, GPU and NIC clocks on one axis (chapter 3)

Two tools and one report:

| Tool | Measures | Output |
|---|---|---|
| `bin/clockcal` | host `CLOCK_MONOTONIC_RAW` vs GPU `%globaltimer`, one fitted window per interval | `clock_windows.csv`, `meta.json`, optional `brackets_XXXXX.bin` |
| `bin/phc_offset` | host `CLOCK_MONOTONIC_RAW` vs NIC PTP hardware clock (PHC) | `phc.csv` |
| `python3 -m analysis.clock_report RUN_DIR [--out DIR] [--phc FILE]` | drift, wander, Allan deviation, residuals, plots | `clock_summary.json`, `clock_*.png` |

## Why CLOCK_MONOTONIC_RAW is the common axis

Every host timestamp in slotbench is `CLOCK_MONOTONIC_RAW`: the undisciplined local oscillator, never
slewed or stepped by NTP, ptp4l or phc2sys. The GPU is tied to it by bracket fits (clockcal, and the
driver's pre/post calibration), the NIC PHC by `PTP_SYS_OFFSET_PRECISE`, which reports the PHC together with
`CLOCK_MONOTONIC_RAW` for the same instant (or by `PTP_SYS_OFFSET_EXTENDED` plus a conversion, below).
So GPU time maps to host raw time and PHC time maps to host raw time, and with PTP the PHC is in turn tied
to the grandmaster. `CLOCK_REALTIME` is only a by-product (phc2sys disciplines it) and is never used as the axis.

## Procedure (6-hour run under thermal load)

1. Prepare the host as for the slot runs (`scripts/setup_os.sh`, `scripts/gpu_lock.sh`): isolated core, clocks
   locked, performance governor. Record whether each step was applied.
2. Optional PTP (with an i210/i225/i226 or similar NIC with a PHC, and a PTP grandmaster on the link):
   `scripts/setup_os.sh --iface IFACE` installs linuxptp, checks `ethtool -T IFACE` for hardware timestamping and
   prints the commands, which are
   ```
   ptp4l -i IFACE -m -s                              # PHC follows the grandmaster
   phc2sys -s IFACE -c CLOCK_REALTIME -O 0 -m        # CLOCK_REALTIME follows the PHC (not needed by us)
   ```
   (`--ptp-units` writes systemd units for them). Then in parallel with clockcal:
   ```
   bin/phc_offset --dev /dev/ptp0 --interval-ms 1000 --out RUN_DIR/phc.csv
   ```
   Find the PHC for an interface with `ethtool -T IFACE` ("PTP Hardware Clock: N" -> `/dev/ptpN`).
3. Start the thermal load (either the adversary, e.g. `bin/adversary --workload sgemm --duty 50 --prio low`, in
   another process, or clockcal's built-in `--load spin`), then:
   ```
   bin/clockcal --out runs/clock/rtx3060_load --duration-s 21600 --interval-s 60 --samples 10000 \
                --method pingpong --core 4 --fifo 90
   ```
   Repeat idle (no load) and with `--method launch` to compare the two bracket methods.
4. `python3 -m analysis.clock_report runs/clock/rtx3060_load` (picks up `phc.csv` from the run dir if present).

clockcal flags: `--out DIR` (required), `--duration-s 21600`, `--interval-s 60`, `--samples 10000` (per window),
`--method pingpong|launch`, `--core -1`, `--fifo 0`, `--gpu 0`, `--raw 0|1`, `--load none|spin`, `--print-config`
(no GPU needed). Unknown flags exit 2. SIGINT/SIGTERM finish the current window's row, then write `meta.json`.
Rows are flushed as they are written, so a killed run keeps every finished window; `meta.json` is written at
start with `"status": "running"`, refreshed every 10 windows and rewritten at the end.

## Bracket methods

- `pingpong` (default): a resident 1-thread kernel spins on a sequence word in mapped pinned memory; the host
  takes t0, writes seq, spins until the kernel echoes it (after writing `%globaltimer`), takes t1. The kernel is
  restarted every 400 samples (as `gpu_run/lockstep.cu` calibrate()); each sample has a 1 s timeout and the stream
  wait a 10 s guard. Typical brackets 1-2 us.
- `launch`: each sample launches a 1-thread kernel that writes `%globaltimer`, then a sequence number after a
  `__threadfence_system()`; the host spins until the sequence appears. The bracket includes the launch call;
  typical 5-20 us.

Both run on the highest-priority stream; `--load spin` runs short-block FMA kernels on the lowest-priority
stream from a separate host thread (started before the main thread is pinned, so it does not inherit the core
or SCHED_FIFO priority). meta.json records the `%globaltimer` update step (min/median/max over 64 steps),
because a coarse step (1 us on some parts) bounds how sharp a single bracket can be.

## clock_windows.csv

One row per window (`#` lines at the top are comments). All window statistics come from `sb::fit_clock`
(`common/clock_fit.h`): keep the tightest 1% of brackets (at least 50), fit host_mid = a (g - g_ref) + b.

| Column | Meaning |
|---|---|
| t_host_ns | host `CLOCK_MONOTONIC_RAW` of the window centre = host_k(g_c), with g_c = (min g + max g) / 2 over the window |
| n, n_kept | brackets collected, brackets used by the fit |
| offset_ns | `[host_k(g_c) - g_c] - [host_ref(g_ref_c) - g_ref_c]`: how the GPU->host offset wandered since the reference window (the first window that fitted) |
| rate_ppm | (a - 1) 1e6 of this window's fit; positive = host clock gains on the GPU clock |
| residual_rms_ns, residual_max_ns | fit residuals over the kept brackets |
| min_width_ns, median_width_ns | bracket widths over all valid brackets of the window |
| eps_ns | mapping error bound: half the median kept width + worst bracket violation |
| gpu_temp_c, sm_clock_mhz | NVML (dlopen of `libnvidia-ml.so.1`, device matched by PCI bus id); empty if NVML is missing |

A window whose fit fails keeps its row with empty fit columns.

**offset_ns definition.** Each window is evaluated with its own fit at its own centre, and the reference is the
reference window's fitted offset at its own centre. The GPU readings are ~1.7e18 ns, so the offset is kept as an
exact integer part (t_ref - g_ref) plus a small double. The fitted *rate* of one window is never
extrapolated to another: a 10000-sample pingpong window lasts only ~20 ms, and its rate is uncertain by several
ppm (a host simulation with 3 ppm true drift gave per-window rates from -1 to +10 ppm), which over 6 hours
would mean tens of milliseconds. Evaluating the first window's mapping at a later GPU reading would add exactly
that error. The slope of offset_ns vs time is the true relative rate; clock_report reports it as `drift_ppm`.

meta.json also carries a global fit over all kept brackets of all windows (`global_fit`, one straight line over
the whole run: its residuals show how far a single calibration would be off after hours), the reference window's
fit, GPU/driver/host details, NVML status, pin/FIFO success, window counts and timeouts.

`--raw 1` also writes `brackets_XXXXX.bin` per window: packed little-endian records of int64 t0, int64 t1,
uint64 g (24 bytes, no header); numpy dtype `[("t0","<i8"),("t1","<i8"),("g","<u8")]`.

## phc_offset

`bin/phc_offset [--dev /dev/ptp0] [--interval-ms 1000] [--count 0] [--out phc.csv] [--samples 25]` writes
`host_monoraw_ns,host_realtime_ns,phc_ns,method,width_ns`, one row per interval (flushed; `--count 0` = until
SIGINT/SIGTERM). The method is chosen once at start:

- `precise`: `PTP_SYS_OFFSET_PRECISE`, a hardware cross-timestamp giving PHC, `CLOCK_REALTIME` and
  `CLOCK_MONOTONIC_RAW` for one instant; width 0 (the driver/hardware cross-timestamp error is not visible to us).
- `extended_monoraw`: `PTP_SYS_OFFSET_EXTENDED` asking for `CLOCK_MONOTONIC_RAW` sandwiches through the clockid
  field that recent kernels (about 6.12/6.13 onwards, as far as we know) accept; older kernels reject it with
  EINVAL and the next method is used. Width = narrowest of `--samples` [sys, phc, sys] sandwiches.
- `extended`: `PTP_SYS_OFFSET_EXTENDED` with `CLOCK_REALTIME` sandwiches. `CLOCK_REALTIME` and
  `CLOCK_MONOTONIC_RAW` are read back to back just before and just after the ioctl; the realtime->raw offset is
  the mean of the two, and width_ns = sandwich width + twice (read bracket + half the before/after offset
  change), i.e. the full width of the host raw-time interval the PHC reading lies in. If phc2sys steps
  `CLOCK_REALTIME` during the ioctl the width shows it.

NIC support (hedged): Intel i225/i226 (igc driver) implement `PTP_SYS_OFFSET_PRECISE` through PCIe PTM only on
kernels whose igc has PTM cross-timestamping and when PTM is enabled end to end (root port and device; check
`lspci -vv` for "PTM"); otherwise EXTENDED is used. The i210 (igb) supports EXTENDED; we are not aware of
PRECISE support there. Headers without the PRECISE/EXTENDED definitions are handled by compiling in the
(stable) ioctl ABI.

## clock_report outputs

`clock_summary.json`: `drift_ppm` (least-squares slope of offset vs time), `rate_between_windows_ppm` and
`rate_fit_ppm` (mean/std/percentiles), `wander_p2p_ns` and `wander_detrended_p2p_ns` / `_rms_ns`, overlapping
Allan deviation of the offset (phase) series at tau = 1, 2, 4, ... x interval (gap-aware; needs >= 3 windows),
residual/eps/width statistics, temperature and SM clock ranges and the rate-vs-temperature correlation, the
global fit from meta.json, and a `phc` block (drift of PHC - raw in ppm, wander, largest step, width statistics,
Allan deviation) when a phc.csv is given. Figures: `clock_offset.png` (offset and detrended offset),
`clock_rate.png` (window and between-window rate, temperature on a twin axis), `clock_residual.png`,
`clock_adev.png`, `clock_phc.png`.

Look for steps in the detrended offset: if the driver ever adjusts `%globaltimer`, it shows as a jump there
rather than as drift.
