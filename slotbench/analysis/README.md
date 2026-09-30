# slotbench analysis

Python 3.10+ with numpy, scipy, matplotlib (Agg) and pandas. Run every command from `slotbench/`.

```
python3 -m analysis.summarize RUN_DIR [RUN_DIR...]    # writes RUN_DIR/summary.json
python3 -m analysis.validate RUNS_ROOT                # table + RUNS_ROOT/validation.json, exit 1 if any run invalid
python3 -m analysis.report RUNS_ROOT --out REPORT_DIR # summaries, validation, figures, summary.csv, report.md
python3 -m analysis.synth OUT_DIR [--slots N]         # FAKE but format-exact run tree for demos/tests
python3 -m analysis.clock_report ...                  # clock calibration runs (separate module)
python3 -m pytest -q analysis/tests                   # tests (use synth; no GPU)
```

`report` works on a partial matrix and on cloud results that carry only `summary.json` (no `slots.bin`):
it summarizes runs whose summary is missing or older than its inputs and reads everything else from
the summaries. Options: `--include-invalid` (plot invalid runs too), `--throttle warn`,
`--idle-p9999-us F`. `validate` takes `--throttle`, `--idle-p9999-us`, `--clock-tol-mhz`, `--cv-warn`.

## Metric definitions (DESIGN.md section 4)

All record times are host `CLOCK_MONOTONIC_RAW` ns; g0/g1 are GPU `%globaltimer` ns.

| Metric | Definition |
|---|---|
| latency | t1 - t0 (primary); a recorded slot misses if latency > deadline (strict) |
| latency_from_boundary | t1 - t_sched |
| wake_overshoot | t_wake - (t_sched - spin_ns), spin_ns from meta.json config (`spin_us`) |
| start_lateness | t0 - t_sched |
| launch_cost | t_launched - t0 |
| gpu_exec | g1 - g0, only records with both stamps non-zero |
| queue_delay | host_of(g0) - t0 using the clock fits (below), records with g0 != 0 |
| skipped boundaries | interior gaps in `slot` (boundaries before the first / after the last record are not counted); each is a miss |
| total slots, misses | total = recorded + skipped; misses = late recorded slots + skipped |
| miss rate | misses / total with an exact two-sided 95% Clopper-Pearson interval (0 misses in n still gives the upper bound 1 - 0.025^(1/n), 3.69e-6 for n = 1e6) |
| quantiles | p50, p90, p99, p99.9, p99.99, max with method "higher": x_sorted[ceil(q (n-1))], exact integer index; always an observed sample, never below the interpolated value (= `numpy.quantile(..., method="higher")`) |
| jitter | p99.99 - p50 |
| relative throughput | adversary units_per_s / mean units_per_s of the valid `SOLO_<W>_r*` runs in the same config dir |

**Clock mapping for queue_delay.** meta.json carries the pre- and post-run fits (common/clock_fit.h).
With both, host_of(g) is the straight line through the two fits' anchors (the fit evaluated at the
median GPU reading of its calibration CSV, else at its g_ref). A single fit's slope comes from a
calibration window well under a second long; extrapolated over a 500 s run, even 1 ppm of slope error
is 500 us (on the synthetic 1e6-slot run, one fit alone is off by up to 0.8 ms and the two-point line
by 25 ns). With one fit only, `queue_delay_us.method` says `pre` or `post`; treat those values with
suspicion. `summary.json` keeps each fit's `eps_ns`, the mapping error bound.

**slots.bin reading.** The header is checked (magic, version, record size). `n_records == 0` means the
driver did not close cleanly: the record count comes from the file size and the run is flagged
`crashed`. An incomplete trailing record is ignored and flagged; either flag invalidates the run.

## summary.json

Identity; header; `counts` (recorded, requested, gaps, meta counters, crashed, exit reason); `miss`;
quantile blocks in us for `latency_us` (plus `jitter`), `latency_from_boundary_us`, `start_lateness_us`,
`launch_cost_us`, `wake_overshoot_us`, `gpu_exec_us`, `queue_delay_us`; `clock_fits`; `histogram` of
latency (edges 1 us to 1 s, 100 bins per decade, bins `[lo, hi)`, plus underflow, overflow and skipped
counts); `worst` (the 200 largest-latency records, every field); `adversary` and relative throughput;
`validation`. Keys are sorted and there are no timestamps, so re-running gives an identical file.

## Validation (DESIGN.md section 8)

Each check is pass / fail / warn / skip with a reason; any fail makes the run invalid:
`status_file` (status says ok), `not_crashed`, `file_integrity`, `ring_overflows == 0`,
`stamp_mismatches == 0`, `recorded_eq_requested`, `slot_order` (strictly increasing), `skipped_consistent`
(meta counter vs gaps: warn), `idle_baseline` (D=0: 0 misses and p99.99 < 300 us), `adversary_present`
(warn), `throttle` (telemetry clocks-event reasons other than GpuIdle/ApplicationsClocksSetting:
power/thermal bits fail, the rest warn; `--throttle warn` downgrades), `clocks_locked` (telemetry SM
clock vs the locked value in run.json, within 15 MHz for 95% of samples; skipped for M5 and when
run.json records locking as NOT SUPPORTED). Repeatability: for cells with more than one valid rep, the
coefficient of variation of p99.99 (warn above 0.10; reported, never invalidates). Telemetry is
judged over the whole file (adversary warm-up included), which is the conservative choice.

## Reading the figures

Each mechanism keeps one colour, marker and line style in every figure.

- **CCDF** (`ccdf_<config>_<workload>_d<duty>`): x = latency (us, log), y = P(latency > x) (log), one
  line per mechanism, reps pooled, dotted line at the deadline. Skipped boundaries count as infinite
  latency, so the curve's height at the deadline is the miss rate, and a line ending in a flat tail is
  the skipped fraction. Lower and further left is better. "hist" in the legend marks a curve redrawn from
  the summary histogram: every sample is placed at its bin's upper edge, so it can only overstate the
  tail, by at most one bin (2.3%) in x.
- **Heatmap** (`heatmap_<config>`): mechanism x duty -> pooled miss rate per workload on a log colour
  scale. Pale green = zero misses, with the 95% upper bound shown (`<ub`); `*` marks D=0 cells taken from
  the idle-adversary run; `n/a` = not run.
- **Pareto** (`pareto_<config>`): x = adversary throughput relative to its solo run (units/s when no
  solo run exists), y = miss rate with 95% CI bars (log). Open down-triangles are zero-miss points drawn
  at their upper bound. Down and to the right is better: the mechanism that gets misses to zero at the
  least throughput cost.
- **Time series** (`timeseries_<config>_worst`): latency of the worst run (highest miss rate) against
  time, as max and median per bin of consecutive slots, so no spike is hidden by downsampling; ticks
  along the top mark skipped boundaries. Needs `slots.bin`.
- **Tables**: `summary.csv` (one row per run, every statistic), `cells.csv` (reps pooled), and both
  tables in `report.md` with the invalid runs, warnings, NOT SUPPORTED items from run.json and
  repeatability.

## Lenient inputs

DESIGN.md does not fix meta.json/run.json key names, so the readers search for them: counters
`recorded`, `skipped_boundaries`/`skipped`, `stamp_mismatches`, `ring_overflows` (preferring a `counts`
object); `slots` and `spin_us` (preferring `config`); clock fits = any object holding `a`, `g_ref` and
`t_ref`, with "pre"/"post" in the key path; lock settings `lock_clocks`/`clocks_locked` and `gc_mhz`;
anything under a key containing `not_supported`, and any string containing `NOT SUPPORTED`. Telemetry
is nvidia-smi CSV with units in the header; both `clocks_event_reasons` and `clocks_throttle_reasons`
column names are accepted. A counter the validator needs but cannot find makes the check fail loudly
rather than pass.
