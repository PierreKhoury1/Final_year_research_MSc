# Instruction table page

`prep.py RUNS_DIR` reads the `instr`, `instr_cotenant` and `instr_mps50` runs of two `gputrace characterize
--profile instr` outputs (edit `GPUS` at the top), computes per-(kind, working set, N) quantiles and log-binned
histograms of the raw bracket samples (shared samples only for the same-SM co-tenant, as the analysis does), the
per-bracket traces inside each kernel (N = 1 and N = max) with the co-tenant coverage of the probe's SM, and
writes `data.json`. The SASS between the clock reads is added by the snippet in the session notes
(`cuobjdump -sass` of the sm_80 / sm_86 builds, the bracket pair picked as `tools/sass_check.py` does).
`template.html` with `/*DATA*/` replaced by `data.json` is the page: dot plot of latency per instruction under
the three conditions, the fit for the selected instruction with p10–p90 ranges, the histogram at the longest
chain, every bracket over kernel time, the bracket decomposed, the SASS, and the full table.
