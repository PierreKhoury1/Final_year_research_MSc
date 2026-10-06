# Notebooks

- `tempotrace_problem.ipynb`: the problem TempoTrace (arXiv 2609.23301) sets out to fix (causal inversions in
  multi-GPU traces when the clocks are only NTP-accurate), reproduced and resolved on one server with gputrace's
  8× A100 NCCL data: raw per-GPU timers, a one-off offset, and the bounded per-GPU mapping, then what the bound lets
  you attribute (ring order at 8 B, socket split at 16 MB), then the independent two-process check (H100). numpy
  for every number, matplotlib for every figure. Built and executed by `build_tempotrace_notebook.py` (run from
  `slotbench/`), so the outputs are reproducible from the datasets in `data/`.
