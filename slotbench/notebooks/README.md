# Notebooks

- `tempotrace_problem.ipynb`: the problem TempoTrace (arXiv 2609.23301) sets out to fix (causal inversions in
  multi-GPU traces when the clocks are only NTP-accurate), reproduced and resolved on one server with gputrace's
  8× A100 NCCL data: raw per-GPU timers, a one-off offset, and the bounded per-GPU mapping, then what the bound lets
  you attribute (ring order at 8 B, socket split at 16 MB), then the independent two-process check (H100). numpy
  for every number, matplotlib for every figure. Built and executed by `build_tempotrace_notebook.py` (run from
  `slotbench/`), so the outputs are reproducible from the datasets in `data/`.
- `kernel_timeline.ipynb`: the flow of execution inside a kernel on one time axis (A100 and RTX 3060 `ktrace`
  datasets): every block of a launch on the host axis with the host-side sync events, the phase-occupancy timeline
  (warps loading / waiting / computing / storing at each instant), one block's eight warps and two barriers in
  exact cycles, phase costs at one and four blocks per SM, who waits for whom at the barrier, the placement bounds
  and checks, and the SASS of each phase. Built by `build_kernel_timeline_notebook.py`.
