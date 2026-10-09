"""Builds FINDINGS.md for the research repo from the 8-9 Oct 2026 tickbound data (tables generated from the data
files, narrative written here). Usage: python make_findings.py REPO_DIR STRICT_JSON"""
import glob, json, os, statistics, sys

REPO, STRICT = sys.argv[1], sys.argv[2]
D = os.path.join(REPO, "tickbound_data")
S = json.load(open(STRICT, encoding="utf-8"))
SUM = json.load(open(os.path.join(D, "summary_2026-10-09.json"), encoding="utf-8"))


def p50(x):
    return x.get("p50") if isinstance(x, dict) else x


def fmt_us(x, nd=2):
    v = p50(x)
    return f"{v / 1e3:.{nd}f}" if isinstance(v, (int, float)) else "-"


# ---------------------------------------------------------------- strict bounds
def strict_rows():
    groups = {}
    for rel, r in S.items():
        if not r or "chord_ns" not in r:
            continue
        box, kind = rel.split("/")[0], rel.split("/")[1]
        cat = {"ktrace": "kernel timing (ktrace)", "ktrace126": "kernel timing (ktrace)", "sweep": "kernel timing (ktrace)",
               "multi": "8-GPU kernel timing", "instr": "instruction / co-tenant / MPS", "reorder126": "reorder E1-E6",
               "reorder122": "reorder E1-E6"}.get(kind, kind)
        groups.setdefault((box, cat), []).append(r)
    out = ["| data | runs | reported (chord) ± ns | strict ± ns | strict / chord |", "|---|---|---|---|---|"]
    for (box, cat), rs in sorted(groups.items()):
        c = [r["chord_ns"] for r in rs]; s = [r["strict_ns"] for r in rs]; q = [r["ratio"] for r in rs]
        out.append(f"| {box} {cat} | {len(rs)} | {min(c):.0f}-{max(c):.0f} | {min(s):.0f}-{max(s):.0f} | {min(q):.2f}-{max(q):.2f} |")
    return "\n".join(out)


# ---------------------------------------------------------------- kernel shapes
def shape_rows():
    out = ["| GPU (toolkit) | shape | FFMA | launches | launch -> first warp µs | block-start spread µs | kernel span µs | cycles / FFMA | last exit -> sync return µs | clock-fit widening ns |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for box, label in (("2026-10-08_a100_cuda12.6", "A100-SXM4 (12.6)"), ("2026-10-08_h100_cuda12.6", "H100 (12.6)")):
        for name in ("ffma64", "ffma256", "ffma1024", "ffma4096"):
            a = json.load(open(os.path.join(D, box, "sweep", name + ".analysis.json"), encoding="utf-8"))
            r, sms = a["result"], a["sms"]
            for B, e in sorted(r["by_blocks"].items(), key=lambda kv: int(kv[0])):
                B = int(B); shape = "1 block" if B == 1 else f"{B // sms} per SM ({B})"
                ks = e.get("kernels"); n = len(ks) if isinstance(ks, list) else ks
                out.append(f"| {label} | {shape} | {r['ffma']} | {n} | {fmt_us(e.get('launch_to_first_entry_ns'))} | "
                           f"{fmt_us(e.get('block_start_spread_ns'))} | {fmt_us(e.get('kernel_span_ns'))} | "
                           f"{p50(e.get('compute_cycles_per_ffma')):.2f} | {fmt_us(e.get('last_exit_to_sync_return_ns'))} | "
                           f"{p50(e.get('rate_inconsistency_ns')) or 0:.1f} |")
    # 9 Oct: the 8-GPU run, median over the 8 GPUs of each GPU's p50
    per = {}
    for g in range(8):
        a = json.load(open(os.path.join(D, "2026-10-09_a100x8_pcie_cuda12.6+12.2", "multi", f"gpu{g}.analysis.json"), encoding="utf-8"))
        for B, e in a["result"]["by_blocks"].items():
            per.setdefault(int(B), []).append(e)
    for B, es in sorted(per.items()):
        def med(key, scale=1e3):
            v = [p50(e.get(key)) for e in es if isinstance(p50(e.get(key)), (int, float))]
            return f"{statistics.median(v) / scale:.2f}" if v else "-"
        ks = es[0].get("kernels"); n = len(ks) if isinstance(ks, list) else ks
        out.append(f"| 8x A100-PCIe (12.6), median of 8 GPUs | {'1 block' if B == 1 else '1 per SM (108)'} | 256 | {n} per GPU | "
                   f"{med('launch_to_first_entry_ns')} | {med('block_start_spread_ns')} | {med('kernel_span_ns')} | "
                   f"{med('compute_cycles_per_ffma', 1)} | {med('last_exit_to_sync_return_ns')} | "
                   f"{max(p50(e.get('rate_inconsistency_ns')) or 0 for e in es):.1f} (max) |")
    return "\n".join(out)


# ---------------------------------------------------------------- Tier B
def tierb_rows():
    s = SUM["tierB"]["edits"]
    out = ["| edit | group | checker | output vs reference (200 launches) | cycles p50 |", "|---|---|---|---|---|"]
    for e in s:
        out.append(f"| `{e['id']}` | {e['group']} | {'legal' if e['legal_per_checker'] else 'illegal'} | "
                   f"{'correct' if e.get('match') else 'WRONG'} ({e.get('mismatching_launches')}/{e.get('launches_compared')} wrong) | {e.get('cycles_p50')} |")
    agree = sum(1 for e in s if bool(e["legal_per_checker"]) == bool(e.get("match")))
    false_legal = sum(1 for e in s if e["legal_per_checker"] and not e.get("match"))
    return "\n".join(out), agree, len(s), false_legal


# ---------------------------------------------------------------- census
def census_rows():
    import collections
    out = ["| GPU, host | runs | %globaltimer step sizes seen at sync edges (count) |", "|---|---|---|"]
    for box in sorted(glob.glob(os.path.join(D, "2026-10-0*"))):
        c = collections.Counter(); n = 0
        for f in glob.glob(os.path.join(box, "**", "*.json"), recursive=True):
            b = os.path.basename(f)
            if b.endswith((".analysis.json", ".trace.json", ".orig.json")) or "sass" in b or "phases" in b or b == "summary.json":
                continue
            try:
                m = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            st = m.get("timer_edge_steps_ns") if isinstance(m, dict) else None
            if st:
                n += 1; c.update(st)
        tot = sum(c.values())
        out.append(f"| {os.path.basename(box)} | {n} | " + ", ".join(f"{k} ns x{v} ({100 * v / tot:.1f}%)" for k, v in sorted(c.items())) + " |")
    return "\n".join(out)


strict_tbl = strict_rows()
shape_tbl = shape_rows()
tierb_tbl, agree, n_edits, false_legal = tierb_rows()
census_tbl = census_rows()
multi = json.load(open(os.path.join(REPO, "figures_2026-10-09", "multi_gpu_all_instants.json"), encoding="utf-8"))
tv = 0
wid = 0.0
for g in range(8):
    a = json.load(open(os.path.join(D, "2026-10-09_a100x8_pcie_cuda12.6+12.2", "multi", f"gpu{g}.analysis.json"), encoding="utf-8"))
    for k in a["result"]["kernels"]:
        tv += int((k.get("ticket") or {}).get("violations") or 0)
        if k["blocks"] == 108:
            wid = max(wid, float(k["sm_fit"].get("rate_inconsistency_ns") or 0))

md = f"""# tickbound findings, 8-9 October 2026 (A100, H100, 8x A100 on one host)

tickbound measures GPU work from inside the kernel: each warp reads the SM cycle counter (`clock64`) and the GPU timer
(`%globaltimer`) at checkpoints, and each run syncs the GPU timer to the host clock (`CLOCK_MONOTONIC_RAW`) before
and after the work, so every checkpoint lands on the host time axis **with a stated error bound**. The SASS of the
binary that measured is checked so that the checkpoints sit where the source says. The tool itself lives in a
separate repository (not yet in git); this repository holds the data, figures, run scripts and this write-up.

Everything below was measured on rented vast.ai machines (8 Oct: $1.07, 9 Oct: $1.59).

| date | hardware | toolkit | what ran | data |
|---|---|---|---|---|
| 8 Oct | 1x A100-SXM4-40GB | CUDA 12.6 | ktrace profile, FFMA x blocks sweep, instr / co-tenant / MPS | `tickbound_data/2026-10-08_a100_cuda12.6` |
| 8 Oct | 1x H100 80GB HBM3 | CUDA 12.2 and 12.6 | same | `tickbound_data/2026-10-08_h100_cuda12.2`, `..._h100_cuda12.6` |
| 9 Oct | 8x A100-PCIE-40GB, one host | CUDA 12.6 and 12.2 | 8-GPU ktrace on one timeline, reorder E1-E7, 30 hand-edited SASS kernels | `tickbound_data/2026-10-09_a100x8_pcie_cuda12.6+12.2` |
| 9 Oct | 1x H100 80GB HBM3 (another host) | CUDA 12.6 and 12.2 | reorder E1-E7, ktrace profile (timer census) | `tickbound_data/2026-10-09_h100_cuda12.6+12.2` |

Raw traces (`*.bin`), timeline viewer pages (`*.trace.json`, `timeline.html`) and uncompressed SASS listings are kept
locally and not committed (size); every analysis result, report, log and gzipped SASS listing is here.

## 1. Eight GPUs on one timeline, instruction-phase level

![8 GPUs on one host clock](figures_2026-10-09/multi_gpu_instruction_timeline.png)

Eight A100s in one host, one process per GPU, each with its own clock sync before and after. All processes started
together and launched the same ktrace kernel (108 blocks x 256 threads, a 256-FFMA chain between checkpoints) on a
shared 2 ms grid of the host clock. The figure shows one grid instant: every block's warp 0, phase by phase (two
dependent loads, barrier, FFMA chain, barrier, store + GPU fence, atomic ticket, flag store), all eight GPUs on the
same host axis, each with its strict placement bound.

Over all {multi['n']} grid instants at which all eight GPUs launched:

- the eight host processes made their launch calls within **30 ns** of each other (median; max 41 ns);
- the kernels started 7.6-8.3 µs after the launch call (per-GPU median), and the kernel starts of the eight GPUs
  spread **0.78 µs** (median; range 0.51-2.70 µs);
- each GPU's placement bound on the host axis is **±0.85-1.20 µs** (strict clock bound 0.79-1.03 µs plus the
  checkpoint's own placement, ≤ 0.17 µs);
- so the start order of two GPUs is proved only when they are more than **1.7-2.4 µs** apart: that held for **11 of
  1120** GPU pairs (in 2 of the {multi['n']} instants, where one GPU lagged);
- ticket-order checks inside the kernels: {tv} violations; clock-fit widening for the 108-block launches: at most {wid:.1f} ns.

What this shows: per-GPU instruction-phase timelines of several GPUs can be put on one host clock with a stated,
checked error on every point; cross-GPU ordering through the host clock is limited to gaps above ~2 µs.

Figure data: `figures_2026-10-09/multi_gpu_instruction_timeline.json`, all instants: `multi_gpu_all_instants.json`.
Method note: the start barrier and the shared launch grid were a run-time patch to the probe for this run
(`tickbound_runs/2026-10-09/patches/gputrace_start_barrier_launch_grid.diff`).

## 2. How exact is each timestamp: strict bounds

The clock sync gives a set of feasible (rate, offset) lines between GPU timer and host clock. The bound reported by
the fit so far (the "chord") was the half-width of the feasible offsets at the fitted rate. The **strict** bound is
the half-extent of the projection of the whole feasible set at the event's time, taken here as its maximum over the
gap between the two sync windows (where the measured work happens). The strict bound is the one to quote.

{strict_tbl}

All runs: `tickbound_data/strict_bounds_2026-10-09.json`. For kernel-timing runs the strict bound is 1.00-1.35x the
reported one (A100 ±0.79-1.03 µs, H100 ±0.55-0.73 µs). Runs with a long or lopsided gap between the sync windows
(the 8 Oct instr / MPS runs, the 9 Oct A100 load-use runs) widen up to 2.6x (worst ±1.40 µs).

Inside one GPU, between SMs, a checkpoint is placed far more tightly: ±20-26 ns on H100 (64 ns timer steps) and
±80-330 ns on A100 (1024 ns timer steps), per block, from the SM's own cycle counter.

## 3. Kernel shapes

Same kernel, three grid shapes (1 block, 1 block per SM, "4 per SM") x four FFMA chain lengths, plus the 9 Oct 8-GPU
run (1 block vs 1 per SM, 40 launches each per GPU). Block size was 256 threads in every run (never varied), and no
memory-heavy kernel shape was tried.

{shape_tbl}

- Launch to first warp does not depend on the shape: A100 5.1-6.8 µs, H100 4.6-5.5 µs. After 50 ms idle it rises to
  9.8 µs (A100) and 24-32 µs (H100). On the 9 Oct PCIe host (8 processes launching together) it was 7.6-8.3 µs.
- A full wave starts within 0.4-0.6 µs on A100 (108 blocks) and 0.2-0.3 µs on H100 (132 blocks).
- "4 blocks per SM" never had four blocks resident: from the trace, at most **3 blocks were resident per SM** at any
  time, on every SM of both GPUs; the fourth ran as a tail wave, which is why those spans are ~4x the 1-per-SM spans.
- A dependent FFMA costs 4.0 cycles alone; with 3 resident blocks the A100 slows to 5.8-6.2 cycles (its FP32 pipe
  saturates), the H100 stays at 4.0-4.5.
- Last warp exit to the host's sync return: 6.3-8.0 µs (A100), 4.6-5.0 µs (H100), for every shape (9 Oct PCIe host:
  8.2-9.3 µs).
- Kernels under ~10 µs need no clock-fit widening (the SMs' cycle-counter lines agree exactly); 28-64 µs kernels
  need 45-64 ns, so their within-GPU placement is empirical rather than hard.

## 4. Instruction-order experiments (E1-E6) and two compilers (E7)

Each experiment times variants that hold the same instructions in a different order; a variant counts only where
the reorder SASS gate shows the intended order in the binary that measured. Same source, two toolkits (CUDA 12.6
and 12.2) on A100 (sm_80) and H100 (sm_90).

| | A100 12.6 | A100 12.2 | H100 12.6 | H100 12.2 |
|---|---|---|---|---|
| gate (295 builds) | 286 pass, 9 fail | 278 pass, 17 fail | 295 pass | 287 pass, 8 fail |

Gate failures are compiler choices, and the analysis drops those builds:
- **CUDA 12.2 emits `BAR.SYNC`, 12.6 emits `BAR.SYNC.DEFER_BLOCKING`** for the same `__syncthreads()` (616 of 616
  barriers in the binary, on sm_80 and sm_90, both disassemblers agree): the barrier experiment is not measured for 12.2.
- On sm_80 (both toolkits), with two loads and 96 or more FFMA, ptxas moves the loaded value into a uniform register
  (`R2UR`) in the middle of the chain, so the intended overlap does not exist: 9 builds dropped.
- On sm_90, one PTX fence is `MEMBAR.ALL.CTA` directly followed by the scoped `MEMBAR` (then `ERRBAR`, `CGAERRBAR`,
  `CCTL.IVALL`). The gate first refused this; it now accepts the adjacent pair as one fence
  (`tickbound_runs/2026-10-09/patches/sass_reorder_sm90_fence_pair.diff`, with a regression test), and the H100 data
  was re-analysed with it.

Measured (cycles unless stated; CUDA 12.6; full one-line results per run in `tickbound_data/summary_2026-10-09.json`):

- **E1 load to use:** latency L1 / L2 / DRAM: A100 50 / 353 / 614, H100 49 / 370 / 771. Independent FFMA hide a load
  up to the predicted break-even N* = (L - d) / 4 (A100 L2: 73 measured vs 81 predicted; H100 L2: 84.8 vs 84.7);
  the largest saving from hoisting a DRAM load: 595 (A100), 634 (H100).
- **E2 write-after-read:** reusing a just-stored register instead of a fresh one costs 0-10 cycles in most cases;
  STG.128 with 32 warps: +8 (A100), +46 (H100); one A100 case (STG.32, 32 warps) measured -28, not yet explained.
- **E3 barrier:** moving work that touches shared memory to after the barrier costs about its full length (+296 at
  M = 64, +1066 at M = 256, against 4M = 256 / 1024), the same on A100 and H100.
- **E4 fence cost:** GPU scope / system scope / acq_rel: A100 410 / 2277 / 413, H100 682 / 1252 / 684. Giving the
  preceding store up to 1024 FFMA to drain before the fence saves ≤ 8 cycles: the fence's cost does not shrink.
- **E5 FFMA throughput:** cycles per FFMA per warp follow max(4 / ILP, p x warps per sub-partition) with p = 2 on
  A100 and p = 1 on H100 (A100 reaches 15.8 at 32 warps, H100 8.0).
- **E6 dependent chains:** measured cycles per dependent link: 4 for the integer / FP32 ALU ops tested (as ptxas
  encodes them), 8 for IABS, 6 (A100) / 8 (H100) for DADD and DFMA, and 13 (A100) / 12 (H100) for IMAD.WIDE, where
  ptxas encodes 10 / 8: the hardware adds 3-4 cycles beyond the encoding. Independent FP32 ops issue every ~2 cycles
  on A100 and every cycle on H100; DADD / DFMA every 4.0 (A100) and 2.2 (H100) cycles.
- **E7 (12.2 vs 12.6):** apart from the barrier form above, the two toolkits measure the same within a few cycles.
  One open point: the A100 acq_rel fence measured 413 (12.6) vs 565 (12.2) with identical fence SASS, so it is not a
  compiler effect; it needs a repeat.

## 5. Hand-edited SASS: does the legality checker match the hardware?

Thirty edits to real sm_80 SASS (CuAssembler), each run 200 times on an A100 against the unedited kernel's output:
legal moves of a load ("slide", k slots), deliberate violations (moved past its consumer, cleared scoreboard wait,
stall counts below the latency), and probes of rules the checker applies conservatively.

{tierb_tbl}

- The checker agreed with the hardware on **{agree} of {n_edits}** edits and called **{false_legal}** broken edit legal.
- The four disagreements are all conservative refusals that ran correctly: a 4-cycle ALU-to-store distance (ptxas
  never uses less than 5; 4 was enough here, twice), a swap of two `LDGSTS` with disjoint targets, and an FFMA moved
  across a barrier with no reader before it.
- Some broken edits fail rarely: a cleared scoreboard wait gave a wrong result in 1 of 200 launches, stall counts
  below the IADD3 latency in 8 of 200 (FFMA: 200 of 200). A short test would call them safe.
- Legal slides of a load: cycles stay flat while the load hides under the FFMA chain, then rise by 4 cycles per slot.

## 6. H100 `%globaltimer` steps are not uniform

The GPU timer does not always advance in 64 ns steps on H100: at the sync edges it stepped by 64, 96 and occasionally
128 ns, on a second, different H100 host as well (A100: always 1024 ns).

{census_tbl}

Any error model for GPU-side timestamps on H100 (including a software clock-correlation fallback) should take the
96 / 128 ns steps into account.

## 7. Relation to TempoTrace

TempoTrace (Elbakoury and Sharma, arXiv:2609.23301, an unreviewed technical white paper) co-designs PTP time
synchronisation with distributed tracing so that events across a GPU cluster are ordered correctly, and builds a
diagnosis layer on top: spans, a causal graph with its critical path, and a rule + XGBoost engine that names the root
cause (compute saturation, KV-cache misses, scheduler preemption, network stalls, ...). GPU events are timestamped at
the network card (GPUDirect RDMA doorbell, PTP-disciplined clock, TAI timescale). By the paper's own provenance
notes, its GPU-to-host figures (0.056 µs residual sigma, 0.218 µs bound) are design targets with the hardware
experiments pending, the large-cluster tables are illustrative, and the diagnosis accuracy comes from a synthetic
corpus. Its own ablation finds that hardware GPU timestamping (instead of software clock correlation) improves
diagnosis only for operations shorter than 20 µs; without PTP (NTP only) diagnosis degrades at every scale.

Where this work sits:
- it is a different layer: no spans or root-cause engine, but measured GPU timing on real A100 / H100 hardware, with
  a strict bound on every timestamp, at instruction-phase level inside kernels and across 8 GPUs, on commodity
  hardware (no special NIC, no PTP);
- the strict per-GPU bound (±0.55-1.03 µs on the host clock) is below TempoTrace's 2.1 µs software baseline and
  2.5-5x wider than its 0.218 µs hardware target; it is one host's clock (`CLOCK_MONOTONIC_RAW`), not TAI;
- two results bear directly on its design: the H100 timer's non-uniform steps (section 6), and the measured limit
  of ordering GPUs through the host clock (~2 µs, section 1), in the sub-20 µs regime where its ablation finds
  hardware timestamps matter.

## 8. Limits and corrections

- One host per run; host clock, not TAI; no external reference clock: the checks are internal (ticket order inside
  the kernel, flag write before the host sees it, feasibility of the clock fit).
- 8 Oct samples are small (5 launches per shape); block size fixed at 256 threads; no memory-heavy kernel shapes.
- The 8 Oct write-up (`tickbound findings and rules 2026-10-08 (A100 + H100) v2.docx`) and figures quote the
  reported (chord) host-clock bounds; the strict values are in section 2 (1.00-1.13x for those kernel-timing runs).
  Its description of TempoTrace should follow section 7 (GPU timestamps taken at the NIC, TAI timescale, the 0.218 µs
  bound is a statistical design target).
- Open: the A100 acq_rel fence difference (section 4), a multi-node or PTP-referenced run.

## Files

- `FINDINGS.md`: this write-up.
- `figures_2026-10-08/`, `figures_2026-10-09/`: figures (PNG + SVG) with their data.
- `tickbound_data/<date>_<gpu>_<toolkit>/`: per run `.json` (run metadata), `.analysis.json`, `.log`, `report.md`,
  SASS gate results, gzipped SASS listings; `summary_2026-10-09.json`, `strict_bounds_2026-10-09.json`.
- `tickbound_runs/2026-10-09/`: the scripts that ran on the rented machines and locally (`onbox_*.sh`, `babysit.sh`,
  `backstop.sh`, `multi_fig.py`, `analyze_9oct.py`, `strict_all.py`) and the two patches.
"""
open(os.path.join(REPO, "FINDINGS.md"), "w", encoding="utf-8", newline="\n").write(md)
print("wrote FINDINGS.md", len(md), "chars")
