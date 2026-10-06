#!/usr/bin/env python3
"""Builds and executes notebooks/kernel_timeline.ipynb. Run from slotbench/: python3 notebooks/build_kernel_timeline_notebook.py
Datasets: data/2026-10-06_a100_ktrace (A100 SXM4) and data/2026-10-06_3060_ktrace (RTX 3060); SASS phase listings
from tools/sass_phases.py are read from the datasets (ktrace_phases.json) when present."""
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip()))
code = lambda s: cells.append(new_code_cell(s.strip()))

md(r"""
# The flow of execution inside a kernel, on one time axis

Cycle-accurate benchmarking of CUDA kernels (Core Velocity Lab's nanobenchmarking post, and the instruction
table in this repository) brackets code between two `CS2R SR_CLOCKLO` reads and reports cycles. It stops at the
SM boundary: "counters have different starting values for different SMs, so we cannot measure inter-block
execution time". This notebook goes one step further with the same two hardware timers every SM has:

- `clock64` (`CS2R SR_CLOCKLO`): one counter per SM, exact to the cycle for every warp and block on that SM;
- `%globaltimer` (`CS2R SR_GLOBALTIMERLO`): the GPU-wide ns timer, quantised to its tick.

gputrace's `ktrace` strategy runs a staged kernel in which **every warp** stamps both timers at eleven
checkpoints (entry, a calibration stamp, two dependent global loads landed, barrier arrival, barrier release, a
256-FFMA dependent chain done, a second barrier, a global store plus `__threadfence`, a grid-wide atomic ticket,
exit). The stamps stay in registers and are written at the end of the warp, so a checkpoint costs one read of each
timer and nothing else. Per SM and per launch, every stamp says that the true time of cycle *c* lies in
*[g, g + tick)*; the set of these windows bounds the SM's cycle-to-ns line exactly as the tick-edge method bounds
the host-GPU mapping, with a half-width far below the tick because consecutive checkpoints are tens of cycles
apart. Within an SM nothing is fitted: warps and blocks compare in cycles. Across SMs the fitted lines place
blocks on the GPU axis with a stated bound, and the run's own clock bound carries them to the host axis, where the
launch call, the mapped flag the last block writes, and the stream synchronisation sit.

Three checks have to hold if the placement is right, and they are evaluated on every launch: a warp's barrier
release never precedes the last arrival in its block (cycles); the grid-wide ticket order equals the time order of
the ticket stamps across SMs (fitted lines); the host sees the flag after the last block's exit (host axis).
Every number below is computed from the raw records with numpy; the figures are matplotlib.
""")

code(r"""
import os, sys, json, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
sys.path.insert(0, ROOT)
from analysis.gputrace import Run
from analysis.ktrace import kernel_timeline, KT_NAMES, KT_PHASES, kt_stamps

C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300', violet='#4a3aa7',
         gray='#9a9a94', ink='#15161a', grid='#e6e5e0', ink2='#55575f')
PHASE_COL = {'load': C['blue'], 'barrier1': C['orange'], 'compute': C['aqua'], 'barrier2': C['orange'], 'store_fence': C['violet'],
             'ticket': C['magenta'], 'tail': C['gray']}
mpl.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 160, 'font.size': 10, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': C['ink2'], 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'xtick.color': C['ink2'], 'ytick.color': C['ink2'], 'axes.labelcolor': C['ink'], 'axes.titlesize': 11,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False, 'lines.linewidth': 2})
DATA = {'A100 SXM4': os.path.join(ROOT, 'data', '2026-10-06_a100_ktrace'), 'RTX 3060': os.path.join(ROOT, 'data', '2026-10-06_3060_ktrace')}
runs = {g: Run(os.path.join(d, 'ktrace')) for g, d in DATA.items() if os.path.exists(os.path.join(d, 'ktrace.json'))}
for g, r in runs.items():
    print(f"{g}: {r.meta['sms']} SMs, tick {r.clock['tick_ns']} ns, host bound ±{r.clock['bound_ns']:.0f} ns, ffma {r.meta['ffma']}, {r.recs.size:,} stamps")
""")

md(r"""
## 1. One launch, every block, on the host axis

One block per SM (B = number of SMs), 256 threads, warp 0 of each block drawn as a bar from its entry to its exit
and coloured by phase; the other seven warps of the block are the thin lines behind it. Blocks are ordered by
their start. The host events of the same launch are the vertical lines.
""")

code(r"""
def launches(run, B):
    enter = run.events('LAUNCH_ENTER')
    return [int(k) for k, a in zip(enter['kernel_id'], enter['a']) if int(a) == B]

def fig_launch(run, gname, kid, ax):
    T = kernel_timeline(run, kid); h = T['host']
    rows = T['rows']; t0 = h['launch_enter']
    us = lambda t: (t - t0) / 1e3
    blocks = {}
    for r in rows: blocks.setdefault(r['block'], []).append(r)
    order = sorted(blocks, key=lambda b: min(r['t_host'][0] for r in blocks[b]))
    for y, b in enumerate(order):
        for r in blocks[b]:
            th = r['t_host']
            if r['warp'] == 0:
                for name, i, j, _ in KT_PHASES:
                    ax.barh(y, us(th[j]) - us(th[i]), left=us(th[i]), height=0.8, color=PHASE_COL[name], lw=0)
            else:
                ax.plot([us(th[0]), us(th[10])], [y, y], color=C['gray'], lw=0.5, alpha=.5, zorder=0)
    for key, lab in (('launch_enter', 'launch call'), ('launch_return', 'call returns'), ('flag_seen', 'host sees flag'), ('sync_return', 'sync returns')):
        if h.get(key): ax.axvline(us(h[key]), color=C['ink2'], lw=0.8, ls=':'); ax.text(us(h[key]), len(order) + 1.5, lab, rotation=90, fontsize=8, va='bottom', ha='center', color=C['ink2'])
    ax.set_ylim(-1, len(order) + 9); ax.set_xlabel('µs after the launch call (host clock)'); ax.set_ylabel('block (ordered by start)')
    ax.set_title(f"{gname}: launch {kid}, {len(order)} blocks on {T['n_sms']} SMs; host bound ±{T['host_view']['bound_ns']:.0f} ns, SM lines ±{T['sm_fit']['bound_ns_max']:.0f} ns max")
    ax.legend(handles=[Patch(color=PHASE_COL[n], label=n) for n in ('load', 'barrier1', 'compute', 'store_fence', 'ticket', 'tail')], loc='lower right', ncol=3, fontsize=8)
    return T

Ts = {}
for gname, run in runs.items():
    B = run.meta['sms']; kid = launches(run, B)[-1]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    Ts[gname] = fig_launch(run, gname, kid, ax)
    plt.show()
    hv = Ts[gname]['host_view']
    print(f"{gname}: launch -> first warp {hv['launch_to_first_entry_ns'] / 1e3:.2f} µs; block starts spread {Ts[gname]['block_start_spread_ns'] / 1e3:.2f} µs; "
          f"kernel span {Ts[gname]['span_gpu_ns'] / 1e3:.2f} µs; last exit -> flag seen {hv['last_exit_to_flag_seen_ns'] / 1e3:.2f} µs -> sync returns {hv['last_exit_to_sync_return_ns'] / 1e3:.2f} µs")
""")

md(r"""
## 2. Inside one block: eight warps, two barriers

Cycles since the block's first stamp, exact (one SM, one counter). Each warp's row shows its phases; the barrier
segments are split into the time the warp *waited* for the last warp to arrive (lighter) and the release
latency after the last arrival (darker). The checkpoint's own cost (checkpoint 0 → 1) has been subtracted from
every phase.
""")

code(r"""
def fig_block(T, gname, b, ax):
    rs = sorted([r for r in T['rows'] if r['block'] == b], key=lambda r: r['warp'])
    c0 = min(r['c'][0] for r in rs); ghz = None
    last1 = max(r['c'][3] for r in rs); last2 = max(r['c'][6] for r in rs)
    for r in rs:
        y = r['warp']; c = r['c'] - c0; cal = r['cal']
        segs = [('entry→loaded', c[1], c[2] - cal, PHASE_COL['load']),
                ('wait', c[3], last1 - c0, '#f6c3ad'), ('release', last1 - c0, c[4] - cal, PHASE_COL['barrier1']),
                ('compute', c[4], c[5] - cal, PHASE_COL['compute']),
                ('wait', c[6], last2 - c0, '#f6c3ad'), ('release', last2 - c0, c[7] - cal, PHASE_COL['barrier2']),
                ('store+fence', c[7], c[8] - cal, PHASE_COL['store_fence']), ('ticket', c[8], c[9] - cal, PHASE_COL['ticket']), ('tail', c[9], c[10] - cal, PHASE_COL['tail'])]
        for name, a, bb, col in segs:
            if bb > a: ax.barh(y, bb - a, left=a, height=0.7, color=col, lw=0)
        for k in range(11): ax.plot([c[k], c[k]], [y - 0.42, y + 0.42], color=C['ink'], lw=0.6)
    ax.set_yticks(range(len(rs))); ax.set_yticklabels([f'warp {r["warp"]}' for r in rs]); ax.invert_yaxis()
    ax.set_xlabel(f'cycles since the block\'s first stamp (SM {rs[0]["sm"]} counter, exact)')
    ax.set_title(f"{gname}: block {b} on SM {rs[0]['sm']}; ticks are the eleven checkpoints")
    ax.legend(handles=[Patch(color=PHASE_COL['load'], label='two dependent loads'), Patch(color='#f6c3ad', label='waiting at the barrier for the last warp'),
                       Patch(color=PHASE_COL['barrier1'], label='barrier release after the last arrival'), Patch(color=PHASE_COL['compute'], label='256 dependent FFMA'),
                       Patch(color=PHASE_COL['store_fence'], label='store + __threadfence'), Patch(color=PHASE_COL['ticket'], label='atomic ticket (warp 0)'), Patch(color=PHASE_COL['tail'], label='third barrier, flag, exit')],
              loc='lower right', ncol=2, fontsize=8)

for gname, T in Ts.items():
    b = max({r['block'] for r in T['rows']} , key=lambda b: max(x['arrival_spread'] for x in T['barriers'] if x['block'] == b))
    fig, ax = plt.subplots(figsize=(11, 3.6)); fig_block(T, gname, b, ax); plt.show()
""")

md(r"""
## 3. Phase costs across all launches: one block per SM against four

Cycles per phase over every warp of every launch, for B = SMs (each block alone on its SM) and B = 4 × SMs
(four blocks, 32 warps, share each SM). The barrier is split into wait and release; the compute chain is the
same 256 dependent FFMA either way, so what moves is contention for the SM.
""")

code(r"""
from analysis.ktrace import a_ktrace
res = {g: a_ktrace(run) for g, run in runs.items()}
order = ['load', 'barrier wait', 'barrier release', 'compute', 'store_fence', 'ticket']
fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 3.8), squeeze=False)
for ax, (gname, R) in zip(axes[0], res.items()):
    Bs = sorted(R['by_blocks'])
    w = 0.38
    for k, B in enumerate(Bs[:2]):
        d = R['by_blocks'][B]
        vals = [d['phases_cycles']['load'], d['barrier_wait_cycles'], d['barrier_latency_cycles'], d['phases_cycles']['compute'], d['phases_cycles']['store_fence'], d['phases_cycles']['ticket']]
        x = np.arange(len(order)) + (k - 0.5) * w
        med = np.array([v['p50'] for v in vals]); p90 = np.array([v['p90'] for v in vals])
        ax.bar(x, med, width=w - 0.04, color=[C['blue'], C['orange']][k], label=f'B = {B} ({B // runs[gname].meta["sms"]} block/SM)', lw=0)
        ax.vlines(x, med, p90, color=C['ink2'], lw=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha='right'); ax.set_yscale('log'); ax.set_ylabel('cycles (median, line to p90)')
    ax.set_title(f'{gname}: phase cost per warp'); ax.legend(fontsize=8)
plt.show()
for gname, R in res.items():
    for B, d in sorted(R['by_blocks'].items()):
        ph = d['phases_cycles']
        print(f"{gname} B={B}: checkpoint cost {d['checkpoint_cost_cycles']['p50']:.0f} cy | load {ph['load']['p50']:.0f} | barrier wait {d['barrier_wait_cycles']['p50']:.0f} (p99 {d['barrier_wait_cycles']['p99']:.0f}) "
              f"release {d['barrier_latency_cycles']['p50']:.0f} | compute {ph['compute']['p50']:.0f} | store+fence {ph['store_fence']['p50']:.0f} | ticket {ph['ticket']['p50']:.0f} cy; "
              f"SM clock {d['sm_ghz']['p50']:.3f} GHz; span {d['kernel_span_ns']['p50'] / 1e3:.1f} µs")
""")

md(r"""
## 4. Who waits for whom at the first barrier

Per block (rows) and warp (columns): cycles the warp waited at the first barrier for the last warp of its block
to arrive, one launch with one block per SM. A column that is consistently dark is a warp that is consistently
late; a flat row is a block whose warps arrived together.
""")

code(r"""
fig, axes = plt.subplots(1, len(Ts), figsize=(5.6 * len(Ts), 4.2), squeeze=False)
for ax, (gname, T) in zip(axes[0], Ts.items()):
    bars = [b for b in T['barriers'] if b['barrier'] == 'barrier1']
    M = np.array([b['wait_per_warp'] for b in bars])
    im = ax.imshow(M, aspect='auto', cmap=mpl.colors.LinearSegmentedColormap.from_list('b', ['#f7f7f5', C['blue']]), interpolation='nearest')
    ax.grid(False); ax.set_xlabel('warp'); ax.set_ylabel('block'); ax.set_xticks(range(M.shape[1]))
    ax.set_title(f'{gname}: wait at barrier 1 (cycles), {M.shape[0]} blocks')
    plt.colorbar(im, ax=ax, shrink=0.8, label='cycles')
    early = sum(b['release_before_last_arrival'] for b in T['barriers'])
    print(f"{gname}: warps released before the last arrival: {early} (must be 0); arrival spread p50 {np.median([b['arrival_spread'] for b in bars]):.0f} cy, max {max(b['arrival_spread'] for b in bars):.0f} cy")
plt.show()
""")

md(r"""
## 5. How precise is the placement, and does it pass its checks

Per SM and launch, the half-width of the feasible cycle-to-ns line (the bound on any cross-SM time), against the
timer tick; the SM clocks the lines imply; and the grid-wide ticket check: the ticket stamp of ticket *k* must not
precede ticket *k−1*'s by more than the two SMs' half-widths.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), gridspec_kw=dict(wspace=0.35))
for gname, run in runs.items():
    col = C['blue'] if gname.startswith('A100') else C['orange']
    bounds, ghz, slack = [], [], []
    for kid in sorted(set(kt_stamps(run)['kid'].tolist())):
        T = kernel_timeline(run, kid)
        bounds += [r['bound_ns'] for r in T['rows'] if r['warp'] == 0]
        ghz += [T['sm_fit']['ghz_p50']]
        tk = sorted([(r['ticket'], r['t_gpu'][9]) for r in T['rows'] if r['warp'] == 0 and r['ticket'] is not None])
        slack += [t1 - t0 for (_, t0), (_, t1) in zip(tk, tk[1:])]
        if T['ticket']['violations']: print(f'{gname} launch {kid}: {T["ticket"]["violations"]} ticket-order violations')
    tick = run.clock['tick_ns']
    axes[0].hist(bounds, bins=40, color=col, alpha=.75, label=f'{gname} (tick {tick} ns)', lw=0)
    axes[1].hist(ghz, bins=20, color=col, alpha=.75, label=gname, lw=0)
    axes[2].hist(np.clip(slack, -200, 2000), bins=60, color=col, alpha=.75, label=gname, lw=0)
    print(f"{gname}: SM-line half-width p50 {np.median(bounds):.0f} ns, max {max(bounds):.0f} ns (tick {tick} ns); SM clock {np.median(ghz):.3f} GHz; "
          f"ticket slack min {min(slack):.0f} ns over {len(slack)} consecutive tickets")
axes[0].set_xlabel('half-width of the SM cycle→ns line (ns)'); axes[0].set_ylabel('blocks'); axes[0].set_title('Cross-SM placement bound'); axes[0].legend(fontsize=8)
axes[1].set_xlabel('SM clock from the fitted line (GHz)'); axes[1].set_title('Implied SM clock per launch'); axes[1].legend(fontsize=8)
axes[2].set_xlabel('ticket k stamp − ticket k−1 stamp (ns, clipped)'); axes[2].set_title('Ticket order against time order (must be ≥ −bounds)'); axes[2].axvline(0, color=C['ink2'], lw=0.8); axes[2].legend(fontsize=8)
plt.show()
""")

md(r"""
## 6. What each phase is, in SASS

The instructions between consecutive checkpoints of the compiled kernel (sm_80 build; the sm_86 build differs
only in the FFMA loop body). The checkpoint reads themselves are excluded. The phase named *tail* holds the third
`__syncthreads` and the last block's flag write.
""")

code(r"""
for gname, d in DATA.items():
    p = os.path.join(d, 'ktrace_phases.json')
    if not os.path.exists(p): continue
    ph = json.load(open(p))['phases']
    print(f'== {gname}')
    for name, x in ph.items():
        ops = ', '.join(f'{k}×{v}' if v > 1 else k for k, v in sorted(x['opcodes'].items(), key=lambda kv: -kv[1]))
        print(f'  {name:28s} {x["n"]:3d} instructions: {ops}')
""")

md(r"""
## What this establishes

Every warp of a running kernel can be placed on one time axis from entry to exit, with its loads, barriers,
compute, store, fence and atomic as measured segments: exact cycles within an SM, and across SMs a bound from the
timer tick that is tens of ns rather than the tick itself, carried to the host axis by the run's clock bound. The
barrier check (release after the last arrival, every block), the ticket check (atomic order equals time order
across SMs) and the flag check (host after GPU) hold on every launch, so the placement is tested rather than
assumed. The next step on this axis is a second GPU (the per-GPU half is already in `gpus`/`nccl`), then a second
host.
""")

nb = new_notebook(cells=cells, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')))
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
ep = ExecutePreprocessor(timeout=900, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': root}})
out = os.path.join(here, 'kernel_timeline.ipynb')
nbformat.write(nb, out)
print('wrote', out)
