"""Figures for the slotbench report (matplotlib Agg; PNG at 150 dpi plus PDF).

Every mechanism keeps one colour, marker and line style in every figure (colour is never the only cue).
Palette: a colour-blind-checked categorical order (worst adjacent CVD dE 9.1); the heatmap uses a
single-hue blue ramp on a log scale with a separate pale-green fill for exactly zero misses.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from . import stats  # noqa: E402

MECH_NAMES = {"M0": "none", "M1": "stream priority", "M2": "MPS", "M3": "time-slice", "M4": "streams (no graph)",
              "M5": "boost clocks", "M6": "MIG"}
MECH_COLORS = {"M0": "#2a78d6", "M1": "#eb6834", "M2": "#1baf7a", "M3": "#eda100", "M4": "#e87ba4",
               "M5": "#008300", "M6": "#4a3aa7"}
MECH_MARKERS = {"M0": "o", "M1": "s", "M2": "^", "M3": "D", "M4": "v", "M5": "P", "M6": "X"}
MECH_LINES = {"M0": "-", "M1": "--", "M2": "-.", "M3": ":", "M4": (0, (5, 1)), "M5": (0, (3, 1, 1, 1, 1, 1)),
              "M6": (0, (1, 1))}
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#d8d7d0"
ZERO_COLOR = "#cfeccf"
SEQ_CMAP = LinearSegmentedColormap.from_list("sb_blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 10, "legend.fontsize": 8, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})


def mech_label(m):
    return f"{m} {MECH_NAMES.get(m, '')}".strip()


def mstyle(m):
    return {"color": MECH_COLORS.get(m, INK2), "marker": MECH_MARKERS.get(m, "o"),
            "linestyle": MECH_LINES.get(m, "-")}


def save(fig, out_base: str) -> list[str]:
    """Write out_base.png (150 dpi) and out_base.pdf; returns the paths."""
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    paths = [out_base + ".png", out_base + ".pdf"]
    fig.savefig(paths[0], dpi=150, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _plain_log_x(ax):
    """Log x axis with plain-number labels (200, 500, 1000) instead of 2x10^2."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _grid(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 1. CCDF

def ccdf_series(item):
    """(x_us, y, n_total, exact) for one pooled cell. item has either 'latency_us' (raw samples) with
    'skipped', or 'hist' (a log histogram incl. its 'skipped')."""
    if item.get("latency_us") is not None:
        v, p, n = stats.ccdf_raw(item["latency_us"], item.get("skipped", 0))
        return v, p, n, True
    h = item["hist"]
    e, y, n = stats.ccdf_from_hist(h, h.get("skipped", 0))
    return e, y, n, False


def plot_ccdf(items: dict, deadline_us: float, title: str, out_base: str) -> list[str]:
    """items: mechanism -> pooled cell dict (see ccdf_series). One line per mechanism, log-log,
    y = P(latency > x) with skipped boundaries counted as +inf latency."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0), layout="constrained")
    xmax, ymin, any_hist = deadline_us * 1.2, 1.0, False
    series = {}
    for m in sorted(items):
        x, y, n, exact = ccdf_series(items[m])
        any_hist |= not exact
        keep = y > 0
        if not keep.any():
            continue
        last = int(np.nonzero(keep)[0][-1])
        # the last non-zero level holds until the next value/edge, where it drops
        x_end = float(x[last + 1]) if last + 1 < x.size else float(x[last])
        series[m] = (x, y, n, exact, keep, x_end)
        xmax = max(xmax, x_end * 1.3)
        ymin = min(ymin, 1.0 / n)
    for m, (x, y, n, exact, keep, x_end) in series.items():
        xs, ys = x[keep], y[keep]
        tail = y[-1] if y.size else 0.0  # level contributed by skipped slots (+inf) / overflow
        if tail > 0:
            xs, ys = np.append(xs, xmax), np.append(ys, tail)
        elif x_end > xs[-1]:
            xs, ys = np.append(xs, x_end), np.append(ys, ys[-1])
        st = mstyle(m)
        misses = items[m].get("misses")
        lab = f"{mech_label(m)} (n={n:,}" + (f", {misses} miss" if misses is not None else "") + \
              ("" if exact else ", hist") + ")"
        ax.step(xs, ys, where="post", color=st["color"], linestyle=st["linestyle"], linewidth=1.6, label=lab)
        ax.plot(xs[-1:], ys[-1:], marker=st["marker"], color=st["color"], markersize=5, linestyle="none")
    ax.axvline(deadline_us, color=INK, linewidth=1.0, linestyle=":")
    ax.text(deadline_us, 1.0, f" deadline {deadline_us:g} us", color=INK, va="top", ha="left", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log_x(ax)
    ax.set_ylim(max(ymin / 3, 1e-9), 1.5)
    ax.set_xlabel("latency t1 - t0 (us)")
    ax.set_ylabel("P(latency > x)")
    ax.set_title(title + ("\n(hist: conservative upper-edge CCDF from 100 bins/decade)" if any_hist else ""))
    _grid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    return save(fig, out_base)


# ---------------------------------------------------------------- 2. heatmap

def _fmt(r):
    return f"{r:.1e}" if r < 0.01 else f"{r:.3f}"


def plot_heatmap(cells: dict, mechanisms: list, duties: list, workloads: list, title: str,
                 out_base: str) -> list[str]:
    """cells[(workload, mechanism, duty)] = {'misses', 'total', 'ci_hi', 'idle': bool}. One panel per
    workload; log colour scale over non-zero rates; exactly 0 misses in pale green annotated with the
    95% upper bound ("<ub"); missing cells blank ("n/a")."""
    rates = [c["misses"] / c["total"] for c in cells.values() if c["total"] and c["misses"]]
    lo = min(rates) if rates else 1e-6
    hi = max(rates) if rates else 1e-2
    if hi <= lo:
        hi = lo * 10
    norm = LogNorm(vmin=lo, vmax=hi)
    nw = max(1, len(workloads))
    fig, axes = plt.subplots(1, nw, figsize=(1.6 + 1.0 * len(duties) * nw + 1.2, 0.55 * len(mechanisms) + 2.0),
                             squeeze=False, layout="constrained")
    for k, (ax, w) in enumerate(zip(axes[0], workloads)):
        for i, m in enumerate(mechanisms):
            for j, d in enumerate(duties):
                c = cells.get((w, m, d))
                if c is None or not c["total"]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="white", edgecolor=GRID))
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color=INK2)
                    continue
                r = c["misses"] / c["total"]
                if c["misses"] == 0:
                    fc, txt, tc = ZERO_COLOR, f"0\n<{c['ci_hi']:.1e}", INK
                else:
                    fc = SEQ_CMAP(norm(r))
                    txt = _fmt(r)
                    tc = "white" if norm(r) > 0.55 else INK
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=fc, edgecolor="white", linewidth=2))
                ax.text(j, i, txt + ("*" if c.get("idle") else ""), ha="center", va="center", fontsize=7, color=tc)
        ax.set_xlim(-.5, len(duties) - .5)
        ax.set_ylim(len(mechanisms) - .5, -.5)
        ax.set_xticks(range(len(duties)), [f"{d}%" for d in duties])
        ax.set_yticks(range(len(mechanisms)), [mech_label(m) for m in mechanisms] if k == 0 else [])
        ax.set_xlabel("adversary duty")
        ax.set_title(w)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=SEQ_CMAP)
    cb = fig.colorbar(sm, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
    cb.set_label("miss rate (log)")
    fig.suptitle(title + "\npale green = 0 misses, value is the 95% upper bound; * = idle-adversary D=0 run",
                 fontsize=9)
    return save(fig, out_base)


# ---------------------------------------------------------------- 3. Pareto

def plot_pareto(points: list, workloads: list, relative: bool, title: str, out_base: str) -> list[str]:
    """points: dicts with workload, mechanism, duty, x (throughput), misses, total, ci_lo, ci_hi.
    y = miss rate with 95% CI bars; zero-miss points drawn at their upper bound as open down-triangles."""
    nw = max(1, len(workloads))
    fig, axes = plt.subplots(1, nw, figsize=(4.6 * nw, 4.4), squeeze=False, layout="constrained")
    lows = [p["ci_hi"] for p in points if p["total"]] + [p["ci_lo"] for p in points if p["total"] and p["ci_lo"] > 0]
    floor = min(lows) if lows else 1e-6
    for ax, w in zip(axes[0], workloads):
        seen = set()
        for p in sorted((p for p in points if p["workload"] == w and p["x"] is not None),
                        key=lambda p: (p["mechanism"], p["duty"])):
            st = mstyle(p["mechanism"])
            lab = mech_label(p["mechanism"]) if p["mechanism"] not in seen else None
            seen.add(p["mechanism"])
            if p["misses"] == 0:
                ax.plot([p["x"]], [p["ci_hi"]], marker="v", markerfacecolor="white", markeredgecolor=st["color"],
                        markersize=8, linestyle="none", label=lab)
                y = p["ci_hi"]
            else:
                y = p["misses"] / p["total"]
                ax.errorbar([p["x"]], [y], yerr=[[y - p["ci_lo"]], [p["ci_hi"] - y]], color=st["color"],
                            marker=st["marker"], markersize=7, linestyle="none", capsize=2, label=lab)
            ax.annotate(f"{p['mechanism']} d{p['duty']}", (p["x"], y), textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=INK2)
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor / 3)
        ax.set_xlabel("adversary throughput relative to solo" if relative else "adversary units/s")
        ax.set_ylabel("slot miss rate (95% CI)")
        ax.set_title(w)
        _grid(ax)
        h, l = ax.get_legend_handles_labels()
        if h:
            hl = sorted(zip(l, h))
            ax.legend([x[1] for x in hl], [x[0] for x in hl], loc="best")
    fig.suptitle(title + "\nopen down-triangle = 0 misses, drawn at the 95% upper bound", fontsize=9)
    return save(fig, out_base)


# ---------------------------------------------------------------- 4. time series

def plot_timeseries(slot, latency_us, period_us: float, deadline_us: float, title: str, out_base: str,
                    n_bins: int = 3000) -> list[str]:
    """Latency against run time for one run; max per bin (never hides a spike) and median per bin;
    skipped boundaries marked along the top."""
    slot = np.asarray(slot, dtype=np.int64)
    lat = np.asarray(latency_us, dtype=np.float64)
    t = (slot - slot[0]) * period_us / 1e6 if slot.size else slot
    idx, mx, med = stats.downsample_max(lat, n_bins)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    ax.plot(t[idx], mx, color=MECH_COLORS["M0"], linewidth=0.9, label=f"max per bin ({max(1, lat.size // n_bins)} slots)")
    ax.plot(t[idx], med, color=INK2, linewidth=0.9, label="median per bin")
    ax.axhline(deadline_us, color=INK, linestyle=":", linewidth=1.0, label=f"deadline {deadline_us:g} us")
    gaps = np.nonzero(np.diff(slot) > 1)[0]
    if gaps.size:
        top = max(float(mx.max()), deadline_us) * 1.6
        ax.plot(t[gaps], np.full(gaps.size, top), "|", color=MECH_COLORS["M1"], markersize=8,
                label=f"skipped boundaries ({int(np.sum(np.diff(slot)[gaps] - 1))})")
    ax.set_yscale("log")
    ax.set_xlabel("time since first recorded slot (s)")
    ax.set_ylabel("latency t1 - t0 (us)")
    ax.set_title(title)
    _grid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    return save(fig, out_base)
