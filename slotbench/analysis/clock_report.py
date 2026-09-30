"""Clock chapter report for a clockcal run (DESIGN.md section 6).

Reads RUN_DIR/clock_windows.csv (+ meta.json) written by bin/clockcal and optionally a phc.csv from
bin/phc_offset, and computes: drift of the GPU->host offset (least-squares slope, in ppm), the
window-to-window rate series (mean/std), peak-to-peak wander (raw and after removing the linear
drift), overlapping Allan deviation of the offset series, and fit residual / eps statistics.
Writes clock_summary.json and PNG figures into --out (default RUN_DIR).

    python3 -m analysis.clock_report RUN_DIR [--out DIR] [--phc FILE]

Sign convention: offset = host - GPU (resp. PHC - host for phc.csv, see phc_summary); a positive
drift in ppm means the host clock gains on the GPU clock. The maths lives in pure functions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

WINDOW_COLUMNS = ["t_host_ns", "n", "n_kept", "offset_ns", "rate_ppm", "residual_rms_ns", "residual_max_ns",
                  "min_width_ns", "median_width_ns", "eps_ns", "gpu_temp_c", "sm_clock_mhz"]
PHC_COLUMNS = ["host_monoraw_ns", "host_realtime_ns", "phc_ns", "method", "width_ns"]


# ---------------------------------------------------------------- readers

def read_windows(path: str) -> pd.DataFrame:
    """clock_windows.csv -> DataFrame ('#' lines are comments; empty cells become NaN)."""
    df = pd.read_csv(path, comment="#", dtype={"t_host_ns": "int64"})
    missing = [c for c in WINDOW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def read_meta(run_dir: str) -> dict:
    p = os.path.join(run_dir, "meta.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def read_phc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"host_monoraw_ns": "int64", "host_realtime_ns": "int64", "phc_ns": "int64"})
    missing = [c for c in PHC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


# ---------------------------------------------------------------- maths (pure)

def to_grid(t_s: np.ndarray, x: np.ndarray, tau0: float) -> np.ndarray:
    """Place samples on a uniform grid of step tau0 (index = round((t - t[0]) / tau0)); missing
    grid points are NaN. Duplicate indices keep the first sample."""
    t_s = np.asarray(t_s, float)
    x = np.asarray(x, float)
    if len(t_s) == 0:
        return np.array([])
    idx = np.rint((t_s - t_s[0]) / tau0).astype(np.int64)
    grid = np.full(int(idx.max()) + 1, np.nan)
    for i, v in zip(idx[::-1], x[::-1]):  # reversed so the first sample wins
        grid[i] = v
    return grid


def overlapping_adev(x, tau0: float, m_list=None):
    """Overlapping Allan deviation from phase (time-error) data x sampled every tau0.

    sigma^2(m*tau0) = sum_i (x[i+2m] - 2 x[i+m] + x[i])^2 / (2 (m tau0)^2 K), summed over the K index
    triples whose three samples are all present (NaN = missing grid point). With x in seconds and tau0 in
    seconds the result is a fractional frequency. Returns (taus, adev, counts) for m with K >= 1.
    """
    x = np.asarray(x, float)
    n = len(x)
    if m_list is None:
        m_list = []
        m = 1
        while 2 * m < n:
            m_list.append(m)
            m *= 2
    taus, devs, counts = [], [], []
    for m in m_list:
        m = int(m)
        if m < 1 or 2 * m >= n:
            continue
        d = x[2 * m:] - 2 * x[m:n - m] + x[:n - 2 * m]
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        tau = m * tau0
        taus.append(tau)
        devs.append(math.sqrt(float(np.sum(d * d)) / (2.0 * tau * tau * len(d))))
        counts.append(len(d))
    return np.array(taus), np.array(devs), np.array(counts, dtype=int)


def linear_drift(t_s, off_ns):
    """Least-squares off = slope * t + c. Returns (drift_ppm, intercept_ns, residual_ns array).
    slope is ns per s, so ppm = slope / 1e3."""
    t_s = np.asarray(t_s, float)
    off_ns = np.asarray(off_ns, float)
    if len(t_s) < 2 or np.ptp(t_s) == 0:
        return float("nan"), float("nan"), off_ns - np.mean(off_ns) if len(off_ns) else off_ns
    tm = t_s.mean()
    slope, c = np.polyfit(t_s - tm, off_ns, 1)
    resid = off_ns - (slope * (t_s - tm) + c)
    return float(slope / 1e3), float(c - slope * tm), resid


def pairwise_rate_ppm(t_s, off_ns) -> np.ndarray:
    """Rate between consecutive samples in ppm: d(off_ns)/d(t_s) / 1e3."""
    t_s = np.asarray(t_s, float)
    off_ns = np.asarray(off_ns, float)
    dt = np.diff(t_s)
    ok = dt > 0
    return np.diff(off_ns)[ok] / dt[ok] / 1e3


def stats(v) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "median": None, "p99": None, "max": None}
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max())}


def _tau0(t_s: np.ndarray, meta_interval) -> float:
    if meta_interval:
        return float(meta_interval)
    d = np.diff(t_s)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else float("nan")


def _adev_block(t_s, off_ns, tau0) -> list:
    if len(t_s) < 3 or not np.isfinite(tau0) or tau0 <= 0:
        return []
    grid = to_grid(t_s, np.asarray(off_ns, float) * 1e-9, tau0)
    taus, devs, counts = overlapping_adev(grid, tau0)
    return [{"tau_s": float(t), "adev": float(a), "adev_ppb": float(a * 1e9), "n_terms": int(k)}
            for t, a, k in zip(taus, devs, counts)]


def window_summary(df: pd.DataFrame, meta: dict | None = None) -> dict:
    """Stability statistics of a clockcal run (windows without a fit are counted, then dropped)."""
    meta = meta or {}
    ok = df[np.isfinite(df["offset_ns"].astype(float))].reset_index(drop=True)
    out = {"n_windows": int(len(df)), "n_windows_fitted": int(len(ok))}
    if len(ok) == 0:
        return out
    t_ns = ok["t_host_ns"].to_numpy(np.int64)
    t_s = (t_ns - t_ns[0]).astype(float) / 1e9
    off = ok["offset_ns"].to_numpy(float)
    drift, c, resid = linear_drift(t_s, off)
    pr = pairwise_rate_ppm(t_s, off)
    tau0 = _tau0(t_s, (meta.get("config") or {}).get("interval_s"))
    out.update({
        "span_s": float(t_s[-1]),
        "tau0_s": tau0,
        "drift_ppm": drift,                       # slope of offset vs time
        "drift_intercept_ns": c,
        "rate_between_windows_ppm": stats(pr),    # from consecutive offsets
        "rate_fit_ppm": stats(ok["rate_ppm"]),    # per-window fits (short windows: noisy)
        "wander_p2p_ns": float(np.ptp(off)),
        "wander_detrended_p2p_ns": float(np.ptp(resid)),
        "wander_detrended_rms_ns": float(np.sqrt(np.mean(resid ** 2))),
        "residual_rms_ns": stats(ok["residual_rms_ns"]),
        "residual_max_ns": stats(ok["residual_max_ns"]),
        "eps_ns": stats(ok["eps_ns"]),
        "min_width_ns": stats(ok["min_width_ns"]),
        "median_width_ns": stats(ok["median_width_ns"]),
        "adev": _adev_block(t_s, off, tau0) if len(ok) >= 3 else [],
    })
    temp = ok["gpu_temp_c"].astype(float).to_numpy()
    if np.isfinite(temp).any():
        out["gpu_temp_c"] = stats(temp)
        # correlate the between-window rate with the mean temperature of each pair
        tm = 0.5 * (temp[1:] + temp[:-1])
        good = np.isfinite(tm) & (np.diff(t_s) > 0)
        if good.sum() >= 3 and np.std(tm[good]) > 0 and np.std(pr) > 0 and len(pr) == good.size:
            out["rate_temp_corr"] = float(np.corrcoef(pr[good], tm[good])[0, 1])
            out["rate_temp_slope_ppm_per_c"] = float(np.polyfit(tm[good], pr[good], 1)[0])
    smc = ok["sm_clock_mhz"].astype(float).to_numpy()
    if np.isfinite(smc).any():
        out["sm_clock_mhz"] = stats(smc)
    return out


def phc_offsets(df: pd.DataFrame):
    """(t_s, offset_ns) with offset = phc - host_monoraw, computed in int64 before going to float
    (the PHC is near 1.7e18 ns, far beyond double's integer precision)."""
    raw = df["host_monoraw_ns"].to_numpy(np.int64)
    off = df["phc_ns"].to_numpy(np.int64) - raw
    return (raw - raw[0]).astype(float) / 1e9, (off - off[0]).astype(float)


def phc_summary(df: pd.DataFrame) -> dict:
    """PHC vs CLOCK_MONOTONIC_RAW: drift_ppm > 0 means the PHC gains on the host."""
    out = {"n": int(len(df)), "methods": {str(k): int(v) for k, v in df["method"].value_counts().items()}}
    if len(df) < 2:
        return out
    t_s, off = phc_offsets(df)
    drift, c, resid = linear_drift(t_s, off)
    steps = np.abs(np.diff(off) - drift * 1e3 * np.diff(t_s))
    out.update({"span_s": float(t_s[-1]), "drift_ppm": drift, "wander_p2p_ns": float(np.ptp(off)),
                "wander_detrended_p2p_ns": float(np.ptp(resid)),
                "max_step_ns": float(steps.max()) if len(steps) else 0.0,
                "width_ns": stats(df["width_ns"]),
                "rate_between_samples_ppm": stats(pairwise_rate_ppm(t_s, off)),
                "adev": _adev_block(t_s, off, _tau0(t_s, None))})
    return out


# ---------------------------------------------------------------- plots

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_plots(df: pd.DataFrame, summary: dict, out_dir: str, phc: pd.DataFrame | None = None) -> list:
    plt = _plt()
    files = []
    ok = df[np.isfinite(df["offset_ns"].astype(float))].reset_index(drop=True)
    if len(ok) == 0:
        return files
    t_ns = ok["t_host_ns"].to_numpy(np.int64)
    th = (t_ns - t_ns[0]).astype(float) / 3600e9
    off = ok["offset_ns"].to_numpy(float)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(th, off / 1e3, ".-", ms=3)
    a1.set_ylabel("offset (us)")
    a1.set_title(f"GPU->host offset, drift {summary.get('drift_ppm', float('nan')):.4f} ppm")
    if len(ok) >= 2:
        _, _, resid = linear_drift(th * 3600, off)
        a2.plot(th, resid, ".-", ms=3)
    a2.set_ylabel("detrended offset (ns)")
    a2.set_xlabel("time (h)")
    fig.tight_layout()
    files.append(_save(fig, out_dir, "clock_offset.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(th, ok["rate_ppm"], ".", ms=3, alpha=0.5, label="window fit")
    if len(ok) >= 2:
        tm = 0.5 * (th[1:] + th[:-1])
        ax.plot(tm, np.diff(off) / np.diff(th * 3600) / 1e3, "-", lw=1, label="between windows")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("rate (ppm)")
    temp = ok["gpu_temp_c"].astype(float).to_numpy()
    if np.isfinite(temp).any():
        tx = ax.twinx()
        tx.plot(th, temp, "r-", lw=1, label="GPU temp")
        tx.set_ylabel("GPU temperature (C)", color="r")
    ax.legend(loc="upper left")
    fig.tight_layout()
    files.append(_save(fig, out_dir, "clock_rate.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    for col in ("residual_rms_ns", "residual_max_ns", "eps_ns", "min_width_ns"):
        ax.plot(th, ok[col], ".-", ms=3, lw=0.8, label=col)
    ax.set_yscale("log")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("ns")
    ax.legend()
    fig.tight_layout()
    files.append(_save(fig, out_dir, "clock_residual.png"))

    if summary.get("adev"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog([a["tau_s"] for a in summary["adev"]], [a["adev"] for a in summary["adev"]], "o-", label="GPU")
        if phc is not None and summary.get("phc", {}).get("adev"):
            p = summary["phc"]["adev"]
            ax.loglog([a["tau_s"] for a in p], [a["adev"] for a in p], "s-", label="PHC")
        ax.set_xlabel("tau (s)")
        ax.set_ylabel("overlapping Allan deviation")
        ax.legend()
        fig.tight_layout()
        files.append(_save(fig, out_dir, "clock_adev.png"))

    if phc is not None and len(phc) >= 2:
        t_s, poff = phc_offsets(phc)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_s / 3600, poff / 1e3, ".-", ms=2)
        ax.set_xlabel("time (h)")
        ax.set_ylabel("PHC - monoraw offset change (us)")
        fig.tight_layout()
        files.append(_save(fig, out_dir, "clock_phc.png"))
    return files


def _save(fig, out_dir, name):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=120)
    _plt().close(fig)
    return p


# ---------------------------------------------------------------- CLI

def _clean(o):
    """NaN/inf -> None so the JSON is strict."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def report(run_dir: str, out_dir: str | None = None, phc_path: str | None = None) -> dict:
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    meta = read_meta(run_dir)
    df = read_windows(os.path.join(run_dir, "clock_windows.csv"))
    summary = {"run_dir": os.path.abspath(run_dir), "method": meta.get("method"),
               "gpu": (meta.get("gpu") or {}).get("name"), "global_fit": meta.get("global_fit"),
               "globaltimer_step_ns": meta.get("globaltimer_step_ns")}
    summary.update(window_summary(df, meta))
    if phc_path is None and os.path.exists(os.path.join(run_dir, "phc.csv")):
        phc_path = os.path.join(run_dir, "phc.csv")
    phc = None
    if phc_path:
        phc = read_phc(phc_path)
        summary["phc"] = phc_summary(phc)
        summary["phc"]["path"] = os.path.abspath(phc_path)
    summary["figures"] = [os.path.basename(p) for p in make_plots(df, summary, out_dir, phc)]
    summary = _clean(summary)
    with open(os.path.join(out_dir, "clock_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="output directory (default RUN_DIR)")
    ap.add_argument("--phc", default=None, help="phc.csv from bin/phc_offset (default RUN_DIR/phc.csv if present)")
    a = ap.parse_args(argv)
    s = report(a.run_dir, a.out, a.phc)
    print(f"windows {s.get('n_windows_fitted')}/{s.get('n_windows')}  drift {s.get('drift_ppm')} ppm  "
          f"detrended p2p {s.get('wander_detrended_p2p_ns')} ns  -> {a.out or a.run_dir}/clock_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
