"""Pure statistics for slotbench (numpy/scipy only, no file IO).

Conventions (see analysis/README.md):
- Quantile method "higher": q-quantile of n sorted samples is x[ceil(q*(n-1))], computed with exact
  rational arithmetic so float rounding cannot move the index. It is always an observed sample and is
  never below the linearly interpolated value, so a tail quantile is never understated. Identical to
  numpy.quantile(x, q, method="higher").
- Misses: a recorded slot misses if latency > deadline (strict). Every skipped boundary (gap in slot
  indices) is also a miss: total = recorded + skipped, misses = late recorded + skipped.
- Miss-rate interval: exact two-sided Clopper-Pearson 95% (0 misses -> [0, 1 - (alpha/2)^(1/n)]).
- CCDFs treat skipped boundaries as +inf latency, so CCDF(deadline) equals the miss rate.
- queue_delay maps g0 to host time through the straight line joining the pre- and post-run clock fits'
  anchors (see queue_delay_ns for why this beats blending the two fitted lines).
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.stats import beta

QUANTILES = (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("p99_9", 0.999), ("p99_99", 0.9999))
QUANTILE_METHOD = "higher"

# Log histogram: edges 1 us .. 1 s, 100 bins per decade (601 edges, 600 bins), in microseconds.
HIST_LO_US, HIST_HI_US, HIST_BINS_PER_DECADE = 1.0, 1e6, 100
HIST_EDGES_US = 10.0 ** (np.arange(0, 6 * HIST_BINS_PER_DECADE + 1) / HIST_BINS_PER_DECADE)


# ---------------------------------------------------------------- quantiles

def quantile_index(n: int, q: float) -> int:
    """Index into the sorted sample for method "higher": ceil(q*(n-1)), exact."""
    if n <= 0:
        raise ValueError("empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q outside [0, 1]")
    return min(n - 1, math.ceil(Fraction(str(q)) * (n - 1)))


def quantile_sorted(xs: np.ndarray, q: float):
    return xs[quantile_index(len(xs), q)]


def quantiles(x, sorted_=False) -> dict:
    """n, min, mean, p50..p99_99 ("higher" method) and max of x (NaN entries when x is empty)."""
    x = np.asarray(x)
    out = {"n": int(x.size)}
    if x.size == 0:
        out.update({k: None for k in ("min", "mean", "max")})
        out.update({k: None for k, _ in QUANTILES})
        return out
    xs = x if sorted_ else np.sort(x, kind="stable")
    out["min"] = float(xs[0])
    out["mean"] = float(np.mean(xs, dtype=np.float64))
    for k, q in QUANTILES:
        out[k] = float(quantile_sorted(xs, q))
    out["max"] = float(xs[-1])
    return out


def bootstrap_quantile_ci(x, q: float, n_boot: int = 200, conf: float = 0.95, seed: int = 0):
    """Percentile-bootstrap CI for the q-quantile ("higher" method). Slow for 1e6 samples; off by default
    in summaries. Note: for extreme q the bootstrap cannot see beyond the sample max."""
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    idx = quantile_index(x.size, q)
    est = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(x, size=x.size, replace=True)
        est[b] = np.partition(s, idx)[idx]
    a = (1 - conf) / 2
    return float(np.quantile(est, a)), float(np.quantile(est, 1 - a))


# ---------------------------------------------------------------- miss rate

def clopper_pearson(k: int, n: int, conf: float = 0.95):
    """Exact two-sided binomial interval for k successes in n trials."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    a = 1 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def slot_gaps(slots) -> dict:
    """Skipped boundaries between recorded slot indices. Only interior gaps count (boundaries before the
    first or after the last recorded slot are not attributable to the run). Also reports whether indices
    were strictly increasing; duplicates/out-of-order records indicate a driver bug."""
    s = np.asarray(slots, dtype=np.uint64)
    if s.size < 2:
        return {"skipped": 0, "monotonic": True, "duplicates": 0, "first": int(s[0]) if s.size else None,
                "last": int(s[-1]) if s.size else None}
    d = np.diff(s.astype(np.int64))
    monotonic = bool(np.all(d > 0))
    if monotonic:
        skipped = int(np.sum(d - 1))
        dup = 0
    else:
        u = np.unique(s)
        dup = int(s.size - u.size)
        skipped = int(u[-1] - u[0] + 1 - u.size)
    return {"skipped": skipped, "monotonic": monotonic, "duplicates": dup,
            "first": int(s.min()), "last": int(s.max())}


def miss_stats(latency_ns, deadline_ns: float, skipped: int, conf: float = 0.95) -> dict:
    lat = np.asarray(latency_ns)
    late = int(np.count_nonzero(lat > deadline_ns))
    total = int(lat.size) + int(skipped)
    misses = late + int(skipped)
    lo, hi = clopper_pearson(misses, total, conf)
    return {"recorded": int(lat.size), "skipped": int(skipped), "total_slots": total, "late": late,
            "misses": misses, "miss_rate": (misses / total) if total else None,
            "miss_rate_ci_lo": lo, "miss_rate_ci_hi": hi, "ci_conf": conf, "ci_method": "clopper-pearson"}


# ---------------------------------------------------------------- clock mapping

def fit_clock(t0, t1, g, keep_frac: float = 0.01, min_keep: int = 50) -> dict:
    """Python port of common/clock_fit.h fit_clock (same keep rule and outputs), used when meta.json has
    no fit but calib CSVs exist, and by synth."""
    t0 = np.asarray(t0, np.int64)
    t1 = np.asarray(t1, np.int64)
    g = np.asarray(g, np.uint64)
    f = {"ok": False, "a": 1.0, "b_ns": 0.0, "g_ref": 0, "t_ref": 0, "n": int(t0.size), "n_kept": 0}
    valid = np.nonzero((t1 >= t0) & (g != 0))[0]
    if valid.size < 2:
        return f
    w = (t1 - t0)[valid]
    ws = np.sort(w)
    keep = max(int(math.ceil(keep_frac * valid.size)), max(min_keep, 2))
    keep = min(keep, valid.size)
    thr = ws[keep - 1]
    kept = valid[w <= thr]
    if kept.size < 2:
        return f
    g_ref, t_ref = int(g[kept[0]]), int(t0[kept[0]])
    X = (g[kept] - np.uint64(g_ref)).astype(np.int64).astype(np.float64)
    Y = 0.5 * (t0[kept] - t_ref).astype(np.float64) + 0.5 * (t1[kept] - t_ref).astype(np.float64)
    mx, my = X.mean(), Y.mean()
    sxx = np.sum((X - mx) ** 2)
    a = float(np.sum((X - mx) * (Y - my)) / sxx) if sxx > 0 else 1.0
    b = float(my - a * mx)
    pred = b + a * X
    lo = (t0[kept] - t_ref).astype(np.float64)
    hi = (t1[kept] - t_ref).astype(np.float64)
    r = pred - Y
    viol = float(np.max(np.maximum(lo - pred, pred - hi)))
    kw = np.sort(hi - lo)
    kmw = float(kw[kw.size // 2])
    f.update({"ok": True, "a": a, "rate_ppm": (a - 1) * 1e6, "b_ns": b, "g_ref": g_ref, "t_ref": t_ref,
              "eps_ns": kmw / 2 + max(viol, 0.0), "resid_rms_ns": float(np.sqrt(np.mean(r * r))),
              "resid_max_ns": float(np.max(np.abs(r))), "min_width_ns": float(ws[0]),
              "median_width_ns": float(ws[ws.size // 2]), "kept_median_width_ns": kmw,
              "n_kept": int(kept.size)})
    return f


def _fit_ok(f) -> bool:
    return bool(f) and bool(f.get("ok", True)) and all(k in f for k in ("a", "g_ref", "t_ref")) \
        and ("b_ns" in f or "b" in f)


def host_minus(fit: dict, g, t) -> np.ndarray:
    """host_of(g) - t in ns (float64), computed relative to t_ref so 1e15-ns absolute times keep
    sub-ns precision: host_of(g) = t_ref + b + a*(g - g_ref)."""
    a = float(fit["a"])
    b = float(fit.get("b_ns", fit.get("b", 0.0)))
    x = (np.asarray(g, np.uint64) - np.uint64(int(fit["g_ref"]))).astype(np.int64).astype(np.float64)
    dt = (np.int64(int(fit["t_ref"])) - np.asarray(t, np.int64)).astype(np.float64)
    return dt + b + a * x


def anchor(fit: dict):
    """(g, host) point where a fit is most trustworthy: its GPU reading anchor_g (centre of the calibration
    window, when known) else g_ref (first kept bracket, inside the window). host is relative to t_ref."""
    g = int(fit.get("anchor_g") or fit["g_ref"])
    a = float(fit["a"])
    b = float(fit.get("b_ns", fit.get("b", 0.0)))
    x = float(np.int64(np.uint64(g) - np.uint64(int(fit["g_ref"]))))
    return g, int(fit["t_ref"]), b + a * x


def queue_delay_ns(g0, t0, fits: dict):
    """queue_delay = host_of(g0) - t0 for records with g0 != 0.

    Pre and post both ok ("two_point"): the mapping is the straight line through the pre-fit anchor and the
    post-fit anchor (see anchor()). This is linear interpolation between the two fits in time, done on
    their anchors rather than by blending the two lines: each fit's slope comes from a calibration window
    of well under a second, and a slope error of 1 ppm extrapolated over a 500 s run is 500 us, whereas
    the two anchors are separated by the whole run, so the joint slope is ~1000x better determined and the
    error stays at the anchors' own error (about eps). One ok fit: that fit alone ("pre"/"post"; beware
    slope extrapolation). None: queue_delay is not computed. Returns (values, mask of g0 != 0, method)."""
    g0 = np.asarray(g0, np.uint64)
    t0 = np.asarray(t0, np.int64)
    mask = g0 != 0
    pre = fits.get("pre") if _fit_ok(fits.get("pre")) else None
    post = fits.get("post") if _fit_ok(fits.get("post")) else None
    if pre is None and post is None:
        return None, mask, "none"
    gs, ts = g0[mask], t0[mask]
    if pre is not None and post is not None:
        ga, ta, ha = anchor(pre)
        gb, tb, hb = anchor(post)
        dg = float(np.int64(np.uint64(gb) - np.uint64(ga)))
        if dg > 0:
            # host(g) = (ta + ha) + (g - ga) * slope, evaluated relative to each record's t0
            slope = (float(tb - ta) + hb - ha) / dg
            x = (gs - np.uint64(ga)).astype(np.int64).astype(np.float64)
            v = (np.int64(ta) - ts).astype(np.float64) + ha + slope * x
            return v, mask, "two_point"
    f = pre if pre is not None else post
    return host_minus(f, gs, ts), mask, "pre" if f is pre else "post"


# ---------------------------------------------------------------- per-slot metrics

def derived_metrics(rec, spin_ns: float | None, fits: dict | None = None) -> dict:
    """DESIGN.md section 4 metrics, all in ns. gpu_exec / queue_delay only for records with non-zero
    stamps (their masks are returned too)."""
    t_sched = rec["t_sched"].astype(np.int64)
    t0 = rec["t0"].astype(np.int64)
    t1 = rec["t1"].astype(np.int64)
    out = {"latency": t1 - t0,
           "latency_from_boundary": t1 - t_sched,
           "start_lateness": t0 - t_sched,
           "launch_cost": rec["t_launched"].astype(np.int64) - t0}
    if spin_ns is not None:
        out["wake_overshoot"] = rec["t_wake"].astype(np.int64) - (t_sched - np.int64(round(spin_ns)))
    g0, g1 = rec["g0"], rec["g1"]
    gm = (g0 != 0) & (g1 != 0)
    out["gpu_exec"] = (g1[gm] - g0[gm]).astype(np.int64)
    out["gpu_exec_mask"] = gm
    if fits:
        qd, qm, how = queue_delay_ns(g0, t0, fits)
        if qd is not None:
            out["queue_delay"] = qd
            out["queue_delay_mask"] = qm
            out["queue_delay_method"] = how
    return out


# ---------------------------------------------------------------- histogram / CCDF

def log_histogram(x_us) -> dict:
    """Counts over HIST_EDGES_US: bin i = [e_i, e_{i+1}); underflow < 1 us; overflow >= 1 s."""
    x = np.asarray(x_us, dtype=np.float64)
    e = HIST_EDGES_US
    idx = np.searchsorted(e, x, side="right") - 1
    under = int(np.count_nonzero(idx < 0))
    over = int(np.count_nonzero(idx >= e.size - 1))
    inside = idx[(idx >= 0) & (idx < e.size - 1)]
    counts = np.bincount(inside, minlength=e.size - 1)
    return {"edges_us": e.tolist(), "counts": counts.astype(int).tolist(), "underflow": under,
            "overflow": over, "bins_per_decade": HIST_BINS_PER_DECADE, "bin_rule": "[lo, hi)"}


def ccdf_raw(x, n_inf: int = 0):
    """Exact empirical CCDF: for each distinct value v, P(X > v); n_inf extra samples at +inf
    (skipped boundaries). Returns (values, probabilities, n_total)."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = xs.size + int(n_inf)
    if xs.size == 0:
        return np.zeros(0), np.zeros(0), n
    v, first = np.unique(xs, return_index=True)
    last_excl = np.append(first[1:], xs.size)  # count of samples <= v
    p = (n - last_excl) / n
    return v, p, n


def ccdf_at(x, points, n_inf: int = 0):
    """Exact P(X > p) for each p in points (skipped boundaries as +inf)."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = xs.size + int(n_inf)
    return (n - np.searchsorted(xs, np.asarray(points, dtype=np.float64), side="right")) / n


def ccdf_from_hist(hist: dict, n_inf: int = 0):
    """Conservative CCDF from a log histogram: every sample is assumed to sit at its bin's upper edge.
    Returns (edges, y) with y_i = (#samples >= e_i + n_inf) / n_total, to be drawn as a step that holds
    y_i on [e_i, e_{i+1}). At an edge this is >= the true P(X > e_i) (equal unless samples lie exactly on
    the edge), and between edges it is an upper bound, so the tail is never understated."""
    counts = np.asarray(hist["counts"], dtype=np.int64)
    edges = np.asarray(hist["edges_us"], dtype=np.float64)
    over = int(hist.get("overflow", 0))
    under = int(hist.get("underflow", 0))
    n = int(counts.sum()) + over + under + int(n_inf)
    ge = np.concatenate([np.cumsum(counts[::-1])[::-1], [0]]) + over + int(n_inf)  # >= e_i for each edge
    return edges, ge / n if n else ge.astype(float), n


def merge_hists(hists):
    """Sum log histograms (same edges)."""
    hists = [h for h in hists if h]
    if not hists:
        return None
    out = dict(hists[0])
    out["counts"] = np.sum([np.asarray(h["counts"]) for h in hists], axis=0).astype(int).tolist()
    out["underflow"] = int(sum(h.get("underflow", 0) for h in hists))
    out["overflow"] = int(sum(h.get("overflow", 0) for h in hists))
    return out


def coefficient_of_variation(values):
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size < 2 or v.mean() == 0:
        return None
    return float(v.std(ddof=1) / v.mean())


def downsample_max(y, n_bins: int):
    """Split y into n_bins consecutive chunks; return (chunk start index, max, median) so the largest
    spike in each chunk is always kept."""
    y = np.asarray(y)
    if y.size <= n_bins:
        i = np.arange(y.size)
        return i, y.astype(np.float64), y.astype(np.float64)
    edges = np.linspace(0, y.size, n_bins + 1).astype(np.int64)
    mx = np.maximum.reduceat(y, edges[:-1]).astype(np.float64)
    med = np.array([np.median(y[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    return edges[:-1], mx, med
