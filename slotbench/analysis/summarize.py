"""Per-run summary: python3 -m analysis.summarize RUN_DIR [RUN_DIR...] writes RUN_DIR/summary.json.

The summary holds everything report.py and validate.py need, so a run can be analysed without its
slots.bin (cloud runs ship summaries only): counts, miss rate + Clopper-Pearson interval, quantiles of
every derived metric (DESIGN.md section 4), a log-binned latency histogram (enough to redraw the CCDF),
the 200 worst records, the adversary throughput and the validation checks. Output is deterministic
(sorted keys, no timestamps).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from . import sbio, stats

SUMMARY_VERSION = 1
N_WORST = 200


def _us(q: dict) -> dict:
    """Quantile dict in ns -> us (n unchanged)."""
    return {k: (v / 1e3 if (v is not None and k != "n") else v) for k, v in q.items()}


def _clean(o):
    """JSON-safe: numpy scalars -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def identity(info: sbio.RunInfo) -> dict:
    return {"name": info.name, "config": info.config, "kind": info.kind, "mechanism": info.mechanism,
            "workload": info.workload, "duty": info.duty, "rep": info.rep, "cell": info.cell}


def adversary_summary(info: sbio.RunInfo) -> dict | None:
    adv = sbio.load_json(info.file("adversary.json"))
    if not isinstance(adv, dict):
        return None
    keep = ("workload", "duty", "period_ms", "prio", "seconds_total", "seconds_active", "units",
            "units_per_s", "tflops", "gb_per_s", "fps", "gpu")
    return {k: adv.get(k) for k in keep if k in adv}


def solo_units_per_s(info: sbio.RunInfo, workload: str):
    """Mean units_per_s over SOLO_<workload>_r* siblings of this run (same config dir), or None."""
    parent = os.path.dirname(info.path)
    vals = []
    try:
        names = sorted(os.listdir(parent))
    except OSError:
        return None
    for n in names:
        p = sbio.parse_cell_name(n)
        if p and p[0] == "solo" and p[2] == workload:
            adv = sbio.load_json(os.path.join(parent, n, "adversary.json"))
            if isinstance(adv, dict) and isinstance(adv.get("units_per_s"), (int, float)):
                vals.append(float(adv["units_per_s"]))
    return float(np.mean(vals)) if vals else None


def load_fits(info: sbio.RunInfo, meta) -> tuple[dict, str]:
    """Clock fits from meta.json; if meta has none, refit from calib_pre/post.csv."""
    fits = {k: dict(v) for k, v in sbio.find_fits(meta).items()}
    src = "meta.json" if fits else "none"
    for k in ("pre", "post"):
        c = sbio.read_calib_csv(info.file(f"calib_{k}.csv"))
        if c is None or not c[0].size:
            continue
        if k not in fits:
            fits[k] = stats.fit_clock(*c)
            src = "calib_csv" if src == "none" else src + "+calib_csv"
        g = np.sort(c[2][c[2] != 0])
        if g.size:
            fits[k]["anchor_g"] = int(g[g.size // 2])  # centre of the calibration window
    return fits, src


def build_summary(run_dir: str, with_validation: bool = True) -> dict:
    """Summary dict for one run dir (raw slots.bin required for cells)."""
    info = sbio.run_info(run_dir)
    if info is None:
        raise ValueError(f"{run_dir}: not a run directory name (<M>_<W>_d<D>_r<rep> or SOLO_<W>_r<rep>)")
    s = {"summary_version": SUMMARY_VERSION, "identity": identity(info)}
    adv = adversary_summary(info)
    s["adversary"] = adv
    if adv and isinstance(adv.get("units_per_s"), (int, float)):
        solo = solo_units_per_s(info, info.workload)
        s["adversary_solo_units_per_s"] = solo
        s["adversary_relative_throughput"] = (adv["units_per_s"] / solo) if solo else None
    if info.kind == "solo":
        s["source"] = "adversary.json"
        if with_validation:
            from . import validate
            s["validation"] = validate.check_run(info, s)
        return _clean(s)

    sf = sbio.read_slots(info.file("slots.bin"))
    meta = sbio.load_json(info.file("meta.json"))
    rec = sf.records
    hdr = sf.header
    deadline_ns = hdr["deadline_ns"]
    s["source"] = "slots.bin"
    s["header"] = {k: hdr[k] for k in ("period_ns", "deadline_ns", "t_start_ns", "n_records")}
    s["deadline_us"] = deadline_ns / 1e3
    s["period_us"] = hdr["period_ns"] / 1e3
    s["file_flags"] = sf.flags
    s["partial_tail_bytes"] = sf.partial_tail_bytes

    gaps = stats.slot_gaps(rec["slot"])
    mc = sbio.meta_counts(meta)
    s["counts"] = {"recorded": int(rec.size), "requested": sbio.requested_slots(meta),
                   "skipped_from_gaps": gaps["skipped"], "slot_monotonic": gaps["monotonic"],
                   "slot_duplicates": gaps["duplicates"], "first_slot": gaps["first"],
                   "last_slot": gaps["last"], "meta": mc, "crashed": sf.crashed or meta is None,
                   "header_crashed": sf.crashed, "meta_present": meta is not None,
                   "exit_reason": sbio.exit_reason(meta)}

    spin = sbio.spin_ns(meta)
    fits, fit_src = load_fits(info, meta)
    d = stats.derived_metrics(rec, spin, fits)
    lat = d["latency"]
    s["miss"] = stats.miss_stats(lat, deadline_ns, gaps["skipped"])

    lq = stats.quantiles(lat)
    s["latency_us"] = _us(lq)
    if lq["n"]:
        s["latency_us"]["jitter"] = (lq["p99_99"] - lq["p50"]) / 1e3
    bq = stats.quantiles(d["latency_from_boundary"])
    s["latency_from_boundary_us"] = _us(bq)
    if bq["n"]:
        s["latency_from_boundary_us"]["jitter"] = (bq["p99_99"] - bq["p50"]) / 1e3
        s["latency_from_boundary_us"]["late"] = int(np.count_nonzero(d["latency_from_boundary"] > deadline_ns))
    for k in ("start_lateness", "launch_cost", "wake_overshoot", "gpu_exec"):
        s[f"{k}_us"] = _us(stats.quantiles(d[k])) if k in d else None
    s["spin_ns"] = spin
    s["stamps_missing"] = int(rec.size - np.count_nonzero(d["gpu_exec_mask"]))
    if "queue_delay" in d:
        s["queue_delay_us"] = _us(stats.quantiles(d["queue_delay"]))
        s["queue_delay_us"]["method"] = d["queue_delay_method"]
    else:
        s["queue_delay_us"] = None
    s["clock_fits"] = {"source": fit_src,
                       **{k: {kk: f.get(kk) for kk in ("ok", "a", "rate_ppm", "b_ns", "g_ref", "t_ref", "anchor_g",
                                                       "eps_ns", "resid_rms_ns", "n", "n_kept")}
                          for k, f in fits.items()}}

    h = stats.log_histogram(lat / 1e3)
    h["skipped"] = gaps["skipped"]
    h["metric"] = "latency_us"
    s["histogram"] = h

    order = np.argsort(-lat, kind="stable")[:N_WORST]
    worst = []
    for i in order:
        r = rec[i]
        row = {f: int(r[f]) for f in sbio.RECORD_DTYPE.names}
        row["index"] = int(i)
        row["latency_us"] = float(lat[i]) / 1e3
        worst.append(row)
    s["worst"] = worst

    if with_validation:
        from . import validate
        s["validation"] = validate.check_run(info, s)
    return _clean(s)


def write_summary(run_dir: str) -> str:
    s = build_summary(run_dir)
    out = os.path.join(run_dir, "summary.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, sort_keys=True, indent=1, allow_nan=False)
        f.write("\n")
    os.replace(tmp, out)
    return out


def needs_summary(info: sbio.RunInfo) -> bool:
    """True when summary.json is missing or older than its inputs (and the inputs exist)."""
    sp = info.file("summary.json")
    src = "adversary.json" if info.kind == "solo" else "slots.bin"
    if not info.has(src):
        return False
    if not os.path.exists(sp):
        return True
    t = os.path.getmtime(sp)
    return any(info.has(n) and os.path.getmtime(info.file(n)) > t
               for n in (src, "meta.json", "adversary.json", "telemetry.csv", "run.json", "status"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+")
    a = ap.parse_args(argv)
    rc = 0
    for d in a.run_dirs:
        try:
            out = write_summary(d)
            s = sbio.load_json(out)
            if s.get("source") == "slots.bin":
                m, l = s["miss"], s["latency_us"]
                p = lambda v: "-" if v is None else f"{v:.1f}"
                g = lambda v: "-" if v is None else f"{v:.3g}"
                print(f"{s['identity']['name']}: n={m['recorded']} skipped={m['skipped']} misses={m['misses']} "
                      f"rate={g(m['miss_rate'])} [{g(m['miss_rate_ci_lo'])},{g(m['miss_rate_ci_hi'])}] "
                      f"p50={p(l['p50'])} p99.99={p(l['p99_99'])} max={p(l['max'])} us -> {out}")
            else:
                print(f"{s['identity']['name']}: -> {out}")
        except Exception as e:  # keep going over the other dirs; nonzero exit at the end
            print(f"{d}: ERROR {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
