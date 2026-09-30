"""Matrix report: python3 -m analysis.report RUNS_ROOT --out REPORT_DIR.

Discovers runs (raw or summary-only), (re)writes missing/stale summary.json files, validates every run,
then writes per config: CCDF per workload/duty, miss-rate heatmap, Pareto, worst-run time series,
REPORT_DIR/summary.csv (one row per run), cells.csv (reps pooled), validation.json and report.md.
Works on a partially complete matrix. Invalid runs are listed and, unless --include-invalid, left out
of the figures and the pooled table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from . import plots, sbio, stats, summarize, validate

RUN_COLS = ["config", "run", "kind", "mechanism", "workload", "duty", "rep", "valid", "failed", "warned",
            "source", "recorded", "requested", "skipped", "total_slots", "misses", "miss_rate", "miss_ci_lo",
            "miss_ci_hi", "p50_us", "p90_us", "p99_us", "p99_9_us", "p99_99_us", "max_us", "jitter_us",
            "lfb_p99_99_us", "lfb_max_us", "wake_p99_us", "wake_max_us", "gpu_exec_p50_us", "gpu_exec_p99_us",
            "gpu_exec_max_us", "queue_delay_p50_us", "queue_delay_p99_us", "queue_delay_max_us",
            "stamps_missing", "units_per_s", "solo_units_per_s", "rel_throughput", "not_supported"]


def not_supported_items(runj) -> list[str]:
    """Strings in run.json saying something was not supported (lenient: keys containing
    'not_supported'/'unsupported', and any string value containing 'NOT SUPPORTED')."""
    out = []
    for p, v in sbio.iter_items(runj or {}):
        key = str(p[-1]).lower() if p else ""
        if ("not_supported" in key or "unsupported" in key) and v:
            if isinstance(v, (list, tuple)):
                out += [str(x) for x in v if x]
            elif isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict):
                out += [f"{k}: {x}" for k, x in v.items() if x]
            elif v is True:
                out.append(".".join(str(x) for x in p))
        elif isinstance(v, str) and "NOT SUPPORTED" in v.upper():
            out.append(v)
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _g(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def run_row(info, s, v, solo_ups, ns) -> dict:
    row = {c: None for c in RUN_COLS}
    row.update(config=info.config, run=info.name, kind=info.kind, mechanism=info.mechanism, workload=info.workload,
               duty=info.duty, rep=info.rep, valid=v["valid"], failed=";".join(v["failed"]),
               warned=";".join(v["warned"]), source=(s or {}).get("source"), not_supported="; ".join(ns))
    adv = (s or {}).get("adversary") or {}
    ups = adv.get("units_per_s")
    row["units_per_s"] = ups
    row["solo_units_per_s"] = solo_ups
    row["rel_throughput"] = (ups / solo_ups) if (ups is not None and solo_ups) else None
    if s and s.get("source") == "slots.bin":
        m, l, b = s["miss"], s.get("latency_us") or {}, s.get("latency_from_boundary_us") or {}
        row.update(recorded=m["recorded"], requested=_g(s, "counts", "requested"), skipped=m["skipped"],
                   total_slots=m["total_slots"], misses=m["misses"], miss_rate=m["miss_rate"],
                   miss_ci_lo=m["miss_rate_ci_lo"], miss_ci_hi=m["miss_rate_ci_hi"],
                   p50_us=l.get("p50"), p90_us=l.get("p90"), p99_us=l.get("p99"), p99_9_us=l.get("p99_9"),
                   p99_99_us=l.get("p99_99"), max_us=l.get("max"), jitter_us=l.get("jitter"),
                   lfb_p99_99_us=b.get("p99_99"), lfb_max_us=b.get("max"),
                   wake_p99_us=_g(s, "wake_overshoot_us", "p99"), wake_max_us=_g(s, "wake_overshoot_us", "max"),
                   gpu_exec_p50_us=_g(s, "gpu_exec_us", "p50"), gpu_exec_p99_us=_g(s, "gpu_exec_us", "p99"),
                   gpu_exec_max_us=_g(s, "gpu_exec_us", "max"),
                   queue_delay_p50_us=_g(s, "queue_delay_us", "p50"), queue_delay_p99_us=_g(s, "queue_delay_us", "p99"),
                   queue_delay_max_us=_g(s, "queue_delay_us", "max"), stamps_missing=s.get("stamps_missing"))
    return row


def collect(root: str, cfg=None, write_summaries=True, log=print):
    """Summaries (written when missing/stale), validation and table rows for every run under root."""
    runs = sbio.discover_runs(root)
    entries = []
    for info in runs:
        err = None
        if write_summaries and summarize.needs_summary(info):
            try:
                summarize.write_summary(info.path)
            except Exception as e:  # noqa: BLE001 - report and carry on with the matrix
                err = f"summarize failed: {e}"
                log(f"{info.name}: {err}")
        s = sbio.load_json(info.file("summary.json"))
        if s is None and not write_summaries:
            s = validate.get_summary(info)
        if s is None and info.kind == "solo":
            s = {"adversary": summarize.adversary_summary(info)}
        v = validate.check_run(info, s, cfg)
        if err:
            v["checks"].append({"name": "summarize", "status": "fail", "reason": err})
            v["valid"] = False
            v["failed"].append("summarize")
        entries.append((info, s, v))
    solo = defaultdict(list)
    for info, s, v in entries:
        ups = _g(s, "adversary", "units_per_s")
        if info.kind == "solo" and v["valid"] and isinstance(ups, (int, float)):
            solo[(info.config, info.workload)].append(float(ups))
    rows = []
    for info, s, v in entries:
        so = solo.get((info.config, info.workload))
        ns = not_supported_items(sbio.load_json(info.file("run.json")))
        rows.append(run_row(info, s, v, float(np.mean(so)) if so else None, ns))
    df = pd.DataFrame(rows, columns=RUN_COLS)
    return entries, df


def pool_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Reps pooled per (config, mechanism, workload, duty): misses and slots summed, CI recomputed,
    worst p99.99/max over reps, mean throughput."""
    cells = df[(df.kind == "cell") & df.total_slots.notna()]
    out = []
    for key, g in cells.groupby(["config", "mechanism", "workload", "duty"], sort=True):
        k, n = int(g.misses.sum()), int(g.total_slots.sum())
        lo, hi = stats.clopper_pearson(k, n)
        out.append({"config": key[0], "mechanism": key[1], "workload": key[2], "duty": int(key[3]),
                    "cell": f"{key[1]}_{key[2]}_d{int(key[3])}", "reps": len(g), "total_slots": n,
                    "misses": k, "miss_rate": k / n if n else None, "miss_ci_lo": lo, "miss_ci_hi": hi,
                    "p50_us_median": float(g.p50_us.median()), "p99_99_us_worst": float(g.p99_99_us.max()),
                    "max_us_worst": float(g.max_us.max()),
                    "units_per_s": float(g.units_per_s.mean()) if g.units_per_s.notna().any() else None,
                    "rel_throughput": float(g.rel_throughput.mean()) if g.rel_throughput.notna().any() else None})
    return pd.DataFrame(out)


def _pooled_latency(infos_and_summaries):
    """Raw latencies (us) when every rep has slots.bin, else merged histograms. Skipped counts summed."""
    skipped = sum(int(s["miss"]["skipped"]) for _, s in infos_and_summaries)
    misses = sum(int(s["miss"]["misses"]) for _, s in infos_and_summaries)
    if all(i.has("slots.bin") for i, _ in infos_and_summaries):
        lat = []
        for i, _ in infos_and_summaries:
            r = sbio.read_slots(i.file("slots.bin")).records
            lat.append((r["t1"].astype(np.int64) - r["t0"].astype(np.int64)) / 1e3)
        return {"latency_us": np.concatenate(lat), "skipped": skipped, "misses": misses}
    h = stats.merge_hists([s.get("histogram") for _, s in infos_and_summaries])
    h = dict(h)
    h["skipped"] = skipped
    return {"hist": h, "misses": misses}


def md_table(df: pd.DataFrame, cols, fmt=None) -> str:
    fmt = fmt or {}

    def f(c, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        if c in fmt:
            return fmt[c](v)
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f(c, r[c]) for c in cols) + " |")
    return "\n".join(lines)


def make_report(root: str, out: str, cfg=None, include_invalid=False, log=print) -> dict:
    os.makedirs(out, exist_ok=True)
    entries, df = collect(root, cfg, log=log)
    by_path = {i.path: (i, s, v) for i, s, v in entries}
    df.to_csv(os.path.join(out, "summary.csv"), index=False)
    use = df if include_invalid else df[df.valid]
    cells = pool_cells(use)
    cells.to_csv(os.path.join(out, "cells.csv"), index=False)
    val = {"runs": {f"{i.config}/{i.name}": v for i, _, v in entries},
           "repeatability": validate.repeatability({i.path: (i, s) for i, s, v in entries if v["valid"]}, cfg)}
    with open(os.path.join(out, "validation.json"), "w") as f:
        json.dump(val, f, sort_keys=True, indent=1)

    figs = defaultdict(list)  # config -> [(caption, png path)]
    for config in sorted(df.config.unique()):
        cfg_runs = [(i, s) for i, s, v in entries if i.config == config and i.kind == "cell" and s
                    and s.get("source") == "slots.bin" and (v["valid"] or include_invalid)]
        if not cfg_runs:
            continue
        deadline = float(cfg_runs[0][1]["deadline_us"])
        period = float(cfg_runs[0][1]["period_us"])
        groups = defaultdict(list)
        for i, s in cfg_runs:
            groups[(i.workload, i.duty, i.mechanism)].append((i, s))
        # (1) CCDF per workload/duty
        for w, d in sorted({(k[0], k[1]) for k in groups}):
            items = {m: _pooled_latency(groups[(w, d, m)]) for (ww, dd, m) in groups if (ww, dd) == (w, d)}
            base = os.path.join(out, f"ccdf_{config}_{w}_d{d}")
            plots.plot_ccdf(items, deadline, f"{config}: {w}, duty {d}% - latency CCDF per mechanism", base)
            figs[config].append((f"CCDF {w} d{d}", base + ".png"))
        # (2) heatmap and (3) Pareto from pooled cells
        cc = cells[cells.config == config]
        if len(cc):
            mechs = sorted(cc.mechanism.unique())
            duties = sorted(cc.duty.unique())
            wls = sorted(w for w in cc.workload.unique() if w != "idle") or ["idle"]
            hm = {}
            for _, r in cc.iterrows():
                c = {"misses": int(r.misses), "total": int(r.total_slots), "ci_hi": r.miss_ci_hi}
                if r.workload == "idle":
                    for w in wls:
                        hm.setdefault((w, r.mechanism, r.duty), {**c, "idle": True})
                else:
                    hm[(r.workload, r.mechanism, r.duty)] = c
            base = os.path.join(out, f"heatmap_{config}")
            plots.plot_heatmap(hm, mechs, [int(d) for d in duties], wls, f"{config}: slot miss rate", base)
            figs[config].append(("Miss-rate heatmap", base + ".png"))
            adv = cc[(cc.workload != "idle") & cc.units_per_s.notna()]
            if len(adv):
                relative = bool(adv.rel_throughput.notna().all())
                pts = [{"workload": r.workload, "mechanism": r.mechanism, "duty": int(r.duty),
                        "x": r.rel_throughput if relative else r.units_per_s, "misses": int(r.misses),
                        "total": int(r.total_slots), "ci_lo": r.miss_ci_lo, "ci_hi": r.miss_ci_hi}
                       for _, r in adv.iterrows()]
                base = os.path.join(out, f"pareto_{config}")
                plots.plot_pareto(pts, sorted(adv.workload.unique()), relative,
                                  f"{config}: adversary throughput vs slot miss rate", base)
                figs[config].append(("Pareto", base + ".png"))
        # (4) worst run time series (needs raw)
        raw = [(i, s) for i, s in cfg_runs if i.has("slots.bin")]
        if raw:
            i, s = max(raw, key=lambda t: (t[1]["miss"]["miss_rate"] or 0, t[1]["latency_us"]["p99_99"] or 0))
            rec = sbio.read_slots(i.file("slots.bin")).records
            lat = (rec["t1"].astype(np.int64) - rec["t0"].astype(np.int64)) / 1e3
            base = os.path.join(out, f"timeseries_{config}_worst")
            plots.plot_timeseries(rec["slot"], lat, period, deadline,
                                  f"{config}: worst run {i.name} (miss rate {s['miss']['miss_rate']:.3g})", base)
            figs[config].append((f"Time series of worst run {i.name}", base + ".png"))

    md = render_md(root, out, df, cells, entries, figs, val, include_invalid)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(md)
    return {"n_runs": len(df), "n_invalid": int((~df.valid).sum()), "figures": dict(figs),
            "by_path": list(by_path)}


def render_md(root, out, df, cells, entries, figs, val, include_invalid) -> str:
    L = ["# slotbench report", "", f"Runs root: `{os.path.abspath(root)}`  ",
         f"Runs found: {len(df)} ({int((df.kind == 'cell').sum())} cells, {int((df.kind == 'solo').sum())} solo "
         f"baselines), invalid: {int((~df.valid).sum())}. "
         + ("Invalid runs are included in figures." if include_invalid else
            "Figures and the pooled table use valid runs only."), "",
         "Metric definitions and how to read each figure: `analysis/README.md`. Miss = latency t1 - t0 > "
         "deadline, plus every skipped slot boundary; intervals are exact 95% Clopper-Pearson; quantiles use "
         "the 'higher' method (always an observed sample).", ""]
    if any((sbio.load_json(i.file("run.json")) or {}).get("synthetic") is True for i, _, _ in entries):
        L += ["**WARNING: this tree contains synthetic (fake) runs generated by analysis.synth.**", ""]
    for config in sorted(df.config.unique()):
        L += [f"## Config `{config}`", ""]
        for cap, p in figs.get(config, []):
            rel = os.path.relpath(p, out)
            L += [f"### {cap}", "", f"![{cap}]({rel}) ([pdf]({rel[:-4]}.pdf))", ""]
        cc = cells[cells.config == config] if len(cells) else cells
        if len(cc):
            L += ["### Cells (reps pooled)", "",
                  md_table(cc, ["cell", "reps", "total_slots", "misses", "miss_rate", "miss_ci_lo", "miss_ci_hi",
                                "p50_us_median", "p99_99_us_worst", "max_us_worst", "units_per_s",
                                "rel_throughput"],
                           {"miss_rate": lambda v: f"{v:.3g}", "miss_ci_lo": lambda v: f"{v:.3g}",
                            "miss_ci_hi": lambda v: f"{v:.3g}"}), ""]
        rr = df[df.config == config]
        L += ["### All runs", "",
              md_table(rr, ["run", "valid", "recorded", "skipped", "misses", "miss_rate", "miss_ci_hi", "p50_us",
                            "p99_us", "p99_9_us", "p99_99_us", "max_us", "jitter_us", "wake_p99_us",
                            "gpu_exec_p99_us", "queue_delay_p99_us", "units_per_s", "rel_throughput"],
                       {"miss_rate": lambda v: f"{v:.3g}", "miss_ci_hi": lambda v: f"{v:.3g}"}), ""]
    L += ["## Invalid runs", ""]
    bad = [(i, v) for i, _, v in entries if not v["valid"]]
    if not bad:
        L += ["None.", ""]
    for i, v in bad:
        reasons = "; ".join(f"{c['name']}: {c['reason']}" for c in v["checks"] if c["status"] == "fail")
        L.append(f"- `{i.config}/{i.name}`: {reasons}")
    warned = [(i, v) for i, _, v in entries if v["warned"]]
    if warned:
        L += ["", "Warnings (do not invalidate):", ""]
        for i, v in warned:
            L.append(f"- `{i.config}/{i.name}`: " + "; ".join(f"{c['name']}: {c['reason']}" for c in v["checks"]
                                                            if c["status"] == "warn"))
    L += ["", "## Not supported (from run.json)", ""]
    ns = df[df.not_supported.fillna("") != ""]
    if not len(ns):
        L += ["Nothing recorded as not supported.", ""]
    for _, r in ns.iterrows():
        L.append(f"- `{r.config}/{r.run}` ({r.mechanism or 'SOLO'}): {r.not_supported}")
    L += ["", "## Repeatability (CV of p99.99 across reps)", ""]
    if not val["repeatability"]:
        L += ["No cell has more than one rep.", ""]
    for rp in val["repeatability"]:
        cv = "n/a" if rp["cv"] is None else f"{rp['cv']:.3f}"
        L.append(f"- `{rp['cell']}`: reps {rp['reps']}, p99.99 {['%.1f' % x for x in rp['p99_99_us']]} us, "
                 f"CV {cv} ({rp['status']})")
    L += ["", "Files: `summary.csv` (per run), `cells.csv` (pooled), `validation.json`.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="slotbench matrix report")
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-invalid", action="store_true", help="also plot/pool invalid runs")
    ap.add_argument("--throttle", choices=["fail", "warn"], default=validate.DEFAULTS["throttle_mode"])
    ap.add_argument("--idle-p9999-us", type=float, default=validate.DEFAULTS["idle_p9999_us"])
    a = ap.parse_args(argv)
    r = make_report(a.root, a.out, {"throttle_mode": a.throttle, "idle_p9999_us": a.idle_p9999_us},
                    a.include_invalid)
    print(f"{r['n_runs']} runs ({r['n_invalid']} invalid); report: {os.path.join(a.out, 'report.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
