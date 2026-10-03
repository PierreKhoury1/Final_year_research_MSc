#!/usr/bin/env python3
"""Re-score cuPHY lockstep trials at any completion deadline.

Every trial leaves one 64-byte record per measured slot (see slotbench/cuphy/README.md) and pre/post clock
anchors in its JSON. A slot's completion latency from its boundary is end - g2pt(target); a slot is a miss at
deadline D when it was skipped (the previous slot was still running) or its latency exceeds D. This script
pools the trials of each launch variant (cpu, cpu_keepalive, gpu, and the contention conditions when present)
and reports misses versus deadline, so the 500 us accounting deadline of the run is not the only view.

Usage: cuphy_deadline_sweep.py OUT_DIR [--output-dir DIR] [--deadlines 150,160,175,200,250,300,400,500]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import struct
import sys
from collections import defaultdict
from fractions import Fraction

RECORD = struct.Struct("<Qq6Q")


def rounded(value):
    return int(round(value))


def trial_latencies(json_path, raw_path):
    """List of (latency_us | None if skipped, start_error_us, exec_us) per measured slot."""
    data = json.load(open(json_path))
    pre, post = data["clock_fit_pre"], data["clock_fit_post"]
    ha = int(pre["t_ref"]) + rounded(Fraction(str(pre["b_ns"])))
    hb = int(post["t_ref"]) + rounded(Fraction(str(post["b_ns"])))
    ga, gb = pre["g_ref"], post["g_ref"]
    rate = float(gb - ga) / float(hb - ha)
    out = []
    for _slot, target, _gt, _la, _lr, start, end, flags in RECORD.iter_unpack(open(raw_path, "rb").read()):
        if flags & 1:
            out.append((None, None, None))
            continue
        tt = ga + rounded(float(target - ha) * rate)
        out.append(((end - tt) / 1000.0, (start - tt) / 1000.0, (end - start) / 1000.0))
    return data, out


def variant_of(name):
    base = os.path.basename(name)
    if "gate_" in base:
        return None
    if "keepalive" in base:
        v = "cpu_keepalive"
    elif "_gpu_" in base or base.startswith("gpu_"):
        v = "gpu"
    elif "_cpu_" in base or base.startswith("cpu_"):
        v = "cpu"
    else:
        return None
    for cond in ("proc_sgemm", "mps_sgemm", "alone"):
        if cond in base:
            return f"{v}/{cond}"
    return v


def pct(sorted_values, p):
    if not sorted_values:
        return float("nan")
    i = int(p / 100.0 * (len(sorted_values) - 1))
    return sorted_values[min(max(i, 0), len(sorted_values) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--deadlines", default="150,160,175,200,250,300,400,500")
    args = ap.parse_args()
    deadlines = [float(x) for x in args.deadlines.split(",")]
    outdir = args.output_dir or os.path.join(args.out_dir, "deadline_sweep")
    os.makedirs(outdir, exist_ok=True)

    pooled = defaultdict(list)
    trials = []
    for raw in sorted(glob.glob(os.path.join(args.out_dir, "*.bin"))):
        name = os.path.basename(raw)[:-4]
        var = variant_of(name)
        js = raw[:-4] + ".json"
        if var is None or not os.path.exists(js):
            continue
        data, lat = trial_latencies(js, raw)
        if data.get("ok") is not True:
            continue
        pooled[var].extend(lat)
        executed = [x[0] for x in lat if x[0] is not None]
        row = {"trial": name, "variant": var, "slots": len(lat), "skipped": len(lat) - len(executed),
               "stalls_over_1ms": sum(1 for x in executed if x > 1000.0) + (len(lat) - len(executed))}
        for d in deadlines:
            row[f"miss@{d:g}"] = row["skipped"] + sum(1 for x in executed if x > d)
        trials.append(row)

    rows = []
    lines = []
    header = (f"{'variant':22s} {'slots':>7s} {'skipped':>7s} {'>1ms':>5s} "
              + " ".join(f"miss@{d:g}".rjust(10) for d in deadlines)
              + "   p50 lat  p99 lat  p99.9 lat  max lat (us)")
    lines.append(header)
    for var in sorted(pooled):
        L = pooled[var]
        n = len(L)
        executed = sorted(x[0] for x in L if x[0] is not None)
        skipped = n - len(executed)
        stalls = sum(1 for x in executed if x > 1000.0) + skipped
        row = {"variant": var, "slots": n, "skipped": skipped, "stalls_over_1ms": stalls,
               "p50_latency_us": pct(executed, 50), "p99_latency_us": pct(executed, 99),
               "p99_9_latency_us": pct(executed, 99.9), "max_latency_us": executed[-1] if executed else float("nan")}
        cols = []
        for d in deadlines:
            m = skipped + sum(1 for x in executed if x > d)
            row[f"miss@{d:g}"] = m
            row[f"miss_pct@{d:g}"] = 100.0 * m / n if n else float("nan")
            cols.append(f"{100.0 * m / n:9.4f}%" if n else "        -")
        rows.append(row)
        lines.append(f"{var:22s} {n:7d} {skipped:7d} {stalls:5d} " + " ".join(cols)
                     + f"   {row['p50_latency_us']:7.1f} {row['p99_latency_us']:8.1f} {row['p99_9_latency_us']:9.1f} {row['max_latency_us']:8.1f}")
    summary = "\n".join(lines) + "\n"
    sys.stdout.write(summary)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(summary)
    with open(os.path.join(outdir, "deadline_sweep.json"), "w") as f:
        json.dump({"deadlines_us": deadlines, "variants": rows, "trials": trials}, f, indent=2)
    if trials:
        with open(os.path.join(outdir, "trials.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trials[0].keys()))
            w.writeheader()
            w.writerows(trials)
    if rows:
        with open(os.path.join(outdir, "variants.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
