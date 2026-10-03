#!/usr/bin/env python3
"""Paired launcher comparison of deadline-miss rates at several completion deadlines.

For a triplet campaign (tNN_pK_<variant>_alone trials from control_cuphy_activity.sh) this re-scores every
trial's raw records at each deadline, forms per-triplet differences of the miss rate (percentage points) between
launch variants, and reports the mean difference with a whole-triplet bootstrap interval. Triplets, not slots, are
the resampling unit; the intervals are exploratory (one host, one vector).

Usage: cuphy_deadline_pairs.py TRIALS_DIR [--output FILE.json] [--deadlines 200,250,300,400,500] [--draws 20000]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import struct
import sys
from fractions import Fraction

RECORD = struct.Struct("<Qq6Q")
VARIANTS = ("cpu", "cpu_keepalive", "gpu")


def miss_rates(json_path, raw_path, deadlines):
    data = json.load(open(json_path))
    pre, post = data["clock_fit_pre"], data["clock_fit_post"]
    ha = int(pre["t_ref"]) + round(Fraction(str(pre["b_ns"])))
    hb = int(post["t_ref"]) + round(Fraction(str(post["b_ns"])))
    ga, gb = pre["g_ref"], post["g_ref"]
    rate = float(gb - ga) / float(hb - ha)
    misses = {d: 0 for d in deadlines}
    n = 0
    for _slot, target, _gt, _la, _lr, _start, end, flags in RECORD.iter_unpack(open(raw_path, "rb").read()):
        n += 1
        if flags & 1:
            for d in deadlines:
                misses[d] += 1
            continue
        latency = (end - (ga + round(float(target - ha) * rate))) / 1000.0
        for d in deadlines:
            if latency > d:
                misses[d] += 1
    return {d: 100.0 * misses[d] / n for d in deadlines}, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trials_dir")
    ap.add_argument("--output", default=None)
    ap.add_argument("--deadlines", default="200,250,300,400,500")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()
    deadlines = [float(x) for x in args.deadlines.split(",")]
    triplets = {}
    slots = None
    for raw in sorted(glob.glob(os.path.join(args.trials_dir, "t*_alone.bin"))):
        name = os.path.basename(raw)[:-4]
        m = re.match(r"t(\d+)_p\d_(cpu_keepalive|cpu|gpu)_alone$", name)
        if not m:
            continue
        data = json.load(open(raw[:-4] + ".json"))
        if data.get("ok") is not True:
            print(f"skipping {name}: ok is not true", file=sys.stderr)
            continue
        rates, n = miss_rates(raw[:-4] + ".json", raw, deadlines)
        slots = n if slots is None else slots
        triplets.setdefault(int(m.group(1)), {})[m.group(2)] = rates
    complete = {t: v for t, v in triplets.items() if all(k in v for k in VARIANTS)}
    if len(complete) < 2:
        print("need at least two complete triplets", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)
    results = []
    lines = [f"{len(complete)} complete triplets, {slots} measured slots per trial; differences in percentage points of "
             "missed boundaries (skips included), mean over triplets [95% whole-triplet bootstrap], triplets where the first is lower"]
    for d in deadlines:
        for a, b in (("gpu", "cpu"), ("gpu", "cpu_keepalive"), ("cpu_keepalive", "cpu")):
            diffs = [complete[t][a][d] - complete[t][b][d] for t in sorted(complete)]
            mean = sum(diffs) / len(diffs)
            boots = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(args.draws))
            lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
            lower = sum(1 for x in diffs if x < 0)
            results.append({"deadline_us": d, "first": a, "second": b, "mean_pp": mean, "ci95_pp": [lo, hi],
                            "first_lower_in": lower, "triplets": len(diffs), "per_triplet_pp": diffs})
            lines.append(f"  deadline {d:5.0f} us  {a:13s} - {b:13s}: {mean:+8.3f} pp  [{lo:+.3f}, {hi:+.3f}]  lower in {lower}/{len(diffs)}")
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"deadlines_us": deadlines, "slots_per_trial": slots, "draws": args.draws, "seed": args.seed,
                       "variant_means_pct": {v: {d: sum(complete[t][v][d] for t in complete) / len(complete) for d in deadlines} for v in VARIANTS},
                       "comparisons": results}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
