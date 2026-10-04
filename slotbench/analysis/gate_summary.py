#!/usr/bin/env python3
"""Summarise the time-aware gating campaign (cloud/onstart_cuphy_lockstep.sh, SB_CUPHY_CAMPAIGN=gate).

Per configuration (sharing x gate mode x GEMM size), pooled over repeats:
  5G side : slots, missed slots (latency from target > deadline, skipped, or failed), p99 / max latency
  AI side : GEMMs per second during the slot window, as a fraction of the same GEMM size running alone
Usage: gate_summary.py OUT_DIR [--json FILE]
"""
import argparse
import glob
import json
import os
import re
import sys

CASE_RE = re.compile(r"^(proc|mps)_(observe|gated)_n(\d+)_r(\d+)$")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def summarise(out):
    alone_ai = {}
    for p in glob.glob(os.path.join(out, "ai_alone_n*.json")):
        d = load(p)
        n = int(re.search(r"_n(\d+)\.json$", p).group(1))
        if d and d.get("ok"):
            alone_ai[n] = d["units_per_s"]
    groups, problems = {}, []
    for p in sorted(glob.glob(os.path.join(out, "*.json"))):
        name = os.path.basename(p)[:-5]
        m = CASE_RE.match(name)
        is_alone = re.match(r"^alone_r\d+$", name)
        if not (m or is_alone):
            continue
        d = load(p)
        if not d or not d.get("ok"):
            problems.append(f"{name}: cuPHY run not ok ({(d or {}).get('error', 'missing')})")
            continue
        key = ("none", "alone", 0) if is_alone else (m.group(1), m.group(2), int(m.group(3)))
        g = groups.setdefault(key, dict(runs=0, slots=0, misses=0, p99=[], max=[], ai_ups=[], ai_busy=[],
                                         overrun=0, ai_units=0, est_us=[]))
        g["runs"] += 1
        g["slots"] += d["slots"]
        g["misses"] += d["misses"]
        g["p99"].append(d["latency_from_target_us"]["p99"])
        g["max"].append(d["latency_from_target_us"]["max"])
        if m:
            a = load(os.path.join(out, name + ".adversary.json"))
            gate = (a or {}).get("gate") or {}
            if not a or not a.get("ok") or not gate.get("timetable_loaded"):
                problems.append(f"{name}: tenant summary missing, not ok, or timetable not loaded")
                continue
            g["ai_ups"].append(gate["units_per_s_in_window"])
            g["ai_busy"].append(gate["gpu_busy_fraction_in_window"])
            g["overrun"] += gate["overrun_units"]
            g["ai_units"] += gate["units_in_window"]
            g["est_us"].append(gate["est_unit_us_final"])
    rows = []
    for (iso, mode, n), g in sorted(groups.items()):
        ups = sum(g["ai_ups"]) / len(g["ai_ups"]) if g["ai_ups"] else None
        rows.append(dict(sharing=iso, gate=mode, gemm_n=n, runs=g["runs"], slots=g["slots"], misses=g["misses"],
                         miss_rate=g["misses"] / g["slots"] if g["slots"] else None,
                         latency_p99_us_worst=max(g["p99"]), latency_max_us=max(g["max"]),
                         ai_units_per_s=ups, ai_alone_units_per_s=alone_ai.get(n),
                         ai_kept=(ups / alone_ai[n]) if ups is not None and alone_ai.get(n) else None,
                         ai_units=g["ai_units"], ai_units_overlapping_slot=g["overrun"],
                         ai_unit_est_us=max(g["est_us"]) if g["est_us"] else None))
    return dict(rows=rows, alone_ai_units_per_s=alone_ai, problems=problems,
                busy_us=(open(os.path.join(out, "busy_us.txt")).read().strip()
                         if os.path.exists(os.path.join(out, "busy_us.txt")) else None))


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--json")
    a = ap.parse_args()
    s = summarise(a.out)
    print(f"busy_us reserved per slot: {s['busy_us']}")
    print(f"{'sharing':7} {'gate':8} {'n':>5} {'runs':>4} {'slots':>6} {'missed':>7} {'miss %':>7} "
          f"{'p99 us':>8} {'max us':>8} {'AI/s':>9} {'AI kept':>8} {'AI over':>8}")
    for r in s["rows"]:
        print(f"{r['sharing']:7} {r['gate']:8} {r['gemm_n']:5d} {r['runs']:4d} {r['slots']:6d} {r['misses']:7d} "
              f"{fmt(100 * r['miss_rate'] if r['miss_rate'] is not None else None, '7.2f')} "
              f"{r['latency_p99_us_worst']:8.1f} {r['latency_max_us']:8.1f} {fmt(r['ai_units_per_s'], '9.1f')} "
              f"{(format(100 * r['ai_kept'], '7.1f') + '%') if r['ai_kept'] is not None else '       -':>8} "
              f"{r['ai_units_overlapping_slot'] if r['gate'] != 'alone' else '-':>8}")
    for p in s["problems"]:
        print("PROBLEM:", p)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(s, f, indent=2)
    return 1 if s["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
