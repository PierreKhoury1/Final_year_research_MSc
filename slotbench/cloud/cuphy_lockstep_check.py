#!/usr/bin/env python3
"""Fail-closed collection gate for a cuPHY lockstep smoke case (no GPU needed)."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import struct

RECORD = struct.Struct("<Qq6Q")


def rounded(value):
    value = Fraction(value)
    return (-1 if value < 0 else 1) * ((2 * abs(value.numerator) + value.denominator) // (2 * value.denominator))


def validate_case(json_path, raw_path, mode, slots, warmup, period_us=None, deadline_us=None):
    data = json.loads(Path(json_path).read_text())
    for key in ("ok", "correctness_before", "correctness_after", "two_point_ok"):
        if data.get(key) is not True:
            raise ValueError(f"{key} is not true: {data.get('error', '')}")
    for key, expected in (("mode", mode), ("slots", slots), ("warmup", warmup), ("total_slots", slots),
                          ("launch_errors", 0), ("timeouts", 0)):
        if data.get(key) != expected:
            raise ValueError(f"{key}: {data.get(key)!r} != {expected!r}")
    for key, expected in (("period_us", period_us), ("deadline_us", deadline_us)):
        if expected is not None and data.get(key) != expected:
            raise ValueError(f"{key}: {data.get(key)!r} != requested {expected!r}")
    raw = Path(raw_path).read_bytes()
    if len(raw) != slots * RECORD.size:
        raise ValueError(f"raw size {len(raw)} != {slots} * {RECORD.size}")
    pre, post = data["clock_fit_pre"], data["clock_fit_post"]
    if pre.get("ok") is not True or post.get("ok") is not True:
        raise ValueError("pre/post fit is invalid")
    ha = rounded(Fraction(pre["t_ref"]) + Fraction(str(pre["b_ns"])))
    hb = rounded(Fraction(post["t_ref"]) + Fraction(str(post["b_ns"])))
    ga, gb = pre["g_ref"], post["g_ref"]
    if hb - ha <= 1_000_000 or gb <= ga:
        raise ValueError("invalid clock anchor baseline")
    rate = float(gb - ga) / float(hb - ha)
    deadline = float(data["deadline_us"]) * 1000
    if not math.isfinite(rate) or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("invalid rate/deadline")
    counts = dict(recorded=0, skipped=0, misses=0)
    for index, row in enumerate(RECORD.iter_unpack(raw)):
        slot, target, _, _, _, start, end, flags = row
        if slot != index + warmup:
            raise ValueError(f"raw slot {slot} != expected {index + warmup}")
        if flags & ~5:
            raise ValueError(f"raw slot {slot}: launch error/timeout/unknown flags {flags}")
        if flags & 1:
            counts["skipped"] += 1
            counts["misses"] += 1
            continue
        if not start or end < start:
            raise ValueError(f"raw slot {slot}: missing/reversed stamps")
        counts["recorded"] += 1
        true_target = ga + rounded(float(target - ha) * rate)
        if ((end - true_target) / 1000.0) * 1000.0 > deadline:
            counts["misses"] += 1
    if counts["recorded"] == 0:
        raise ValueError("no recorded slots")
    for key, expected in counts.items():
        if data.get(key) != expected:
            raise ValueError(f"{key}: JSON {data.get(key)!r} != raw {expected}")
    if not math.isclose(float(data["miss_rate"]), counts["misses"] / slots, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("miss_rate disagrees with raw counts")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--period-us", type=float)
    parser.add_argument("--deadline-us", type=float)
    args = parser.parse_args()
    try:
        counts = validate_case(args.json, args.raw, args.mode, args.slots, args.warmup, args.period_us, args.deadline_us)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        parser.exit(1, f"cuPHY lockstep validation failed: {exc}\n")
    print(json.dumps(dict(valid=True, **counts), sort_keys=True))


if __name__ == "__main__":
    main()
