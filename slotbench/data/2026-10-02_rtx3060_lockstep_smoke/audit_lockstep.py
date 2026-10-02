#!/usr/bin/env python3
"""Read-only, one-off audit of lockstep JSON and little-endian 64-byte raw records.

Usage: python audit_lockstep.py RESULTS_DIRECTORY
Exit 0 means all discovered cells passed the audit; 1 means invalid/mismatched
cells; 2 means no cells or an unreadable input directory. No files are written.
"""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import struct


RECORD = struct.Struct("<Qq6Q")
PAIR_FIELDS = ("slot_variant", "phy", "period_us", "deadline_us", "load")
METRICS = ("start_error_us", "launch_precision_us", "exec_us")


def llround(value):
    """C++ round-to-nearest, ties away from zero, without large absolute floats."""
    value = value if isinstance(value, Fraction) else Fraction(value)
    sign = -1 if value < 0 else 1
    numerator = abs(value.numerator)
    return sign * ((2 * numerator + value.denominator) // (2 * value.denominator))


def mapping(data):
    pre, post = data["clock_fit_pre"], data["clock_fit_post"]
    if not pre["ok"] or not post["ok"] or not data.get("two_point_ok"):
        raise ValueError("invalid pre/post clock fit or two-point mapping")
    # h = round(t_ref + b_ns). Keep t_ref integral; never convert GPU epochs
    # (~1.7e18 ns) to float. This also preserves sub-nanosecond b_ns values.
    ha = llround(Fraction(int(pre["t_ref"])) + Fraction(str(pre["b_ns"])))
    hb = llround(Fraction(int(post["t_ref"])) + Fraction(str(post["b_ns"])))
    ga, gb = int(pre["g_ref"]), int(post["g_ref"])
    if hb - ha <= 1_000_000 or gb <= ga:
        raise ValueError("invalid two-point anchor baseline/rate")
    rate = float(gb - ga) / float(hb - ha)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("nonfinite/nonpositive two-point rate")
    return lambda target: ga + llround(float(target - ha) * rate)


def statistics(values):
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    result = {"n": len(ordered), "min": ordered[0], "max": ordered[-1]}
    for label, percentile in (("p50", 50), ("p90", 90), ("p99", 99),
                              ("p99_9", 99.9), ("p99_99", 99.99)):
        result[label] = ordered[math.ceil(percentile / 100.0 * (len(ordered) - 1))]
    return result


def audit(path, root):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    issues = []
    result = {"name": str(path.relative_to(root).with_suffix("")), "data": data,
              "issues": issues, "stats": {}, "counts": {}, "N": data.get("slots", "?")}
    if data.get("ok") is not True:
        issues.append("JSON ok is false: " + str(data.get("error") or "no reason provided"))
    for key in PAIR_FIELDS:
        if key not in data:
            issues.append("missing workload field " + key)
    if data.get("executive_finished") != 1:
        issues.append("executive did not finish normally")
    if data.get("executive_errors", 0) or data.get("inproc_error", 0):
        issues.append("executive or in-process load reported errors")
    raw_path = path.with_suffix(".bin")
    if not raw_path.is_file():
        issues.append("missing raw .bin file")
        result["counts"] = dict(raw=0, recorded=0, skipped=0, errors=0, timeouts=0,
                                missing=data.get("slots", "?"), missing_stamps=0, misses=None)
        return result
    raw = raw_path.read_bytes()
    count, trailing = divmod(len(raw), RECORD.size)
    if trailing:
        issues.append(f"raw file has {trailing} trailing bytes")
    slots, warmup = data.get("slots"), data.get("warmup")
    if type(slots) is not int or slots <= 0 or type(warmup) is not int or warmup < 0:
        issues.append("invalid slots/warmup metadata")
        return result
    if count != slots:
        issues.append(f"raw count {count} != slots {slots}")
    counts = dict(raw=count, recorded=0, skipped=0, errors=0, timeouts=0,
                  missing=max(0, slots - count), missing_stamps=0, misses=0)
    result["counts"] = counts
    try:
        gpu_of = mapping(data)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        issues.append("clock mapping: " + str(exc))
        gpu_of = None
    samples = {key: [] for key in METRICS}
    samples["latency_from_target_us"] = []
    deadline = data.get("deadline_us")
    if not isinstance(deadline, (int, float)) or not math.isfinite(deadline) or deadline <= 0:
        issues.append("invalid deadline")
        deadline = None
    period = data.get("period_us")
    period_ns = llround(float(period) * 1000) if isinstance(period, (int, float)) and math.isfinite(period) else None
    previous_target = None
    for index, row in enumerate(RECORD.iter_unpack(raw[:count * RECORD.size])):
        slot, target, g_target, launch, launch_done, g0, g1, flags = row
        if slot != warmup + index:
            issues.append(f"raw index {index}: slot {slot} != {warmup + index}")
        if previous_target is not None and period_ns is not None and target - previous_target != period_ns:
            issues.append(f"raw index {index}: target period differs from JSON")
        previous_target = target
        if flags & ~15:
            issues.append(f"raw index {index}: unknown flag bits {flags:#x}")
        if flags & 1:
            counts["skipped"] += 1
            counts["misses"] += 1
            continue
        if flags & 2:
            counts["errors"] += 1
            counts["misses"] += 1
            continue
        if not g0 or not g1:
            counts["missing_stamps"] += 1
        if flags & 8 or not g0 or not g1:
            counts["timeouts"] += 1
            counts["misses"] += 1
            continue
        counts["recorded"] += 1
        if g1 < g0:
            issues.append(f"raw index {index}: end stamp precedes start")
        if launch_done < launch:
            issues.append(f"raw index {index}: launch completion precedes launch")
        samples["launch_precision_us"].append((g0 - g_target) / 1000.0)
        samples["exec_us"].append((g1 - g0) / 1000.0)
        if gpu_of is not None:
            true_target = gpu_of(target)
            samples["start_error_us"].append((g0 - true_target) / 1000.0)
            latency = (g1 - true_target) / 1000.0
            samples["latency_from_target_us"].append(latency)
            if deadline is not None and latency * 1000.0 > deadline * 1000.0:
                counts["misses"] += 1
    if counts["errors"] or counts["timeouts"] or not counts["recorded"]:
        issues.append("launch errors, timeouts, or no recorded slots")
    for key, json_key in (("recorded", "recorded"), ("skipped", "skipped"),
                          ("errors", "launch_errors"), ("timeouts", "timeouts"), ("raw", "total_slots")):
        if data.get(json_key) != counts[key]:
            issues.append(f"JSON {json_key}={data.get(json_key)!r} != raw {counts[key]}")
    if data.get("total_slots") != slots:
        issues.append("JSON total_slots denominator differs from slots")
    if gpu_of is not None and deadline is not None:
        if data.get("misses") != counts["misses"]:
            issues.append(f"JSON misses={data.get('misses')!r} != raw {counts['misses']}")
        rate = data.get("miss_rate")
        if count and (not isinstance(rate, (int, float)) or not math.isfinite(rate)
                      or not math.isclose(rate, counts["misses"] / count, rel_tol=1e-12, abs_tol=1e-15)):
            issues.append("JSON miss_rate differs from raw misses/raw count")
    else:
        counts["misses"] = None
    for name, values in samples.items():
        stats = statistics(values)
        result["stats"][name] = stats
        reported = data.get(name, {})
        if reported.get("n") != stats["n"]:
            issues.append(f"JSON {name}.n differs from raw sample count")
        for key in ("p50", "p99", "max"):
            if key in stats:
                value = reported.get(key)
                # The driver's absolute host-anchor rounding can differ by 1 ns
                # from centered arithmetic; allow that quantization in metrics.
                if not isinstance(value, (int, float)) or not math.isclose(value, stats[key], rel_tol=1e-12, abs_tol=0.001000001):
                    issues.append(f"JSON {name}.{key}={value!r} != raw {stats[key]:.9g}")
    return result


def audit_pairs(cells):
    grouped = {}
    for cell in cells:
        key, changed = re.subn(r"(^|[_/-])(cpu|gpu)(?=[_/-]|$)", r"\1{mode}", cell["name"], count=1)
        if changed:
            grouped.setdefault(key, []).append(cell)
    messages = []
    for key, group in sorted(grouped.items()):
        cpu = [c for c in group if c["data"].get("mode") == "cpu"]
        gpu = [c for c in group if c["data"].get("mode") == "gpu"]
        if len(cpu) != 1 or len(gpu) != 1:
            messages.append(f"{key}: incomplete/ambiguous pair; no CPU/GPU timing conclusion")
            continue
        differences = [f for f in PAIR_FIELDS if cpu[0]["data"].get(f) != gpu[0]["data"].get(f)]
        if differences:
            issue = "CPU/GPU workload mismatch: " + ", ".join(differences)
            for cell in group:
                cell["issues"].append(issue)
        valid = not any(c["issues"] for c in group)
        messages.append(f"{key}: " + ("matching workloads; both cells valid" if valid else "invalid/not comparable; no timing conclusion"))
    return messages


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    if not root.is_dir():
        parser.error("results directory does not exist")
    cells = []
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".adversary.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or data.get("mode") not in ("cpu", "gpu"):
                continue
            cells.append(audit(path, root))
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            if path.with_suffix(".bin").exists() or re.match(r"(?:cpu|gpu)[_-]", path.stem):
                cells.append({"name": str(path.relative_to(root)), "data": {}, "N": "?", "counts": {},
                              "stats": {}, "issues": ["unreadable/malformed result: " + str(exc)]})
    for path in sorted(root.rglob("*.bin")):
        if not path.with_suffix(".json").exists() and re.match(r"(?:cpu|gpu)[_-]", path.stem):
            cells.append({"name": str(path.relative_to(root)), "data": {}, "N": "?", "counts": {},
                          "stats": {}, "issues": ["raw file has no matching case JSON"]})
    if not cells:
        print("No lockstep case JSON files found.")
        return 2
    pairs = audit_pairs(cells)
    print("Timing in us; each triplet is p50/p99/max (ceil percentile convention). Invalid timing is suppressed.")
    print(f"{'case':32} {'status':7} {'start_error':>26} {'launch_precision':>26} {'exec':>26} {'misses/N':>14}")
    for cell in cells:
        valid = not cell["issues"]
        columns = []
        for metric in METRICS:
            values = cell["stats"].get(metric, {})
            columns.append("/".join(f"{values[k]:.3f}" for k in ("p50", "p99", "max")) if valid and values.get("n") else "--")
        counts = cell["counts"]
        misses = counts.get("misses")
        print(f"{cell['name']:32} {'VALID' if valid else 'INVALID':7} " + " ".join(f"{v:>26}" for v in columns)
              + f" {str(misses) if misses is not None else '?'}/{cell['N']}")
        print("  counts: " + ", ".join(f"{k}={counts.get(k, '?')}" for k in
              ("raw", "recorded", "skipped", "errors", "timeouts", "missing", "missing_stamps")))
        for issue in dict.fromkeys(cell["issues"]):
            print("  INVALID: " + issue)
    for message in pairs:
        print("PAIR " + message)
    print("missing = absent raw records; missing_stamps = non-skipped/non-error rows with zero stamps (also timeouts).")
    return int(any(cell["issues"] for cell in cells))


if __name__ == "__main__":
    raise SystemExit(main())
