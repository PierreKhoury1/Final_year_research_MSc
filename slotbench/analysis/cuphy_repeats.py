#!/usr/bin/env python3
"""Audit randomized cuPHY replay trials and summarize paired trial-level effects.

Usage: python cuphy_repeats.py DATA_DIRECTORY --output-dir ANALYSIS_DIRECTORY

The input experiment.json enumerates the timed trials; correctness gates and
other JSON files are never inferred to be measurements. All-target miss rates
include skips. Within-trial timing quantiles include executed targets only.
Bootstrap units are whole CPU/GPU trial pairs, never individual target slots.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import statistics
import struct
import sys
import xml.etree.ElementTree as ET


RECORD = struct.Struct("<Qq6Q")
CONDITIONS = ("alone", "proc_sgemm", "mps_sgemm")
TIMING_KEYS = (
    "start_error_us", "launch_precision_us", "target_pred_error_us",
    "launch_error_us", "launch_call_us", "launch_to_start_us", "exec_us",
    "latency_from_target_us",
)
SETTING_KEYS = (
    "gpu", "sm", "driver", "runtime", "stream_priority", "slots", "warmup",
    "period_us", "deadline_us", "implementation", "slot_variant",
)
RUN_SETTING_KEYS = (
    "test_vector_sha256", "aerial_commit", "isolation", "workload",
    "adversary_mps_percentage", "command",
)
TRIAL_METRICS = (
    "miss_percentage", "p99_start_us", "p99_phy_us", "p99_completion_us",
    "executed", "skipped", "executed_late", "on_time",
)


def collection_checker():
    path = Path(__file__).resolve().parents[1] / "cloud" / "cuphy_lockstep_check.py"
    spec = importlib.util.spec_from_file_location("_cuphy_collection_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_case


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact leaves the collected directory: {relative}")
    return path


def round_away(value) -> int:
    """C++ llround semantics, preserving integer clock anchors."""
    exact = Fraction(value)
    magnitude = (2 * abs(exact.numerator) + exact.denominator) // (2 * exact.denominator)
    return magnitude if exact >= 0 else -magnitude


def quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    result = {"n": len(ordered)}
    if not ordered:
        return result
    result.update(min=ordered[0], max=ordered[-1])
    for name, percent in (("p50", 50), ("p90", 90), ("p99", 99), ("p99_9", 99.9)):
        result[name] = ordered[math.ceil(percent * (len(ordered) - 1) / 100)]
    return result


def recompute_raw(data: dict, raw: bytes) -> dict:
    """Separate implementation of raw counts, centered mapping, and quantiles."""
    count = int(data["slots"])
    if len(raw) != count * RECORD.size:
        raise ValueError("Raw length does not match measured target count")
    pre, post = data["clock_fit_pre"], data["clock_fit_post"]
    ha = int(pre["t_ref"]) + round_away(Fraction(str(pre["b_ns"])))
    hb = int(post["t_ref"]) + round_away(Fraction(str(post["b_ns"])))
    ga, gb = int(pre["g_ref"]), int(post["g_ref"])
    if hb <= ha or gb <= ga:
        raise ValueError("Clock anchors do not advance")
    rate = float(gb - ga) / float(hb - ha)
    deadline = float(data["deadline_us"])
    eps = max(float(pre["eps_ns"]), float(post["eps_ns"])) / 1000
    tick = float(data["globaltimer_tick_ns"]) / 1000
    if not all(math.isfinite(x) and x > 0 for x in (rate, deadline, eps, tick)):
        raise ValueError("Invalid clock calibration, timer tick, or deadline")
    vectors = {key: [] for key in TIMING_KEYS}
    counts = dict(recorded=0, skipped=0, misses=0, executed_late=0, on_time=0)
    near_eps = near_eps_tick = 0
    minimum_margin = math.inf
    previous_target = previous_end = None
    for index, row in enumerate(RECORD.iter_unpack(raw)):
        slot, target, predicted, launch, launch_done, start, end, flags = row
        if slot != index + data["warmup"]:
            raise ValueError(f"Unexpected raw slot index {slot}")
        if previous_target is not None and target - previous_target != round_away(data["period_us"] * 1000):
            raise ValueError(f"Target schedule spacing changed at slot {slot}")
        previous_target = target
        if flags not in (0, 1, 4):
            raise ValueError(f"Invalid flags {flags} at slot {slot}")
        if flags & 1:
            if previous_end is not None and predicted > previous_end:
                raise ValueError(f"Skip is not covered by previous execution at slot {slot}")
            counts["skipped"] += 1
            counts["misses"] += 1
            continue
        if flags != (4 if data["mode"] == "gpu" else 0):
            raise ValueError(f"Launcher flag disagrees with mode at slot {slot}")
        if not start or not end or end < start or launch_done < launch:
            raise ValueError(f"Missing or reversed timestamps at slot {slot}")
        if previous_end is not None and (start < previous_end or predicted <= previous_end):
            raise ValueError(f"Graph overlap or skip-policy violation at slot {slot}")
        previous_end = end
        corrected = ga + round_away(float(target - ha) * rate)
        ns = (start - corrected, start - predicted, predicted - corrected,
              launch - predicted, launch_done - launch, start - launch,
              end - start, end - corrected)
        for key, value in zip(TIMING_KEYS, ns):
            vectors[key].append(value / 1000)
        latency = ns[-1] / 1000
        late = latency > deadline
        counts["recorded"] += 1
        counts["misses"] += int(late)
        counts["executed_late"] += int(late)
        counts["on_time"] += int(not late)
        margin = abs(latency - deadline)
        minimum_margin = min(minimum_margin, margin)
        near_eps += int(margin <= eps)
        near_eps_tick += int(margin <= eps + tick)
    if counts["recorded"] == 0:
        raise ValueError("No executed measurements")
    for key in ("recorded", "skipped", "misses"):
        if counts[key] != data[key]:
            raise ValueError(f"Independent raw {key} differs from summary")
    stats = {key: quantiles(values) for key, values in vectors.items()}
    maximum_difference = 0.0
    for key, result in stats.items():
        for statistic, expected in result.items():
            observed = data[key][statistic]
            difference = abs(float(observed) - expected)
            if not math.isfinite(float(observed)) or difference > 0.000001:
                raise ValueError(f"Independent raw {key}.{statistic}: {expected} != JSON {observed}")
            maximum_difference = max(maximum_difference, difference)
    return {
        "counts": counts, "stats": stats,
        "max_stat_difference_us": maximum_difference,
        "calibration": {
            "pre_eps_us": float(pre["eps_ns"]) / 1000,
            "post_eps_us": float(post["eps_ns"]) / 1000,
            "max_fit_eps_us": eps, "timer_tick_us": tick,
            "within_fit_eps": near_eps, "within_fit_eps_plus_tick": near_eps_tick,
            "minimum_deadline_margin_us": minimum_margin,
            "two_point_rate_ppm": (rate - 1) * 1e6,
            "anchor_baseline_s": (hb - ha) / 1e9,
        },
    }


def endpoint_log(path: Path) -> dict:
    content = path.read_text(encoding="utf-8", errors="replace")
    checks = re.findall(r"Lockstep validation: payload/CB errors (\d+), CRC errors (\d+)", content)
    if len(checks) != 2 or any(a != "0" or b != "0" for a, b in checks):
        raise ValueError("Expected exactly two zero-error endpoint validation log records")
    cb = re.findall(r"Error CBs (\d+), Mismatched CBs (\d+), MismatchedCRC CBs (\d+), Total CBs (\d+)", content)
    tb = re.findall(r"MismatchedCRC TBs (\d+), Total TBs (\d+)", content)
    if len(cb) != 2 or any(any(int(v) for v in row[:3]) or int(row[3]) <= 0 for row in cb):
        raise ValueError("Expected two positive-size, zero-error code-block reference checks")
    if len(tb) != 2 or any(int(errors) or int(total) <= 0 for errors, total in tb):
        raise ValueError("Expected two positive-size, zero-error transport-block CRC checks")
    return {
        "checks": 2, "CBs_each": [int(row[3]) for row in cb],
        "TBs_each": [int(row[1]) for row in tb], "zero_errors": True,
        "realtime_priority_denied": "Failed to set scheduling algo" in content,
        "nvlog_configuration_error": "Failed to load yaml file" in content,
        "process_pid": (lambda ids: int(ids[0]) if ids else None)(
            re.findall(r"CON\[(\d+)\s*\].*\[NVLOG.CPP\] Using", content)),
        "sha256": sha256(path),
    }


def telemetry(path: Path) -> dict:
    root = ET.parse(path).getroot()
    gpu = root.find("gpu")
    if gpu is None:
        raise ValueError("nvidia-smi telemetry XML contains no GPU")
    result = {"sha256": sha256(path), "gpu_uuid": gpu.findtext("uuid"),
              "product_name": gpu.findtext("product_name")}
    for group in ("clocks", "applications_clocks", "max_clocks", "temperature",
                  "clocks_event_reasons", "clocks_throttle_reasons", "gpu_power_readings", "power_readings"):
        node = gpu.find(group)
        if node is not None:
            result[group] = {child.tag: child.text for child in node}
    return result


def epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def continuous_telemetry(path: Path, start_ns: int, end_ns: int, interval_ms: int) -> dict:
    """Separate actual replay-window samples from initialization/calibration."""
    samples, dropped = [], 0
    with path.open(encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            row = {key.strip().split(" [", 1)[0]: value.strip()
                   for key, value in source.items() if key is not None and value is not None}
            try:
                time = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
            except (KeyError, ValueError):
                dropped += 1
                continue
            sample = {"epoch_s": time, "uuid": row.get("uuid"), "pstate": row.get("pstate")}
            for name in ("temperature.gpu", "power.draw", "power.limit", "clocks.current.sm",
                         "clocks.current.memory", "utilization.gpu", "utilization.memory"):
                match = re.match(r"^(-?\d+(?:\.\d+)?)", row.get(name, ""))
                if match:
                    sample[name] = float(match.group(1))
            # NVIDIA 595 reports the new event-reasons name for the throttle alias.
            reason = row.get("clocks_event_reasons.active", row.get("clocks_throttle_reasons.active", ""))
            try:
                sample["clock_event_mask"] = int(reason, 0)
            except ValueError:
                pass
            samples.append(sample)
    if not samples:
        raise ValueError("No usable continuous GPU telemetry samples")
    if any(b["epoch_s"] <= a["epoch_s"] for a, b in zip(samples, samples[1:])):
        raise ValueError("Continuous telemetry timestamps do not increase")
    start, end = start_ns / 1e9, end_ns / 1e9
    active = [sample for sample in samples if start <= sample["epoch_s"] <= end]

    def summarize_samples(rows):
        summary = {"samples": len(rows)}
        if not rows:
            return summary
        summary.update(first_epoch_s=rows[0]["epoch_s"], last_epoch_s=rows[-1]["epoch_s"],
                       gpu_uuids=sorted({r["uuid"] for r in rows if r.get("uuid")}),
                       pstates=sorted({r["pstate"] for r in rows if r.get("pstate")}))
        for name in ("temperature.gpu", "power.draw", "power.limit", "clocks.current.sm",
                     "clocks.current.memory", "utilization.gpu", "utilization.memory"):
            values = [row[name] for row in rows if name in row]
            if values:
                summary[name] = {"min": min(values), "median": statistics.median(values), "max": max(values)}
        masks = [row["clock_event_mask"] for row in rows if "clock_event_mask" in row]
        summary["clock_event_masks_observed"] = [hex(mask) for mask in sorted(set(masks))]
        summary["samples_with_clock_event_bits"] = sum(mask != 0 for mask in masks)
        summary["maximum_sample_gap_ms"] = max((b["epoch_s"] - a["epoch_s"]) * 1000
                                              for a, b in zip(rows, rows[1:])) if len(rows) > 1 else None
        return summary

    result = {"sha256": sha256(path), "timestamp_timezone": "UTC (remote container)",
              "requested_interval_ms": interval_ms, "dropped_rows": dropped,
              "replay_window": summarize_samples(active), "whole_process": summarize_samples(samples),
              "replay_window_start_ns": start_ns, "replay_window_end_ns": end_ns,
              "scope": "Replay window includes warmup and measured targets; whole process also includes setup and calibration. Sampling cannot exclude shorter clock excursions."}
    tolerance = interval_ms / 1000 * 3
    result["replay_window_covered"] = bool(active and active[0]["epoch_s"] - start <= tolerance
                                           and end - active[-1]["epoch_s"] <= tolerance
                                           and (result["replay_window"]["maximum_sample_gap_ms"] or 0) <= interval_ms * 3)
    return result


def adversary_evidence(root: Path, entry: dict, data: dict, receiver_pid=None) -> dict:
    if entry["condition"] == "alone":
        return {"required": False}
    path = artifact(root, entry.get("adversary_json", f"{entry['name']}.adversary.json"))
    timeline = artifact(root, entry.get("adversary_timeline", f"{entry['name']}.adversary.csv"))
    adversary = read_json(path)
    if not adversary.get("ok") or adversary.get("workload") != "sgemm" or adversary.get("duty") != 100:
        raise ValueError("Adversary failed or differs from the requested SGEMM load")
    if adversary.get("units", 0) <= 0 or adversary.get("active_fraction", 0) < 0.8:
        raise ValueError("Adversary did not sustain the requested busy load")
    percentage = adversary.get("mps_active_thread_percentage")
    if entry["condition"] == "mps_sgemm" and str(percentage) != "50":
        raise ValueError("MPS adversary lacks the requested 50% active-thread cap")
    if entry["condition"] == "proc_sgemm" and percentage is not None:
        raise ValueError("Non-MPS adversary unexpectedly has an MPS thread cap")
    origin = epoch(adversary["end_time"]) - adversary["seconds_total"]
    start = data["run_wall_start_ns"] / 1e9
    end = data["run_wall_end_ns"] / 1e9
    if origin > start or epoch(adversary["end_time"]) < end:
        raise ValueError("Adversary measured interval does not cover the receiver run")
    overlaps = []
    previous = 0.0
    with timeline.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            t = float(row["t_s"])
            overlap = min(origin + t, end) - max(origin + previous, start)
            if overlap > 0:
                units = int(row["units"])
                if units <= 0:
                    raise ValueError("Adversary has an empty activity bin overlapping the receiver run")
                overlaps.append({"t_s": t, "units": units, "overlap_s": overlap})
            previous = t
    if not overlaps or origin + previous < end - 0.01:
        raise ValueError("Adversary timeline does not span the receiver run")
    result = {
        "required": True, "units": adversary["units"],
        "active_fraction": adversary["active_fraction"], "mps_percentage": percentage,
        "positive_overlapping_bins": overlaps, "sha256": sha256(path),
        "timeline_sha256": sha256(timeline),
        "scope": "Approximately one-second activity bins; not per-slot isolation evidence.",
    }
    if entry["condition"] == "mps_sgemm":
        server = root / f"mps-block-{entry['block_index']}" / "server.log"
        if not server.exists():
            server = root / "mps_logs" / "server.log"
        result["MPS_server_membership_confirmed"] = False
        if server.exists() and receiver_pid is not None:
            content = server.read_text(encoding="utf-8", errors="replace")
            members = {int(pid) for pid in re.findall(r"Client \{PID: (\d+), Context ID: \d+\} connected", content)}
            result["MPS_server_membership_confirmed"] = (adversary.get("pid") in members and receiver_pid in members)
            result["MPS_server_log_sha256"] = sha256(server)
    return result


def audit_trial(root: Path, entry: dict, experiment: dict, checker=None) -> dict:
    result = {key: entry[key] for key in
              ("name", "case_index", "block_index", "pair_index", "condition", "mode")}
    result.update(valid=False, errors=[], warnings=[])
    try:
        jp, rp = artifact(root, entry["json"]), artifact(root, entry["raw"])
        data = read_json(jp)
        checker = checker or collection_checker()
        result["collection_gate"] = checker(
            jp, rp, entry["mode"], experiment["slots"], experiment["warmup"],
            experiment["period_us"], experiment["deadline_us"])
        result.update(recompute_raw(data, rp.read_bytes()))
        result["hashes"] = {"json": sha256(jp), "raw": sha256(rp)}
        result["settings"] = {key: data[key] for key in SETTING_KEYS}
        run_path = artifact(root, entry["run"])
        run = read_json(run_path)
        if run.get("exit_code") != 0:
            raise ValueError("Run manifest has a nonzero exit code")
        for key in ("mode", "slots", "warmup", "period_us", "deadline_us"):
            if run.get(key) != data[key]:
                raise ValueError(f"Run manifest {key} disagrees with measured case")
        for key in ("name", "case_index", "block_index", "pair_index", "condition", "mode"):
            if key in run and run[key] != entry[key]:
                raise ValueError(f"Run manifest {key} disagrees with trial schedule")
        for key in ("test_vector_sha256", "aerial_commit"):
            if not run.get(key):
                raise ValueError(f"Missing provenance field {key}")
            if experiment.get(key) is not None and run[key] != experiment[key]:
                raise ValueError(f"Trial {key} differs from global experiment provenance")
        result["run_settings"] = {key: run.get(key) for key in RUN_SETTING_KEYS}
        result["hashes"]["run"] = sha256(run_path)
        result["run_start_utc"] = data.get("run_start_utc")
        result["run_end_utc"] = data.get("run_end_utc")
        result["run_wall_start_ns"] = data.get("run_wall_start_ns")
        result["run_wall_end_ns"] = data.get("run_wall_end_ns")
        log_path = artifact(root, entry.get("pusch_log", f"{entry['name']}.pusch.log"))
        result["correctness"] = endpoint_log(log_path)
        if result["correctness"]["realtime_priority_denied"]:
            result["warnings"].append("CPU real-time scheduling request was denied.")
        if result["correctness"]["nvlog_configuration_error"]:
            result["warnings"].append("The receiver logged an nvlog configuration error.")
        scheduler_path = artifact(root, entry.get("scheduler", f"{entry['name']}.scheduler.json"))
        if scheduler_path.exists():
            result["scheduler"] = read_json(scheduler_path)
            if result["scheduler"].get("observed"):
                if result["scheduler"].get("affinity_matches_requested") is not True:
                    raise ValueError("Observed worker CPU affinity differs from requested affinity")
            else:
                result["warnings"].append("Worker scheduling policy could not be observed.")
        else:
            result["warnings"].append("Missing direct worker scheduling-policy observation.")
        result["telemetry"] = {}
        for key, suffix in (("telemetry_before", "before"), ("telemetry_after", "after")):
            path = artifact(root, entry.get(key, f"{entry['name']}.telemetry.{suffix}.xml"))
            if path.exists():
                try:
                    result["telemetry"][suffix] = telemetry(path)
                except (ValueError, ET.ParseError, OSError) as error:
                    result["warnings"].append(f"Unusable {suffix} clock telemetry: {error}")
            else:
                result["warnings"].append(f"Missing {suffix} clock telemetry.")
        uuids = {v["gpu_uuid"] for v in result["telemetry"].values() if v.get("gpu_uuid")}
        if len(uuids) > 1:
            raise ValueError("GPU UUID changed between telemetry snapshots")
        continuous_path = artifact(root, entry.get("telemetry_continuous", f"{entry['name']}.telemetry.continuous.csv"))
        if continuous_path.exists():
            try:
                result["telemetry_continuous"] = continuous_telemetry(
                    continuous_path, data["run_wall_start_ns"], data["run_wall_end_ns"], experiment.get("telemetry_interval_ms", 100))
                if not result["telemetry_continuous"]["replay_window_covered"]:
                    result["warnings"].append("Continuous clock telemetry has insufficient replay-window coverage or long gaps.")
            except (ValueError, OSError, TypeError, KeyError) as error:
                result["warnings"].append(f"Unusable continuous clock telemetry: {error}")
        else:
            result["warnings"].append("Missing continuous GPU clock telemetry.")
        result["adversary"] = adversary_evidence(root, entry, data, result["correctness"]["process_pid"])
        if entry["condition"] == "mps_sgemm" and not result["adversary"]["MPS_server_membership_confirmed"]:
            raise ValueError("MPS server membership was not confirmed for both PHY and adversary processes")
        c, s = result["counts"], result["stats"]
        result["metrics"] = {
            "miss_percentage": 100 * c["misses"] / experiment["slots"],
            "p99_start_us": s["start_error_us"]["p99"],
            "p99_phy_us": s["exec_us"]["p99"],
            "p99_completion_us": s["latency_from_target_us"]["p99"],
            "executed": c["recorded"], "skipped": c["skipped"],
            "executed_late": c["executed_late"], "on_time": c["on_time"],
        }
        if result["calibration"]["within_fit_eps_plus_tick"]:
            result["warnings"].append("Some deadline classifications are near the empirical clock-fit bound plus timer tick.")
        result["valid"] = True
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        result["errors"].append(str(error))
    return result


def across_trials(values: list[float]) -> dict:
    return {"n_trials": len(values), "min": min(values),
            "median": statistics.median(values), "max": max(values),
            "mean": statistics.fmean(values)}


def linear_quantile(ordered: list[float], probability: float) -> float:
    index = probability * (len(ordered) - 1)
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def bootstrap_pairs(differences: list[float], seed: int, draws: int = 20000) -> dict:
    if len(differences) < 2 or draws < 100:
        raise ValueError("Paired bootstrap requires at least two pairs and 100 draws")
    rng = random.Random(seed)
    n = len(differences)
    distribution = sorted(statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
                          for _ in range(draws))
    return {
        "estimand": "mean paired GPU-minus-CPU difference",
        "resampling_unit": "whole trial pair", "n_pairs": n,
        "estimate": statistics.fmean(differences),
        "lower": linear_quantile(distribution, 0.025),
        "upper": linear_quantile(distribution, 0.975),
        "confidence_level": 0.95, "draws": draws, "seed": seed,
        "method": "exploratory paired percentile bootstrap; no multiplicity correction",
        "warning": "Few trial pairs on one host; interval omits within-trial quantile and clock-calibration uncertainty.",
    }


def summarize(trials: list[dict], repeats: int, seed: int, draws: int = 20000) -> dict:
    modes, pairs, effects, errors = [], [], [], []
    for condition_index, condition in enumerate(CONDITIONS):
        for mode in ("cpu", "gpu"):
            members = [t for t in trials if t["condition"] == condition and t["mode"] == mode and t["valid"]]
            if len(members) != repeats:
                errors.append(f"{condition}/{mode}: {len(members)} valid trials, expected {repeats}")
            if members:
                modes.append({"condition": condition, "mode": mode,
                              "metrics": {key: across_trials([t["metrics"][key] for t in members])
                                          for key in TRIAL_METRICS}})
        condition_pairs = []
        for pair_index in range(1, repeats + 1):
            members = [t for t in trials if t["condition"] == condition and t["pair_index"] == pair_index]
            pair = {"condition": condition, "pair_index": pair_index, "valid": False, "errors": []}
            cpu = [t for t in members if t["mode"] == "cpu"]
            gpu = [t for t in members if t["mode"] == "gpu"]
            if len(cpu) != 1 or len(gpu) != 1 or not all(t["valid"] for t in members):
                pair["errors"].append("Pair is incomplete, duplicated, or contains a failed audit")
            else:
                c, g = cpu[0], gpu[0]
                pair.update(cpu=c["name"], gpu=g["name"],
                            order="CPU then GPU" if c["case_index"] < g["case_index"] else "GPU then CPU")
                if c["block_index"] != g["block_index"] or abs(c["case_index"] - g["case_index"]) != 1:
                    pair["errors"].append("CPU/GPU trials are not adjacent members of the same block")
                for section in ("settings", "run_settings"):
                    for key in c[section]:
                        if c[section][key] != g[section].get(key):
                            pair["errors"].append(f"CPU/GPU {section}.{key} differ")
                if all(t.get("scheduler", {}).get("observed") for t in (c, g)):
                    for key in ("policy", "rt_priority", "affinity", "nice"):
                        if c["scheduler"].get(key) != g["scheduler"].get(key):
                            pair["errors"].append(f"CPU/GPU observed worker scheduler.{key} differ")
                cu = c.get("telemetry", {}).get("before", {}).get("gpu_uuid")
                gu = g.get("telemetry", {}).get("before", {}).get("gpu_uuid")
                if cu and gu and cu != gu:
                    pair["errors"].append("CPU/GPU hardware UUIDs differ")
                pair["valid"] = not pair["errors"]
                if pair["valid"]:
                    pair["differences"] = {key: g["metrics"][key] - c["metrics"][key]
                                           for key in TRIAL_METRICS}
                    pair["calibration_comparison_scale_us"] = sum(
                        t["calibration"]["max_fit_eps_us"] + t["calibration"]["timer_tick_us"]
                        for t in (c, g))
                    pair["p99_start_difference_comparable_to_clock_scale"] = (
                        abs(pair["differences"]["p99_start_us"]) <= pair["calibration_comparison_scale_us"])
                    condition_pairs.append(pair)
            pairs.append(pair)
            errors.extend(f"{condition}/pair{pair_index}: {error}" for error in pair["errors"])
        if len(condition_pairs) == repeats:
            metrics = {}
            for metric_index, metric in enumerate(TRIAL_METRICS):
                values = [pair["differences"][metric] for pair in condition_pairs]
                metrics[metric] = {**across_trials(values), "differences": values}
                if repeats >= 2:
                    metrics[metric]["exploratory_95pct_ci"] = bootstrap_pairs(
                        values, seed + condition_index * 100 + metric_index, draws)
            effects.append({"condition": condition, "direction": "GPU minus CPU",
                            "unit_for_miss_percentage": "percentage points", "metrics": metrics})
    return {"mode_summaries": modes, "pairs": pairs, "paired_effects": effects, "errors": errors}


def planned_order(seed: int, repeats: int) -> list[tuple]:
    """Independently replay the documented balanced runner randomization."""
    if repeats % 2:
        raise ValueError("Balanced launcher order requires an even number of pairs")
    rng = random.Random(seed)
    first_modes = {}
    for condition in CONDITIONS:
        first_modes[condition] = ["cpu", "gpu"] * (repeats // 2)
        rng.shuffle(first_modes[condition])
    expected, block = [], 0
    for pair in range(1, repeats + 1):
        conditions = list(CONDITIONS)
        rng.shuffle(conditions)
        for condition in conditions:
            block += 1
            first = first_modes[condition][pair - 1]
            for mode in (first, "gpu" if first == "cpu" else "cpu"):
                expected.append((block, pair, condition, mode))
    return expected


def analyze(manifest: Path, draws: int = 20000) -> dict:
    experiment = read_json(manifest)
    root = manifest.parent
    schedule = experiment["schedule"]
    repeats = int(experiment["repeats"])
    errors, warnings = [], []
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if len(schedule) != repeats * len(CONDITIONS) * 2:
        errors.append("Schedule does not contain repeats × three conditions × two launchers")
    if [row["case_index"] for row in schedule] != list(range(1, len(schedule) + 1)):
        errors.append("Schedule case indices are not unique and sequential")
    names = [row["name"] for row in schedule]
    if len(set(names)) != len(names):
        errors.append("Duplicate case names in the schedule")
    randomization_verified = False
    if "balanced" in experiment.get("randomization", ""):
        expected = planned_order(int(experiment["seed"]), repeats)
        actual = [(row["block_index"], row["pair_index"], row["condition"], row["mode"]) for row in schedule]
        randomization_verified = actual == expected
        if not randomization_verified:
            errors.append("Actual schedule disagrees with the seeded balanced randomization")
    for row in schedule:
        if row["condition"] not in CONDITIONS or row["mode"] not in ("cpu", "gpu"):
            raise ValueError("Unexpected scheduled condition or launcher")
    checker = collection_checker()
    trials = [audit_trial(root, entry, experiment, checker) for entry in schedule]
    for trial in trials:
        errors.extend(f"{trial['name']}: {message}" for message in trial["errors"])
    valid_trials = [trial for trial in trials if trial["valid"]]
    for first, second in zip(valid_trials, valid_trials[1:]):
        if first.get("run_wall_end_ns") and second.get("run_wall_start_ns") and first["run_wall_end_ns"] > second["run_wall_start_ns"]:
            errors.append(f"Actual run chronology overlaps or disagrees with schedule: {first['name']} / {second['name']}")
    if valid_trials:
        reference = valid_trials[0]["settings"]
        for trial in valid_trials[1:]:
            for key in SETTING_KEYS:
                if trial["settings"][key] != reference[key]:
                    errors.append(f"Experiment-wide setting {key} changed in {trial['name']}")
    summary = summarize(trials, repeats, int(experiment["seed"]) + 7304, draws)
    errors.extend(summary.pop("errors"))
    clocks_locked = experiment.get("clocks_locked")
    if clocks_locked is not True:
        warnings.append("GPU clocks were not locked." if clocks_locked is False else
                        "GPU clock-lock state is unknown; no affirmative control evidence supplied.")
    if any(not t.get("telemetry") or len(t["telemetry"]) < 2 for t in trials):
        warnings.append("At least one trial is missing usable before/after GPU telemetry.")
    if any(not t.get("telemetry_continuous") for t in trials):
        warnings.append("At least one trial is missing usable continuous GPU clock telemetry.")
    warnings.append("Before/after snapshots and sampled continuous telemetry do not exclude shorter clock excursions.")
    warnings.append("Clock-fit epsilon and timer tick are empirical measurement scales, not statistical confidence intervals.")
    return {
        "valid": not errors, "experiment": experiment,
        "randomization_verified": randomization_verified,
        "manifest_sha256": sha256(manifest), "trials": trials, **summary,
        "errors": errors, "warnings": warnings,
        "method": {
            "within_trial_quantile": "ceil(p*(n-1)/100), executed measurements only",
            "miss_denominator": "all scheduled measured boundaries, including skips",
            "across_trial_statistics": "min, median, max and mean of individual trial metrics; no pooling",
            "bootstrap": "Resample complete CPU/GPU trial pairs and calculate their mean GPU-minus-CPU difference.",
            "bootstrap_draws": draws,
            "inference": f"Exploratory: {repeats} short paired trials per condition on one host do not establish production behavior.",
            "correctness": "Two endpoint checks per trial; intermediate outputs are not checked individually.",
            "scope": "Fixed-vector NVIDIA cuPHY PUSCH replay; excludes live RAN ingress, changing setup, and live HARQ.",
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "cuphy_repeats.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = []
    for trial in result["trials"]:
        row = {key: trial[key] for key in ("name", "condition", "pair_index", "case_index", "mode", "valid")}
        row.update(trial.get("metrics", {}))
        row.update({key: trial.get("calibration", {}).get(key) for key in
                    ("max_fit_eps_us", "timer_tick_us", "within_fit_eps", "within_fit_eps_plus_tick")})
        row["errors"] = " | ".join(trial["errors"])
        row["warnings"] = " | ".join(trial["warnings"])
        rows.append(row)
    write_csv(output / "trials.csv", rows)
    write_csv(output / "mode_summary.csv", [
        {"condition": group["condition"], "mode": group["mode"], "metric": key, **value}
        for group in result["mode_summaries"] for key, value in group["metrics"].items()])
    write_csv(output / "paired_trials.csv", [
        {"condition": pair["condition"], "pair_index": pair["pair_index"], "valid": pair["valid"],
         "order": pair.get("order"), **pair.get("differences", {}), "errors": " | ".join(pair["errors"])}
        for pair in result["pairs"]])
    rows = []
    for condition in result["paired_effects"]:
        for metric, value in condition["metrics"].items():
            ci = value.get("exploratory_95pct_ci", {})
            rows.append({"condition": condition["condition"], "metric": metric, "direction": "GPU minus CPU",
                         "n_pairs": value["n_trials"], "min": value["min"], "median": value["median"],
                         "max": value["max"], "mean": value["mean"],
                         "exploratory_mean_ci_lower": ci.get("lower"),
                         "exploratory_mean_ci_upper": ci.get("upper"),
                         "units": "percentage points" if metric == "miss_percentage" else "µs" if metric.endswith("_us") else "boundaries"})
    write_csv(output / "paired_summary.csv", rows)


def plots(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11})
    names = {"alone": "Idle", "proc_sgemm": "Separate SGEMM", "mps_sgemm": "SGEMM + MPS"}
    for metric, ylabel, filename in (
        ("miss_percentage", "Deadline misses (% of all targets)", "repeated_deadline_misses"),
        ("p99_start_us", "p99 start error (µs; executed targets)", "repeated_p99_start"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=metric == "miss_percentage")
        for ax, condition in zip(axes, CONDITIONS):
            for pair in result["pairs"]:
                if pair["condition"] != condition or not pair["valid"]:
                    continue
                c = next(t for t in result["trials"] if t["name"] == pair["cpu"])
                g = next(t for t in result["trials"] if t["name"] == pair["gpu"])
                values = [c["metrics"][metric], g["metrics"][metric]]
                ax.plot([0, 1], values, color="#AEB8C6", linewidth=1, zorder=1)
                ax.scatter([0, 1], values, c=["#D66A43", "#087F8C"], s=45, zorder=2)
            ax.set_xticks([0, 1], ["CPU launcher", "GPU launcher"])
            ax.set_xlim(-0.3, 1.3)
            ax.set_title(names[condition])
            ax.grid(axis="y", color="#DEE2E6")
            ax.spines[["top", "right"]].set_visible(False)
            if metric == "miss_percentage":
                ax.set_ylim(-3, 103)
        axes[0].set_ylabel(ylabel)
        fig.text(.5, .02, "One point per trial; lines join randomized CPU/GPU pairs. Single-host exploratory experiment.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .07, 1, 1))
        fig.savefig(output / f"{filename}.png", dpi=200)
        fig.savefig(output / f"{filename}.svg")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="experiment.json or directory containing it")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    manifest = args.input / "experiment.json" if args.input.is_dir() else args.input
    if not manifest.exists() and args.input.is_dir():
        manifest = args.input / "out" / "experiment.json"
    try:
        if args.bootstrap_draws < 100:
            raise ValueError("At least 100 bootstrap draws are required")
        result = analyze(manifest, args.bootstrap_draws)
        write_outputs(result, args.output_dir)
        if args.plots:
            plots(result, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        parser.exit(2, f"cuPHY repeated-trial analysis failed: {error}\n")
    print(json.dumps({"valid": result["valid"], "trials": len(result["trials"]),
                      "pairs": len(result["pairs"]), "errors": result["errors"],
                      "output_dir": str(args.output_dir)}))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
