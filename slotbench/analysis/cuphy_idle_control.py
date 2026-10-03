#!/usr/bin/env python3
"""Analyze the separately collected CPU/GPU/CPU-keepalive idle control.

Retain the underlying launcher mode and the explicit experimental variant.
Whole triplets, rather than slots, are the units of exploratory resampling.
This program never pools the activity control with the primary experiment.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
import sys

try:
    from . import cuphy_repeats as base
except ImportError:
    import cuphy_repeats as base


VARIANTS = ("cpu", "gpu", "cpu_keepalive")
CONTRASTS = (
    ("gpu_minus_cpu", "gpu", "cpu"),
    ("keepalive_minus_cpu", "cpu_keepalive", "cpu"),
    ("gpu_minus_keepalive", "gpu", "cpu_keepalive"),
)
EXTRA_METRICS = {
    "p50_start_us": ("start_error_us", "p50"),
    "p50_phy_us": ("exec_us", "p50"),
    "p50_launch_to_start_us": ("launch_to_start_us", "p50"),
    "p99_launch_to_start_us": ("launch_to_start_us", "p99"),
    "p99_launch_call_us": ("launch_call_us", "p99"),
    "p99_launch_error_us": ("launch_error_us", "p99"),
}


def expected_orders(seed: int, repeats: int = 6) -> list[tuple[str, ...]]:
    orders = list(itertools.permutations(VARIANTS)) * (repeats // 6)
    random.Random(seed).shuffle(orders)
    return orders


def validate_keepalive(data: dict, raw: bytes, variant: str) -> dict:
    expected = variant == "cpu_keepalive"
    active = data.get("keepalive_active")
    detail = data.get("keepalive", {})
    if active is not expected:
        raise ValueError("Keepalive activity disagrees with the explicitly named variant")
    if not expected:
        if detail.get("requested", False):
            raise ValueError("Ordinary CPU/GPU trial unexpectedly requested a keepalive")
        return {"required": False, "active": False}
    if data["mode"] != "cpu":
        raise ValueError("Keepalive variant must retain the CPU launcher mode")
    for key, value in (("requested", True), ("blocks", 1), ("threads_per_block", 1),
                       ("stop_reason", "host_stop"), ("covers_replay", True),
                       ("stream_nonblocking", True), ("stream_priority", data["stream_priority"]),
                       ("stop_poll_interval_ns", 100000)):
        if detail.get(key) != value:
            raise ValueError(f"Keepalive {key}: {detail.get(key)!r} != {value!r}")
    rows = list(base.RECORD.iter_unpack(raw))
    executed = [row for row in rows if not row[-1] & 1]
    if not executed:
        raise ValueError("Keepalive control has no completed measurements")
    start = int(detail["start_gpu_ns"])
    end = int(detail["end_gpu_ns"])
    first_target = rows[0][2]
    last_end = max(row[6] for row in executed)
    watchdog = int(detail["watchdog_deadline_gpu_ns"])
    if not 0 < start <= first_target <= last_end <= end < watchdog:
        raise ValueError("Keepalive stamps do not cover the measured target/execution interval before watchdog expiry")
    if detail.get("last_phy_end_gpu_ns") is not None and detail["last_phy_end_gpu_ns"] != last_end:
        raise ValueError("Keepalive's final PHY completion disagrees with raw records")
    return {
        "required": True, "active": True, "detail": detail,
        "measured_window_independently_covered": True,
        "warmup_coverage": "Receiver covers_replay assertion; raw file omits warmup records.",
        "first_measured_target_gpu_ns": first_target, "last_measured_completion_gpu_ns": last_end,
    }


def audit_control_trial(root: Path, entry: dict, experiment: dict, checker) -> dict:
    normalized = dict(entry)
    normalized.setdefault("condition", "alone")
    normalized.setdefault("pair_index", entry["triplet_index"])
    normalized.setdefault("block_index", entry["triplet_index"])
    trial = base.audit_trial(root, normalized, experiment, checker)
    trial.update(variant=entry["variant"], triplet_index=entry["triplet_index"], position=entry["position"])
    if not trial["valid"]:
        return trial
    try:
        data = base.read_json(base.artifact(root, entry["json"]))
        raw = base.artifact(root, entry["raw"]).read_bytes()
        trial["keepalive"] = validate_keepalive(data, raw, entry["variant"])
        for metric, (key, quantile) in EXTRA_METRICS.items():
            trial["metrics"][metric] = trial["stats"][key][quantile]
        clocks = trial.get("telemetry_continuous", {}).get("replay_window", {})
        for metric, key in (("sampled_sm_clock_median_mhz", "clocks.current.sm"),
                            ("sampled_power_median_w", "power.draw")):
            if key in clocks:
                trial["metrics"][metric] = clocks[key]["median"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        trial["valid"] = False
        trial["errors"].append(str(error))
    return trial


def summarize_triplets(trials: list[dict], repeats: int, seed: int, draws: int) -> dict:
    errors, groups, triplets, contrasts = [], [], [], []
    for variant in VARIANTS:
        members = [trial for trial in trials if trial["variant"] == variant and trial["valid"]]
        if len(members) != repeats:
            errors.append(f"{variant}: {len(members)} valid trials, expected {repeats}")
        if members:
            metrics = set.intersection(*(set(member["metrics"]) for member in members))
            groups.append({"variant": variant, "mode": "gpu" if variant == "gpu" else "cpu",
                           "metrics": {key: base.across_trials([member["metrics"][key] for member in members])
                                       for key in sorted(metrics)}})
    for index in range(1, repeats + 1):
        members = [trial for trial in trials if trial["triplet_index"] == index]
        triplet = {"triplet_index": index, "valid": False, "errors": []}
        by_variant = {trial["variant"]: trial for trial in members}
        if len(members) != 3 or set(by_variant) != set(VARIANTS) or not all(trial["valid"] for trial in members):
            triplet["errors"].append("Triplet is incomplete, duplicated, or contains an invalid trial")
        else:
            ordered = sorted(members, key=lambda trial: trial["case_index"])
            triplet["order"] = [trial["variant"] for trial in ordered]
            triplet["cases"] = {variant: trial["name"] for variant, trial in by_variant.items()}
            expected_indices = list(range(3 * index - 2, 3 * index + 1))
            if [trial["case_index"] for trial in ordered] != expected_indices:
                triplet["errors"].append("Triplet cases are not consecutive in the intended round")
            if [trial["position"] for trial in ordered] != [1, 2, 3]:
                triplet["errors"].append("Recorded positions disagree with case order")
            reference = by_variant["cpu"]
            for variant in ("gpu", "cpu_keepalive"):
                other = by_variant[variant]
                for section in ("settings", "run_settings"):
                    for key, value in reference[section].items():
                        if other[section].get(key) != value:
                            triplet["errors"].append(f"{variant}/{section}.{key} differs from CPU reference")
                if all(trial.get("scheduler", {}).get("observed") for trial in (reference, other)):
                    for key in ("policy", "rt_priority", "affinity", "nice"):
                        if reference["scheduler"].get(key) != other["scheduler"].get(key):
                            triplet["errors"].append(f"{variant}/scheduler.{key} differs from CPU reference")
            triplet["valid"] = not triplet["errors"]
            if triplet["valid"]:
                metrics = set.intersection(*(set(member["metrics"]) for member in members))
                triplet["differences"] = {
                    name: {metric: by_variant[a]["metrics"][metric] - by_variant[b]["metrics"][metric]
                           for metric in sorted(metrics)} for name, a, b in CONTRASTS}
        triplets.append(triplet)
        errors.extend(f"triplet{index}: {error}" for error in triplet["errors"])
    if len(triplets) == repeats and all(triplet["valid"] for triplet in triplets):
        for contrast_index, (name, a, b) in enumerate(CONTRASTS):
            metrics = set.intersection(*(set(triplet["differences"][name]) for triplet in triplets))
            result = {"contrast": name, "direction": f"{a} minus {b}", "metrics": {}}
            for metric_index, metric in enumerate(sorted(metrics)):
                values = [triplet["differences"][name][metric] for triplet in triplets]
                bootstrap = base.bootstrap_pairs(values, seed + contrast_index * 100 + metric_index, draws)
                bootstrap.update(estimand=f"mean within-triplet {a}-minus-{b} difference",
                                 resampling_unit="whole triplet", n_triplets=bootstrap.pop("n_pairs"))
                result["metrics"][metric] = {**base.across_trials(values), "differences": values,
                                             "exploratory_95pct_ci": bootstrap}
            contrasts.append(result)
    return {"variant_summaries": groups, "triplets": triplets, "contrasts": contrasts, "errors": errors}


def analyze(manifest: Path, draws: int = 20000) -> dict:
    experiment = base.read_json(manifest)
    schedule = experiment["schedule"]
    repeats = int(experiment["repeats"])
    errors, warnings = [], []
    if repeats % 6 != 0 or len(schedule) != 3 * repeats:
        raise ValueError("This balanced activity-control protocol requires a multiple of six triplets, three cases each")
    if [entry["case_index"] for entry in schedule] != list(range(1, 3 * repeats + 1)):
        errors.append("Schedule case indices are not sequential")
    if len({entry["name"] for entry in schedule}) != 3 * repeats:
        errors.append("Trial names are duplicated")
    for entry in schedule:
        if entry["variant"] not in VARIANTS:
            raise ValueError("Unknown activity-control variant")
        expected_mode = "gpu" if entry["variant"] == "gpu" else "cpu"
        if entry["mode"] != expected_mode or entry.get("condition", "alone") != "alone":
            raise ValueError("Activity variant was mislabeled as a different launcher or load condition")
    actual_orders = [tuple(entry["variant"] for entry in schedule if entry["triplet_index"] == index)
                     for index in range(1, repeats + 1)]
    balanced = sorted(actual_orders) == sorted(list(itertools.permutations(VARIANTS)) * (repeats // 6))
    seeded = actual_orders == expected_orders(int(experiment["seed"]), repeats)
    if not balanced:
        errors.append("Control does not use each of the six variant-order permutations equally often")
    if not seeded:
        errors.append("Control ordering disagrees with the seeded permutation protocol")
    checker = base.collection_checker()
    trials = [audit_control_trial(manifest.parent, entry, experiment, checker) for entry in schedule]
    errors.extend(f"{trial['name']}: {error}" for trial in trials for error in trial["errors"])
    valid = [trial for trial in trials if trial["valid"]]
    for a, b in zip(valid, valid[1:]):
        if a["run_wall_end_ns"] > b["run_wall_start_ns"]:
            errors.append("Observed trial chronology disagrees with planned order")
    summary = summarize_triplets(trials, repeats, int(experiment["seed"]) + 7304, draws)
    errors.extend(summary.pop("errors"))
    if experiment.get("clocks_locked") is not True:
        warnings.append("GPU clocks were not locked; keepalive is an activity control, not a hardware clock lock.")
    warnings.extend([
        "The keepalive occupies one GPU thread/block and can perturb resource use and power; it does not isolate every launcher difference.",
        "Whole triplets are exploratory resampling units; no slots are pooled and no primary-experiment observations are included.",
        "Bootstrap intervals exclude within-trial quantile and clock-calibration uncertainty and are not corrected for multiple contrasts.",
        "Telemetry is sampled at 100 ms; reported replay-window clocks include warmup and measured targets, and can miss shorter excursions.",
        "Correctness checks occur before and after each replay series; intermediate decoded outputs are not individually checked.",
    ])
    return {"valid": not errors, "experiment": experiment, "manifest_sha256": base.sha256(manifest),
            "balanced_order_verified": balanced, "seeded_order_verified": seeded,
            "trials": trials, **summary, "errors": errors, "warnings": warnings,
            "method": {"within_trial_quantile": "ceil(p*(n-1)/100), executed targets only",
                       "miss_denominator": "all measured target boundaries, including skips",
                       "across_trial_statistics": "min, median, max and mean of per-trial metrics",
                       "bootstrap_draws": draws, "resampling_unit": "whole triplet"}}


def write_outputs(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "cuphy_idle_control.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    base.write_csv(output / "control_trials.csv", [
        {**{key: trial[key] for key in ("name", "case_index", "triplet_index", "position", "variant", "mode", "valid")},
         **trial.get("metrics", {}), "errors": " | ".join(trial["errors"]), "warnings": " | ".join(trial["warnings"])}
        for trial in result["trials"]])
    base.write_csv(output / "control_variant_summary.csv", [
        {"variant": group["variant"], "metric": metric, **summary}
        for group in result["variant_summaries"] for metric, summary in group["metrics"].items()])
    rows = []
    for contrast in result["contrasts"]:
        for metric, summary in contrast["metrics"].items():
            ci = summary["exploratory_95pct_ci"]
            rows.append({"contrast": contrast["contrast"], "direction": contrast["direction"], "metric": metric,
                         "n_triplets": summary["n_trials"], **{key: summary[key] for key in ("min", "median", "max", "mean")},
                         "exploratory_mean_ci_lower": ci["lower"], "exploratory_mean_ci_upper": ci["upper"]})
    base.write_csv(output / "control_contrasts.csv", rows)
    base.write_csv(output / "control_triplet_differences.csv", [
        {"triplet_index": triplet["triplet_index"], "contrast": contrast, **values}
        for triplet in result["triplets"] if triplet["valid"]
        for contrast, values in triplet["differences"].items()])


def plots(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11})
    ordered_variants = ("cpu", "cpu_keepalive", "gpu")
    labels = ("CPU", "CPU + keepalive", "GPU")
    colors = ("#D66A43", "#8861A5", "#087F8C")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title, unit in zip(axes,
            ("p99_start_us", "p50_phy_us", "sampled_sm_clock_median_mhz"),
            ("p99 start error", "Median PHY interval", "Sampled median SM clock"), ("µs", "µs", "MHz")):
        for triplet in result["triplets"]:
            if not triplet["valid"]:
                continue
            values = []
            for variant in ordered_variants:
                trial = next(trial for trial in result["trials"] if trial["name"] == triplet["cases"][variant])
                values.append(trial["metrics"].get(metric))
            if any(value is None for value in values):
                continue
            ax.plot(range(3), values, color="#AEB8C6", linewidth=1, zorder=1)
            ax.scatter(range(3), values, color=colors, s=45, zorder=2)
        ax.set_xticks(range(3), labels, rotation=12)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xlim(-.25, 2.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DEE2E6")
    fig.text(.5, .015, "One point per trial; lines join a balanced triplet. Activity control only: no hardware clock lock.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(output / "idle_activity_control.png", dpi=200)
    fig.savefig(output / "idle_activity_control.svg")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    manifest = args.input / "experiment.json" if args.input.is_dir() else args.input
    try:
        if args.bootstrap_draws < 100:
            raise ValueError("At least 100 bootstrap draws are required")
        result = analyze(manifest, args.bootstrap_draws)
        write_outputs(result, args.output_dir)
        if args.plots:
            plots(result, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        parser.exit(2, f"cuPHY idle activity-control analysis failed: {error}\n")
    print(json.dumps({"valid": result["valid"], "trials": len(result["trials"]),
                      "triplets": len(result["triplets"]), "errors": result["errors"], "output_dir": str(args.output_dir)}))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
