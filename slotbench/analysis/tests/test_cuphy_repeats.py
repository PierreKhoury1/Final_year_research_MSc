"""Independent archived-trace audits and paired (not slot-pooled) statistics."""
import json
from pathlib import Path
import shutil

import pytest

from analysis import cuphy_repeats as repeats


ARCHIVE = Path(__file__).resolve().parents[2] / "data" / "2026-10-02_a100_cuphy_lockstep"
OUT = ARCHIVE / "out"
CASES = ("cpu_alone", "gpu_alone", "cpu_proc_sgemm", "gpu_proc_sgemm",
         "cpu_mps_sgemm", "gpu_mps_sgemm")


@pytest.mark.parametrize("name", CASES)
def test_independent_raw_recomputation_matches_archived_audit(name):
    data = repeats.read_json(OUT / f"{name}.json")
    actual = repeats.recompute_raw(data, (OUT / f"{name}.bin").read_bytes())
    previous = repeats.read_json(ARCHIVE / "audit.json")["cases"][name]
    for key in ("recorded", "skipped", "misses"):
        assert actual["counts"][key] == previous["counts"][key]
    assert actual["stats"] == previous["stats"]
    assert actual["max_stat_difference_us"] == 0
    assert actual["calibration"]["within_fit_eps"] == previous["fit"]["within_fit_eps"]
    assert actual["calibration"]["within_fit_eps_plus_tick"] == previous["fit"]["within_eps_plus_tick"]


def test_tampered_quantile_fails_independent_audit_when_collection_gate_passes(tmp_path):
    data = repeats.read_json(OUT / "cpu_alone.json")
    data["start_error_us"]["p99"] += 1
    changed = tmp_path / "tampered.json"
    changed.write_text(json.dumps(data), encoding="utf-8")
    assert repeats.collection_checker()(changed, OUT / "cpu_alone.bin", "cpu", 1000, 100)["misses"] == 0
    with pytest.raises(ValueError, match=r"start_error_us.p99"):
        repeats.recompute_raw(data, (OUT / "cpu_alone.bin").read_bytes())


def test_invalid_skip_policy_is_detected():
    name = "gpu_proc_sgemm"
    data = repeats.read_json(OUT / f"{name}.json")
    rows = list(repeats.RECORD.iter_unpack((OUT / f"{name}.bin").read_bytes()))
    previous_end = None
    for index, row in enumerate(rows):
        if row[-1] & 1 and previous_end is not None:
            changed = list(row)
            changed[2] = previous_end + 1
            rows[index] = tuple(changed)
            break
        if not row[-1] & 1:
            previous_end = row[6]
    raw = b"".join(repeats.RECORD.pack(*row) for row in rows)
    with pytest.raises(ValueError, match="Skip is not covered"):
        repeats.recompute_raw(data, raw)


def test_endpoint_checks_reject_nonzero_crc(tmp_path):
    content = (OUT / "cpu_alone.pusch.log").read_text(encoding="utf-8")
    changed = tmp_path / "case.log"
    changed.write_text(content.replace("CRC errors 0", "CRC errors 1", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="endpoint validation"):
        repeats.endpoint_log(changed)


def test_paired_bootstrap_resamples_whole_differences_and_is_reproducible():
    constant = repeats.bootstrap_pairs([-3.0] * 6, seed=73, draws=1000)
    assert constant["n_pairs"] == 6
    assert constant["lower"] == constant["upper"] == constant["estimate"] == -3
    values = [-10, -2, 0, 1, 5, 12]
    first = repeats.bootstrap_pairs(values, seed=73, draws=1000)
    assert first == repeats.bootstrap_pairs(values, seed=73, draws=1000)
    assert min(values) <= first["lower"] <= first["estimate"] <= first["upper"] <= max(values)
    assert first["resampling_unit"] == "whole trial pair"


def mock_trials(n=6):
    trials = []
    for condition_index, condition in enumerate(repeats.CONDITIONS):
        for pair_index in range(1, n + 1):
            block = condition_index * n + pair_index
            for position, mode in enumerate(("cpu", "gpu") if pair_index % 2 else ("gpu", "cpu")):
                metric = pair_index ** 2 + (2 if mode == "gpu" else 0)
                trials.append({
                    "name": f"{condition}_{pair_index}_{mode}", "condition": condition,
                    "pair_index": pair_index, "block_index": block,
                    "case_index": 2 * block - 1 + position, "mode": mode, "valid": True,
                    "settings": {"slots": 5000}, "run_settings": {"test_vector_sha256": "equal"},
                    "metrics": {key: metric for key in repeats.TRIAL_METRICS},
                    "calibration": {"max_fit_eps_us": 2, "timer_tick_us": 1.024},
                })
    return trials


def test_across_trial_statistics_preserve_pairs_without_pooling():
    result = repeats.summarize(mock_trials(), repeats=6, seed=73, draws=1000)
    assert not result["errors"]
    assert len(result["paired_effects"]) == 3
    assert len(result["pairs"]) == 18
    for group in result["paired_effects"]:
        for metric in group["metrics"].values():
            assert metric["n_trials"] == 6
            assert metric["min"] == metric["median"] == metric["max"] == 2
            assert metric["exploratory_95pct_ci"]["estimate"] == 2
    cpu_idle = result["mode_summaries"][0]["metrics"]["p99_start_us"]
    assert cpu_idle["median"] == (9 + 16) / 2
    assert cpu_idle["n_trials"] == 6


def test_mismatched_pair_settings_prevent_condition_effect_estimate():
    trials = mock_trials()
    trials[1]["settings"] = {"slots": 3000}
    result = repeats.summarize(trials, repeats=6, seed=73, draws=100)
    assert any("settings.slots differ" in error for error in result["errors"])
    assert "alone" not in [condition["condition"] for condition in result["paired_effects"]]


def test_missing_pair_is_reported_without_silently_changing_replicate_count():
    trials = mock_trials()
    trials.pop(0)
    result = repeats.summarize(trials, repeats=6, seed=73, draws=100)
    assert any("5 valid trials, expected 6" in error for error in result["errors"])
    assert not result["pairs"][0]["valid"]


def test_full_analysis_and_csv_outputs_with_archived_cases(tmp_path):
    data_dir = tmp_path / "data"
    shutil.copytree(OUT, data_dir)
    schedule = []
    for index, name in enumerate(CASES):
        mode, condition = name.split("_", 1)
        schedule.append({"name": name, "case_index": index + 1, "block_index": index // 2 + 1,
                         "pair_index": 1, "condition": condition, "mode": mode,
                         "json": f"{name}.json", "raw": f"{name}.bin", "run": f"{name}.run.json"})
    experiment = {"seed": 73, "repeats": 1, "slots": 1000, "warmup": 100,
                  "period_us": 500, "deadline_us": 500, "clocks_locked": False, "schedule": schedule}
    manifest = data_dir / "experiment.json"
    manifest.write_text(json.dumps(experiment), encoding="utf-8")
    result = repeats.analyze(manifest, draws=100)
    assert result["valid"], result["errors"]
    assert len(result["trials"]) == 6
    assert len(result["pairs"]) == 3
    assert any("not locked" in warning for warning in result["warnings"])
    assert any("missing usable" in warning for warning in result["warnings"])
    assert result["paired_effects"][1]["metrics"]["miss_percentage"]["median"] == -40.5
    output = tmp_path / "analysis"
    repeats.write_outputs(result, output)
    assert {p.name for p in output.iterdir()} == {
        "cuphy_repeats.json", "trials.csv", "mode_summary.csv", "paired_trials.csv", "paired_summary.csv"}
    assert repeats.read_json(output / "cuphy_repeats.json")["valid"] is True


def test_artifact_paths_cannot_escape_the_collection(tmp_path):
    with pytest.raises(ValueError, match="leaves"):
        repeats.artifact(tmp_path, "../unrelated.json")


def test_continuous_telemetry_excludes_setup_and_calibration_samples(tmp_path):
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "timestamp, uuid, pstate, clocks.current.sm [MHz], power.draw [W], clocks_event_reasons.active\n"
        "2026/10/02 20:00:00.000, GPU-1, P2, 210 MHz, 35 W, 0x0\n"
        "2026/10/02 20:00:00.100, GPU-1, P0, 1100 MHz, 260 W, 0x0\n"
        "2026/10/02 20:00:00.200, GPU-1, P0, 1200 MHz, 270 W, 0x4\n"
        "2026/10/02 20:00:00.300, GPU-1, P2, 300 MHz, 40 W, 0x0\n",
        encoding="utf-8")
    origin_ns = int(repeats.epoch("2026-10-02T20:00:00Z") * 1e9)
    result = repeats.continuous_telemetry(telemetry, origin_ns + 100_000_000,
                                          origin_ns + 200_000_000, 100)
    assert result["replay_window"]["samples"] == 2
    assert result["whole_process"]["samples"] == 4
    assert result["replay_window"]["clocks.current.sm"]["min"] == 1100
    assert result["whole_process"]["clocks.current.sm"]["min"] == 210
    assert result["replay_window"]["samples_with_clock_event_bits"] == 1
    assert result["replay_window_covered"] is True


def test_continuous_telemetry_reports_gaps(tmp_path):
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text("timestamp,uuid\n2026/10/02 20:00:00.000,GPU-1\n"
                         "2026/10/02 20:00:01.000,GPU-1\n", encoding="utf-8")
    origin_ns = int(repeats.epoch("2026-10-02T20:00:00Z") * 1e9)
    result = repeats.continuous_telemetry(telemetry, origin_ns, origin_ns + 1_000_000_000, 100)
    assert result["replay_window_covered"] is False
    assert result["replay_window"]["maximum_sample_gap_ms"] == 1000


def test_seeded_order_is_balanced_and_reproducible():
    order = repeats.planned_order(7304, 6)
    assert len(order) == 36
    assert order == repeats.planned_order(7304, 6)
    assert order != repeats.planned_order(7305, 6)
    for condition in repeats.CONDITIONS:
        first_modes = [row[3] for index, row in enumerate(order) if index % 2 == 0 and row[2] == condition]
        assert first_modes.count("cpu") == first_modes.count("gpu") == 3
