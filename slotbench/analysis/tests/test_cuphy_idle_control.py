"""Keepalive coverage and separately paired three-arm activity-control statistics."""
import itertools

import pytest

from analysis import cuphy_idle_control as control
from analysis import cuphy_repeats as base


def keepalive_fixture():
    raw = base.RECORD.pack(1000, 1_000_000, 10_000_000, 10_000_001,
                           10_000_002, 10_000_003, 10_000_100, 0)
    data = {"mode": "cpu", "keepalive_active": True, "stream_priority": -2,
            "keepalive": {"requested": True, "blocks": 1, "threads_per_block": 1,
                          "stop_reason": "host_stop", "covers_replay": True,
                          "stream_nonblocking": True, "stream_priority": -2, "stop_poll_interval_ns": 100000,
                          "start_gpu_ns": 9_000_000, "end_gpu_ns": 11_000_000,
                          "watchdog_deadline_gpu_ns": 20_000_000}}
    return data, raw


def test_keepalive_measured_coverage_is_checked_independently():
    data, raw = keepalive_fixture()
    assert control.validate_keepalive(data, raw, "cpu_keepalive")["measured_window_independently_covered"]
    data["keepalive"]["end_gpu_ns"] = 10_000_099
    with pytest.raises(ValueError, match="do not cover"):
        control.validate_keepalive(data, raw, "cpu_keepalive")


@pytest.mark.parametrize("field,value", [("stop_reason", "watchdog"), ("covers_replay", False),
                                         ("blocks", 2), ("threads_per_block", 32)])
def test_keepalive_configuration_or_termination_mismatch_is_rejected(field, value):
    data, raw = keepalive_fixture()
    data["keepalive"][field] = value
    with pytest.raises(ValueError, match=field):
        control.validate_keepalive(data, raw, "cpu_keepalive")


def test_keepalive_does_not_relabel_cpu_as_gpu():
    data, raw = keepalive_fixture()
    with pytest.raises(ValueError, match="activity disagrees"):
        control.validate_keepalive(data, raw, "cpu")
    data["mode"] = "gpu"
    with pytest.raises(ValueError, match="retain the CPU launcher"):
        control.validate_keepalive(data, raw, "cpu_keepalive")


def test_each_variant_order_occurs_once():
    orders = control.expected_orders(20261003)
    assert sorted(orders) == sorted(itertools.permutations(control.VARIANTS))
    assert orders == control.expected_orders(20261003)
    for position in range(3):
        for variant in control.VARIANTS:
            assert sum(order[position] == variant for order in orders) == 2


def fixture_trials():
    trials = []
    for index, order in enumerate(control.expected_orders(20261003), 1):
        for position, variant in enumerate(order, 1):
            value = index ** 2 + {"cpu": 0, "gpu": -8, "cpu_keepalive": -3}[variant]
            trials.append({"variant": variant, "triplet_index": index, "position": position,
                           "name": f"r{index}_{variant}", "case_index": 3 * index - 3 + position,
                           "valid": True, "settings": {"slots": 5000}, "run_settings": {"command": ["same"]},
                           "metrics": {"p99_start_us": value, "miss_percentage": 0}})
    return trials


def test_contrasts_resample_triplets_and_preserve_underlying_variants():
    result = control.summarize_triplets(fixture_trials(), 6, 20261003, 1000)
    assert not result["errors"]
    assert len(result["triplets"]) == 6
    assert len(result["contrasts"]) == 3
    values = {item["contrast"]: item["metrics"]["p99_start_us"] for item in result["contrasts"]}
    assert values["gpu_minus_cpu"]["mean"] == -8
    assert values["keepalive_minus_cpu"]["mean"] == -3
    assert values["gpu_minus_keepalive"]["mean"] == -5
    for summary in values.values():
        ci = summary["exploratory_95pct_ci"]
        assert ci["resampling_unit"] == "whole triplet"
        assert ci["n_triplets"] == 6
        assert ci["lower"] == ci["upper"] == ci["estimate"]
    groups = {item["variant"]: item for item in result["variant_summaries"]}
    assert groups["cpu_keepalive"]["mode"] == "cpu"
    assert groups["gpu"]["mode"] == "gpu"


def test_failed_control_trial_prevents_all_contrast_estimates():
    trials = fixture_trials()
    trials[0]["valid"] = False
    result = control.summarize_triplets(trials, 6, 20261003, 100)
    assert result["errors"]
    assert not result["triplets"][0]["valid"]
    assert result["contrasts"] == []


def test_settings_changes_in_a_triplet_are_not_ignored():
    trials = fixture_trials()
    trials[0]["settings"]["slots"] = 3000
    result = control.summarize_triplets(trials, 6, 20261003, 100)
    assert any("settings.slots" in error for error in result["errors"])
    assert not result["contrasts"]
