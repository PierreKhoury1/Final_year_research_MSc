"""Randomized cuPHY orchestration tests; no CUDA, cloud or driver mutations."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

import pytest

CLOUD = Path(__file__).resolve().parents[1]
SCRIPT = CLOUD / "repeat_cuphy_lockstep.sh"


def shell(command, env, timeout=15):
    return subprocess.run(["bash", "-c", f"source {shlex.quote(str(SCRIPT))}\n{command}"],
                          env=dict(os.environ, **env), capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def environment(tmp_path):
    out, logs = tmp_path / "out-repeat-test", tmp_path / "logs-repeat-test"
    out.mkdir()
    logs.mkdir()
    return dict(W=str(tmp_path), OUT=str(out), LOGS=str(logs), REPEAT_SEED="20261002",
                REPEATS="6", SLOTS="5000", WARMUP="1000", PERIOD="500", DEADLINE="500",
                TV_SHA="verified-vector", RUN_ID="test", OBSERVER_CPU="1")


def read_plan(env):
    return json.loads((Path(env["OUT"]) / "experiment.json").read_text())


def test_source_and_syntax_do_not_run_experiment(tmp_path):
    syntax = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    result = shell("declare -F repeat_main repeat_schedule repeat_plan", dict(SB_CUPHY_WORKDIR=str(tmp_path)))
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_plan_has_balanced_adjacent_pairs_and_exact_reproducibility(environment, tmp_path):
    result = shell("repeat_plan", environment)
    assert result.returncode == 0, result.stderr
    plan = read_plan(environment)
    assert len(plan["schedule"]) == 36
    assert plan["slots"] == 5000 and plan["warmup"] == 1000
    assert plan["clocks_locked"] is None
    assert len({r["name"] for r in plan["schedule"]}) == 36
    first = {c: [] for c in ("alone", "proc_sgemm", "mps_sgemm")}
    for a, b in zip(plan["schedule"][::2], plan["schedule"][1::2]):
        assert (a["block_index"], a["pair_index"], a["condition"]) == (b["block_index"], b["pair_index"], b["condition"])
        assert {a["mode"], b["mode"]} == {"cpu", "gpu"}
        first[a["condition"]].append(a["mode"])
    assert all(modes.count("cpu") == modes.count("gpu") == 3 for modes in first.values())
    for pair in range(1, 7):
        assert {r["condition"] for r in plan["schedule"] if r["pair_index"] == pair} == set(first)
    second = tmp_path / "second"
    second.mkdir()
    other = dict(environment, OUT=str(second))
    assert shell("repeat_plan", other).returncode == 0
    assert read_plan(other) == plan
    different = tmp_path / "different"
    different.mkdir()
    other = dict(environment, OUT=str(different), REPEAT_SEED="20261003")
    assert shell("repeat_plan", other).returncode == 0
    assert read_plan(other)["schedule"] != plan["schedule"]


def test_plan_refuses_overwrite_and_preserves_clock_evidence(environment):
    out = Path(environment["OUT"])
    evidence = dict(clocks_locked=True, command=["nvidia-smi", "-lgc", "1110,1110"], result="success")
    (out / "clock_control.json").write_text(json.dumps(evidence))
    assert shell("repeat_plan", environment).returncode == 0
    before = (out / "experiment.json").read_bytes()
    assert read_plan(environment)["clock_control"] == evidence
    result = shell("repeat_plan", environment)
    assert result.returncode != 0
    assert (out / "experiment.json").read_bytes() == before


@pytest.mark.parametrize("pgrep_status,expected", [(0, False), (1, True), (2, False)])
def test_mps_clean_gate_distinguishes_present_absent_and_scan_error(environment, pgrep_status, expected):
    result = shell(f"pgrep() {{ return {pgrep_status}; }}\nrepeat_no_mps", environment)
    assert (result.returncode == 0) == expected


def test_schedule_cleans_mps_between_blocks_and_stops_on_failed_case(environment):
    result = shell('''
repeat_plan
MPS_ON=0
repeat_no_mps() { [[ $MPS_ON == 0 ]]; }
repeat_start_mps() { [[ $MPS_ON == 0 ]]; MPS_ON=1; echo start >> "$OUT/events"; }
repeat_stop_mps() { if [[ $MPS_ON == 1 ]]; then echo stop >> "$OUT/events"; MPS_ON=0; fi; }
repeat_case() {
    if [[ $3 == mps ]]; then [[ $MPS_ON == 1 ]]; else [[ $MPS_ON == 0 ]]; fi
    printf '%s %s %s %s\n' "$@" >> "$OUT/cases"
}
repeat_schedule
[[ $MPS_ON == 0 ]]
''', environment)
    assert result.returncode == 0, result.stderr
    out = Path(environment["OUT"])
    assert (out / "events").read_text().splitlines() == ["start", "stop"] * 6
    assert len((out / "cases").read_text().splitlines()) == 36
    result = shell('''
repeat_no_mps() { return 0; }
repeat_start_mps() { return 0; }
repeat_stop_mps() { return 0; }
repeat_case() { echo "$1" >> "$OUT/failure-cases"; return 7; }
repeat_schedule
touch "$OUT/continued-after-failure"
''', environment)
    assert result.returncode == 7
    assert len((out / "failure-cases").read_text().splitlines()) == 1
    assert not (out / "continued-after-failure").exists()


def test_finish_archives_for_ssh_without_base64_and_refuses_overwrite(environment):
    (Path(environment["OUT"]) / "raw.bin").write_bytes(b"raw-timestamps")
    commands = "stop_adversary() { :; }; repeat_stop_mps() { :; }; repeat_no_mps() { :; }; repeat_finish ok"
    result = shell(commands, environment)
    assert result.returncode == 0, result.stderr
    assert "SLOTBENCH-BEGIN" not in result.stdout
    assert "archive=" in result.stdout
    archive = Path(environment["W"]) / "collection-repeat-test.tar.gz"
    before = archive.read_bytes()
    with tarfile.open(archive) as f:
        assert "out-repeat-test/raw.bin" in f.getnames()
        completion = json.load(f.extractfile("out-repeat-test/completion.json"))
        assert completion["status"] == "ok"
    assert shell(commands, environment).returncode != 0
    assert archive.read_bytes() == before


@pytest.mark.parametrize("variable,value", [("SB_CUPHY_CPU", "2;false"), ("SB_ADVERSARY_CPU", "-1")])
def test_affinity_arguments_are_rejected_before_launch(environment, variable, value):
    env = dict(environment, **{variable: value}, TV="unused", PUSCH="/does/not/exist", ADV="/does/not/exist")
    command = "run_case bad cpu none none" if variable == "SB_CUPHY_CPU" else "start_adversary bad none"
    result = shell(command, env)
    assert result.returncode == 2
    assert "Invalid " + variable in result.stdout
    assert not (Path(environment["OUT"]) / "bad.pusch.log").exists()


@pytest.mark.parametrize("observed,affinity,policy,expected", [
    (True, True, 0, True), (False, True, 0, False),
    (True, False, 0, False), (True, True, 2, False)])
def test_case_manifest_requires_actual_matching_scheduler(environment, observed, affinity, policy, expected):
    assert shell("repeat_plan", environment).returncode == 0
    out = Path(environment["OUT"])
    name = read_plan(environment)["schedule"][0]["name"]
    (out / (name + ".run.json")).write_text("{}")
    (out / "gate_cpu_alone.scheduler.json").write_text(json.dumps(dict(observed=True, policy=0, rt_priority=0)))
    (out / (name + ".scheduler.json")).write_text(json.dumps(dict(
        observed=observed, affinity_matches_requested=affinity, policy=policy, rt_priority=0)))
    result = shell(f"repeat_record_case {name}\ntouch \"$OUT/continued\"", environment)
    assert (result.returncode == 0) == expected
    assert (out / "continued").exists() == expected
    assert json.loads((out / (name + ".run.json")).read_text())["name"] == name
