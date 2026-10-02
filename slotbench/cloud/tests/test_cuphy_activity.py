"""Activity-control scheduling preserves mode identity and balances order."""
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / "control_cuphy_activity.sh"


def run_shell(tmp_path, command):
    (tmp_path / "cpu_selection.json").write_text(json.dumps({"phy_cpu_requested": 1}))
    env = dict(os.environ, OUT=str(tmp_path), REPEAT_SEED="20261003", SLOTS="5000",
               WARMUP="1000", PERIOD="500", DEADLINE="500", TV_SHA="vector", OBSERVER_CPU="3")
    return subprocess.run(["bash", "-c", f"source {shlex.quote(str(SCRIPT))}\n{command}"],
                          env=env, capture_output=True, text=True, timeout=10)


def test_all_orders_once_with_cpu_identity_preserved(tmp_path):
    result = run_shell(tmp_path, "repeat_plan")
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "experiment.json").read_text())
    assert manifest["experiment_kind"] == "idle_keepalive_control"
    cases = manifest["schedule"]
    assert len(cases) == 18
    orders = {tuple(row["variant"] for row in cases[i:i+3]) for i in range(0, 18, 3)}
    assert orders == set(itertools.permutations(("cpu", "gpu", "cpu_keepalive")))
    for row in cases:
        assert row["condition"] == "alone"
        assert row["mode"] == ("gpu" if row["variant"] == "gpu" else "cpu")
        assert row["pair_index"] == row["triplet_index"] == row["block_index"]


def test_control_flag_only_enables_cpu_keeper_and_gate_is_separate(tmp_path):
    result = run_shell(tmp_path, '''
repeat_plan
repeat_no_mps() { :; }
activity_validate_case() { :; }
repeat_case() { printf '%s %s %s %s\n' "$1" "$2" "$SB_CUPHY_LOCKSTEP_CPU_KEEPALIVE" "$SLOTS" >> "$OUT/observed"; }
repeat_schedule
[[ $SB_CUPHY_LOCKSTEP_CPU_KEEPALIVE == 0 ]]
''')
    assert result.returncode == 0, result.stderr
    rows = [line.split() for line in (tmp_path / "observed").read_text().splitlines()]
    assert rows[0] == ["gate_cpu_keepalive_alone", "cpu", "1", "64"]
    assert len(rows) == 19
    for name, mode, flag, slots in rows[1:]:
        assert slots == "5000"
        assert flag == ("1" if "cpu_keepalive" in name else "0")
        if flag == "1":
            assert mode == "cpu"


def test_activity_script_is_sourceable_without_launch(tmp_path):
    result = run_shell(tmp_path, "declare -F activity_main")
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "experiment.json").exists()


def test_keeper_assignment_cannot_silently_be_an_ordinary_cpu_run(tmp_path):
    (tmp_path / "case.json").write_text(json.dumps({
        "keepalive_active": False, "keepalive": {"requested": False}}))
    result = run_shell(tmp_path, "activity_validate_case case cpu_keepalive")
    assert result.returncode != 0
    assert "differs from the assigned variant" in result.stderr
