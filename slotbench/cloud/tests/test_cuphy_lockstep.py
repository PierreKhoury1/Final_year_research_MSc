"""cuPHY bootstrap gates with fake local cases; no cloud, driver, or network calls."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys

import pytest

CLOUD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLOUD))
import cuphy_lockstep_check as check
import vast


@pytest.fixture
def case(tmp_path):
    ga, ha = 1_700_000_000_000_000_037, 1_000_000_007
    data = dict(ok=True, correctness_before=True, correctness_after=True, two_point_ok=True,
                mode="cpu", slots=2, warmup=1, total_slots=2, launch_errors=0, timeouts=0,
                recorded=1, skipped=1, misses=1, miss_rate=0.5, period_us=500, deadline_us=500,
                clock_fit_pre=dict(ok=True, t_ref=ha, g_ref=ga, b_ns=0),
                clock_fit_post=dict(ok=True, t_ref=ha+10_000_000, g_ref=ga+10_000_000, b_ns=0))
    rows = [(1, ha+2_000_000, ga+2_000_000, ga+2_000_000, ga+2_000_001,
             ga+2_000_003, ga+2_010_003, 4),
            (2, ha+2_500_000, ga+2_500_000, 0, 0, 0, 0, 1)]
    jp, rp = tmp_path / "case.json", tmp_path / "case.bin"
    jp.write_text(json.dumps(data))
    rp.write_bytes(b"".join(struct.pack("<Qq6Q", *row) for row in rows))
    return jp, rp, data, rows


def test_case_gate_accepts_matching_correct_raw(case):
    jp, rp, _, _ = case
    assert check.validate_case(jp, rp, "cpu", 2, 1) == dict(recorded=1, skipped=1, misses=1)


@pytest.mark.parametrize("field,value", [("correctness_before", False), ("correctness_after", False),
                                        ("correctness_after", 1), ("launch_errors", 1),
                                        ("recorded", 2), ("misses", 0), ("total_slots", 1)])
def test_case_gate_rejects_mismatch(case, field, value):
    jp, rp, data, _ = case
    data[field] = value
    jp.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        check.validate_case(jp, rp, "cpu", 2, 1)


def test_case_gate_rejects_short_raw_and_missing_stamps(case):
    jp, rp, _, rows = case
    rp.write_bytes(rp.read_bytes()[:-1])
    with pytest.raises(ValueError, match="raw size"):
        check.validate_case(jp, rp, "cpu", 2, 1)
    rows[0] = (*rows[0][:5], 0, 0, 0)
    rp.write_bytes(b"".join(struct.pack("<Qq6Q", *row) for row in rows))
    with pytest.raises(ValueError, match="stamps"):
        check.validate_case(jp, rp, "cpu", 2, 1)


def shell(command, env):
    script = CLOUD / "onstart_cuphy_lockstep.sh"
    return subprocess.run(["bash", "-c", f"source {shlex.quote(str(script))}\n{command}"],
                          env=dict(os.environ, **env), capture_output=True, text=True, timeout=10)


def test_bootstrap_source_and_syntax_do_not_execute_main(tmp_path):
    env = dict(W=str(tmp_path), LOGS=str(tmp_path / "logs"), OUT=str(tmp_path / "out"))
    result = shell("declare -F main apply_adapter gcmake deps tv", env)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("correctness_ok", [True, False])
def test_run_case_only_continues_after_successful_correctness_gate(tmp_path, case, correctness_ok):
    jp, rp, data, _ = case
    out = tmp_path / "out"
    out.mkdir()
    cloud = tmp_path / "sb" / "slotbench" / "cloud"
    cloud.mkdir(parents=True)
    (cloud / "cuphy_lockstep_check.py").write_bytes((CLOUD / "cuphy_lockstep_check.py").read_bytes())
    data["correctness_after"] = correctness_ok
    jp.write_text(json.dumps(data))
    fake = tmp_path / "fake_pusch"
    fake.write_text("#!/usr/bin/env bash\ncp \"$FIXTURE_JSON\" \"$SB_CUPHY_LOCKSTEP_OUT\"\n"
                    "cp \"$FIXTURE_RAW\" \"$SB_CUPHY_LOCKSTEP_RAW\"\n")
    fake.chmod(0o755)
    env = dict(W=str(tmp_path), OUT=str(out), PUSCH=str(fake), TV="unused.h5", TV_SHA="abc",
               SLOTS="2", WARMUP="1", PERIOD="500", DEADLINE="500", CASE_TIMEOUT="10",
               FIXTURE_JSON=str(jp), FIXTURE_RAW=str(rp))
    result = shell('run_case cpu_alone cpu none none\ntouch "$OUT/should-not-run"', env)
    assert (result.returncode == 0) == correctness_ok
    assert (out / "should-not-run").exists() == correctness_ok
    if not correctness_ok:
        assert "correctness_after" in (out / "cpu_alone.check.log").read_text()
    manifest = json.loads((out / "cpu_alone.run.json").read_text())
    assert manifest["aerial_commit"] == "4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c"
    assert manifest["test_vector_sha256"] == "abc"


def test_finish_collects_complete_logs_raw_and_markers(tmp_path):
    logs, out = tmp_path / "logs", tmp_path / "out"
    logs.mkdir()
    out.mkdir()
    payload = b"full untruncated build output\n" * 20_000
    (logs / "build.log").write_bytes(payload)
    (out / "cpu_alone.bin").write_bytes(b"raw timestamps")
    digest = hashlib.sha256(payload).hexdigest()
    result = shell('sleep() { return 0; }; finish ok', dict(W=str(tmp_path), LOGS=str(logs), OUT=str(out)))
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) < 20_000
    assert vast.scan_markers(result.stdout)[:2] == (True, "ok")
    collected = tmp_path / "collected"
    report = vast.collect_text(result.stdout, str(collected), out=lambda message: None)
    assert report["done"] and not report["blocks_bad"]
    assert hashlib.sha256((collected / "logs/build.log").read_bytes()).hexdigest() == digest
    assert (collected / "out/cpu_alone.bin").read_bytes() == b"raw timestamps"
    assert hashlib.sha256((logs / "build.log").read_bytes()).hexdigest() == digest


def test_finish_rejects_oversized_collection_without_truncating_sources(tmp_path):
    logs, out = tmp_path / "logs", tmp_path / "out"
    logs.mkdir()
    out.mkdir()
    payload = os.urandom(1_200_000)
    (logs / "build.log").write_bytes(payload)
    result = shell('sleep() { return 0; }; finish ok', dict(W=str(tmp_path), LOGS=str(logs), OUT=str(out)))
    assert "collection-too-large" in result.stdout
    assert vast.scan_markers(result.stdout)[:2] == (True, "error-collection-size")
    assert not vast.extract_blocks(result.stdout)
    assert (tmp_path / "collection.tar.gz").is_file()
    assert (logs / "build.log").read_bytes() == payload
