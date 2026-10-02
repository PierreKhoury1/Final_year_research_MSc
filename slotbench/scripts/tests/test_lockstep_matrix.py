"""Lockstep orchestration regressions using fake GPU binaries (Linux/Bash, no GPU)."""
import json
import os
from pathlib import Path
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "scripts" / "lockstep_matrix.sh"
CLOUD = ROOT / "cloud" / "onstart_lockstep.sh"
pytestmark = pytest.mark.skipif(os.name == "nt", reason="requires Linux process and Bash semantics")


def executable(path, source):
    path.write_text(source)
    path.chmod(0o755)


FAKE_CLIENT = r'''#!/usr/bin/env python3
import json, os, signal, sys
from pathlib import Path
args = sys.argv[1:]
def arg(name):
    # Match the driver's last-option-wins parsing.
    return args[len(args) - 1 - args[::-1].index(name) + 1]
kind = Path(sys.argv[0]).name
entry = dict(kind=kind, args=args,
             mps={k: v for k, v in os.environ.items() if k.startswith("CUDA_MPS_")})
with open(os.environ["FAKE_CALLS"], "a") as stream:
    stream.write(json.dumps(entry) + "\n")
if kind == "adversary":
    Path(arg("--timeline")).write_text("time,work\n1,1\n2,1\n3,1\n")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        signal.pause()
variant = arg("--slot-variant")
if os.environ.get("FAKE_WRONG_VARIANT") and arg("--mode") == "gpu":
    variant = "no_cublas" if variant == "full" else "full"
if os.environ.get("FAKE_NO_JSON"):
    print('CUDA failed: "unsupported" \\ graph')
else:
    Path(arg("--out")).write_text(json.dumps(dict(ok=True, slot_variant=variant)))
sys.exit(int(os.environ.get("FAKE_DRIVER_RC", "0")))
'''


FAKE_MPS = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
command = "start" if sys.argv[1:] == ["-d"] else sys.stdin.read().strip()
with open(os.environ["FAKE_CALLS"], "a") as stream:
    stream.write(json.dumps(dict(kind="mps", command=command,
                                pipe=os.environ.get("CUDA_MPS_PIPE_DIRECTORY"))) + "\n")
if command == "start":
    Path(os.environ["CUDA_MPS_LOG_DIRECTORY"], "control.log").write_text("daemon startup log\n")
    sys.exit(int(os.environ.get("FAKE_MPS_RC", "0")))
'''


@pytest.fixture
def matrix_env(tmp_path):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    for name in ("lockstep_driver", "adversary"):
        executable(fakebin / name, FAKE_CLIENT)
    executable(fakebin / "nvidia-cuda-mps-control", FAKE_MPS)
    executable(fakebin / "nvidia-smi", "#!/bin/bash\necho 'Fake GPU'\n")
    executable(fakebin / "sleep", "#!/bin/bash\nexec /bin/sleep 0.03\n")
    executable(fakebin / "pgrep", '#!/bin/bash\n[[ ${REAL_PGREP:-0} == 1 ]] && exec /usr/bin/pgrep "$@"\nexit "${FAKE_PGREP_RC:-1}"\n')
    calls = tmp_path / "calls.jsonl"
    calls.touch()
    env = dict(os.environ, PATH=f"{fakebin}:{os.environ['PATH']}", FAKE_CALLS=str(calls),
               TMPDIR=str(tmp_path), CUDA_MPS_PIPE_DIRECTORY="/inherited/pipe",
               CUDA_MPS_LOG_DIRECTORY="/inherited/log", CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="25",
               CUDA_MPS_CLIENT_PRIORITY="1", CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=1G",
               CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING="1")
    return dict(bin=fakebin, calls=calls, env=env, out=tmp_path / "out")


def run_matrix(fixture, *extra, **env_extra):
    return subprocess.run(
        ["bash", str(MATRIX), "--out", str(fixture["out"]), "--bin", str(fixture["bin"]),
         "--slots", "1", "--core", "-1", "--workloads", "sgemm", *extra],
        env=dict(fixture["env"], **env_extra), text=True, capture_output=True, timeout=20,
    )


def calls(fixture):
    return [json.loads(line) for line in fixture["calls"].read_text().splitlines()]


@pytest.mark.parametrize("variant", ["full", "no_cublas"])
def test_matrix_keeps_variants_equal_and_isolates_mps(matrix_env, variant):
    # Existing output from another run must not affect this invocation's status/summary.
    matrix_env["out"].mkdir()
    (matrix_env["out"] / "old_failed_case.json").write_text('{"ok": false}')
    extra = [] if variant == "full" else ["--slot-variant", variant]
    result = run_matrix(matrix_env, *extra)
    assert result.returncode == 0, result.stdout + result.stderr
    entries = calls(matrix_env)
    drivers = [entry for entry in entries if entry["kind"] == "lockstep_driver"]
    assert len(drivers) == 8
    for driver in drivers:
        args = driver["args"]
        assert args[args.index("--slot-variant") + 1] == variant
    for client in [entry for entry in entries if entry["kind"] != "mps"]:
        args = client["args"]
        name = Path(args[args.index("--out") + 1]).name
        if "_mps_" in name:
            assert client["mps"]["CUDA_MPS_PIPE_DIRECTORY"] != "/inherited/pipe"
            if client["kind"] == "adversary":
                assert client["mps"]["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "50"
            else:
                assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in client["mps"]
        else:
            assert client["mps"] == {}
    mps = [entry for entry in entries if entry["kind"] == "mps"]
    assert [entry["command"] for entry in mps] == ["start", "quit"]
    assert mps[0]["pipe"] == mps[1]["pipe"]
    assert (matrix_env["out"] / "mps_logs" / "control.log").read_text() == "daemon startup log\n"
    assert "old_failed_case" not in result.stdout


def test_existing_long_named_mps_is_refused_without_terminating_it(matrix_env):
    daemon = subprocess.Popen(["bash", "-c", "exec -a nvidia-cuda-mps-control /bin/sleep 30"])
    try:
        time.sleep(0.05)
        result = run_matrix(matrix_env, REAL_PGREP="1")
        assert result.returncode == 3, result.stdout + result.stderr
        assert "existing MPS daemon" in result.stdout
        assert daemon.poll() is None
        assert calls(matrix_env) == []
    finally:
        daemon.terminate()
        daemon.wait(timeout=5)


def test_failed_mps_process_scan_is_not_treated_as_no_daemon(matrix_env):
    result = run_matrix(matrix_env, FAKE_PGREP_RC="2")
    assert result.returncode == 3, result.stdout + result.stderr
    assert "cannot inspect MPS processes" in result.stdout
    assert calls(matrix_env) == []


@pytest.mark.parametrize("failure", [{"FAKE_DRIVER_RC": "7"}, {"FAKE_WRONG_VARIANT": "1"},
                                    {"FAKE_NO_JSON": "1"}, {"FAKE_MPS_RC": "1"}])
def test_failed_or_mismatched_case_fails_matrix_and_keeps_artifacts(matrix_env, failure):
    matrix_env["out"].mkdir()
    (matrix_env["out"] / "cpu_alone.bin").write_bytes(b"stale raw output")
    result = run_matrix(matrix_env, **failure)
    assert result.returncode == 1, result.stdout + result.stderr
    assert len([entry for entry in calls(matrix_env) if entry["kind"] == "lockstep_driver"]) >= 6
    for mode in ("cpu", "gpu"):
        data = json.loads((matrix_env["out"] / f"{mode}_alone.json").read_text())
        assert "slot_variant" in data
        assert (matrix_env["out"] / f"{mode}_alone.log").exists()
    assert (matrix_env["out"] / "mps_logs" / "control.log").exists()
    assert not (matrix_env["out"] / "cpu_alone.bin").exists()
    if "FAKE_WRONG_VARIANT" in failure:
        assert "slot variant mismatch" in result.stdout
    if "FAKE_MPS_RC" in failure:
        assert not json.loads((matrix_env["out"] / "gpu_mps_sgemm.json").read_text())["ok"]
        assert [entry["command"] for entry in calls(matrix_env) if entry["kind"] == "mps"] == ["start"]


@pytest.mark.parametrize("matrix_rc", [0, 7])
def test_cloud_reports_matrix_status_and_preserves_log(tmp_path, matrix_rc):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    executable(scripts / "lockstep_matrix.sh", f"#!/bin/bash\necho matrix-output\nexit {matrix_rc}\n")
    logs, out = tmp_path / "logs", tmp_path / "out"
    logs.mkdir()
    out.mkdir()
    (logs / "header.txt").write_text("machine details\n")
    (out / "header.txt").write_text("matrix settings\n")
    # Exercise the real launch/finish dispatch without package installation, cloning, or billing watchdog.
    tail = CLOUD.read_text().split('log "running matrix', 1)[1]
    code = '''set -euo pipefail
log() { echo "$*"; }
finish() { printf 'finished:%s\\n' "$1"; }
log "running matrix''' + tail
    result = subprocess.run(["bash", "-c", code], cwd=tmp_path,
                            env=dict(os.environ, LOGS=str(logs), OUT=str(out)),
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    expected = "ok" if matrix_rc == 0 else f"error-matrix-{matrix_rc}"
    assert f"finished:{expected}" in result.stdout
    assert (logs / "matrix.log").read_text() == "matrix-output\n"
    assert (out / "cloud_header.txt").read_text() == "machine details\n"
    assert (out / "header.txt").read_text() == "matrix settings\n"


def test_cloud_preserves_failed_build_step_log(tmp_path):
    function = "step() {" + CLOUD.read_text().split("step() {", 1)[1].split("\n}", 1)[0] + "\n}"
    code = 'set -euo pipefail\nlog() { :; }\n' + function + '\nstep build bash -c "echo compiler-error; exit 7"'
    result = subprocess.run(["bash", "-c", code], env=dict(os.environ, LOGS=str(tmp_path)),
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 7
    assert (tmp_path / "current.log").read_text() == "compiler-error\n"
    assert (tmp_path / "build.log").read_text() == "compiler-error\n"
