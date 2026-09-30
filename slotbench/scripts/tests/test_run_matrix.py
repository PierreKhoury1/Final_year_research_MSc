"""Tests for scripts/run_matrix.py and scripts/run_one.sh (no GPU: fake binaries and nvidia tools).

Run: cd slotbench && python -m pytest -q scripts/tests
"""
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
import run_matrix as rm  # noqa: E402

CONFIGS = ROOT / "configs"
RUN_ONE = SCRIPTS / "run_one.sh"


def cfg_of(name):
    return rm.load_config(str(CONFIGS / f"{name}.toml"))


def names(cells):
    return [c["name"] for c in cells]


def write_toml(tmp_path, body, name="t.toml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


# ------------------------------------------------------------------ expansion
def test_quick_expansion_order():
    assert names(rm.expand(cfg_of("quick"))) == [
        "M0_idle_d0_r1", "M0_sgemm_d100_r1",
        "M1_idle_d0_r1", "M1_sgemm_d100_r1",
        "M4_idle_d0_r1", "M4_sgemm_d100_r1",
    ]


def test_home_lab_expansion():
    cells = rm.expand(cfg_of("home_lab"))
    n = names(cells)
    assert len(n) == 3 + 6 * (1 + 3 * 4) == 81
    assert n[:3] == ["SOLO_sgemm", "SOLO_llm", "SOLO_vision"]
    assert n[3:9] == ["M0_idle_d0_r1", "M0_sgemm_d25_r1", "M0_sgemm_d50_r1", "M0_sgemm_d75_r1",
                      "M0_sgemm_d100_r1", "M0_llm_d25_r1"]
    assert n[-1] == "M5_vision_d100_r1"
    assert len(set(n)) == len(n)
    assert [c["mechanism"] for c in cells[3::13]] == ["M0", "M1", "M2", "M3", "M4", "M5"]
    solo = cells[0]
    assert solo["mechanism"] == "SOLO" and solo["duty"] == 100 and solo["rep"] is None


def test_cloud_expansion():
    n = names(rm.expand(cfg_of("cloud")))
    assert len(n) == 3 + 5 * (1 + 3 * 2) == 38
    assert not any(x.startswith("M3") for x in n)
    assert "M2_idle_d0_r1" in n and "M5_vision_d50_r1" in n


def test_d0_dedup_with_repeats(tmp_path):
    p = write_toml(tmp_path, """
        [run]
        repeats = 2
        [matrix]
        mechanisms = ["M0", "M1"]
        workloads = ["sgemm", "llm"]
        duties = [0, 50]
    """)
    n = names(rm.expand(rm.load_config(str(p))))
    assert n == [
        "M0_idle_d0_r1", "M0_idle_d0_r2", "M0_sgemm_d50_r1", "M0_sgemm_d50_r2",
        "M0_llm_d50_r1", "M0_llm_d50_r2",
        "M1_idle_d0_r1", "M1_idle_d0_r2", "M1_sgemm_d50_r1", "M1_sgemm_d50_r2",
        "M1_llm_d50_r1", "M1_llm_d50_r2",
    ]
    assert sum(1 for x in n if "idle" in x) == 4
    n3 = names(rm.expand(rm.load_config(str(p)), repeat_from=3))
    assert n3[:2] == ["M0_idle_d0_r3", "M0_idle_d0_r4"]


def test_idle_cell_properties():
    idle = [c for c in rm.expand(cfg_of("home_lab")) if c["duty"] == 0]
    assert len(idle) == 6
    assert all(c["workload"] == "idle" for c in idle)


def test_bad_config(tmp_path):
    p = write_toml(tmp_path, """
        [matrix]
        mechanisms = ["M9"]
    """)
    with pytest.raises(rm.ConfigError):
        rm.load_config(str(p))
    p = write_toml(tmp_path, """
        [run]
        slotz = 5
    """)
    with pytest.raises(rm.ConfigError):
        rm.load_config(str(p))
    assert rm.main([str(p), "--dry-run"]) == 2


def test_estimate():
    cfg = cfg_of("home_lab")
    cells = rm.expand(cfg)
    solo = rm.estimate_s(cfg, cells[0])
    assert solo == pytest.approx(300 + 30 + 120)
    run = rm.estimate_s(cfg, cells[4])
    assert run == pytest.approx(300 + 30 + 30 + 1e6 * 500e-6)


def test_overrides(tmp_path):
    p = write_toml(tmp_path, """
        [run]
        slots = 1000
        [matrix]
        mechanisms = ["M0", "M3"]
        workloads = ["sgemm"]
        duties = [100]
        [overrides.M3]
        slots = 77
    """)
    cfg = rm.load_config(str(p))
    cells = rm.expand(cfg)
    c0 = rm.build_command(cfg, cells[0], "/x", None, "run_one.sh")
    c3 = rm.build_command(cfg, cells[1], "/x", None, "run_one.sh")
    assert c0[c0.index("--slots") + 1] == "1000"
    assert c3[c3.index("--slots") + 1] == "77"


# ------------------------------------------------------------------ dry run
def dry(config, *extra, out_root=None):
    args = [sys.executable, str(SCRIPTS / "run_matrix.py"), str(config), "--dry-run"]
    if out_root:
        args += ["--out-root", str(out_root)]
    r = subprocess.run(args + list(extra), capture_output=True, text=True, check=True)
    return r.stdout


def plan_blocks(text):
    """cell name -> {'cmd': run_one line, 'driver': ..., 'adversary': ..., ...}"""
    blocks = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"# \[\d+/\d+\] (\S+)", line)
        if m:
            cur = blocks.setdefault(m.group(1), {})
            continue
        if cur is None or not line:
            continue
        if line.startswith("#   "):
            k, _, v = line[4:].partition(": ")
            cur[k] = v
        elif not line.startswith("#"):
            cur["cmd"] = line
    return blocks


def test_dry_run_flags_per_mechanism(tmp_path):
    p = write_toml(tmp_path, """
        [run]
        sizes = "--ldpc-cb 40 --ldpc-iters 10"
        driver_core = 4
        collector_core = 5
        fifo = 99
        lock_clocks = true
        [matrix]
        mechanisms = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
        workloads = ["sgemm"]
        duties = [50]
        solo = true
    """)
    b = plan_blocks(dry(p, out_root=tmp_path / "runs"))
    assert list(b) == ["SOLO_sgemm"] + [f"M{i}_sgemm_d50_r1" for i in range(7)]
    d = {m: b[f"{m}_sgemm_d50_r1"] for m in ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]}
    for m, blk in d.items():
        assert "--ldpc-cb 40 --ldpc-iters 10" in blk["driver"]
        assert "--core 4" in blk["driver"] and "--collector-core 5" in blk["driver"]
        assert "--fifo 99" in blk["driver"]
        assert f"--mechanism {m}" in blk["cmd"]
        assert "--workload sgemm --duty 50" in blk["adversary"]
    assert "--prio default" in d["M0"]["driver"] and "--prio default" in d["M0"]["adversary"]
    assert "--prio high" in d["M1"]["driver"] and "--prio low" in d["M1"]["adversary"]
    assert "--mode graph" in d["M1"]["driver"]
    assert "--prio high" in d["M4"]["driver"] and "--prio low" in d["M4"]["adversary"]
    assert "--mode streams" in d["M4"]["driver"]
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50" in d["M2"]["adversary"]
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in d["M2"]["driver"]
    assert "CUDA_MPS_PIPE_DIRECTORY" in d["M2"]["driver"] and "--prio high" in d["M2"]["driver"]
    assert "mps" in d["M2"] and "mps" not in d["M1"]
    assert "--set-timeslice=1" in d["M3"]["timeslice"] and "--prio high" in d["M3"]["driver"]
    assert "unlock" in d["M5"]["clocks"] and "lock --gpu" in d["M1"]["clocks"]
    assert "--prio high" in d["M5"]["driver"] and "--prio low" in d["M5"]["adversary"]
    assert "mig" in d["M6"]
    solo = b["SOLO_sgemm"]
    assert "driver" not in solo and "--duty 100" in solo["adversary"] and "--seconds 120" in solo["adversary"]
    for blk in d.values():
        assert "--out " in blk["driver"] and "--timeline " in blk["adversary"]


def test_dry_run_idle_cell_and_home_lab(tmp_path):
    b = plan_blocks(dry(CONFIGS / "home_lab.toml", out_root=tmp_path))
    assert len(b) == 81
    idle = b["M1_idle_d0_r1"]
    assert "--workload idle --duty 0" in idle["adversary"]
    assert "--slots 1000000" in idle["driver"] and "--warmup-s 300" in idle["cmd"]
    assert not (tmp_path / "home_lab").exists(), "dry run must not create directories"


def test_only_filter(tmp_path):
    b = plan_blocks(dry(CONFIGS / "home_lab.toml", "--only", "M4_*_d100_r1,SOLO_llm", out_root=tmp_path))
    assert list(b) == ["SOLO_llm", "M4_sgemm_d100_r1", "M4_llm_d100_r1", "M4_vision_d100_r1"]
    r = subprocess.run([sys.executable, str(SCRIPTS / "run_matrix.py"), str(CONFIGS / "quick.toml"),
                        "--dry-run", "--only", "nomatch*"], capture_output=True, text=True)
    assert r.returncode == 2


# ------------------------------------------------------------------ resume with a fake run_one
FAKE_RUN_ONE = """#!/usr/bin/env bash
# fake run_one.sh: logs its args, writes status (ok unless the cell name matches FAKE_FAIL).
out=""; cell=""
args=("$@")
while [[ $# -gt 0 ]]; do
  case $1 in --out) out=$2; shift 2;; --cell) cell=$2; shift 2;; *) shift;; esac
done
mkdir -p "$out"
echo "$cell ${args[*]}" >> "$FAKE_CALLS"
trap 'kill $! 2>/dev/null; echo invalid:interrupted > "$out/status"; exit 130' INT
if [[ -n ${FAKE_SLEEP:-} ]]; then sleep "$FAKE_SLEEP" & wait $!; fi
if [[ -n ${FAKE_FAIL:-} && $cell == $FAKE_FAIL ]]; then echo "invalid:driver_rc1" > "$out/status"; exit 1; fi
echo ok > "$out/status"
"""


@pytest.fixture
def fake_run_one(tmp_path):
    p = tmp_path / "fake_run_one.sh"
    p.write_text(FAKE_RUN_ONE)
    p.chmod(0o755)
    calls = tmp_path / "calls.txt"
    env = dict(os.environ, FAKE_CALLS=str(calls))
    return p, calls, env


def run_matrix(config, out_root, run_one, env, *extra):
    args = [sys.executable, str(SCRIPTS / "run_matrix.py"), str(config), "--out-root", str(out_root),
            "--run-one", str(run_one)] + list(extra)
    return subprocess.run(args, capture_output=True, text=True, env=env)


def test_resume_skips_ok_cells(tmp_path, fake_run_one):
    run_one, calls, env = fake_run_one
    runs = tmp_path / "runs"
    pre = runs / "quick" / "M0_sgemm_d100_r1"
    pre.mkdir(parents=True)
    (pre / "status").write_text("ok\n")
    stale = runs / "quick" / "M1_idle_d0_r1"
    stale.mkdir(parents=True)
    (stale / "status").write_text("invalid:driver_rc1\n")

    r = run_matrix(CONFIGS / "quick.toml", runs, run_one, dict(env, FAKE_FAIL="M4_sgemm*"))
    assert r.returncode == 1, r.stdout + r.stderr
    called = [line.split()[0] for line in calls.read_text().splitlines()]
    assert called == ["M0_idle_d0_r1", "M1_idle_d0_r1", "M1_sgemm_d100_r1", "M4_idle_d0_r1", "M4_sgemm_d100_r1"]
    first = calls.read_text().splitlines()[0]
    assert "--mechanism M0 --workload idle --duty 0 --slots 20000 --warmup-s 10 --settle-s 5" in first

    mj = json.loads((runs / "quick" / "matrix.json").read_text())
    assert [c["name"] for c in mj["plan"]] == names(rm.expand(cfg_of("quick")))
    assert mj["outcomes"]["M1_idle_d0_r1"]["status"] == "ok"
    assert mj["outcomes"]["M4_sgemm_d100_r1"]["status"] == "invalid:driver_rc1"
    log = (runs / "quick" / "matrix.log").read_text()
    assert "M0_sgemm_d100_r1: status ok, skipped" in log
    assert re.search(r"^\[\d{4}-\d\d-\d\dT", log, re.M)

    # Second pass: only the failed cell runs again.
    calls.write_text("")
    r = run_matrix(CONFIGS / "quick.toml", runs, run_one, env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert [line.split()[0] for line in calls.read_text().splitlines()] == ["M4_sgemm_d100_r1"]
    mj = json.loads((runs / "quick" / "matrix.json").read_text())
    assert mj["outcomes"]["M4_sgemm_d100_r1"]["status"] == "ok"
    assert "M0_idle_d0_r1" in mj["outcomes"]  # outcomes from the earlier pass are kept


def test_ctrl_c_stops_cleanly(tmp_path, fake_run_one):
    run_one, calls, env = fake_run_one
    runs = tmp_path / "runs"
    args = [sys.executable, str(SCRIPTS / "run_matrix.py"), str(CONFIGS / "quick.toml"),
            "--out-root", str(runs), "--run-one", str(run_one)]
    p = subprocess.Popen(args, env=dict(env, FAKE_SLEEP="30"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    deadline = time.time() + 10
    while time.time() < deadline and not (calls.exists() and calls.read_text()):
        time.sleep(0.05)
    time.sleep(0.2)
    p.send_signal(signal.SIGINT)
    out, err = p.communicate(timeout=20)
    assert p.returncode == 130, out + err
    assert len(calls.read_text().splitlines()) == 1
    mj = json.loads((runs / "quick" / "matrix.json").read_text())
    assert mj["outcomes"]["M0_idle_d0_r1"]["status"] == "invalid:interrupted"
    assert (runs / "quick" / "M0_idle_d0_r1" / "status").read_text().strip() == "invalid:interrupted"


# ------------------------------------------------------------------ run_one.sh end to end with fakes
FAKE_SMI = r"""#!/usr/bin/env bash
# fake nvidia-smi: logs every call; FAKE_SMI_FAIL lists operations that fail (lgc,lmc,timeslice,...)
echo "nvidia-smi $*" >> "$FAKE_LOG"
fail() { [[ ,${FAKE_SMI_FAIL:-}, == *,$1,* ]]; }
args="$*"
case $args in
  *compute-policy*--set-timeslice*) fail timeslice && { echo "Setting timeslice is not supported"; exit 3; }; exit 0 ;;
  *compute-policy*) echo "timeslice: default"; exit 0 ;;
  *--query-supported-clocks*) printf '7501, 2100\n7501, 1800\n7501, 1320\n405, 210\n'; exit 0 ;;
  *-lms*)
    f=""; prev=""
    for a in "$@"; do [[ $prev == -f ]] && f=$a; prev=$a; done
    echo "pid $$ telemetry" >> "$FAKE_PIDS"
    echo "timestamp, temperature.gpu" > "$f"
    while :; do echo "2026/01/01 00:00:00.000, 50" >> "$f"; sleep 0.2; done ;;
  *--query-gpu=name*) echo "Fake GPU"; exit 0 ;;
  *default_applications*) echo "[N/A]"; exit 0 ;;
  *--query-gpu=*) echo "1000"; exit 0 ;;
  *-pm*) fail pm && exit 4; exit 0 ;;
  *-lgc*) fail lgc && { echo "not permitted"; exit 4; }; exit 0 ;;
  *-lmc*) fail lmc && { echo "not supported"; exit 3; }; exit 0 ;;
  *-rgc*|*-rmc*) exit 0 ;;
  *) echo "fake nvidia-smi output"; exit 0 ;;
esac
"""

FAKE_MPS = r"""#!/usr/bin/env bash
# fake nvidia-cuda-mps-control: -d spawns a process named nvidia-cuda-mps-control; stdin commands
echo "mps $* pipe=${CUDA_MPS_PIPE_DIRECTORY:-}" >> "$FAKE_LOG"
p=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
if [[ ${1:-} == -d ]]; then
  mkdir -p "$p"
  (exec -a nvidia-cuda-mps-control sleep 300) &
  echo $! > "$p/daemon.pid"
  echo "pid $! mps" >> "$FAKE_PIDS"
  touch "$p/control"
  exit 0
fi
read -r cmd
case $cmd in
  quit) [[ -f $p/daemon.pid ]] && kill "$(cat "$p/daemon.pid")" 2>/dev/null; rm -f "$p/control" "$p/daemon.pid" ;;
  get_server_list) echo 4242 ;;
esac
exit 0
"""

FAKE_DRIVER = r"""#!/usr/bin/env bash
# fake slot_driver: FAKE_DRV_SLEEP, FAKE_DRV_RC, FAKE_RING (ring_overflows), FAKE_NO_META
out=""
for ((i = 1; i <= $#; i++)); do [[ ${!i} == --out ]] && { j=$((i + 1)); out=${!j}; }; done
echo "driver $* MPS_PCT=${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-} PIPE=${CUDA_MPS_PIPE_DIRECTORY:-} CVD=${CUDA_VISIBLE_DEVICES:-}" >> "$FAKE_LOG"
echo "pid $$ driver" >> "$FAKE_PIDS"
meta() { [[ -n ${FAKE_NO_META:-} ]] || echo "{\"counts\": {\"recorded\": 10, \"ring_overflows\": ${FAKE_RING:-0}}, \"exit_reason\": \"$1\"}" > "$out/meta.json"; }
trap 'kill $! 2>/dev/null; meta signal; exit 0' TERM INT
sleep "${FAKE_DRV_SLEEP:-0.5}" & wait $!
meta done
exit "${FAKE_DRV_RC:-0}"
"""

FAKE_ADV = r"""#!/usr/bin/env bash
# fake adversary: runs --seconds S or until SIGTERM; FAKE_ADV_DIE=S exits 1 early (main run only)
out=""; secs=""
for ((i = 1; i <= $#; i++)); do
  j=$((i + 1))
  [[ ${!i} == --out ]] && out=${!j}
  [[ ${!i} == --seconds ]] && secs=${!j}
done
echo "adversary $* MPS_PCT=${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-} PIPE=${CUDA_MPS_PIPE_DIRECTORY:-}" >> "$FAKE_LOG"
echo "pid $$ adversary" >> "$FAKE_PIDS"
summary() { echo '{"units": 5, "units_per_s": 1.0}' > "$out"; }
trap 'kill $! 2>/dev/null; summary; exit 0' TERM INT
if [[ -n $secs ]]; then sleep "$secs" & wait $!; summary; exit 0; fi
if [[ -n ${FAKE_ADV_DIE:-} ]]; then sleep "$FAKE_ADV_DIE" & wait $!; exit 1; fi
while :; do sleep 0.1 & wait $!; done
"""


@pytest.fixture
def fakes(tmp_path):
    fb = tmp_path / "fakebin"
    fb.mkdir()
    for name, body in [("nvidia-smi", FAKE_SMI), ("nvidia-cuda-mps-control", FAKE_MPS)]:
        (fb / name).write_text(body)
        (fb / name).chmod(0o755)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in [("slot_driver", FAKE_DRIVER), ("adversary", FAKE_ADV)]:
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)
    log = tmp_path / "fake.log"
    pids = tmp_path / "pids.txt"
    log.touch()
    pids.touch()
    env = dict(os.environ, PATH=f"{fb}:{os.environ['PATH']}", FAKE_LOG=str(log), FAKE_PIDS=str(pids),
               SB_PYTHON=sys.executable)
    for k in ("MIG_DRIVER_UUID", "MIG_ADV_UUID", "CUDA_MPS_PIPE_DIRECTORY"):
        env.pop(k, None)
    return {"bin": bindir, "log": log, "pids": pids, "env": env, "tmp": tmp_path}


def run_one(f, mech, workload="sgemm", duty=50, extra=(), env_extra=None, wait=True, out=None):
    out = out or f["tmp"] / "runs" / f"{mech}_{workload}_d{duty}_r1"
    args = ["bash", str(RUN_ONE), "--out", str(out), "--mechanism", mech, "--workload", workload,
            "--duty", str(duty), "--slots", "100", "--warmup-s", "0.2", "--settle-s", "0.2",
            "--lock-clocks", "1", "--bin", str(f["bin"]), "--solo-seconds", "0.3",
            "--driver-flags", "--ldpc-cb 5"] + list(extra)
    env = dict(f["env"], **(env_extra or {}))
    if not wait:
        return subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), out
    r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=120)
    return r, out


def status(out):
    return (out / "status").read_text().strip()


def assert_no_leftovers(f):
    time.sleep(0.2)
    for line in f["pids"].read_text().splitlines():
        pid = int(line.split()[1])
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        st = Path(f"/proc/{pid}/stat").read_text().split()[2] if Path(f"/proc/{pid}/stat").exists() else "Z"
        assert st == "Z", f"left-over process: {line}"


def smi_calls(f):
    return [x for x in f["log"].read_text().splitlines() if x.startswith("nvidia-smi")]


def test_run_one_m1_ok(fakes):
    r, out = run_one(fakes, "M1")
    assert r.returncode == 0, r.stderr
    assert status(out) == "ok"
    rj = json.loads((out / "run.json").read_text())
    assert rj["mechanism"] == "M1" and rj["duty"] == 50 and rj["status"] == "ok"
    assert rj["settings"]["driver_prio"] == "high" and rj["settings"]["adversary_prio"] == "low"
    assert rj["settings"]["ring_overflows"] == "0"
    assert rj["exit_codes"]["driver"] == 0
    whats = [c["what"] for c in rj["commands"]]
    for w in ("env_capture", "clocks_lock", "warmup_adversary", "telemetry", "adversary", "driver", "clocks_unlock"):
        assert w in whats, w
    drv = [c["cmd"] for c in rj["commands"] if c["what"] == "driver"][0]
    assert "--prio high" in drv and "--mode graph" in drv and "--ldpc-cb 5" in drv
    log = fakes["log"].read_text()
    assert re.search(r"^driver .*--prio high", log, re.M)
    assert re.search(r"^adversary .*--prio low .*--timeline", log, re.M)
    assert re.search(r"^adversary .*--seconds 0.2 .*warmup_adversary.json", log, re.M)
    smi = smi_calls(fakes)
    assert any("-lgc 1320,1320" in c for c in smi)  # 80% of 2100 -> highest supported <= 1680
    assert any("-lmc 7501,7501" in c for c in smi)
    assert any("-rgc" in c for c in smi)  # restored
    for fn in ("env.txt", "telemetry.csv", "adversary.json", "warmup_adversary.json", "meta.json", "setup.log"):
        assert (out / fn).exists(), fn
    assert not (out / ".runlog.tsv").exists()
    assert_no_leftovers(fakes)


def test_run_one_m4_streams(fakes):
    r, out = run_one(fakes, "M4")
    assert r.returncode == 0, r.stderr
    assert re.search(r"^driver .*--prio high --mode streams", fakes["log"].read_text(), re.M)


def test_run_one_m5_unlocks_only(fakes):
    r, out = run_one(fakes, "M5")
    assert r.returncode == 0, r.stderr
    smi = smi_calls(fakes)
    assert not any("-lgc" in c for c in smi)
    assert any("-rgc" in c for c in smi)


def test_run_one_ring_overflow_invalid(fakes):
    r, out = run_one(fakes, "M1", env_extra={"FAKE_RING": "3"})
    assert r.returncode == 1
    assert status(out) == "invalid:ring_overflows=3"
    assert_no_leftovers(fakes)


def test_run_one_driver_failure(fakes):
    r, out = run_one(fakes, "M0", env_extra={"FAKE_DRV_RC": "2", "FAKE_NO_META": "1"})
    assert r.returncode == 1
    assert status(out) == "invalid:driver_rc2,no_meta_json"
    assert_no_leftovers(fakes)


def test_run_one_adversary_died(fakes):
    r, out = run_one(fakes, "M1", env_extra={"FAKE_ADV_DIE": "0.3", "FAKE_DRV_SLEEP": "1"})
    assert r.returncode == 1
    assert "adversary_exited_early" in status(out)
    assert_no_leftovers(fakes)


def test_run_one_clock_lock_unsupported_is_recorded(fakes):
    r, out = run_one(fakes, "M1", env_extra={"FAKE_SMI_FAIL": "lgc,lmc"})
    assert r.returncode == 0, r.stderr
    rj = json.loads((out / "run.json").read_text())
    assert any(u["what"] == "clocks_lock" and "NOT SUPPORTED" in u["detail"] for u in rj["unsupported"])


def test_run_one_timeslice(fakes):
    r, out = run_one(fakes, "M3")
    assert r.returncode == 0, r.stderr
    smi = smi_calls(fakes)
    i_set = next(i for i, c in enumerate(smi) if "--set-timeslice=1" in c)
    i_rst = next(i for i, c in enumerate(smi) if "--set-timeslice=0" in c)
    assert i_rst > i_set


def test_run_one_timeslice_unsupported(fakes):
    r, out = run_one(fakes, "M3", env_extra={"FAKE_SMI_FAIL": "timeslice"})
    assert r.returncode == 3
    assert status(out) == "invalid:not_supported:timeslice"
    assert not re.search(r"^driver ", fakes["log"].read_text(), re.M)
    rj = json.loads((out / "run.json").read_text())
    assert rj["unsupported"][-1]["what"] == "timeslice"
    assert any("-rgc" in c for c in smi_calls(fakes))  # clocks still restored
    assert_no_leftovers(fakes)


def test_run_one_mps(fakes):
    r, out = run_one(fakes, "M2")
    assert r.returncode == 0, r.stderr + (out / "setup.log").read_text()
    log = fakes["log"].read_text()
    adv = [x for x in log.splitlines() if x.startswith("adversary")]
    drv = [x for x in log.splitlines() if x.startswith("driver")]
    assert all("MPS_PCT=50" in x for x in adv) and len(adv) == 2
    assert "MPS_PCT= " in drv[0]
    pipe = re.search(r"PIPE=(\S+)", drv[0]).group(1)
    assert all(f"PIPE={pipe}" in x for x in adv)
    assert "mps -d" in log and "--prio high" in drv[0]
    assert not Path(pipe).exists()  # run-specific MPS dir removed
    assert_no_leftovers(fakes)  # daemon stopped


def test_run_one_stale_mps_daemon_invalidates(fakes):
    # A daemon on an unknown pipe dir cannot be stopped by the reset; the run must refuse.
    d = subprocess.Popen(["bash", "-c", "exec -a nvidia-cuda-mps-control sleep 30"])
    try:
        time.sleep(0.2)
        r, out = run_one(fakes, "M1")
        assert r.returncode == 1
        assert status(out) == "invalid:mps_daemon_still_running"
    finally:
        d.kill()
        d.wait()


def test_run_one_mig_unsupported_without_uuids(fakes):
    r, out = run_one(fakes, "M6")
    assert r.returncode == 3
    assert status(out) == "invalid:not_supported:mig"


def test_run_one_mig(fakes):
    r, out = run_one(fakes, "M6", env_extra={"MIG_DRIVER_UUID": "MIG-aaa", "MIG_ADV_UUID": "MIG-bbb"})
    assert r.returncode == 0, r.stderr
    assert re.search(r"^driver .*--gpu 0 .*CVD=MIG-aaa", fakes["log"].read_text(), re.M)


def test_run_one_solo(fakes):
    r, out = run_one(fakes, "SOLO", duty=37)
    assert r.returncode == 0, r.stderr
    assert status(out) == "ok"
    log = fakes["log"].read_text()
    assert not re.search(r"^driver ", log, re.M)
    assert re.search(r"^adversary .*--duty 100 .*--seconds 0.3 ", log, re.M)
    assert_no_leftovers(fakes)


def test_run_one_interrupt(fakes):
    p, out = run_one(fakes, "M1", env_extra={"FAKE_DRV_SLEEP": "30"}, wait=False)
    deadline = time.time() + 20
    while time.time() < deadline and not re.search(r"^driver ", fakes["log"].read_text(), re.M):
        time.sleep(0.05)
    time.sleep(0.2)
    p.send_signal(signal.SIGINT)
    p.communicate(timeout=60)
    assert p.returncode == 130
    assert status(out) == "invalid:interrupted"
    meta = json.loads((out / "meta.json").read_text())
    assert meta["exit_reason"] == "signal"  # driver was stopped with a signal and still wrote meta
    assert any("-rgc" in c for c in smi_calls(fakes))
    assert json.loads((out / "run.json").read_text())["status"] == "invalid:interrupted"
    assert_no_leftovers(fakes)


def test_run_one_moves_previous_attempt(fakes):
    out = fakes["tmp"] / "runs" / "cell"
    out.mkdir(parents=True)
    (out / "meta.json").write_text('{"counts": {"ring_overflows": 0}}')
    (out / "status").write_text("invalid:old\n")
    r, _ = run_one(fakes, "M1", out=out, env_extra={"FAKE_DRV_RC": "1", "FAKE_NO_META": "1"})
    assert r.returncode == 1
    assert status(out) == "invalid:driver_rc1,no_meta_json"  # stale meta.json not used
    old = list((out / "old_attempts").iterdir())
    assert len(old) == 1 and (old[0] / "meta.json").exists()


def test_run_one_usage_errors(fakes):
    r, _ = run_one(fakes, "M9")
    assert r.returncode == 2
    r, _ = run_one(fakes, "M1", duty=101)
    assert r.returncode == 2


def test_print_plan_has_no_side_effects(fakes):
    out = fakes["tmp"] / "nothere" / "cell"
    r = subprocess.run(["bash", str(RUN_ONE), "--out", str(out), "--mechanism", "M4", "--workload", "llm",
                        "--duty", "25", "--slots", "10", "--warmup-s", "1", "--settle-s", "1", "--print-plan"],
                       capture_output=True, text=True, env=fakes["env"])
    assert r.returncode == 0
    assert "--mode streams" in r.stdout
    assert not out.parent.exists()
    assert fakes["log"].read_text() == ""
