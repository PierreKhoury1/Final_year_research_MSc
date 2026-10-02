#!/bin/bash
# Reuse an already built, pinned cuPHY checkout and hash-verified TC7304 vector.
# No downloads, builds, clock changes, rental actions, or stdout archive payload.
#   bash slotbench/cloud/repeat_cuphy_lockstep.sh
# Sourceable for host tests; SB_CUPHY_WORKDIR defaults to /workspace/cuphy-lockstep.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/onstart_cuphy_lockstep.sh"

repeat_plan() {
    python3 - "$OUT" "$REPEAT_SEED" "$REPEATS" "$SLOTS" "$WARMUP" "$PERIOD" "$DEADLINE" "$TV_SHA" "$ACAR_COMMIT" "$OBSERVER_CPU" <<'PY'
import json, os, pathlib, random, sys
out, seed, repeats, slots, warmup, period, deadline, vector, revision, observer_cpu = sys.argv[1:]
out = pathlib.Path(out)
rng = random.Random(int(seed))
schedule, block = [], 0
first_modes = {}
for condition in ("alone", "proc_sgemm", "mps_sgemm"):
    first_modes[condition] = ["cpu", "gpu"] * (int(repeats) // 2)
    rng.shuffle(first_modes[condition])
for pair in range(1, int(repeats) + 1):
    conditions = ["alone", "proc_sgemm", "mps_sgemm"]
    rng.shuffle(conditions)
    for condition in conditions:
        block += 1
        first = first_modes[condition][pair-1]
        modes = [first, "gpu" if first == "cpu" else "cpu"]
        for mode in modes:
            name = f"r{pair:02d}_b{block:02d}_{mode}_{condition}"
            schedule.append(dict(case_index=len(schedule)+1, block_index=block,
                pair_index=pair, condition=condition, mode=mode, name=name,
                json=name+".json", raw=name+".bin", run=name+".run.json",
                pusch_log=name+".pusch.log", scheduler=name+".scheduler.json",
                telemetry_before=name+".telemetry.before.xml",
                telemetry_after=name+".telemetry.after.xml",
                telemetry_continuous=name+".telemetry.continuous.csv"))
clock_file = out / "clock_control.json"
clock = json.loads(clock_file.read_text()) if clock_file.exists() else {
    "clocks_locked": None, "command": None, "result": "not changed by repeat runner"}
cpu_file = out / "cpu_selection.json"
cpu_control = json.loads(cpu_file.read_text()) if cpu_file.exists() else None
if clock.get("clocks_locked") is not None and type(clock["clocks_locked"]) is not bool:
    raise SystemExit("clock_control.clocks_locked must be a boolean or null")
manifest = dict(schema_version=1, seed=int(seed), repeats=int(repeats),
    randomization="each round shuffles conditions; each condition has balanced, randomly assigned CPU/GPU-first pairs",
    slots=int(slots), warmup=int(warmup), period_us=float(period), deadline_us=float(deadline),
    test_vector_sha256=vector, aerial_commit=revision,
    clocks_locked=clock.get("clocks_locked"), clock_control=clock,
    gates=["gate_cpu_alone", "gate_gpu_alone"], schedule=schedule,
    telemetry_note="Before/after snapshots and 100 ms continuous sampling; the observer may add measurement overhead.",
    telemetry_interval_ms=100, observer_cpu=int(observer_cpu),
    phy_cpu_requested=int(os.environ.get("SB_CUPHY_CPU", "0")),
    adversary_cpu_requested=int(os.environ["SB_ADVERSARY_CPU"]) if os.environ.get("SB_ADVERSARY_CPU") else None,
    cpu_control=cpu_control,
    scheduler_note="Read-only /proc observation after the PHY reports worker affinity; observation time is recorded.",
    adversary_mps_percentage=50)
with (out / "experiment.json").open("x") as f:
    json.dump(manifest, f, indent=2)
with (out / "schedule.tsv").open("x") as f:
    for row in schedule:
        f.write("\t".join(str(row[k]) for k in
            ("case_index", "block_index", "pair_index", "condition", "mode", "name")) + "\n")
PY
}

repeat_no_mps() {
    local rc=0
    pgrep -af '^([^[:space:]]*/)?nvidia-cuda-mps-(control|server)([[:space:]]|$)' >> "$LOGS/mps-state.log" || rc=$?
    if [[ $rc -ne 1 ]]; then
        log "MPS state is not clean (pgrep status $rc)"
        return 1
    fi
    printf '%s clean\n' "$(date -u +%FT%T.%NZ)" >> "$LOGS/mps-state.log"
}

repeat_start_mps() {
    local block=$1
    repeat_no_mps
    MPS_PIPE="$W/mps-repeat-$RUN_ID-b$block"
    MPS_LOG="$OUT/mps-block-$block"
    mkdir "$MPS_PIPE" "$MPS_LOG"
    # Mark ownership before starting so even a partially failed start is cleaned up.
    MPS_ON=1
    timeout 20 env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" \
        nvidia-cuda-mps-control -d >> "$LOGS/mps.log" 2>&1
    local i
    for ((i=0; i<20; i++)); do
        [[ -e $MPS_PIPE/control ]] && return 0
        sleep 1
    done
    log "MPS control socket did not appear"
    return 1
}

repeat_stop_mps() {
    [[ ${MPS_ON:-0} == 1 ]] || return 0
    echo quit | timeout 10 env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" nvidia-cuda-mps-control >> "$LOGS/mps.log" 2>&1
    local i rc
    for ((i=0; i<20; i++)); do
        rc=0
        pgrep -f '^([^[:space:]]*/)?nvidia-cuda-mps-(control|server)([[:space:]]|$)' >/dev/null || rc=$?
        if [[ $rc -eq 1 ]]; then MPS_ON=0; repeat_no_mps; return 0; fi
        [[ $rc -eq 0 ]] || return 1
        sleep 1
    done
    log "MPS processes survived requested shutdown"
    return 1
}

repeat_telemetry() {
    local name=$1 position=$2
    date -u +%FT%T.%NZ > "$OUT/$name.telemetry.$position.utc"
    timeout 15 nvidia-smi -q -x > "$OUT/$name.telemetry.$position.xml"
    timeout 15 nvidia-smi --query-gpu=timestamp,uuid,name,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,clocks.applications.graphics,clocks.applications.memory,utilization.gpu,utilization.memory \
        --format=csv > "$OUT/$name.telemetry.$position.csv"
}

repeat_observe_scheduler() {
    python3 - "$OUT/$1.pusch.log" "$OUT/$1.scheduler.json" "$OUT/$1.observer.stop" "$CASE_TIMEOUT" "$OBSERVER_CPU" "${SB_CUPHY_CPU:-0}" <<'PY'
import datetime, json, os, pathlib, re, sys, time
log, output, stop, limit, observer_cpu, expected_cpu = sys.argv[1:]
os.sched_setaffinity(0, {int(observer_cpu)})
stop = pathlib.Path(stop)
result = {"observed": False, "reason": "worker affinity not observed before case ended"}
deadline = time.monotonic() + int(limit) + 100
while time.monotonic() < deadline and not stop.exists():
    try:
        text = pathlib.Path(log).read_text(errors="replace")
    except FileNotFoundError:
        text = ""
    match = re.search(r"pid (\d+) set affinity to CPU (\d+)", text)
    if match:
        tid = int(match[1])
        try:
            status = pathlib.Path(f"/proc/{tid}/status").read_text()
            result = dict(observed=True, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                tid=tid, reported_cpu=int(match[2]), status=status,
                policy=os.sched_getscheduler(tid), rt_priority=os.sched_getparam(tid).sched_priority,
                affinity=sorted(os.sched_getaffinity(tid)), nice=os.getpriority(os.PRIO_PROCESS, tid),
                policy_names={0:"SCHED_OTHER", 1:"SCHED_FIFO", 2:"SCHED_RR", 3:"SCHED_BATCH", 5:"SCHED_IDLE"})
            result["expected_cpu"] = int(expected_cpu)
            result["affinity_matches_requested"] = result["affinity"] == [int(expected_cpu)]
        except (OSError, ProcessLookupError) as e:
            result = dict(observed=False, tid=tid, reason=str(e))
        break
    time.sleep(0.1)
with open(output, "w") as f:
    json.dump(result, f, indent=2)
PY
}

repeat_case() {
    local name=$1 mode=$2 isolation=$3 workload=$4
    [[ ! -e $OUT/$name.json && ! -e $OUT/$name.run.json && ! -e $OUT/$name.bin ]]
    repeat_telemetry "$name" before
    # One read-only NVML/nvidia-smi sampler per case, kept off the PHY CPU when possible.
    taskset -c "$OBSERVER_CPU" timeout --signal=TERM --kill-after=5 "$((CASE_TIMEOUT + 100))" \
        nvidia-smi --query-gpu=timestamp,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,clocks_throttle_reasons.active \
        --format=csv --loop-ms=100 > "$OUT/$name.telemetry.continuous.csv" 2> "$OUT/$name.telemetry.continuous.log" &
    TELEMETRY_PID=$!
    repeat_observe_scheduler "$name" &
    OBSERVER_PID=$!
    # run_case preserves process, adversary, correctness and raw-record gates.
    # Do not call it inside a conditional, which would disable Bash errexit.
    run_case "$name" "$mode" "$isolation" "$workload"
    kill -0 "$TELEMETRY_PID"
    kill -TERM "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=""
    touch "$OUT/$name.observer.stop"
    wait "$OBSERVER_PID"
    OBSERVER_PID=""
    repeat_telemetry "$name" after
    repeat_record_case "$name"
}

repeat_record_case() {
    python3 - "$OUT" "$1" <<'PY'
import json, pathlib, sys
out, name = pathlib.Path(sys.argv[1]), sys.argv[2]
experiment = json.loads((out / "experiment.json").read_text())
p = out / (name + ".run.json")
run = json.loads(p.read_text())
row = next((r for r in experiment["schedule"] if r["name"] == name), None)
run.update(row or {"condition":"alone", "gate":True})
run.update(seed=experiment["seed"], cuBB_SDK=str(pathlib.Path(out.parent, "acar")),
           clocks_locked=experiment["clocks_locked"], cpu_control=experiment.get("cpu_control"))
p.write_text(json.dumps(run, indent=2))
scheduler = json.loads((out / (name + ".scheduler.json")).read_text())
if scheduler.get("observed") is not True:
    raise SystemExit("PHY worker scheduler was not observed; refusing further cases")
if scheduler.get("affinity_matches_requested") is not True:
    raise SystemExit("Observed PHY worker affinity differs from requested CPU")
if name != "gate_cpu_alone":
    baseline = json.loads((out / "gate_cpu_alone.scheduler.json").read_text())
    for field in ("policy", "rt_priority"):
        if scheduler[field] != baseline[field]:
            raise SystemExit(f"Observed PHY {field} differs from the initial gate")
PY
}

repeat_finish() {
    local status=$1 rc=${2:-0} archive="$W/collection-repeat-$RUN_ID.tar.gz"
    trap - ERR TERM INT
    set +e
    [[ -z ${OBSERVER_PID:-} ]] || { kill -TERM "$OBSERVER_PID" 2>/dev/null; wait "$OBSERVER_PID" 2>/dev/null; }
    [[ -z ${TELEMETRY_PID:-} ]] || { kill -TERM "$TELEMETRY_PID" 2>/dev/null; wait "$TELEMETRY_PID" 2>/dev/null; }
    [[ -z ${STEP_PID:-} ]] || { kill -TERM "$STEP_PID" 2>/dev/null; wait "$STEP_PID" 2>/dev/null; }
    stop_adversary
    repeat_stop_mps || { status=error-mps-cleanup; rc=1; }
    repeat_no_mps || { status=error-mps-state; rc=1; }
    [[ -z ${WATCHDOG_PID:-} ]] || { kill -TERM "$WATCHDOG_PID" 2>/dev/null; wait "$WATCHDOG_PID" 2>/dev/null; }
    python3 - "$OUT/completion.json" "$status" "$rc" <<'PY'
import datetime, json, sys
with open(sys.argv[1], "w") as f:
    json.dump(dict(status=sys.argv[2], exit_code=int(sys.argv[3]),
        ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), f, indent=2)
PY
    if [[ $LOGS == "$W/"* && $OUT == "$W/"* && ! -e $archive ]]; then
        tar -czf "$archive" -C "$W" "${LOGS#"$W/"}" "${OUT#"$W/"}" || rc=1
        sha256sum "$archive" > "$archive.sha256" || rc=1
        log "Archive for SSH collection: $archive ($(stat -c %s "$archive") bytes)"
    else
        log "Refusing unsafe or existing archive path: $archive"; rc=1
    fi
    echo "=====SLOTBENCH-DONE status=$status archive=$archive====="
    return "$rc"
}

repeat_failure() {
    local rc=$1 line=$2 command=$3
    # ERR is inherited by command substitutions. Let the parent archive once.
    if [[ $BASHPID != "${REPEAT_MAIN_PID:-$BASHPID}" ]]; then exit "$rc"; fi
    trap - ERR TERM INT
    echo "=====SLOTBENCH-ERROR $line $command====="
    tail -n 25 "$CURRENT_LOG" 2>/dev/null || true
    repeat_finish error "$rc" || true
    exit "$rc"
}

repeat_cpu_setup() {
    # Validate against the initial full affinity, then narrow only this runner.
    # The PHY worker and explicitly pinned observers can expand their affinity
    # later within the cgroup's actual cpuset.
    python3 - "$OUT" "${SB_TELEMETRY_CPU:-}" "${SB_CUPHY_CPU:-0}" "${SB_ADVERSARY_CPU:-}" "${SB_BACKGROUND_CPUS:-}" <<'PY'
import json, os, pathlib, re, sys
out, observer_arg, phy_arg, adv_arg, background_arg = sys.argv[1:]
out = pathlib.Path(out)
allowed = sorted(os.sched_getaffinity(0))
phy = int(phy_arg)
adv = int(adv_arg) if adv_arg else None
if phy not in allowed or (adv is not None and adv not in allowed):
    raise SystemExit("Requested PHY/adversary CPU is outside the initial allowed affinity")
observer = int(observer_arg) if observer_arg else next((c for c in reversed(allowed) if c not in (phy, adv)), allowed[0])
if observer not in allowed:
    raise SystemExit("SB_TELEMETRY_CPU is outside the initial allowed affinity")
background = []
if background_arg:
    if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", background_arg):
        raise SystemExit("Invalid SB_BACKGROUND_CPUS: expected a CPU list such as 2,4-7")
    cpus = set()
    for part in background_arg.split(","):
        ends = [int(v) for v in part.split("-")]
        first, last = ends[0], ends[-1]
        if first > last or last-first+1 > len(allowed):
            raise SystemExit("Invalid SB_BACKGROUND_CPUS range")
        cpus.update(range(first, last+1))
    if not cpus.issubset(allowed):
        raise SystemExit("SB_BACKGROUND_CPUS is outside the initial allowed affinity")
    background = sorted(cpus)
with (out / "cpu_selection.json").open("x") as f:
    json.dump(dict(initial_allowed_affinity=allowed, phy_cpu_requested=phy,
        adversary_cpu_requested=adv, telemetry_cpu_requested=observer,
        background_cpulist=background_arg or None, background_cpus_requested=background), f, indent=2)
with (out / "observer_cpu.txt").open("x") as f:
    f.write(str(observer) + "\n")
PY
    read -r OBSERVER_CPU < "$OUT/observer_cpu.txt"
    if [[ -n ${SB_BACKGROUND_CPUS:-} ]]; then
        taskset -pc "$SB_BACKGROUND_CPUS" "$$" > "$OUT/runner_affinity.log"
    fi
    python3 - "$OUT/cpu_selection.json" "$$" <<'PY'
import json, os, pathlib, sys
p = pathlib.Path(sys.argv[1])
data = json.loads(p.read_text())
data["runner_affinity_after"] = sorted(os.sched_getaffinity(int(sys.argv[2])))
if data["background_cpus_requested"] and data["runner_affinity_after"] != data["background_cpus_requested"]:
    raise SystemExit("Runner affinity did not match requested background CPUs")
p.write_text(json.dumps(data, indent=2))
PY
}

repeat_schedule() {
    local case_index block pair condition mode name previous_block=0 isolation workload
    while IFS=$'\t' read -r case_index block pair condition mode name; do
        if [[ $block != "$previous_block" ]]; then
            repeat_stop_mps
            repeat_no_mps
            if [[ $condition == mps_sgemm ]]; then repeat_start_mps "$block"; fi
            previous_block=$block
        fi
        isolation=none; workload=sgemm
        [[ $condition != alone ]] || workload=none
        [[ $condition != mps_sgemm ]] || isolation=mps
        log "Measured case $case_index/$((REPEATS * 6)): pair=$pair block=$block condition=$condition"
        repeat_case "$name" "$mode" "$isolation" "$workload"
    done < "$OUT/schedule.tsv"
    repeat_stop_mps
    repeat_no_mps
}

repeat_main() {
    export W="${SB_CUPHY_WORKDIR:-/workspace/cuphy-lockstep}"
    export S="$W/acar" ACAR_COMMIT cuBB_SDK="$W/acar"
    export SB_CUPHY_LINE_BUFFERED=1
    REPEATS=${SB_CUPHY_REPEATS:-6}; REPEAT_SEED=${SB_CUPHY_REPEAT_SEED:-20261002}
    SLOTS=${SB_CUPHY_LOCKSTEP_SLOTS:-5000}; WARMUP=${SB_CUPHY_LOCKSTEP_WARMUP:-1000}
    PERIOD=${SB_CUPHY_LOCKSTEP_PERIOD_US:-500}; DEADLINE=${SB_CUPHY_LOCKSTEP_DEADLINE_US:-500}
    CASE_TIMEOUT=${SB_CUPHY_CASE_TIMEOUT_S:-180}
    local overall_timeout=${SB_CUPHY_REPEAT_TIMEOUT_S:-1800}
    [[ $REPEATS =~ ^[0-9]+$ && $REPEAT_SEED =~ ^[0-9]+$ && $SLOTS =~ ^[0-9]+$ && $WARMUP =~ ^[0-9]+$ && $CASE_TIMEOUT =~ ^[0-9]+$ && $overall_timeout =~ ^[0-9]+$ ]]
    (( REPEATS >= 2 && REPEATS <= 20 && REPEATS % 2 == 0 && SLOTS >= 1 && SLOTS <= 5000 && WARMUP <= 1000 && CASE_TIMEOUT >= 10 && CASE_TIMEOUT <= 600 && overall_timeout >= 60 && overall_timeout <= 7200 ))
    [[ $PERIOD == 500 && $DEADLINE == 500 ]]
    [[ $(git -C "$S" rev-parse HEAD) == "$ACAR_COMMIT" ]]
    PUSCH="$W/build/cuPHY/examples/pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe"
    ADV="$W/sb/slotbench/bin/adversary"
    TV="$W/tv/GPU_test_input/TVnr_7304_PUSCH_gNB_CUPHY_s0p0.h5"
    [[ -x $PUSCH && -x $ADV && -s $TV && -s $S/cuPHY/nvlog/config/nvlog_config.yaml ]]
    TV_SHA=$(sha256sum "$TV" | cut -d' ' -f1)
    [[ $TV_SHA == 85796206a068087c3e2e03bafb7a22118fa7f27ce87f94f00afcf1ec4ac106ca ]]
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
    export LOGS="$W/logs-repeat-$RUN_ID" OUT="$W/out-repeat-$RUN_ID"
    mkdir "$LOGS" "$OUT"
    cd "$W"
    REPEAT_MAIN_PID=$BASHPID
    CURRENT_LOG="$LOGS/header.txt"; ADV_PID=""; STEP_PID=""; OBSERVER_PID=""; TELEMETRY_PID=""; WATCHDOG_PID=""; MPS_ON=0
    trap 'repeat_failure "$?" "$LINENO" "$BASH_COMMAND"' ERR
    trap 'repeat_failure 130 "$LINENO" interrupted' TERM INT
    # A Python sleep has no separate sleep child to leave after cancellation.
    python3 -c 'import os,signal,sys,time; time.sleep(int(sys.argv[1])); os.kill(int(sys.argv[2]),signal.SIGTERM)' "$overall_timeout" "$$" &
    WATCHDOG_PID=$!
    { date -u; nvidia-smi; uname -a; lscpu; cat /proc/self/status; } > "$CURRENT_LOG" 2>&1
    repeat_cpu_setup
    repeat_no_mps
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_MPS_CLIENT_PRIORITY \
        CUDA_MPS_PINNED_DEVICE_MEM_LIMIT CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING
    local adapter="$W/sb/slotbench/cuphy" patch="$W/applied-adapter.patch"
    [[ -s $patch ]]
    git -C "$S" apply --reverse --check "$patch"
    cmp "$patch" "$adapter/patches/cuphy-lockstep.patch"
    cmp "$adapter/cuphy_lockstep.cu" "$S/cuPHY/examples/pusch_rx_multi_pipe/cuphy_lockstep.cu"
    cmp "$adapter/cuphy_lockstep_stamps.cu" "$S/cuPHY/src/cuphy_channels/cuphy_lockstep_stamps.cu"
    cp "$patch" "$OUT/applied-adapter.patch"
    sha256sum "$patch" "$adapter"/cuphy_lockstep*.cu "$adapter"/cuphy_lockstep*.h > "$OUT/adapter_sha256.txt"
    sha256sum "$PUSCH" "$ADV" > "$OUT/binary_sha256.txt"
    sha256sum "$TV" > "$OUT/test_vector_sha256.txt"
    sha256sum "$S/cuPHY/nvlog/config/nvlog_config.yaml" > "$OUT/nvlog_config_sha256.txt"
    git -C "$W/sb" rev-parse HEAD > "$OUT/slotbench_commit.txt"
    printf '%s\n' "$ACAR_COMMIT" > "$OUT/aerial_commit.txt"
    [[ -z ${SB_CUPHY_CLOCK_EVIDENCE:-} ]] || cp "$SB_CUPHY_CLOCK_EVIDENCE" "$OUT/clock_control.json"
    repeat_plan
    local measured_slots=$SLOTS
    SLOTS=64
    repeat_case gate_cpu_alone cpu none none
    repeat_case gate_gpu_alone gpu none none
    SLOTS=$measured_slots
    repeat_schedule
    repeat_finish ok
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then repeat_main "$@"; fi
