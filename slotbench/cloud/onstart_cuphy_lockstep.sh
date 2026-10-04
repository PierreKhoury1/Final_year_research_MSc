#!/bin/bash
# CUDA 13.3 / Ubuntu 24.04 / A100: actual cuPHY TC7304 periodic replay.
# Usage: onstart_cuphy_lockstep.sh [CONFIG [BRANCH [REPO]]]
# CONFIG is reserved for vast.py. The adapter and upstream revision are pinned.
set -Eeuo pipefail

ACAR_COMMIT=4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c
log() { echo "[cuphy-lockstep $(date -u +%H:%M:%S)] $*"; }

stop_adversary() {
    [[ -n ${ADV_PID:-} ]] || return 0
    kill -TERM "$ADV_PID" 2>/dev/null || true
    local i
    for ((i=0; i<20; i++)); do kill -0 "$ADV_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$ADV_PID" 2>/dev/null || true
    wait "$ADV_PID" 2>/dev/null || true
    ADV_PID=""
}

stop_telemetry() {
    [[ -n ${TEL_PID:-} ]] || return 0
    kill "$TEL_PID" 2>/dev/null || true
    wait "$TEL_PID" 2>/dev/null || true
    TEL_PID=""
}

cleanup() {
    stop_adversary
    stop_telemetry
    if [[ ${MPS_ON:-0} -eq 1 ]]; then
        echo quit | timeout 10 env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" nvidia-cuda-mps-control >> "$LOGS/mps.log" 2>&1 || true
        MPS_ON=0
    fi
    if [[ -d ${MPS_LOG:-/nonexistent} ]]; then cp -a "$MPS_LOG" "$OUT/mps_logs" 2>/dev/null || true; fi
}

finish() {
    local status=$1 archive="${COLLECTION_ARCHIVE:-$W/collection.tar.gz}" bytes lines sha
    trap - ERR TERM INT
    [[ -n ${STEP_PID:-} ]] && kill -TERM "$STEP_PID" 2>/dev/null || true
    cleanup
    # Preserve complete originals. Gzip all logs/results once, excluding the 64 MB
    # input vector (its upstream revision, generation log and SHA256 are recorded).
    [[ $LOGS == "$W/"* && $OUT == "$W/"* ]]
    tar --warning=no-file-changed -cf - -C "$W" "${LOGS#"$W/"}" "${OUT#"$W/"}" | gzip -9 > "$archive"
    bytes=$(stat -c %s "$archive")
    lines=$(( (4 * ((bytes + 2) / 3) + 75) / 76 + 2 ))
    sha=$(sha256sum "$archive" | cut -d' ' -f1)
    if (( lines > 12000 )); then
        # The controller only ever sees the last 20000 log lines, so a large archive goes out as parts of
        # 800 kB (about 11000 lines each) with a pause between them long enough for the controller to poll
        # (vast.py harvests every valid block it sees; SB_PART_GAP_S must exceed twice its --poll-s).
        local parts_dir="$W/parts" n i part
        rm -rf "$parts_dir"; mkdir -p "$parts_dir"
        split -b 800k -d -a 2 --additional-suffix=.bin "$archive" "$parts_dir/p"
        n=$(ls "$parts_dir" | wc -l)
        echo "=====SLOTBENCH-PARTS cuphy-lockstep/results $n $sha====="
        log "archive is $bytes bytes: sending $n parts, ${SB_PART_GAP_S:-120} s apart"
        i=0
        for part in "$parts_dir"/p*.bin; do
            i=$((i + 1))
            echo "=====SLOTBENCH-BEGIN cuphy-lockstep/results.part$(printf %02d "$i") $(sha256sum "$part" | cut -d' ' -f1)====="
            base64 -w 76 "$part"
            echo "=====SLOTBENCH-END cuphy-lockstep/results.part$(printf %02d "$i")====="
            if (( i < n )); then sleep "${SB_PART_GAP_S:-120}"; fi
        done
        echo "=====SLOTBENCH-PARTS cuphy-lockstep/results $n $sha====="
    else
        echo "=====SLOTBENCH-BEGIN cuphy-lockstep/results $sha====="
        base64 -w 76 "$archive"
        echo "=====SLOTBENCH-END cuphy-lockstep/results====="
    fi
    echo "=====SLOTBENCH-DONE status=$status====="
    if [[ ${SB_CUPHY_FINISH_EXIT:-0} == 1 ]]; then return; fi
    # Leave time for the controller to collect, then stop the container so a lost controller cannot
    # leave a finished GPU instance billing (an exited instance only pays for storage).
    sleep "${SB_POST_DONE_GRACE_S:-900}"
    echo "=====SLOTBENCH-SELF-STOP====="
    kill -TERM 1; sleep 10; kill -KILL 1
    sleep infinity
}

step() { # step NAME TIMEOUT_SECONDS COMMAND...
    local name=$1 limit=$2 elapsed=0 rc=0
    shift 2
    log "$name (limit ${limit}s)"
    CURRENT_LOG="$LOGS/$name.log"
    timeout --signal=TERM --kill-after=20 "$limit" "$@" > "$CURRENT_LOG" 2>&1 &
    STEP_PID=$!
    while kill -0 "$STEP_PID" 2>/dev/null; do
        sleep 5; elapsed=$((elapsed + 5))
        if (( elapsed % 300 == 0 )); then log "$name: ${elapsed}s: $(tail -n 1 "$CURRENT_LOG" | cut -c1-160)"; fi
    done
    wait "$STEP_PID" || rc=$?
    STEP_PID=""
    log "$name finished rc=$rc after about ${elapsed}s"
    return "$rc"
}

clone_sources() {
    git clone -q --depth 1 --branch "$SB_BRANCH" "$SB_REPO" "$W/sb"
    git init -q "$S"
    git -C "$S" remote add origin https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git
    GIT_LFS_SKIP_SMUDGE=1 git -C "$S" fetch -q --depth 1 origin "$ACAR_COMMIT"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$S" checkout -q FETCH_HEAD
    [[ $(git -C "$S" rev-parse HEAD) == "$ACAR_COMMIT" ]]
    git -C "$W/sb" rev-parse HEAD > "$OUT/slotbench_commit.txt"
    printf '%s\n' "$ACAR_COMMIT" > "$OUT/aerial_commit.txt"
}

apply_adapter() {
    local adapter="$W/sb/slotbench/cuphy" example="$S/cuPHY/examples/pusch_rx_multi_pipe" channels="$S/cuPHY/src/cuphy_channels"
    local patch="$adapter/patches/cuphy-lockstep.patch"
    [[ -s $patch ]]
    git -C "$S" apply --check "$patch"
    git -C "$S" apply "$patch"
    cp "$patch" "$OUT/applied-adapter.patch"
    cp "$patch" "$W/applied-adapter.patch"
    cp "$adapter/cuphy_lockstep.cu" "$adapter/cuphy_lockstep.h" "$example/"
    cp "$W/sb/slotbench/common/clock_fit.h" "$W/sb/slotbench/common/host_time.h" "$W/sb/slotbench/common/json_writer.h" \
        "$W/sb/slotbench/common/timetable.h" "$example/"
    cp "$adapter/cuphy_lockstep_stamps.cu" "$adapter/cuphy_lockstep_stamps.h" "$channels/"
    sha256sum "$patch" "$adapter"/cuphy_lockstep*.cu "$adapter"/cuphy_lockstep*.h > "$OUT/adapter_sha256.txt"
    git -C "$S" diff --stat > "$OUT/upstream_patch_stat.txt"
}

start_adversary() {
    local name=$1 isolation=$2 rows=0 i
    local extra=() affinity=()
    if [[ -n ${SB_ADVERSARY_CPU:-} ]]; then
        [[ $SB_ADVERSARY_CPU =~ ^[0-9]+$ ]] || { log "Invalid SB_ADVERSARY_CPU"; return 2; }
        affinity=(taskset -c "$SB_ADVERSARY_CPU")
    fi
    [[ $isolation == mps ]] && extra+=(CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50)
    env "${extra[@]}" "${affinity[@]}" "$ADV" --workload sgemm --size "${ADV_SIZE:-4096}" --duty 100 --prio default --gpu 0 \
        "${ADV_EXTRA[@]}" --seconds 600 --timeline "$OUT/$name.adversary.csv" --out "$OUT/$name.adversary.json" > "$OUT/$name.adversary.log" 2>&1 &
    ADV_PID=$!
    for ((i=0; i<90; i++)); do
        sleep 1
        kill -0 "$ADV_PID" 2>/dev/null || { log "adversary exited before readiness"; return 1; }
        rows=$(awk -F, 'NR>1 && $2+0>0 {n++} END {print n+0}' "$OUT/$name.adversary.csv" 2>/dev/null || echo 0)
        (( rows >= 3 )) && return 0
    done
    log "adversary did not report completed work within 90s"
    return 1
}

run_case() {
    local name=$1 mode=$2 isolation=$3 workload=$4 rc=0 start end
    local extra=() launch_prefix=() pusch_args=(-i "$TV" -m 1 -r 1)
    case ${SB_CUPHY_LINE_BUFFERED:-0} in
        0) ;;
        1) launch_prefix=(stdbuf -oL -eL) ;;
        *) log "Invalid SB_CUPHY_LINE_BUFFERED"; return 2 ;;
    esac
    if [[ -n ${SB_CUPHY_CPU:-} ]]; then
        [[ $SB_CUPHY_CPU =~ ^[0-9]+$ ]] || { log "Invalid SB_CUPHY_CPU"; return 2; }
        pusch_args+=(-c "$SB_CUPHY_CPU")
    fi
    [[ $isolation == mps ]] && extra+=(CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG")
    if [[ -n ${GATE_TT:-} ]]; then
        rm -f "$GATE_TT"
        extra+=(SB_CUPHY_LOCKSTEP_TIMETABLE="$GATE_TT" SB_CUPHY_LOCKSTEP_BUSY_US="$BUSY_US")
    fi
    if [[ $workload == sgemm ]]; then
        # never run the 5G slot without its tenant in a contended case: that would pass as "contended" data
        if ! start_adversary "$name" "$isolation"; then
            log "case $name: tenant failed to start; cuPHY not run"
            stop_adversary
            echo "tenant failed to start" > "$OUT/$name.skipped"
            return 1
        fi
    fi
    log "case $name: launcher=$mode isolation=$isolation workload=$workload"
    start=$(date -u +%FT%TZ)
    CURRENT_LOG="$OUT/$name.pusch.log"
    env "${extra[@]}" SB_CUPHY_LOCKSTEP_MODE="$mode" SB_CUPHY_LOCKSTEP_LABEL="$name" SB_CUPHY_LOCKSTEP_SLOTS="$SLOTS" \
        SB_CUPHY_LOCKSTEP_WARMUP="$WARMUP" SB_CUPHY_LOCKSTEP_PERIOD_US="$PERIOD" \
        SB_CUPHY_LOCKSTEP_DEADLINE_US="$DEADLINE" SB_CUPHY_LOCKSTEP_OUT="$OUT/$name.json" \
        SB_CUPHY_LOCKSTEP_RAW="$OUT/$name.bin" timeout --signal=TERM --kill-after=20 "$CASE_TIMEOUT" \
        "${launch_prefix[@]}" "$PUSCH" "${pusch_args[@]}" > "$CURRENT_LOG" 2>&1 &
    STEP_PID=$!
    wait "$STEP_PID" || rc=$?
    STEP_PID=""
    end=$(date -u +%FT%TZ)
    if [[ $workload == sgemm ]]; then
        kill -0 "$ADV_PID" 2>/dev/null || { log "adversary died during $name"; rc=1; }
        stop_adversary
        python3 - "$OUT/$name.adversary.json" > "$OUT/$name.adversary.check.log" 2>&1 <<'PY' || rc=1
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
if data.get("ok") is not True or data.get("units", 0) <= 0:
    raise SystemExit("adversary did not finish successfully with completed work")
print("adversary summary valid")
PY
    fi
    python3 - "$OUT/$name.run.json" "$name" "$mode" "$isolation" "$workload" "$start" "$end" "$rc" "$TV_SHA" "$ACAR_COMMIT" "$SLOTS" "$WARMUP" "$PERIOD" "$DEADLINE" "$TV" "$PUSCH" "${SB_ADVERSARY_CPU:-}" "${SB_CUPHY_LINE_BUFFERED:-0}" "${pusch_args[@]}" <<'PY'
import json, sys
p, name, mode, isolation, workload, start, end, rc, tv_sha, commit, slots, warmup, period, deadline, tv, pusch, adversary_cpu, line_buffered, *args = sys.argv[1:]
with open(p, "w") as f:
    json.dump(dict(name=name, mode=mode, isolation=isolation, workload=workload, start_utc=start,
                   end_utc=end, exit_code=int(rc), test_vector_sha256=tv_sha, aerial_commit=commit,
                   slots=int(slots), warmup=int(warmup), period_us=float(period), deadline_us=float(deadline),
                   command=(["stdbuf", "-oL", "-eL"] if line_buffered=="1" else []) + [pusch, *args],
                   line_buffered=line_buffered=="1",
                   phy_cpu_requested=int(args[args.index("-c")+1]) if "-c" in args else 0,
                   adversary_cpu_requested=int(adversary_cpu) if adversary_cpu and workload!="none" else None,
                   adversary_mps_percentage=50 if isolation=="mps" and workload!="none" else None), f, indent=2)
PY
    (( rc == 0 )) || { log "$name process failed rc=$rc"; return "$rc"; }
    python3 "$W/sb/slotbench/cloud/cuphy_lockstep_check.py" "$OUT/$name.json" "$OUT/$name.bin" \
        --mode "$mode" --slots "$SLOTS" --warmup "$WARMUP" --period-us "$PERIOD" --deadline-us "$DEADLINE" > "$OUT/$name.check.log" 2>&1
    log "$name: $(cat "$OUT/$name.check.log")"
}

main() {
    export DEBIAN_FRONTEND=noninteractive
    export SB_BRANCH="${2:-${SB_BRANCH:-codex/continue-lockstep}}"
    export SB_REPO="${3:-${SB_REPO:-https://github.com/PierreKhoury1/Final_year_research_MSc}}"
    export W="${SB_CUPHY_WORKDIR:-/workspace/cuphy-lockstep}"
    export S="$W/acar" LOGS="$W/logs" OUT="$W/out" ACAR_COMMIT
    mkdir -p "$W" "$LOGS" "$OUT"
    cd "$W"
    CURRENT_LOG="$LOGS/header.txt"; ADV_PID=""; STEP_PID=""; MPS_ON=0; ADV_EXTRA=(); GATE_TT=""; TEL_PID=""
    trap 'echo "=====SLOTBENCH-ERROR $LINENO $BASH_COMMAND====="; tail -n 25 "$CURRENT_LOG" 2>/dev/null || true; finish error' ERR
    trap 'echo "=====SLOTBENCH-ERROR 0 interrupted====="; finish interrupted' TERM INT
    SLOTS=${SB_CUPHY_LOCKSTEP_SLOTS:-1000}; WARMUP=${SB_CUPHY_LOCKSTEP_WARMUP:-100}
    PERIOD=${SB_CUPHY_LOCKSTEP_PERIOD_US:-500}; DEADLINE=${SB_CUPHY_LOCKSTEP_DEADLINE_US:-500}
    CASE_TIMEOUT=${SB_CUPHY_CASE_TIMEOUT_S:-180}
    [[ $SLOTS =~ ^[0-9]+$ && $WARMUP =~ ^[0-9]+$ && $CASE_TIMEOUT =~ ^[0-9]+$ ]]
    (( SLOTS >= 1 && SLOTS <= 5000 && WARMUP <= 1000 && CASE_TIMEOUT >= 10 && CASE_TIMEOUT <= 600 ))
    [[ $PERIOD =~ ^[0-9]+([.][0-9]+)?$ && $DEADLINE =~ ^[0-9]+([.][0-9]+)?$ ]]
    awk -v p="$PERIOD" -v d="$DEADLINE" 'BEGIN {exit !(p >= 1 && p <= 1000000 && d > 0)}'
    local main_pid=$$ watchdog_s
    watchdog_s=$(awk -v h="${SB_MAX_HOURS:-1.5}" 'BEGIN {printf "%d",h*3600}')
    (( watchdog_s >= 60 ))
    (sleep "$watchdog_s"; kill -TERM "$main_pid"; sleep 60; kill -TERM 1; sleep 10; kill -KILL 1) &
    { date -u; nvidia-smi; nproc; free -g; df -h /; } > "$LOGS/header.txt" 2>&1
    head -14 "$LOGS/header.txt"
    local gpu_cc jobs trt=10.14.1.48-1+cuda13.0
    gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    [[ $gpu_cc == 80 ]] || { log "This bootstrap is validated for A100 SM80 only"; return 3; }
    jobs=${SB_CUPHY_JOBS:-${JOBS:-$(( $(nproc) > 32 ? 16 : ($(nproc) > 8 ? $(nproc) / 2 : 4) ))}}
    [[ $jobs =~ ^[0-9]+$ ]]
    (( jobs >= 1 && jobs <= 64 ))
    step apt 1200 bash -c 'apt-get -o Acquire::Retries=3 update -y && apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        git git-lfs cmake ninja-build build-essential pkg-config libhdf5-dev hdf5-tools libyaml-dev python3-pip python3-venv \
        numactl wget unzip ca-certificates curl aria2 procps "libnvinfer10=$1" "libnvinfer-headers-dev=$1"' _ "$trt"
    export -f clone_sources apply_adapter gcmake deps tv
    step clone 300 bash -e -o pipefail -c clone_sources
    step adapter 60 bash -e -o pipefail -c apply_adapter
    step deps 600 bash -e -o pipefail -c deps
    mkdir -p "$W/wrapper"
    cat > "$W/wrapper/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
project(cuphy_only LANGUAGES C CXX ASM CUDA)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}/cccl)
set(ENV{cuBB_SDK} ${ACAR_SRC})
set(ENABLE_CUMAC OFF CACHE BOOL "" FORCE)
set(NVIPC_FMTLOG_ENABLE ON)
add_definitions(-DNVIPC_FMTLOG_ENABLE)
add_subdirectory(${ACAR_SRC}/cuPHY cuPHY)
CMAKE
    # Only TensorRT's shared runtime and headers are required; avoid its 2.9GB static development archive.
    local trt_so=/usr/lib/x86_64-linux-gnu/libnvinfer.so.10
    [[ -e $trt_so ]]
    step configure 180 cmake -S "$W/wrapper" -B "$W/build" -GNinja -DACAR_SRC="$S" \
        -DCMAKE_TOOLCHAIN_FILE="$S/cuPHY/cmake/toolchains/x86-64" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=80-real -DBUILD_DOCS=OFF -DENABLE_TESTS=OFF -DNVINFER:FILEPATH="$trt_so"
    step cuphy_build 1200 cmake --build "$W/build" --target cuphy_ex_pusch_rx_multi_pipe -- -j"$jobs"
    PUSCH="$W/build/cuPHY/examples/pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe"
    [[ -x $PUSCH ]]
    step adversary_build 180 make -C "$W/sb/slotbench" SM=80 CUDA_HOME=/usr/local/cuda bin/adversary
    ADV="$W/sb/slotbench/bin/adversary"
    # Repeated campaigns upload the audited cached vector over SSH, then run
    # repeat_cuphy_lockstep.sh. Avoid downloading MATLAB or starting smoke cases.
    if [[ ${SB_CUPHY_PREPARE_ONLY:-0} == 1 ]]; then
        sha256sum "$PUSCH" "$ADV" > "$OUT/executable_sha256.txt"
        date -u +%FT%TZ > "$W/prepared.txt"
        log "Build prepared for SSH vector upload and repeated experiments"
        finish prepared
        return
    fi
    step tv "${TV_TIMEOUT_S:-2400}" bash -e -o pipefail -c tv
    TV="$W/tv/GPU_test_input/TVnr_7304_PUSCH_gNB_CUPHY_s0p0.h5"
    [[ -s $TV ]]
    TV_SHA=$(sha256sum "$TV" | cut -d' ' -f1)
    printf '%s  %s\n' "$TV_SHA" "$TV" > "$OUT/test_vector_sha256.txt"
    h5dump -H "$TV" > "$OUT/test_vector_schema.txt"
    if [[ ${SB_CUPHY_CAMPAIGN:-} == harq ]]; then harq_campaign; finish ok; return; fi
    local mps_scan_rc=0
    if [[ ${SB_CUPHY_CAMPAIGN:-} == gate ]]; then gate_campaign; finish ok; return; fi
    pgrep -f '^([^[:space:]]*/)?nvidia-cuda-mps-(control|server)([[:space:]]|$)' >/dev/null || mps_scan_rc=$?
    if [[ $mps_scan_rc -eq 0 ]]; then
        log "Existing MPS process makes non-MPS phases invalid"; return 3
    elif [[ $mps_scan_rc -ne 1 ]]; then
        log "Cannot inspect MPS processes; refusing non-MPS phases"; return 3
    fi
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_MPS_CLIENT_PRIORITY \
        CUDA_MPS_PINNED_DEVICE_MEM_LIMIT CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING
    run_case cpu_alone cpu none none
    run_case gpu_alone gpu none none
    # Any failed correctness/raw gate above stops execution before contention tests.
    run_case cpu_proc_sgemm cpu none sgemm
    run_case gpu_proc_sgemm gpu none sgemm
    MPS_PIPE="$W/mps-pipe"; MPS_LOG="$W/mps-log"
    mkdir -p "$MPS_PIPE" "$MPS_LOG"
    timeout 20 env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" nvidia-cuda-mps-control -d > "$LOGS/mps.log" 2>&1
    MPS_ON=1
    sleep 2
    [[ -e $MPS_PIPE/control ]]
    run_case cpu_mps_sgemm cpu mps sgemm
    run_case gpu_mps_sgemm gpu mps sgemm
    finish ok
}

# HARQ-budget campaign (SB_CUPHY_CAMPAIGN=harq): idle GPU, three launch variants (CPU, CPU with a GPU
# keep-alive kernel, GPU) in randomised triplets, a tight completion deadline (SB_CUPHY_LOCKSTEP_DEADLINE_US,
# default 200 us) and SB_CUPHY_REPEATS triplets of SB_CUPHY_LOCKSTEP_SLOTS slots. Everything after the build
# reuses the audited repeat/activity runners; the raw records allow any deadline to be re-scored afterwards.
harq_campaign() {
    local hc="$W/host-controls" d
    mkdir -p "$hc"
    log "harq campaign: host probe, clock-lock attempt (recorded, never assumed), then control_cuphy_activity.sh"
    bash "$W/sb/slotbench/cloud/cuphy_host_probe.sh" "$hc/probe" 0 > "$LOGS/host_probe.log" 2>&1 || log "host probe rc=$?"
    python3 - "$hc/clock_control.json" <<'PY'
import json, subprocess, sys
def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()
sm_rc, sm_out = run(["nvidia-smi", "-lgc", "1110,1110"])
mem_rc, mem_out = run(["nvidia-smi", "-lmc", "1215,1215"])
pl_rc, pl_out = run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader"])
locked = sm_rc == 0 and "not supported" not in sm_out.lower() and "permission" not in sm_out.lower()
json.dump(dict(clocks_locked=locked, requested_sm_mhz=1110, requested_memory_mhz=1215,
               sm_command=["nvidia-smi", "-lgc", "1110,1110"], sm_command_rc=sm_rc, sm_result=sm_out,
               memory_command=["nvidia-smi", "-lmc", "1215,1215"], memory_command_rc=mem_rc, memory_result=mem_out,
               power_limit=pl_out, interpretation="Use sampled actual clocks; do not infer a lock from return codes."),
          open(sys.argv[1], "w"), indent=2)
print("clocks_locked", locked, "|", sm_out[:100])
PY
    if [[ -s $hc/probe/recommended_cpus.env ]]; then
        # shellcheck disable=SC1091
        source "$hc/probe/recommended_cpus.env"
        export SB_CUPHY_CPU SB_ADVERSARY_CPU SB_TELEMETRY_CPU SB_BACKGROUND_CPUS
    fi
    export SB_CUPHY_CLOCK_EVIDENCE="$hc/clock_control.json"
    export SB_CUPHY_TV_SHA256="$TV_SHA"   # the vector generated above (its log and schema are in OUT)
    export SB_CUPHY_REPEATS="${SB_CUPHY_REPEATS:-12}" SB_CUPHY_REPEAT_SEED="${SB_CUPHY_REPEAT_SEED:-20261003}"
    export SB_CUPHY_LOCKSTEP_SLOTS="${SB_CUPHY_LOCKSTEP_SLOTS:-5000}" SB_CUPHY_LOCKSTEP_WARMUP="${SB_CUPHY_LOCKSTEP_WARMUP:-1000}"
    export SB_CUPHY_LOCKSTEP_DEADLINE_US="${SB_CUPHY_LOCKSTEP_DEADLINE_US:-200}" SB_CUPHY_LOCKSTEP_PERIOD_US=500
    export SB_CUPHY_REPEAT_TIMEOUT_S="${SB_CUPHY_REPEAT_TIMEOUT_S:-3000}"
    step harq_campaign "$((SB_CUPHY_REPEAT_TIMEOUT_S + 120))" bash "$W/sb/slotbench/cloud/control_cuphy_activity.sh" || log "campaign rc=$?"
    # the runner writes out-repeat-*/logs-repeat-* under W; fold them into OUT/LOGS for the archive
    for d in "$W"/out-repeat-*; do [[ -d $d ]] && cp -a "$d" "$OUT/$(basename "$d")"; done
    for d in "$W"/logs-repeat-*; do [[ -d $d ]] && cp -a "$d" "$LOGS/$(basename "$d")"; done
    cp -a "$hc" "$OUT/host-controls" 2>/dev/null || true
    for d in "$OUT"/out-repeat-*; do
        [[ -d $d ]] || continue
        if python3 "$W/sb/slotbench/analysis/cuphy_deadline_sweep.py" "$d" --output-dir "$d/deadline_sweep" > "$LOGS/deadline_sweep.log" 2>&1; then
            cat "$d/deadline_sweep/summary.txt"
        else
            log "deadline sweep rc=$?"
        fi
    done
}

# Time-aware gating campaign (SB_CUPHY_CAMPAIGN=gate). The cuPHY slot (CPU launcher, 500 us period)
# publishes its timetable; the SGEMM tenant either ignores it (--gate-mode observe: ungated control,
# counted over the same window) or only issues a GEMM when it should finish before the next slot
# (--gate-mode on). Sharing: time-slicing between processes (proc) and MPS (tenant at 50% threads).
# busy_us comes from a separate calibration run of cuPHY alone: p99.9 of completion latency over slots
# launched on time (host launch stalls excluded) + SB_GATE_MARGIN_US, and must leave room in the period.
# Per repeat the cases of each sharing block run in a seeded random order (schedule.txt); every case
# logs the tenant's in-window GEMMs (units.bin) so overlap with the slot can be measured afterwards.
gate_campaign() {
    local hc="$W/host-controls" r n iso sizes repeats seed rc_scan=0
    mkdir -p "$hc"
    # same MPS hygiene as the default campaign: non-MPS cases are invalid if a daemon is already up
    pgrep -f '^([^[:space:]]*/)?nvidia-cuda-mps-(control|server)([[:space:]]|$)' >/dev/null || rc_scan=$?
    if [[ $rc_scan -ne 1 ]]; then log "MPS process present or cannot inspect (rc=$rc_scan); refusing"; return 3; fi
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_MPS_CLIENT_PRIORITY \
        CUDA_MPS_PINNED_DEVICE_MEM_LIMIT CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING
    bash "$W/sb/slotbench/cloud/cuphy_host_probe.sh" "$hc/probe" 0 > "$LOGS/host_probe.log" 2>&1 || log "host probe rc=$?"
    if [[ -s $hc/probe/recommended_cpus.env ]]; then
        # shellcheck disable=SC1091
        source "$hc/probe/recommended_cpus.env"
        export SB_CUPHY_CPU SB_ADVERSARY_CPU
    fi
    [[ -n ${SB_CUPHY_CPU:-} && -n ${SB_ADVERSARY_CPU:-} ]] || { log "no CPU pinning available; refusing"; return 3; }
    cp -a "$hc" "$OUT/host-controls" 2>/dev/null || true
    sizes=${SB_GATE_SIZES:-"1024 768 512"}; repeats=${SB_GATE_REPEATS:-4}; seed=${SB_GATE_SEED:-20261004}
    [[ $repeats =~ ^[0-9]+$ ]] && (( repeats >= 1 && repeats <= 10 )) || { log "invalid SB_GATE_REPEATS"; return 2; }
    [[ $seed =~ ^[0-9]+$ ]] || { log "invalid SB_GATE_SEED"; return 2; }
    for n in $sizes; do
        [[ $n =~ ^[0-9]+$ ]] && (( n >= 64 && n <= 8192 )) || { log "invalid size $n in SB_GATE_SIZES"; return 2; }
    done
    # GPU clocks cannot be locked in these containers: record what they actually were
    { nvidia-smi -lgc 1410,1410; echo "rc=$?"; } > "$OUT/clock_lock_attempt.txt" 2>&1 || true
    taskset -c "${SB_TELEMETRY_CPU:-0}" nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
        --format=csv,noheader -lms 250 > "$OUT/gpu_telemetry.csv" 2>&1 &
    TEL_PID=$!
    GATE_TT="$W/slot_timetable.bin"; BUSY_US=300
    mkdir -p "$W/units"
    # up to 3 calibration runs (names stay outside the analysis patterns); each must pass its own check
    local k busy=""
    BUSY_US=300   # placeholder published by the calibration runs themselves (no tenant reads it)
    for k in 1 2 3; do
        run_case "calib_alone_$k" cpu none none || continue
        grep -q '"valid": true' "$OUT/calib_alone_$k.check.log" || { log "calib_alone_$k failed its check"; continue; }
        busy=$(python3 "$W/sb/slotbench/analysis/gate_summary.py" --busy "$OUT/calib_alone_$k" \
            --margin-us "${SB_GATE_MARGIN_US:-30}" --period-us "$PERIOD") && break
        log "calib_alone_$k: busy_us derivation refused"
        busy=""
    done
    [[ $busy =~ ^[0-9]+$ ]] || { log "no usable calibration run"; return 1; }
    BUSY_US=$busy
    log "gate: busy_us=$BUSY_US (p99.9 on-time completion + margin), sizes=$sizes, repeats=$repeats, seed=$seed"
    printf '%s\n' "$BUSY_US" > "$OUT/busy_us.txt"
    local affinity=(taskset -c "$SB_ADVERSARY_CPU") mps_env=() order case
    : > "$OUT/schedule.txt"
    for iso in none mps; do
        if [[ $iso == mps ]]; then
            MPS_PIPE="$W/mps-pipe"; MPS_LOG="$W/mps-log"
            mkdir -p "$MPS_PIPE" "$MPS_LOG"
            timeout 20 env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" nvidia-cuda-mps-control -d > "$LOGS/mps.log" 2>&1
            MPS_ON=1; sleep 2
            [[ -e $MPS_PIPE/control ]]
            mps_env=(CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50)
        fi
        for n in $sizes; do   # tenant alone, same sharing setting: the throughput it would get without 5G
            env "${mps_env[@]}" "${affinity[@]}" "$ADV" --workload sgemm --size "$n" --duty 100 --sync spin --seconds 5 \
                --out "$OUT/ai_alone_${iso/none/proc}_n$n.json" > "$OUT/ai_alone_${iso/none/proc}_n$n.log" 2>&1 \
                || log "ai_alone ${iso} n=$n rc=$?"
        done
        for ((r=1; r<=repeats; r++)); do
            mapfile -t order < <(python3 - "$seed" "$iso" "$r" $sizes <<'PY'
import random, sys
seed, iso, r, *sizes = sys.argv[1:]
cases = ["alone"] + [f"{m}:{n}" for n in sizes for m in ("observe", "on")]
random.Random(f"{seed}-{iso}-{r}").shuffle(cases)
print("\n".join(cases))
PY
)
            for case in "${order[@]}"; do
                echo "${iso} r$r $case" >> "$OUT/schedule.txt"
                if [[ $case == alone ]]; then
                    run_case "alone_${iso/none/proc}_r$r" cpu "$iso" none || log "case rc=$?"
                    continue
                fi
                local mode=${case%%:*} name
                n=${case##*:}
                name="${iso/none/proc}_${mode/on/gated}_n${n}_r$r"
                ADV_SIZE=$n
                # full unit logs stay on the instance (analysed there); the archive keeps the first 2000 units
                ADV_EXTRA=(--gate "$GATE_TT" --gate-mode "$mode" --sync spin --gate-log "$W/units/$name.units.bin")
                run_case "$name" cpu "$iso" sgemm || log "case rc=$?"
                ADV_EXTRA=()
                [[ -s $W/units/$name.units.bin ]] && head -c 32000 "$W/units/$name.units.bin" > "$OUT/$name.units.head.bin"
            done
        done
    done
    stop_telemetry
    python3 "$W/sb/slotbench/analysis/gate_summary.py" "$OUT" --units-dir "$W/units" --json "$OUT/gate_summary.json" \
        > "$OUT/gate_summary.txt" 2>&1 \
        || log "summary rc=$?"
    cat "$OUT/gate_summary.txt"
}

# The pinned dependency/vector-generation functions below reuse onstart_cuphy.sh's
# previously exercised recipe. Tests can source this file without starting a GPU run.

gcmake() { local n=$1 u=$2 c=$3; shift 3
    rm -rf "$W/deps/$n"; git init -q "$W/deps/$n"; git -C "$W/deps/$n" fetch -q --depth 1 "$u" "$c"
    git -C "$W/deps/$n" checkout -q FETCH_HEAD
    if [ "$n" = fmtlog ]; then (cd "$W/deps/$n" && git apply "$S/cuPHY-CP/container/patches/fmtlog.patch" \
        && cp fmtlog.h fmtlog-inl.h /usr/local/include/); fi
    cmake -S "$W/deps/$n" -B "$W/deps/$n/b" -GNinja -DCMAKE_BUILD_TYPE=Release "$@"
    cmake --build "$W/deps/$n/b"; cmake --install "$W/deps/$n/b"; }
deps() {
    gcmake fmt       https://github.com/fmtlib/fmt.git          e69e5f977d458f2650bb346dadf2ad30c5320281 -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DFMT_TEST=OFF -DFMT_DOC=OFF
    gcmake fmtlog    https://github.com/MengRao/fmtlog.git      acd521b1a64480354136a745c511358da1ec7dc5 -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    gcmake gsl-lite  https://github.com/gsl-lite/gsl-lite.git   56dab5ce071c4ca17d3e0dbbda9a94bd5a1cbca1
    gcmake wise_enum https://github.com/quicknir/wise_enum.git  34ac79f7ea2658a148359ce82508cc9301e31dd3
    gcmake CLI11     https://github.com/CLIUtils/CLI11.git      4160d259d961cd393fd8d67590a8c7d210207348 -DCLI11_BUILD_TESTS=OFF -DCLI11_BUILD_EXAMPLES=OFF
    gcmake yaml-cpp  https://github.com/jbeder/yaml-cpp.git     f7320141120f720aecc4c32be25586e7da9eb978 -DYAML_CPP_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    ldconfig
    wget -q https://developer.download.nvidia.com/compute/cuFFTDx/redist/cuFFTDx/cuda13/nvidia-mathdx-26.03.0-cuda13.tar.gz
    tar xzf nvidia-mathdx-26.03.0-cuda13.tar.gz -C /usr/local --strip-components=1
    rm nvidia-mathdx-26.03.0-cuda13.tar.gz
}

tv() {
    cd "$W"
    if [[ ! -f $W/matlab-runtime-installed || ! -d /usr/local/MATLAB/MATLAB_Runtime/R2026a/runtime/glnxa64 ]]; then
    apt-get install -y --no-install-recommends default-jre libxfont2 x11-xkb-utils xkb-data libxcomposite1 libnss3 \
        libxrandr-dev libatk1.0-0 libatk-bridge2.0-0 libx11-xcb-dev libxcb-dri3-0 libxcursor-dev libxdamage-dev \
        libxi-dev libdrm-dev libgbm-dev libasound-dev libcups2-dev libxtst-dev
    local runtime_url=https://ssd.mathworks.com/supportfiles/downloads/R2026a/Release/4/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2026a_Update_4_glnxa64.zip
    aria2c --allow-overwrite=true --auto-file-renaming=false --continue=true --max-connection-per-server=8 --split=8 \
        --min-split-size=16M --max-tries=3 --retry-wait=5 --summary-interval=120 --console-log-level=warn \
        --dir="$W" --out=mcr.zip "$runtime_url" || wget -q -c "$runtime_url" -O "$W/mcr.zip"
    rm -rf mcr && mkdir mcr && (cd mcr && unzip -q ../mcr.zip && ./install -mode silent -agreeToLicense yes)
    rm -rf mcr mcr.zip
    touch "$W/matlab-runtime-installed"
    fi
    local whl=aerial_mcore-0.20261.508652.508652-py3-none-any.whl
    mkdir -p "$S/5GModel/aerial_mcore/aerial_pkg/dist"
    (cd "$S" && git lfs install --local && git lfs pull --include="5GModel/aerial_mcore/aerial_pkg/dist/*.whl") \
        || wget -q -O "$S/5GModel/aerial_mcore/aerial_pkg/dist/$whl" \
            "https://media.githubusercontent.com/media/NVIDIA/aerial-cuda-accelerated-ran/$ACAR_COMMIT/5GModel/aerial_mcore/aerial_pkg/dist/$whl"
    python3 -m venv "$W/venv"
    "$W/venv/bin/pip" install -q numpy pyyaml h5py "$S"/5GModel/aerial_mcore/aerial_pkg/dist/aerial_mcore-*.whl
    mkdir -p "$W/tv" && cd "$W/tv"
    export PS1="${PS1:-}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    # shellcheck disable=SC1091
    source "$S/5GModel/aerial_mcore/scripts/setup.sh"
    "$W/venv/bin/python" -c "import aerial_mcore as M, matlab; e = M.initialize(); print(e.testCompGenTV_pusch(matlab.double([7304]), 'genTV', nargout=4))"
    find "$W/tv" -name '*.h5' -exec ls -la {} \;
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
