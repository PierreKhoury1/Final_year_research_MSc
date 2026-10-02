#!/bin/bash
# On the SAME retained rental, after updating the slotbench checkout/helper files:
#   bash slotbench/cloud/debug_cuphy_lockstep.sh both    # CPU/GPU alone
#   bash slotbench/cloud/debug_cuphy_lockstep.sh matrix  # all six cells
# Reuses build/dependencies/TC7304. Does not fetch Git or modify prior results.
# SB_CUPHY_WORKDIR defaults to /workspace/cuphy-lockstep; SB_CUPHY_JOBS defaults16.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/onstart_cuphy_lockstep.sh"

refresh_adapter() {
    local adapter="$W/sb/slotbench/cuphy" patch="$W/sb/slotbench/cuphy/patches/cuphy-lockstep.patch"
    local previous="$W/applied-adapter.patch" example="$S/cuPHY/examples/pusch_rx_multi_pipe"
    [[ -f $previous ]] || previous="$W/out/applied-adapter.patch"
    [[ -s $previous && -s $patch ]]
    cp "$previous" "$OUT/previous-adapter.patch"
    # Require the expected prior patch before modifying tracked sources.
    git -C "$S" apply --reverse --check "$previous"
    if ! cmp -s "$previous" "$patch"; then
        git -C "$S" apply --reverse "$previous"
        if ! git -C "$S" apply --check "$patch"; then
            git -C "$S" apply "$previous"
            echo "Updated patch failed check; previous patch restored" >&2
            return 1
        fi
        if ! git -C "$S" apply "$patch"; then
            git -C "$S" apply "$previous"
            return 1
        fi
    fi
    cp "$patch" "$W/applied-adapter.patch"
    cp "$patch" "$OUT/applied-adapter.patch"
    cp "$adapter/cuphy_lockstep.cu" "$adapter/cuphy_lockstep.h" "$example/"
    cp "$W/sb/slotbench/common/clock_fit.h" "$W/sb/slotbench/common/host_time.h" "$W/sb/slotbench/common/json_writer.h" "$example/"
    cp "$adapter/cuphy_lockstep_stamps.cu" "$adapter/cuphy_lockstep_stamps.h" "$S/cuPHY/src/cuphy_channels/"
    sha256sum "$patch" "$adapter"/cuphy_lockstep*.cu "$adapter"/cuphy_lockstep*.h > "$OUT/adapter_sha256.txt"
    git -C "$W/sb" rev-parse HEAD > "$OUT/slotbench_commit.txt"
    printf '%s\n' "$ACAR_COMMIT" > "$OUT/aerial_commit.txt"
}

debug_main() {
    local selection=${1:-both} run_id jobs=${SB_CUPHY_JOBS:-16} previous_vector_hash="" mps_scan_rc=0
    [[ $selection == both || $selection == matrix ]]
    export W="${SB_CUPHY_WORKDIR:-/workspace/cuphy-lockstep}"
    export S="$W/acar" ACAR_COMMIT
    [[ -d $W && -f $W/build/CMakeCache.txt ]]
    [[ $(git -C "$S" rev-parse HEAD) == "$ACAR_COMMIT" ]]
    [[ $jobs =~ ^[0-9]+$ ]] && (( jobs >= 1 && jobs <= 64 ))
    run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
    export LOGS="$W/logs-debug-$run_id" OUT="$W/out-debug-$run_id"
    mkdir "$LOGS" "$OUT"
    cd "$W"
    COLLECTION_ARCHIVE="$W/collection-debug-$run_id.tar.gz"
    export SB_CUPHY_FINISH_EXIT=1
    CURRENT_LOG="$LOGS/header.txt"; ADV_PID=""; STEP_PID=""; MPS_ON=0
    trap 'echo "=====SLOTBENCH-ERROR $LINENO $BASH_COMMAND====="; finish error' ERR
    trap 'echo "=====SLOTBENCH-ERROR 0 interrupted====="; finish interrupted; exit 130' TERM INT
    { date -u; nvidia-smi; git -C "$S" rev-parse HEAD; } > "$LOGS/header.txt" 2>&1
    export -f refresh_adapter tv
    step refresh_adapter 60 bash -e -o pipefail -c refresh_adapter
    step incremental_build 1200 cmake --build "$W/build" --target cuphy_ex_pusch_rx_multi_pipe -- -j"$jobs"
    PUSCH="$W/build/cuPHY/examples/pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe"
    ADV="$W/sb/slotbench/bin/adversary"
    if [[ ! -x $ADV ]]; then
        step adversary_build 180 make -C "$W/sb/slotbench" SM=80 CUDA_HOME=/usr/local/cuda bin/adversary
    fi
    [[ -x $PUSCH && -x $ADV ]]
    TV="$W/tv/GPU_test_input/TVnr_7304_PUSCH_gNB_CUPHY_s0p0.h5"
    if [[ ! -s $TV ]]; then
        # An initial compile failure may have occurred before vector generation.
        step tv "${TV_TIMEOUT_S:-2400}" bash -e -o pipefail -c tv
    fi
    [[ -s $TV ]]
    TV_SHA=$(sha256sum "$TV" | cut -d' ' -f1)
    if [[ -f $W/out/test_vector_sha256.txt ]]; then
        read -r previous_vector_hash _ < "$W/out/test_vector_sha256.txt"
        [[ $TV_SHA == "$previous_vector_hash" ]]
    fi
    printf '%s  %s\n' "$TV_SHA" "$TV" > "$OUT/test_vector_sha256.txt"
    SLOTS=${SB_CUPHY_LOCKSTEP_SLOTS:-1000}; WARMUP=${SB_CUPHY_LOCKSTEP_WARMUP:-100}
    PERIOD=${SB_CUPHY_LOCKSTEP_PERIOD_US:-500}; DEADLINE=${SB_CUPHY_LOCKSTEP_DEADLINE_US:-500}
    CASE_TIMEOUT=${SB_CUPHY_CASE_TIMEOUT_S:-180}
    [[ $SLOTS =~ ^[0-9]+$ && $WARMUP =~ ^[0-9]+$ && $CASE_TIMEOUT =~ ^[0-9]+$ ]]
    (( SLOTS >= 1 && SLOTS <= 5000 && WARMUP <= 1000 && CASE_TIMEOUT >= 10 && CASE_TIMEOUT <= 600 ))
    [[ $PERIOD =~ ^[0-9]+([.][0-9]+)?$ && $DEADLINE =~ ^[0-9]+([.][0-9]+)?$ ]]
    awk -v p="$PERIOD" -v d="$DEADLINE" 'BEGIN {exit !(p >= 1 && p <= 1000000 && d > 0)}'
    pgrep -f '^([^[:space:]]*/)?nvidia-cuda-mps-(control|server)([[:space:]]|$)' >/dev/null || mps_scan_rc=$?
    [[ $mps_scan_rc -eq 1 ]]
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_MPS_CLIENT_PRIORITY \
        CUDA_MPS_PINNED_DEVICE_MEM_LIMIT CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING
    run_case cpu_alone cpu none none
    run_case gpu_alone gpu none none
    if [[ $selection == matrix ]]; then
        run_case cpu_proc_sgemm cpu none sgemm
        run_case gpu_proc_sgemm gpu none sgemm
        MPS_PIPE="$W/mps-pipe"; MPS_LOG="$W/mps-log-debug-$run_id"
        mkdir -p "$MPS_PIPE" "$MPS_LOG"
        timeout 20 env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" nvidia-cuda-mps-control -d > "$LOGS/mps.log" 2>&1
        MPS_ON=1
        sleep 2
        [[ -e $MPS_PIPE/control ]]
        run_case cpu_mps_sgemm cpu mps sgemm
        run_case gpu_mps_sgemm gpu mps sgemm
    fi
    finish ok
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then debug_main "$@"; fi
