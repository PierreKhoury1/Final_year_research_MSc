#!/bin/bash
# Rented-GPU campaign for gputrace: every strategy once (plus launch after idle gaps and with a graph), analysis
# one-liners in the log, results archived. Usage: onstart_gputrace.sh [CONFIG [BRANCH [REPO]]] (CONFIG ignored)
# Env: SB_GT_ITERS (default 2000), SB_GT_SECONDS (timeslice/clocks window, default 8), SB_GT_SKIP (regex of
# strategies to skip), SB_GT_ONLY (regex: run only the matching names, e.g. '^(gpus|nccl)'), SB_GT_REPEAT
# (campaign repeats, default 1). On a host with more than one GPU it also runs gpus (per-GPU timer mapping) and
# nccl (gputrace_nccl: all-reduce spans on every GPU on one bounded axis).
set -Eeuo pipefail
log() { echo "[slotbench $(date -u +%H:%M:%S)] $*"; }
W=/workspace/gputrace; OUT=$W/out; mkdir -p "$OUT"; cd "$W"
BRANCH=${2:-${SB_BRANCH:-main}}; REPO=${3:-${SB_REPO:-https://github.com/PierreKhoury1/Final_year_research_MSc}}
ITERS=${SB_GT_ITERS:-2000}; SECS=${SB_GT_SECONDS:-8}; SKIP=${SB_GT_SKIP:-^$}; ONLY=${SB_GT_ONLY:-.}; REPEAT=${SB_GT_REPEAT:-1}

emit_archive() {   # emit_archive NAME TARARGS... : archive, print as (multi-part) base64 blocks
    local name=$1; shift
    local archive=$W/$name.tar.gz sha n i=0 part
    tar --warning=no-file-changed -czf "$archive" "$@" || true
    sha=$(sha256sum "$archive" | cut -d' ' -f1)
    rm -rf "$W/parts"; mkdir -p "$W/parts"
    split -b 400k -d -a 2 --additional-suffix=.bin "$archive" "$W/parts/p"
    n=$(ls "$W/parts" | wc -l)
    echo "=====SLOTBENCH-PARTS gputrace/$name $n $sha====="
    for part in "$W"/parts/p*.bin; do
        i=$((i + 1))
        echo "=====SLOTBENCH-BEGIN gputrace/$name.part$(printf %02d "$i") $(sha256sum "$part" | cut -d' ' -f1)====="
        base64 -w 76 "$part"
        echo "=====SLOTBENCH-END gputrace/$name.part$(printf %02d "$i")====="
        if (( i < n )); then sleep "${SB_PART_GAP_S:-50}"; fi
    done
    echo "=====SLOTBENCH-PARTS gputrace/$name $n $sha====="
}
emit() {   # small summary block first (analysis JSON, logs, meta), then the raw data; then DONE and self-stop
    local status=$1
    mkdir -p "$W/summary"; rm -rf "$W/summary"/*
    cp "$OUT"/*.json "$OUT"/*.log "$OUT"/*.err "$OUT"/*.txt "$W/summary/" 2>/dev/null || true
    rm -f "$W/summary"/*.gpu.bin
    emit_archive summary -C "$W" summary
    sleep "${SB_PART_GAP_S:-50}"
    emit_archive results -C "$W" out
    echo "=====SLOTBENCH-DONE status=$status====="
    sleep "${SB_POST_DONE_GRACE_S:-120}"; echo "=====SLOTBENCH-SELF-STOP====="; kill -TERM 1; sleep 10; kill -KILL 1
}
trap 'echo "=====SLOTBENCH-ERROR $LINENO $BASH_COMMAND====="; emit error' ERR

log "apt"
apt-get -o Acquire::Retries=3 update -qq > "$OUT/apt.log" 2>&1
apt-get install -y -qq --no-install-recommends git python3 python3-numpy ca-certificates >> "$OUT/apt.log" 2>&1
log "clone $BRANCH"
git clone -q --depth 1 --branch "$BRANCH" "$REPO" "$W/sb"
git -C "$W/sb" rev-parse HEAD > "$OUT/commit.txt"
SB=$W/sb/slotbench
{ date -u; nvidia-smi; nvidia-smi -q | grep -iE "persistence|link width|link gen|compute mode|clocks" ; nproc; lscpu | head -20; uname -a; } > "$OUT/header.txt" 2>&1 || true
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
log "build sm_$cc"
nvcc -O2 -lineinfo -std=c++17 -arch=sm_$cc -I"$SB/common" -I"$SB/gputrace" -o "$W/gputrace" "$SB/gputrace/gputrace.cu" -lpthread > "$OUT/build.log" 2>&1
NP=$(nproc); CORE=$(( NP > 4 ? 2 : 0 )); CCORE=$(( NP > 4 ? 3 : 1 ))
# the raw files of long strategies are large; keep only what the analysis needs plus the json/log
run_bin() {   # run_bin BINARY NAME args...
    local bin=$1 name=$2; shift 2
    if [[ "$name" =~ $SKIP || ! "$name" =~ $ONLY ]]; then return 0; fi
    log "run $name: $*"
    local p="$OUT/$name"
    if ! "$bin" --out "$p" --core "$CORE" --clock-core "$CCORE" "$@" > "$p.log" 2>&1; then
        log "$name FAILED (see $name.log)"; tail -3 "$p.log" | sed 's/^/[slotbench]   /'; return 0
    fi
    tail -1 "$p.log" | sed 's/^/[slotbench]   /'
    python3 "$SB/analysis/gputrace.py" "$p" --json "$p.analysis.json" 2> "$p.analysis.err" | sed 's/^/[slotbench]   /' \
        || { log "analysis $name failed"; tail -2 "$p.analysis.err" | sed 's/^/[slotbench]   /'; }
    sleep 3
}
run() { run_bin "$W/gputrace" "$@"; }
NGPU=$(nvidia-smi -L | wc -l)
if (( NGPU > 1 )); then   # NCCL for the multi-GPU tracer: the CUDA devel images normally ship it; install if not
    if [[ ! -e /usr/include/nccl.h ]]; then
        log "installing NCCL"
        apt-get install -y -qq --no-install-recommends libnccl2 libnccl-dev >> "$OUT/apt.log" 2>&1 || log "NCCL install failed"
    fi
    if [[ -e /usr/include/nccl.h ]]; then
        nvcc -O2 -lineinfo -std=c++17 -arch=sm_$cc -I"$SB/common" -I"$SB/gputrace" -o "$W/gputrace_nccl" \
            "$SB/gputrace/gputrace_nccl.cu" -lnccl -lpthread >> "$OUT/build.log" 2>&1 || log "gputrace_nccl build failed"
    fi
    nvidia-smi topo -m > "$OUT/topo.txt" 2>&1 || true
    sed 's/^/[slotbench] topo /' "$OUT/topo.txt" | head -6
fi
for rep in $(seq 1 "$REPEAT"); do
    R=""; (( REPEAT > 1 )) && R="_r$rep"
    run "launch$R"           --strategy launch --iters "$ITERS"
    run "launch_idle100$R"   --strategy launch --iters "$ITERS" --idle-us 100
    run "launch_idle2000$R"  --strategy launch --iters 1000 --idle-us 2000
    run "launch_idle50000$R" --strategy launch --iters 200 --idle-us 50000
    run "launch_depth8$R"    --strategy launch --iters 500 --depth 8
    run "launch_graph$R"     --strategy launch --iters "$ITERS" --graph 1
    run "launch_spin2000$R"  --strategy launch --iters 1000 --idle-us 2000 --idle-spin 1
    run "notify$R"           --strategy notify --iters "$ITERS" --dur-us 20
    run "dispatch$R"         --strategy dispatch --dur-us 200 --reps 5
    run "dispatch_t64$R"     --strategy dispatch --dur-us 200 --reps 3 --threads 64 --blocks 1,sm,2sm,8sm
    run "dispatch_busy$R"    --strategy dispatch --dur-us 200 --reps 3 --spin-mode 1 --spin-param 32 --blocks sm,2sm,8sm,32sm
    run "dispatch_timer$R"   --strategy dispatch --dur-us 200 --reps 3 --spin-mode 2 --blocks sm,2sm,8sm,32sm
    run "launch_graph8$R"    --strategy launch --iters 500 --depth 8 --graph 1
    run "dispatch_smem$R"    --strategy dispatch --dur-us 200 --reps 3 --smem 32768 --blocks sm,2sm,8sm
    for d in 200 500 2000 10000; do   # how long the second stream waits, vs the first kernel's block length
        run "concurrency_a${d}$R"      --strategy concurrency --dur-us "$d" --dur-b-us 200 --offset-us 100 --reps 5
        run "concurrency_a${d}_prio$R" --strategy concurrency --dur-us "$d" --dur-b-us 200 --offset-us 100 --reps 5 --priority 1
    done
    run "clocks$R"           --strategy clocks --seconds 4 --sample-us 100
    for idle in 0 100 2000 50000; do
        run "ramp_idle${idle}$R" --strategy ramp --idle-us "$idle" --dur-us 3000 --sample-us 20 --reps 40
    done
    run "copy$R"             --strategy copy --iters 1000
    if (( NGPU > 1 )); then
        run "gpus$R"         --strategy gpus --reps 3 --sync-rounds 3 --sync-per-phase 1000
        if [[ -x $W/gputrace_nccl ]]; then
            NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH run_bin "$W/gputrace_nccl" "nccl$R" --iters 200 \
                --sizes 8,4096,65536,1048576,16777216,134217728
            grep -hE "via (P2P|SHM|NET)|Channel 00" "$OUT/nccl$R.log" 2>/dev/null | head -3 | sed 's/^/[slotbench]   nccl transport: /' || true
        fi
    fi
    run "timeslice$R"        --strategy timeslice --seconds "$SECS" --gap-us 20 --hog-dur-us 5000
    # the same two processes under MPS (hog limited to 50 % of the SMs): no time-slicing expected, partition visible
    if command -v nvidia-cuda-mps-control >/dev/null 2>&1 && ! pgrep -f 'nvidia-cuda-mps-(control|server)' >/dev/null; then
        MPS_PIPE=$W/mps-pipe; MPS_LOG=$W/mps-log; mkdir -p "$MPS_PIPE" "$MPS_LOG"
        if timeout 20 env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" nvidia-cuda-mps-control -d > "$OUT/mps.log" 2>&1; then
            export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG"
            run "timeslice_mps50$R" --strategy timeslice --seconds "$SECS" --gap-us 20 --hog-dur-us 5000 --hog-mps-pct 50
            run "timeslice_mps100$R" --strategy timeslice --seconds "$SECS" --gap-us 20 --hog-dur-us 5000
            run "launch_mps$R"      --strategy launch --iters "$ITERS"
            echo quit | timeout 10 env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" nvidia-cuda-mps-control >> "$OUT/mps.log" 2>&1 || true
            unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
            cp -a "$MPS_LOG" "$OUT/mps_logs" 2>/dev/null || true
        else log "MPS daemon did not start (see mps.log)"; fi
    else log "no MPS control binary or a daemon is already running: MPS variant skipped"; fi
done
# compact: keep raw bins of the long strategies only if small
find "$OUT" -name '*.gpu.bin' -size +40M -print -delete | sed 's/^/[slotbench] dropped large raw /' || true
emit ok
