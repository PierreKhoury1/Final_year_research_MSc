#!/bin/bash
# Rented-GPU campaign for gputrace: every strategy once (plus launch after idle gaps and with a graph), analysis
# one-liners in the log, results archived. Usage: onstart_gputrace.sh [CONFIG [BRANCH [REPO]]] (CONFIG ignored)
# Env: SB_GT_ITERS (default 2000), SB_GT_SECONDS (timeslice/clocks window, default 8), SB_GT_SKIP (regex of
# strategies to skip), SB_GT_REPEAT (campaign repeats, default 1).
set -Eeuo pipefail
log() { echo "[slotbench $(date -u +%H:%M:%S)] $*"; }
W=/workspace/gputrace; OUT=$W/out; mkdir -p "$OUT"; cd "$W"
BRANCH=${2:-${SB_BRANCH:-main}}; REPO=${3:-${SB_REPO:-https://github.com/PierreKhoury1/Final_year_research_MSc}}
ITERS=${SB_GT_ITERS:-2000}; SECS=${SB_GT_SECONDS:-8}; SKIP=${SB_GT_SKIP:-^$}; REPEAT=${SB_GT_REPEAT:-1}

emit() {
    local status=$1 archive=$W/results.tar.gz sha n i=0 part
    tar --warning=no-file-changed -czf "$archive" -C "$W" out || true
    sha=$(sha256sum "$archive" | cut -d' ' -f1)
    rm -rf "$W/parts"; mkdir -p "$W/parts"
    split -b 800k -d -a 2 --additional-suffix=.bin "$archive" "$W/parts/p"
    n=$(ls "$W/parts" | wc -l)
    echo "=====SLOTBENCH-PARTS gputrace/results $n $sha====="
    for part in "$W"/parts/p*.bin; do
        i=$((i + 1))
        echo "=====SLOTBENCH-BEGIN gputrace/results.part$(printf %02d "$i") $(sha256sum "$part" | cut -d' ' -f1)====="
        base64 -w 76 "$part"
        echo "=====SLOTBENCH-END gputrace/results.part$(printf %02d "$i")====="
        if (( i < n )); then sleep "${SB_PART_GAP_S:-50}"; fi
    done
    echo "=====SLOTBENCH-PARTS gputrace/results $n $sha====="
    echo "=====SLOTBENCH-DONE status=$status====="
    sleep "${SB_POST_DONE_GRACE_S:-900}"; echo "=====SLOTBENCH-SELF-STOP====="; kill -TERM 1; sleep 10; kill -KILL 1
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
run() {   # run NAME strategy args...
    local name=$1; shift
    if [[ "$name" =~ $SKIP ]]; then log "skip $name"; return 0; fi
    log "run $name: $*"
    local p="$OUT/$name"
    if ! "$W/gputrace" --out "$p" --core "$CORE" --clock-core "$CCORE" "$@" > "$p.log" 2>&1; then
        log "$name FAILED (see $name.log)"; tail -3 "$p.log" | sed 's/^/[slotbench]   /'; return 0
    fi
    tail -1 "$p.log" | sed 's/^/[slotbench]   /'
    python3 "$SB/analysis/gputrace.py" "$p" --json "$p.analysis.json" 2> "$p.analysis.err" | sed 's/^/[slotbench]   /' \
        || { log "analysis $name failed"; tail -2 "$p.analysis.err" | sed 's/^/[slotbench]   /'; }
    sleep 3
}
for rep in $(seq 1 "$REPEAT"); do
    R=""; (( REPEAT > 1 )) && R="_r$rep"
    run "launch$R"           --strategy launch --iters "$ITERS"
    run "launch_idle100$R"   --strategy launch --iters "$ITERS" --idle-us 100
    run "launch_idle2000$R"  --strategy launch --iters 1000 --idle-us 2000
    run "launch_idle50000$R" --strategy launch --iters 200 --idle-us 50000
    run "launch_depth8$R"    --strategy launch --iters 500 --depth 8
    run "launch_graph$R"     --strategy launch --iters "$ITERS" --graph 1
    run "notify$R"           --strategy notify --iters "$ITERS" --dur-us 20
    run "dispatch$R"         --strategy dispatch --dur-us 200 --reps 5
    run "dispatch_t64$R"     --strategy dispatch --dur-us 200 --reps 3 --threads 64 --blocks 1,sm,2sm,8sm
    run "dispatch_smem$R"    --strategy dispatch --dur-us 200 --reps 3 --smem 49152 --blocks sm,2sm,8sm
    run "concurrency$R"      --strategy concurrency --dur-us 2000 --dur-b-us 200 --offset-us 500 --reps 5
    run "concurrency_prio$R" --strategy concurrency --dur-us 2000 --dur-b-us 200 --offset-us 500 --reps 5 --priority 1
    run "clocks$R"           --strategy clocks --seconds "$SECS" --sample-us 100
    run "copy$R"             --strategy copy --iters 1000
    run "timeslice$R"        --strategy timeslice --seconds "$SECS" --gap-us 20
done
# compact: keep raw bins of the long strategies only if small
find "$OUT" -name '*.gpu.bin' -size +40M -print -delete | sed 's/^/[slotbench] dropped large raw /' || true
emit ok
