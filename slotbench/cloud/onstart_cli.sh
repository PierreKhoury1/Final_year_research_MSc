#!/bin/bash
# Rented-GPU run of the gputrace command-line tool itself: clone, `gputrace characterize --profile $SB_GT_PROFILE`,
# then ship the output directory (report.md, summary.json, timeline.html, traces, raw records) through the log.
# Usage: onstart_cli.sh [CONFIG [BRANCH [REPO]]] (CONFIG ignored). Env: SB_GT_PROFILE (quick|full|sharing|instr|
# multi, default quick), SB_GT_MPS (auto|on|off, default auto).
set -Eeuo pipefail
log() { echo "[slotbench $(date -u +%H:%M:%S)] $*"; }
W=/workspace/gputrace; OUT=$W/out; mkdir -p "$OUT"; cd "$W"
BRANCH=${2:-${SB_BRANCH:-main}}; REPO=${3:-${SB_REPO:-https://github.com/PierreKhoury1/Final_year_research_MSc}}
PROFILE=${SB_GT_PROFILE:-quick}; MPS=${SB_GT_MPS:-auto}

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
emit() {   # the report, analysis and logs first (small), then everything else; then DONE and self-stop
    local status=$1
    mkdir -p "$W/summary"; rm -rf "$W/summary"/*
    cp "$OUT"/*.md "$OUT"/*.json "$OUT"/*.log "$OUT"/*.txt "$W/summary/" 2>/dev/null || true
    rm -f "$W/summary"/*.trace.json
    emit_archive summary -C "$W" summary
    sleep "${SB_PART_GAP_S:-50}"
    find "$OUT" -name '*.gpu.bin' -size +40M -print -delete | sed 's/^/[slotbench] dropped large raw /' || true
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
log "gputrace characterize --profile $PROFILE --mps $MPS"
status=ok
"$SB/gputrace/gputrace" characterize --profile "$PROFILE" --mps "$MPS" --out "$OUT" 2>&1 | tee "$OUT/characterize.log" | sed 's/^/[slotbench]   /' || status=error
cp "$W"/sb/slotbench/bin/gputrace_sm* "$W/" 2>/dev/null || true
log "report:"; sed 's/^/[slotbench]   /' "$OUT/report.md" 2>/dev/null | head -80 || true
emit "$status"
