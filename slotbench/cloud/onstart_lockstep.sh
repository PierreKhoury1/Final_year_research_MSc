#!/bin/bash
# vast.ai onstart: build slotbench and run scripts/lockstep_matrix.sh (CPU-launched vs GPU-self-launched slot
# under AI load). Results stream back as =====SLOTBENCH-BEGIN lockstep/results <sha>===== blocks.
# Usage: onstart_lockstep.sh [CONFIG [BRANCH [REPO]]]   (CONFIG unused; env SB_SLOTS, SB_SIZES, SB_MATRIX_ARGS optional)
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
SB_BRANCH="${2:-${SB_BRANCH:-claude/optimistic-ptolemy-r42xrh}}"
SB_REPO="${3:-${SB_REPO:-https://github.com/PierreKhoury1/Final_year_research_MSc}}"
W=/root/sb; LOGS=$W/logs; OUT=$W/out
mkdir -p "$W" "$LOGS" "$OUT"
cd "$W"

# Watchdog: stop the container (and GPU billing) after SB_MAX_HOURS even if the controller dies.
WATCHDOG_S=$(awk -v h="${SB_MAX_HOURS:-2.5}" 'BEGIN { printf "%d", h * 3600 }')
( sleep "$WATCHDOG_S"; echo "=====SLOTBENCH-ERROR 0 watchdog-${SB_MAX_HOURS:-2.5}h====="; kill -TERM 1; sleep 60; kill -KILL 1 ) &

log() { echo "[lockstep-cloud $(date -u +%H:%M:%S)] $*"; }
emit_block() {
    local name="$1" root="$2" tmp sha
    shift 2
    tmp="$(mktemp)"
    tar -czf "$tmp" -C "$root" "$@" 2>/dev/null || { log "tar failed for $name"; rm -f "$tmp"; return 0; }
    sha="$(sha256sum "$tmp" | cut -d' ' -f1)"
    echo "=====SLOTBENCH-BEGIN $name $sha====="; base64 -w 76 "$tmp"; echo "=====SLOTBENCH-END $name====="
    rm -f "$tmp"
}
finish() {
    local status="$1"
    trap - ERR
    # The controller only sees the last 20000 log lines (~1.5 MB of base64), so the raw per-slot .bin files
    # (1.9 MB each) stay on the instance; the JSON summaries carry every metric. Results go last so that they
    # survive if anything is cut.
    emit_block "_logs/lockstep" "$W" logs || true
    [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && emit_block "lockstep/results" "$W" --exclude='*.bin' out || true
    if [ "$status" = ok ]; then echo "=====SLOTBENCH-DONE====="; else echo "=====SLOTBENCH-DONE status=$status====="; fi
    sleep infinity
}
trap 'echo "=====SLOTBENCH-ERROR $LINENO $BASH_COMMAND====="; tail -n 30 "$LOGS/current.log" 2>/dev/null | sed "s/^/  | /"; finish error' ERR
trap 'echo "=====SLOTBENCH-ERROR 0 SIGTERM====="; finish term' TERM
step() {
    local name="$1" rc=0
    shift
    log "$name"
    : > "$LOGS/current.log"
    "$@" >>"$LOGS/current.log" 2>&1 || rc=$?
    cat "$LOGS/current.log" >> "$LOGS/$name.log"
    return "$rc"
}

{ date -u; nvidia-smi; nproc; free -g; } > "$LOGS/header.txt" 2>&1 || true
head -12 "$LOGS/header.txt"
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

step apt bash -c "apt-get update -y && apt-get install -y --no-install-recommends git build-essential python3 ca-certificates"
step clone bash -c "rm -rf $W/repo && git clone -q --depth 1 -b $SB_BRANCH $SB_REPO $W/repo"
cd "$W/repo/slotbench"
step build make SM="$GPU_CC" CUDA_HOME=/usr/local/cuda bin/lockstep_driver bin/adversary
log "running matrix (slots ${SB_SLOTS:-30000})"
# shellcheck disable=SC2086
matrix_rc=0
bash scripts/lockstep_matrix.sh --out "$OUT" --slots "${SB_SLOTS:-30000}" --gpu 0 ${SB_MATRIX_ARGS:-} 2>&1 | tee "$LOGS/matrix.log" || matrix_rc=$?
cp "$LOGS/header.txt" "$OUT/cloud_header.txt" || true
if [[ $matrix_rc -ne 0 ]]; then
    log "matrix failed (exit $matrix_rc); preserving partial results and logs"
    finish "error-matrix-$matrix_rc"
else
    finish ok
fi
