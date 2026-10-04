#!/bin/bash
# vast.ai onstart: GPU %globaltimer stability versus the host clock over hours, in three GPU states
# (idle, resident spinning load that keeps the clocks up, idle again), with 1 Hz nvidia-smi telemetry.
# Results stream back as =====SLOTBENCH-BEGIN clock/results <sha>===== blocks (CSV rows only, small).
# Usage: onstart_clock.sh [CONFIG [BRANCH [REPO]]]  (CONFIG unused; env SB_CLOCK_PHASE_S, SB_CLOCK_INTERVAL_S,
#        SB_CLOCK_SAMPLES, SB_CLOCK_PHASES optional)
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
SB_BRANCH="${2:-${SB_BRANCH:-claude/optimistic-ptolemy-r42xrh}}"
SB_REPO="${3:-${SB_REPO:-https://github.com/PierreKhoury1/Final_year_research_MSc}}"
W=/root/sb; LOGS=$W/logs; OUT=$W/out
mkdir -p "$W" "$LOGS" "$OUT"
cd "$W"

WATCHDOG_S=$(awk -v h="${SB_MAX_HOURS:-3.5}" 'BEGIN { printf "%d", h * 3600 }')
( sleep "$WATCHDOG_S"; echo "=====SLOTBENCH-ERROR 0 watchdog-${SB_MAX_HOURS:-3.5}h====="; kill -TERM 1; sleep 60; kill -KILL 1 ) &

log() { echo "[clock-cloud $(date -u +%H:%M:%S)] $*"; }
emit_block() {
    local name="$1" root="$2" tmp sha
    shift 2
    tmp="$(mktemp)"
    tar -czf "$tmp" -C "$root" "$@" 2>/dev/null || { log "tar failed for $name"; rm -f "$tmp"; return 0; }
    sha="$(sha256sum "$tmp" | cut -d' ' -f1)"
    echo "=====SLOTBENCH-BEGIN $name $sha====="; base64 -w 76 "$tmp"; echo "=====SLOTBENCH-END $name====="
    rm -f "$tmp"
}
TEL_PID=""
finish() {
    local status="$1"
    trap - ERR TERM
    [[ -n $TEL_PID ]] && kill -TERM "$TEL_PID" 2>/dev/null || true
    emit_block "_logs/clock" "$W" logs || true
    [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && emit_block "clock/results" "$W" --exclude='*.bin' out || true
    if [ "$status" = ok ]; then echo "=====SLOTBENCH-DONE====="; else echo "=====SLOTBENCH-DONE status=$status====="; fi
    sleep infinity
}
trap 'echo "=====SLOTBENCH-ERROR $LINENO $BASH_COMMAND====="; tail -n 30 "$LOGS/current.log" 2>/dev/null | sed "s/^/  | /"; finish error' ERR
trap 'echo "=====SLOTBENCH-ERROR 0 SIGTERM====="; finish term' TERM
step() { local name="$1"; shift; log "$name"; : > "$LOGS/current.log"; "$@" >>"$LOGS/current.log" 2>&1; cat "$LOGS/current.log" >> "$LOGS/$name.log"; }

{ date -u; nvidia-smi; nproc; free -g; } > "$LOGS/header.txt" 2>&1 || true
head -12 "$LOGS/header.txt"
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

step apt bash -c "apt-get update -y && apt-get install -y --no-install-recommends git build-essential python3 ca-certificates"
step clone bash -c "rm -rf $W/repo && git clone -q --depth 1 -b $SB_BRANCH $SB_REPO $W/repo"
cd "$W/repo/slotbench"
step build make SM="$GPU_CC" CUDA_HOME=/usr/local/cuda bin/clockcal

# clock-lock attempt, recorded (containers normally refuse it)
{ nvidia-smi -lgc 1410,1410 2>&1 || true; nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.limit,temperature.gpu --format=csv; } > "$OUT/clock_lock_attempt.txt" 2>&1 || true

# 1 Hz telemetry for the whole run
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,clocks_event_reasons.active \
    --format=csv -l 1 > "$OUT/telemetry.csv" 2>&1 &
TEL_PID=$!

PHASE_S="${SB_CLOCK_PHASE_S:-2400}"; INTERVAL_S="${SB_CLOCK_INTERVAL_S:-10}"; SAMPLES="${SB_CLOCK_SAMPLES:-2000}"
PHASES="${SB_CLOCK_PHASES:-idle spin idle}"
i=0
for ph in $PHASES; do
    i=$((i + 1))
    name=$(printf 'p%d_%s' "$i" "$ph")
    log "phase $name: ${PHASE_S}s, window every ${INTERVAL_S}s of $SAMPLES brackets"
    date -u +%FT%TZ > "$OUT/$name.start_utc"
    timeout -k 30 "$((PHASE_S + 300))" bin/clockcal --out "$OUT/$name" --duration-s "$PHASE_S" --interval-s "$INTERVAL_S" \
        --samples "$SAMPLES" --method pingpong --gpu 0 --load "$( [ "$ph" = spin ] && echo spin || echo none )" \
        > "$OUT/$name.log" 2>&1 || log "phase $name rc=$?"
    date -u +%FT%TZ > "$OUT/$name.end_utc"
    tail -n 2 "$OUT/$name.log" || true
done
cp "$LOGS/header.txt" "$OUT/" || true
finish ok
