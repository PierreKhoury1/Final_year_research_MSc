#!/usr/bin/env bash
# telemetry.sh OUT.csv [--gpu N] [--interval-ms 1000]
# Logs nvidia-smi GPU telemetry (temperature, SM/mem clocks, power, utilisation, clock-event
# reasons) to OUT.csv until killed. The clock-event field was renamed in newer drivers
# (clocks_throttle_reasons.active -> clocks_event_reasons.active); we probe which one works.
# Exit 3 if nvidia-smi is missing or cannot query the GPU.
set -euo pipefail

usage() { echo "usage: $0 OUT.csv [--gpu N] [--interval-ms MS]" >&2; exit 2; }

[[ $# -ge 1 ]] || usage
out=$1; shift
gpu=0
interval_ms=1000
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) gpu=${2:?}; shift 2 ;;
    --interval-ms) interval_ms=${2:?}; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NOT SUPPORTED: nvidia-smi not found (telemetry)" >&2
  exit 3
fi

base="timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu"
reason=""
for f in clocks_event_reasons.active clocks_throttle_reasons.active; do
  if nvidia-smi -i "$gpu" --query-gpu="$f" --format=csv,noheader >/dev/null 2>&1; then
    reason=$f
    break
  fi
done
fields=$base
if [[ -n $reason ]]; then
  fields="$base,$reason"
else
  echo "warning: no clock-event/throttle reason field supported; logging without it" >&2
fi
if ! nvidia-smi -i "$gpu" --query-gpu="$base" --format=csv,noheader >/dev/null 2>&1; then
  echo "NOT SUPPORTED: nvidia-smi --query-gpu on GPU $gpu" >&2
  exit 3
fi

# exec so that killing this script's PID stops the logger directly.
exec nvidia-smi -i "$gpu" --query-gpu="$fields" --format=csv -lms "$interval_ms" -f "$out"
