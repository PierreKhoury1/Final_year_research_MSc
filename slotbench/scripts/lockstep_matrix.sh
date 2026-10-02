#!/usr/bin/env bash
# Lockstep matrix: CPU-launched vs GPU-self-launched slot under no load, in-process AI load, a separate
# AI process without isolation, and a separate AI process under MPS (adversary capped at 50%).
# Usage: lockstep_matrix.sh --out DIR [--slots N] [--sizes "..."] [--gpu N] [--bin DIR]
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=""; slots=30000; sizes="${SB_SIZES:---subcarriers 612 --ldpc-cb 0 --ldpc-iters 3}"; gpu=0; bin="$here/../bin"
while [[ $# -gt 0 ]]; do
  case $1 in
    --out) out=$2; shift 2 ;;
    --slots) slots=$2; shift 2 ;;
    --sizes) sizes=$2; shift 2 ;;
    --gpu) gpu=$2; shift 2 ;;
    --bin) bin=$2; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[[ -n $out ]] || { echo "--out required" >&2; exit 2; }
mkdir -p "$out"
read -ra xsz <<< "$sizes"
log() { echo "[lockstep $(date -u +%H:%M:%S)] $*"; }

adv_pid=""
mps_on=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/sb-mps CUDA_MPS_LOG_DIRECTORY=/tmp/sb-mps-log
cleanup() {
  if [[ -n $adv_pid ]]; then kill -TERM "$adv_pid" 2>/dev/null || true; wait "$adv_pid" 2>/dev/null || true; adv_pid=""; fi
  if [[ $mps_on -eq 1 ]]; then echo quit | nvidia-cuda-mps-control 2>/dev/null || true; mps_on=0; fi
}
trap cleanup EXIT

start_adv() {  # start_adv WORKLOAD MPS(0|1) NAME
  local w=$1 mps=$2 name=$3
  local env=()
  [[ $mps -eq 1 ]] && env+=(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50)
  env "${env[@]}" "$bin/adversary" --workload "$w" --duty 100 --prio default --gpu "$gpu" \
      --out "$out/$name.adversary.json" > "$out/$name.adversary.log" 2>&1 &
  adv_pid=$!
  sleep 12
  kill -0 "$adv_pid" 2>/dev/null || { log "adversary died: $(tail -n 3 "$out/$name.adversary.log")"; adv_pid=""; return 1; }
}
stop_adv() {
  [[ -n $adv_pid ]] || return 0
  kill -TERM "$adv_pid" 2>/dev/null || true
  wait "$adv_pid" 2>/dev/null || true
  adv_pid=""
}

run_case() {  # run_case NAME MODE LOAD(none|stream)
  local name=$1 mode=$2 load=$3
  log "case $name (mode $mode, in-process load $load)"
  "$bin/lockstep_driver" --mode "$mode" --load "$load" --slots "$slots" --gpu "$gpu" --label "$name" \
      --out "$out/$name.json" --raw "$out/$name.bin" "${xsz[@]}" 2>&1 | tee "$out/$name.log" || true
}

{ nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem --format=csv; uname -r; nproc; } > "$out/header.txt" 2>&1 || true

# 1. alone
run_case cpu_alone   cpu none
run_case gpu_alone   gpu none
# 2. AI in the same process (same CUDA context, low-priority stream)
run_case cpu_inproc  cpu stream
run_case gpu_inproc  gpu stream
# 3. AI in a separate process, no isolation (time-slicing)
for w in sgemm llm; do
  if start_adv "$w" 0 "cpu_proc_$w"; then run_case "cpu_proc_$w" cpu none; fi; stop_adv
  if start_adv "$w" 0 "gpu_proc_$w"; then run_case "gpu_proc_$w" gpu none; fi; stop_adv
done
# 4. AI in a separate process under MPS (both clients; adversary capped at 50% of SMs)
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if nvidia-cuda-mps-control -d > "$out/mps.log" 2>&1; then
  mps_on=1
  sleep 2
  for w in sgemm llm; do
    if start_adv "$w" 1 "cpu_mps_$w"; then run_case "cpu_mps_$w" cpu none; fi; stop_adv
    if start_adv "$w" 1 "gpu_mps_$w"; then run_case "gpu_mps_$w" gpu none; fi; stop_adv
  done
  echo quit | nvidia-cuda-mps-control || true
  mps_on=0
else
  log "MPS daemon did not start: NOT SUPPORTED here"
fi

log "summary"
python3 - "$out" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, "*.json"))):
    if f.endswith(".adversary.json"): continue
    d = json.load(open(f))
    se = d.get("start_error_us", {}) or {}
    ex = d.get("exec_us", {}) or {}
    rows.append((os.path.basename(f)[:-5], d.get("ok"), se.get("p50"), se.get("p99"), se.get("p99_9"), se.get("max"),
                 ex.get("p50"), d.get("misses"), d.get("total_slots"), d.get("skipped"), d.get("error", "")))
print(f"{'case':16s} {'ok':>3s} {'start p50':>10s} {'p99':>9s} {'p99.9':>9s} {'max':>10s} {'exec p50':>9s} {'miss':>8s} {'of':>7s} {'skip':>6s}")
for r in rows:
    def f(x): return "   -" if x is None else f"{x:.1f}"
    print(f"{r[0]:16s} {str(r[1]):>3s} {f(r[2]):>10s} {f(r[3]):>9s} {f(r[4]):>9s} {f(r[5]):>10s} {f(r[6]):>9s} {str(r[7]):>8s} {str(r[8]):>7s} {str(r[9]):>6s} {r[10]}")
PY
