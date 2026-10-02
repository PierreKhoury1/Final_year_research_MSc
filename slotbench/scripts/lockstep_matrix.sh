#!/usr/bin/env bash
# Lockstep matrix: CPU-launched vs GPU-self-launched slot under no load, in-process AI load, a separate
# AI process without isolation, and a separate AI process under MPS (adversary capped at 50%).
# Usage: lockstep_matrix.sh --out DIR [--slots N] [--sizes "..."] [--gpu N] [--bin DIR] [--core N] [--fifo P]
#                           [--period-us N] [--deadline-us N] [--workloads "sgemm llm"] [--slot-variant full|no_cublas]
# Every case leaves DIR/<case>.json; a case whose driver failed or timed out gets a stub {"ok": false, "error": ...}.
# CPU and GPU always use the same explicitly selected slot; any failed/unsupported case makes the matrix exit nonzero.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=""; slots=30000; sizes="${SB_SIZES:---subcarriers 612 --ldpc-cb 0 --ldpc-iters 3}"; gpu=0; bin="$here/../bin"
core=""; fifo=80; period_us=500; deadline_us=500; workloads="sgemm llm"; slot_variant=full
while [[ $# -gt 0 ]]; do
  case $1 in
    --out) out=$2; shift 2 ;;
    --slots) slots=$2; shift 2 ;;
    --sizes) sizes=$2; shift 2 ;;
    --gpu) gpu=$2; shift 2 ;;
    --bin) bin=$2; shift 2 ;;
    --core) core=$2; shift 2 ;;
    --fifo) fifo=$2; shift 2 ;;
    --period-us) period_us=$2; shift 2 ;;
    --deadline-us) deadline_us=$2; shift 2 ;;
    --workloads) workloads=$2; shift 2 ;;
    --slot-variant) slot_variant=$2; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[[ -n $out ]] || { echo "--out required" >&2; exit 2; }
[[ $slot_variant == full || $slot_variant == no_cublas ]] || { echo "--slot-variant must be full or no_cublas" >&2; exit 2; }
mkdir -p "$out"
read -ra xsz <<< "$sizes"
read -ra xwl <<< "$workloads"
log() { echo "[lockstep $(date -u +%H:%M:%S)] $*"; }

# Timing thread on a core of its own when the host has one to spare (the load thread takes core+1).
if [[ -z $core ]]; then
  ncpu=$(nproc)
  if (( ncpu >= 4 )); then core=1; else core=-1; fi
fi
# Wall-clock cap per case: 3x the nominal run plus start-up/calibration slack.
case_timeout=$(( slots * period_us * 3 / 1000000 + 180 ))

adv_pid=""
mps_on=0
mps_root=""; mps_pipe=""; mps_log=""
case_failed=0
case_names=()
save_mps_logs() {
  if [[ -n $mps_log && -d $mps_log ]]; then
    mkdir -p "$out/mps_logs"
    cp -a "$mps_log/." "$out/mps_logs/" 2>/dev/null || true
  fi
}
cleanup() {
  if [[ -n $adv_pid ]]; then kill -TERM "$adv_pid" 2>/dev/null || true; wait "$adv_pid" 2>/dev/null || true; adv_pid=""; fi
  if [[ $mps_on -eq 1 ]]; then echo quit | CUDA_MPS_PIPE_DIRECTORY=$mps_pipe nvidia-cuda-mps-control 2>/dev/null || true; mps_on=0; fi
  save_mps_logs
  if [[ -n $mps_root ]]; then rm -rf -- "$mps_root"; fi
}
trap cleanup EXIT

# A stale MPS daemon from an earlier run would silently turn phases 1-3 into MPS cases.
# Linux comm names are truncated to 15 characters, so pgrep -x cannot match these names.
stale_mps() { pgrep -f '^([^[:space:]]*/)?nvidia-cuda-mps-(control|server)([[:space:]]|$)' >/dev/null 2>&1; }
mps_scan_rc=0
stale_mps || mps_scan_rc=$?
if [[ $mps_scan_rc -eq 0 ]]; then
  log "existing MPS daemon found; refusing to run non-MPS phases (stop it explicitly before running this matrix)"
  exit 3
elif [[ $mps_scan_rc -ne 1 ]]; then
  log "cannot inspect MPS processes (pgrep exit $mps_scan_rc); refusing to assume non-MPS execution"
  exit 3
fi
# Shells launched from an MPS session may inherit client caps or another daemon's pipe.
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
  CUDA_MPS_PINNED_DEVICE_MEM_LIMIT CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING CUDA_MPS_CLIENT_PRIORITY

start_adv() {  # start_adv WORKLOAD MPS(0|1) NAME   -- returns 1 if the adversary never produced work
  local w=$1 mps=$2 name=$3
  local env=() tl="$out/$name.adversary.csv"
  rm -f "$tl" "$out/$name.adversary.json"
  if [[ $mps -eq 1 ]]; then
    env+=(CUDA_MPS_PIPE_DIRECTORY=$mps_pipe CUDA_MPS_LOG_DIRECTORY=$mps_log CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50)
  fi
  env "${env[@]}" "$bin/adversary" --workload "$w" --duty 100 --prio default --gpu "$gpu" \
      --timeline "$tl" --out "$out/$name.adversary.json" > "$out/$name.adversary.log" 2>&1 &
  adv_pid=$!
  # ready = at least 3 timeline rows with work done (the models/buffers are allocated and kernels are flowing)
  local i rows=0
  for ((i = 0; i < 90; i++)); do
    sleep 1
    kill -0 "$adv_pid" 2>/dev/null || { log "adversary died: $(tail -n 3 "$out/$name.adversary.log" | tr '\n' ' ')"; wait "$adv_pid" 2>/dev/null || true; adv_pid=""; return 1; }
    rows=$(awk -F, 'NR > 1 && $2 + 0 > 0 { n++ } END { print n + 0 }' "$tl" 2>/dev/null || echo 0)
    (( rows >= 3 )) && break
  done
  if (( rows < 3 )); then log "adversary $w produced no work in 90 s"; kill -TERM "$adv_pid" 2>/dev/null || true; wait "$adv_pid" 2>/dev/null || true; adv_pid=""; return 1; fi
  sleep 3   # let it settle at steady state
}
stop_adv() {
  [[ -n $adv_pid ]] || return 0
  kill -TERM "$adv_pid" 2>/dev/null || true
  wait "$adv_pid" 2>/dev/null || true
  adv_pid=""
}

stub() {  # stub NAME MODE REASON
  python3 - "$1" "$2" "$3" "$slot_variant" > "$out/$1.json" <<'PY'
import json, sys
json.dump(dict(ok=False, label=sys.argv[1], mode=sys.argv[2], error=sys.argv[3], slot_variant=sys.argv[4]), sys.stdout)
print()
PY
  case_failed=1
}
run_case() {  # run_case NAME MODE LOAD(none|stream)
  local name=$1 mode=$2 load=$3 rc=0
  case_names+=("$name")
  log "case $name (mode $mode, in-process load $load, timeout ${case_timeout}s)"
  rm -f "$out/$name.json" "$out/$name.bin"
  timeout -k 20 "$case_timeout" "$bin/lockstep_driver" --mode "$mode" --load "$load" --slots "$slots" --gpu "$gpu" \
      --period-us "$period_us" --deadline-us "$deadline_us" --core "$core" --fifo "$fifo" --label "$name" \
      --out "$out/$name.json" --raw "$out/$name.bin" "${xsz[@]}" --slot-variant "$slot_variant" > "$out/$name.log" 2>&1 || rc=$?
  tail -n 2 "$out/$name.log"
  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    log "case $name TIMED OUT (rc $rc)"; stub "$name" "$mode" "timeout after ${case_timeout}s"
  elif [[ ! -s "$out/$name.json" ]]; then
    log "case $name FAILED (rc $rc, no JSON)"; stub "$name" "$mode" "driver exit $rc without JSON: $(tail -n 1 "$out/$name.log" | tr -d '"\\')"
  elif [[ $rc -ne 0 ]]; then
    log "case $name reported failure (rc $rc)"
    case_failed=1
  fi
  # a crashed executive may leave the device wedged for the next case; give the driver time to go away
  sleep 2
}
skip_case() { case_names+=("$1"); rm -f "$out/$1.bin"; log "case $1 SKIPPED: $3"; stub "$1" "$2" "skipped: $3"; }

{ nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,compute_mode --format=csv; uname -r; nproc; echo "core=$core fifo=$fifo period_us=$period_us deadline_us=$deadline_us slots=$slots sizes=$sizes slot_variant=$slot_variant"; } > "$out/header.txt" 2>&1 || true

# 1. alone
run_case cpu_alone   cpu none
run_case gpu_alone   gpu none
# 2. AI in the same process (same CUDA context, low-priority stream)
run_case cpu_inproc  cpu stream
run_case gpu_inproc  gpu stream
# 3. AI in a separate process, no isolation (time-slicing)
for w in "${xwl[@]}"; do
  for m in cpu gpu; do
    if start_adv "$w" 0 "${m}_proc_$w"; then run_case "${m}_proc_$w" "$m" none; else skip_case "${m}_proc_$w" "$m" "adversary $w not ready"; fi
    stop_adv
  done
done
# 4. AI in a separate process under MPS (both clients; adversary capped at 50% of SMs). The MPS environment is
#    exported to the driver only here.
mps_root=$(mktemp -d "${TMPDIR:-/tmp}/sb-lockstep-mps.XXXXXX")
mps_pipe=$mps_root/pipe; mps_log=$mps_root/log
mkdir -p "$mps_pipe" "$mps_log"
if CUDA_MPS_PIPE_DIRECTORY=$mps_pipe CUDA_MPS_LOG_DIRECTORY=$mps_log nvidia-cuda-mps-control -d > "$out/mps.log" 2>&1; then
  mps_on=1
  sleep 2
  export CUDA_MPS_PIPE_DIRECTORY=$mps_pipe CUDA_MPS_LOG_DIRECTORY=$mps_log
  for w in "${xwl[@]}"; do
    for m in cpu gpu; do
      if start_adv "$w" 1 "${m}_mps_$w"; then run_case "${m}_mps_$w" "$m" none; else skip_case "${m}_mps_$w" "$m" "adversary $w not ready under MPS"; fi
      stop_adv
    done
  done
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  echo quit | CUDA_MPS_PIPE_DIRECTORY=$mps_pipe nvidia-cuda-mps-control || true
  mps_on=0
  save_mps_logs
else
  log "MPS daemon did not start: NOT SUPPORTED here"
  for w in "${xwl[@]}"; do for m in cpu gpu; do skip_case "${m}_mps_$w" "$m" "MPS daemon did not start"; done; done
fi

log "summary"
python3 - "$out" "$slot_variant" "$case_failed" "${case_names[@]}" <<'PY'
import json, os, sys
out, slot_variant, failed = sys.argv[1:4]
failed = int(failed)
rows = []
for name in sorted(sys.argv[4:]):
    f = os.path.join(out, name + ".json")
    try: d = json.load(open(f))
    except Exception as e: d = {"ok": False, "error": f"unreadable JSON: {e}"}
    if d.get("ok") is not True:
        failed = 1
    elif d.get("slot_variant") != slot_variant:
        failed = 1
        d = dict(d, ok=False, error=f"slot variant mismatch: requested {slot_variant}, got {d.get('slot_variant')}")
    se = d.get("start_error_us") or {}
    lp = d.get("launch_precision_us") or {}
    ex = d.get("exec_us") or {}
    rows.append((os.path.basename(f)[:-5], d.get("ok"), se.get("p50"), se.get("p99"), se.get("max"),
                 lp.get("p50"), lp.get("p99"), ex.get("p50"), d.get("misses"), d.get("total_slots"), d.get("skipped"), d.get("error", "")))
print(f"{'case':16s} {'ok':>5s} {'start p50':>10s} {'p99':>9s} {'max':>10s} {'prec p50':>9s} {'p99':>9s} {'exec p50':>9s} {'miss':>8s} {'of':>7s} {'skip':>6s}")
def f(x): return "   -" if x is None else f"{x:.1f}"
for r in rows:
    print(f"{r[0]:16s} {str(r[1]):>5s} {f(r[2]):>10s} {f(r[3]):>9s} {f(r[4]):>10s} {f(r[5]):>9s} {f(r[6]):>9s} {f(r[7]):>9s} {str(r[8]):>8s} {str(r[9]):>7s} {str(r[10]):>6s} {r[11]}")
sys.exit(1 if failed else 0)
PY
