#!/usr/bin/env bash
# shellcheck disable=SC2329  # cleanup/as_root/write_run_json are invoked via traps and run_logged
# run_one.sh: one cell of the run matrix, following the run procedure end to end.
#
#   run_one.sh --out DIR --mechanism M0..M6|SOLO --workload sgemm|llm|vision|real-llm|real-vision|idle --duty D
#              --slots N --warmup-s S --settle-s S
#              [--driver-core C] [--collector-core C] [--fifo P]
#              [--driver-flags "..."] [--adversary-flags "..."] [--solo-seconds S]
#              [--lock-clocks 0|1] [--gpu N] [--bin DIR] [--cell NAME] [--rep N] [--print-plan]
#
# Procedure: move any previous attempt in DIR to DIR/old_attempts/<time>/; env_capture -> env.txt;
# reset MPS (stop daemons; a daemon that survives makes the run invalid); lock clocks (or unlock
# for M5); mechanism setup (M2 MPS daemon, M3 time-slice, M6 MIG UUIDs); warm-up adversary for
# --warmup-s (same workload/duty/env, output warmup_adversary.json); telemetry in background;
# adversary in background; settle --settle-s; driver for N slots; then stop adversary (SIGTERM,
# SIGKILL after a timeout), stop telemetry, and (EXIT trap, so also on failure or Ctrl-C) restore
# time-slice, MPS and clocks. Validity -> DIR/status "ok" or "invalid:<reason>". Everything applied,
# unsupported, and every command run goes to DIR/run.json. SOLO: adversary alone at D=100 for
# --solo-seconds, no driver. --print-plan prints the resolved commands and exits (no side effects).
#
# Mechanisms (DESIGN.md section 7): M0 prio default/default; M1 driver high, adversary low;
# M2 MPS + adversary CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50, driver high, adversary default;
# M3 nvidia-smi compute-policy --set-timeslice=1 (restored to 0 = default), driver high, adversary
# default; M4 = M1 + driver --mode streams; M5 = M1 with clocks unlocked; M6 = M1 priorities with
# driver/adversary on MIG instances $MIG_DRIVER_UUID / $MIG_ADV_UUID (CUDA_VISIBLE_DEVICES, --gpu 0).
# Flag strings in --driver-flags/--adversary-flags are split on whitespace (no quoting).
# Exit: 0 ok, 1 invalid, 2 usage, 3 not supported, 130 interrupted.
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(dirname "$here")
PY=${SB_PYTHON:-python3}

usage() { sed -n '2,30p' "$0" >&2; exit 2; }

out=""; mech=""; workload=""; duty=""; slots=""; warmup_s=""; settle_s=""
driver_core=-1; collector_core=-1; fifo=0; driver_flags=""; adversary_flags=""
solo_seconds=60; lock_clocks=0; gpu=0; bin="$root/bin"; cell=""; rep=""; print_plan=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --out) out=${2:?}; shift 2 ;;
    --mechanism) mech=${2:?}; shift 2 ;;
    --workload) workload=${2:?}; shift 2 ;;
    --duty) duty=${2:?}; shift 2 ;;
    --slots) slots=${2:?}; shift 2 ;;
    --warmup-s) warmup_s=${2:?}; shift 2 ;;
    --settle-s) settle_s=${2:?}; shift 2 ;;
    --driver-core) driver_core=${2:?}; shift 2 ;;
    --collector-core) collector_core=${2:?}; shift 2 ;;
    --fifo) fifo=${2:?}; shift 2 ;;
    --driver-flags) driver_flags=${2-}; shift 2 ;;
    --adversary-flags) adversary_flags=${2-}; shift 2 ;;
    --solo-seconds) solo_seconds=${2:?}; shift 2 ;;
    --lock-clocks) lock_clocks=${2:?}; shift 2 ;;
    --gpu) gpu=${2:?}; shift 2 ;;
    --bin) bin=${2:?}; shift 2 ;;
    --cell) cell=${2:?}; shift 2 ;;
    --rep) rep=${2:?}; shift 2 ;;
    --print-plan) print_plan=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

bad() { echo "run_one.sh: $*" >&2; exit 2; }
isnum() { [[ $1 =~ ^[0-9]+([.][0-9]+)?$ ]]; }
isint() { [[ $1 =~ ^-?[0-9]+$ ]]; }
[[ -n $out ]] || bad "--out is required"
[[ $mech =~ ^(M[0-6]|SOLO)$ ]] || bad "--mechanism must be M0..M6 or SOLO (got '$mech')"
[[ $workload =~ ^(sgemm|llm|vision|real-llm|real-vision|idle)$ ]] \
  || bad "--workload must be sgemm|llm|vision|real-llm|real-vision|idle (got '$workload')"
[[ $mech == SOLO ]] && duty=100
if ! [[ $duty =~ ^[0-9]+$ ]] || (( duty > 100 )); then bad "--duty must be an integer 0..100"; fi
[[ $mech == SOLO ]] && slots=${slots:-0}
[[ $slots =~ ^[0-9]+$ ]] || bad "--slots must be an integer"
isnum "${warmup_s:-x}" || bad "--warmup-s must be a number of seconds"
isnum "${settle_s:-x}" || bad "--settle-s must be a number of seconds"
isnum "$solo_seconds" || bad "--solo-seconds must be a number"
for v in "$driver_core" "$collector_core" "$fifo" "$gpu"; do isint "$v" || bad "cores/fifo/gpu must be integers"; done
[[ $lock_clocks =~ ^[01]$ ]] || bad "--lock-clocks must be 0 or 1"
[[ -n $cell ]] || cell=$(basename "$out")
if [[ $print_plan -eq 1 ]]; then
  out=$(realpath -m "$out")
else
  mkdir -p "$out" || bad "cannot create $out"
  out=$(cd "$out" && pwd)
fi

# ------------------------------------------------------------------ resolve the plan
drv_prio=high; adv_prio=low; drv_mode=graph; do_lock=$lock_clocks; do_unlock=0
mig=0; use_mps=0; timeslice=0
case $mech in
  M0) drv_prio=default; adv_prio=default ;;
  M1) ;;
  M2) use_mps=1; adv_prio=default ;;
  M3) timeslice=1; adv_prio=default ;;
  M4) drv_mode=streams ;;
  M5) do_lock=0; do_unlock=1 ;;
  M6) mig=1 ;;
  SOLO) adv_prio=default ;;
esac

mpsdir=/tmp/sb-mps-$$
drv_env=(); adv_env=()
drv_gpu=$gpu; adv_gpu=$gpu
if [[ $use_mps -eq 1 ]]; then
  drv_env+=("CUDA_MPS_PIPE_DIRECTORY=$mpsdir/pipe" "CUDA_MPS_LOG_DIRECTORY=$out/mps_log")
  adv_env+=("CUDA_MPS_PIPE_DIRECTORY=$mpsdir/pipe" "CUDA_MPS_LOG_DIRECTORY=$out/mps_log"
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50")
fi
if [[ $mig -eq 1 ]]; then
  drv_env+=("CUDA_VISIBLE_DEVICES=${MIG_DRIVER_UUID:-}")
  adv_env+=("CUDA_VISIBLE_DEVICES=${MIG_ADV_UUID:-}")
  drv_gpu=0; adv_gpu=0
fi

read -ra xdrv <<< "$driver_flags"
read -ra xadv <<< "$adversary_flags"
drv_cmd=("$bin/slot_driver" --out "$out" --slots "$slots" --prio "$drv_prio" --mode "$drv_mode"
         --gpu "$drv_gpu" --label "$cell")
(( driver_core >= 0 )) && drv_cmd+=(--core "$driver_core")
(( collector_core >= 0 )) && drv_cmd+=(--collector-core "$collector_core")
(( fifo > 0 )) && drv_cmd+=(--fifo "$fifo")
drv_cmd+=("${xdrv[@]}")

if [[ $workload == real-* ]]; then
  # real models (Qwen2.5 LLM, YOLOv8n) need torch; SB_AI_PYTHON points at the venv that has it
  adv_base=("${SB_AI_PYTHON:-python3}" "$here/../adversary/real/real_ai.py" --workload "$workload" --duty "$duty"
            --prio "$adv_prio" --gpu "$adv_gpu")
else
  adv_base=("$bin/adversary" --workload "$workload" --duty "$duty" --prio "$adv_prio" --gpu "$adv_gpu")
fi
warm_cmd=("${adv_base[@]}" --seconds "$warmup_s" --out "$out/warmup_adversary.json" "${xadv[@]}")
if [[ $mech == SOLO ]]; then
  adv_cmd=("${adv_base[@]}" --seconds "$solo_seconds" --out "$out/adversary.json"
           --timeline "$out/adversary_timeline.csv" "${xadv[@]}")
else
  adv_cmd=("${adv_base[@]}" --out "$out/adversary.json" --timeline "$out/adversary_timeline.csv" "${xadv[@]}")
fi

q() { local s; printf -v s '%q ' "$@"; printf '%s' "${s% }"; }
# qe ENVNAME CMDNAME: quoted command line with an "env VAR=..." prefix when the env array is non-empty.
qe() {
  local -n e=$1 c=$2
  if [[ ${#e[@]} -gt 0 ]]; then q env "${e[@]}" "${c[@]}"; else q "${c[@]}"; fi
}

if [[ $print_plan -eq 1 ]]; then
  echo "cell: $cell  mechanism: $mech  workload: $workload  duty: $duty"
  [[ $do_lock -eq 1 ]] && echo "clocks: $(q "$here/gpu_lock.sh" lock --gpu "$gpu")"
  [[ $do_unlock -eq 1 ]] && echo "clocks: $(q "$here/gpu_lock.sh" unlock --gpu "$gpu")"
  [[ $use_mps -eq 1 ]] && echo "mps: $(q "$here/mps.sh" start --gpu "$gpu" --dir "$mpsdir" --log-dir "$out/mps_log")"
  [[ $timeslice -eq 1 ]] && echo "timeslice: nvidia-smi -i $gpu compute-policy --set-timeslice=1 (restore 0)"
  [[ $mig -eq 1 ]] && echo "mig: driver on \$MIG_DRIVER_UUID, adversary on \$MIG_ADV_UUID"
  isnum "$warmup_s" && [[ $warmup_s != 0 ]] && echo "warmup: $(qe adv_env warm_cmd)"
  echo "adversary: $(qe adv_env adv_cmd)"
  [[ $mech != SOLO ]] && echo "settle: ${settle_s}s" && echo "driver: $(qe drv_env drv_cmd)"
  exit 0
fi

# ------------------------------------------------------------------ bookkeeping
runlog=$out/.runlog.tsv
setup_log=$out/setup.log
now_iso() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }
# rec KIND KEY VALUE: KIND in setting|command|unsupported|note|exit.
rec() { local v=${3-}; v=${v//$'\t'/ }; v=${v//$'\n'/ | }; printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$v" "$(now_iso)" >> "$runlog"; }
log() { printf '[%s] %s\n' "$(now_iso)" "$*" | tee -a "$setup_log" >&2; }

# Move a previous attempt aside so stale files can never validate this one.
prev=$out/old_attempts/$(date -u +%Y%m%dT%H%M%SZ)
for f in "$out"/* "$out"/.[!.]*; do
  if [[ -e $f && $(basename "$f") != old_attempts ]]; then
    mkdir -p "$prev" && mv "$f" "$prev"/
  fi
done
: > "$runlog"
started=$(now_iso)
rec setting started "$started"

status_written=0
write_status() { printf '%s\n' "$1" > "$out/status"; status_written=1; rec setting status "$1"; log "status: $1"; }

# run_logged WHAT CMD...: run a setup command, record it and its output; returns its exit code.
run_logged() {
  local what=$1; shift
  local o rc
  rec command "$what" "$(q "$@")"
  o=$("$@" 2>&1); rc=$?
  printf '$ %s\n%s\n(rc=%d)\n' "$(q "$@")" "$o" "$rc" >> "$setup_log"
  rec exit "$what" "$rc"
  last_out=$o
  return $rc
}

# as_root CMD...: run directly as root, else via passwordless sudo, else fail with rc 4.
as_root() {
  if [[ $(id -u) -eq 0 ]]; then "$@"
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then sudo -n "$@"
  else echo "not root and passwordless sudo unavailable" >&2; return 4
  fi
}

alive() {  # alive PID: running and not a zombie
  [[ -n ${1-} ]] && kill -0 "$1" 2>/dev/null || return 1
  local st; st=$(ps -o stat= -p "$1" 2>/dev/null)
  [[ -n $st && $st != Z* ]]
}

# stop_proc PID NAME TIMEOUT_S: SIGTERM, wait, SIGKILL after timeout; sets stop_rc.
stop_proc() {
  local pid=$1 name=$2 to=$3 i=0
  stop_rc=""
  [[ -n $pid ]] || return 0
  if alive "$pid"; then
    kill -TERM "$pid" 2>/dev/null
    while alive "$pid" && (( i < to * 10 )); do sleep 0.1; i=$((i + 1)); done
    if alive "$pid"; then
      kill -KILL "$pid" 2>/dev/null
      rec note "$name" "killed with SIGKILL after ${to}s"
      log "$name did not stop in ${to}s: SIGKILL"
    fi
  fi
  wait "$pid" 2>/dev/null; stop_rc=$?
  rec exit "$name" "$stop_rc"
}

drv_pid=""; adv_pid=""; warm_pid=""; tel_pid=""
mps_started=0; timeslice_set=0; clocks_locked=0; interrupted=0; fail_reason=""

write_run_json() {
  "$PY" - "$runlog" "$out/run.json" <<'PY' || echo "warning: could not write run.json" >&2
import json, sys
log, dst = sys.argv[1], sys.argv[2]
d = {"settings": {}, "commands": [], "exit_codes": {}, "unsupported": [], "notes": []}
for line in open(log, encoding="utf-8", errors="replace"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 4:
        continue
    kind, key, val, t = parts
    if kind == "setting":
        d["settings"][key] = val
    elif kind == "command":
        d["commands"].append({"what": key, "cmd": val, "t": t})
    elif kind == "exit":
        d["exit_codes"][key] = int(val) if val.lstrip("-").isdigit() else val
    elif kind == "unsupported":
        d["unsupported"].append({"what": key, "detail": val, "t": t})
    elif kind == "note":
        d["notes"].append({"what": key, "detail": val, "t": t})
s = d["settings"]
for k in ("mechanism", "workload", "cell", "status"):
    if k in s:
        d[k] = s[k]
for k in ("duty", "rep", "slots"):
    if s.get(k, "").isdigit():
        d[k] = int(s[k])
with open(dst + ".tmp", "w") as f:
    json.dump(d, f, indent=1, sort_keys=True)
import os
os.replace(dst + ".tmp", dst)
PY
}

cleanup() {
  local rc=$?
  trap '' INT TERM
  [[ -n $drv_pid ]] && stop_proc "$drv_pid" driver 60
  [[ -n $warm_pid ]] && stop_proc "$warm_pid" warmup_adversary 30
  [[ -n $adv_pid ]] && stop_proc "$adv_pid" adversary 30
  [[ -n $tel_pid ]] && stop_proc "$tel_pid" telemetry 5
  if [[ $timeslice_set -eq 1 ]]; then
    run_logged restore_timeslice as_root nvidia-smi -i "$gpu" compute-policy --set-timeslice=0 \
      || rec unsupported restore_timeslice "could not restore time-slice to default: $last_out"
  fi
  if [[ $mps_started -eq 1 ]]; then
    run_logged mps_stop "$here/mps.sh" stop --dir "$mpsdir" || rec note mps_stop "$last_out"
    [[ $mpsdir == /tmp/sb-mps-* ]] && rm -rf -- "$mpsdir"
  fi
  if [[ $clocks_locked -eq 1 ]]; then
    run_logged clocks_unlock "$here/gpu_lock.sh" unlock --gpu "$gpu" || rec note clocks_unlock "$last_out"
  fi
  if [[ $status_written -eq 0 ]]; then
    if [[ $interrupted -eq 1 ]]; then write_status "invalid:interrupted"
    else write_status "invalid:${fail_reason:-aborted_rc$rc}"
    fi
  fi
  rec setting ended "$(now_iso)"
  write_run_json
  rm -f "$runlog"
}
trap cleanup EXIT
trap 'trap "" INT TERM; interrupted=1; log "interrupted"; exit 130' INT TERM

# fail REASON [RC]: invalid run, stop now (cleanup restores everything).
fail() { fail_reason=$1; write_status "invalid:$1"; exit "${2:-1}"; }

# waitsleep S: interruptible sleep.
waitsleep() { sleep "$1" & wait $!; }

for kv in "mechanism=$mech" "workload=$workload" "duty=$duty" "slots=$slots" "warmup_s=$warmup_s" \
          "settle_s=$settle_s" "gpu=$gpu" "lock_clocks=$lock_clocks" "cell=$cell" "rep=$rep" \
          "driver_core=$driver_core" "collector_core=$collector_core" "fifo=$fifo" \
          "driver_flags=$driver_flags" "adversary_flags=$adversary_flags" "bin=$bin" \
          "driver_prio=$drv_prio" "adversary_prio=$adv_prio" "driver_mode=$drv_mode" \
          "host=$(hostname)" "user=$(id -un)"; do
  rec setting "${kv%%=*}" "${kv#*=}"
done
[[ $mech == SOLO ]] && rec setting solo_seconds "$solo_seconds"
log "cell $cell: $mech $workload d$duty -> $out"

# ------------------------------------------------------------------ environment + resets
[[ -x ${adv_cmd[0]} ]] || fail "missing_binary:${adv_cmd[0]}" 2
[[ $mech == SOLO || -x ${drv_cmd[0]} ]] || fail "missing_binary:${drv_cmd[0]}" 2
run_logged env_capture "$here/env_capture.sh" "$out/env.txt" || rec note env_capture "failed: $last_out"

# Reset MPS state: no daemon may survive from an earlier run (it would silently put this run in MPS mode).
run_logged mps_reset "$here/mps.sh" stop --all
if pgrep -f '(^|/)nvidia-cuda-mps-(control|server)( |$)' >/dev/null 2>&1; then
  fail "mps_daemon_still_running"
fi

# Clocks.
if [[ $do_unlock -eq 1 ]]; then
  run_logged clocks_unlock_m5 "$here/gpu_lock.sh" unlock --gpu "$gpu"
  r=$?; [[ $r -ne 0 ]] && rec unsupported clocks_unlock "rc=$r: $last_out"
elif [[ $do_lock -eq 1 ]]; then
  clocks_locked=1
  run_logged clocks_lock "$here/gpu_lock.sh" lock --gpu "$gpu"
  r=$?; [[ $r -ne 0 ]] && rec unsupported clocks_lock "rc=$r: $(grep 'NOT SUPPORTED' <<< "$last_out" | tr '\n' ' ')"
fi
if run_logged clocks_status "$here/gpu_lock.sh" status --gpu "$gpu"; then
  while IFS='=' read -r k v; do [[ -n $k ]] && rec setting "clock_$k" "$v"; done <<< "$last_out"
fi
rec setting clocks_locked_requested "$do_lock"

# Mechanism-specific setup.
if [[ $use_mps -eq 1 ]]; then
  mps_started=1
  run_logged mps_start "$here/mps.sh" start --gpu "$gpu" --dir "$mpsdir" --log-dir "$out/mps_log" \
    || { rec unsupported mps "$last_out"; fail "not_supported:mps" 3; }
  rec setting mps_pipe_dir "$mpsdir/pipe"
  rec setting adversary_active_thread_percentage 50
fi
if [[ $timeslice -eq 1 ]]; then
  run_logged compute_policy_before nvidia-smi -i "$gpu" compute-policy -l || true
  if run_logged set_timeslice as_root nvidia-smi -i "$gpu" compute-policy --set-timeslice=1; then
    timeslice_set=1
    rec setting timeslice 1
  else
    rec unsupported timeslice "$last_out"
    fail "not_supported:timeslice" 3
  fi
fi
if [[ $mig -eq 1 ]]; then
  if [[ -z ${MIG_DRIVER_UUID:-} || -z ${MIG_ADV_UUID:-} ]]; then
    rec unsupported mig "MIG_DRIVER_UUID/MIG_ADV_UUID not set (needs a MIG-enabled A100/H100)"
    fail "not_supported:mig" 3
  fi
  rec setting mig_driver_uuid "$MIG_DRIVER_UUID"
  rec setting mig_adversary_uuid "$MIG_ADV_UUID"
fi

# ------------------------------------------------------------------ warm-up
if [[ $warmup_s != 0 && $warmup_s != 0.0 ]]; then
  log "warm-up ${warmup_s}s"
  rec command warmup_adversary "$(qe adv_env warm_cmd)"
  env "${adv_env[@]}" "${warm_cmd[@]}" > "$out/warmup_adversary.log" 2>&1 &
  warm_pid=$!
  wait "$warm_pid"; r=$?; warm_pid=""
  rec exit warmup_adversary "$r"
  [[ $r -eq 0 ]] || fail "warmup_adversary_rc$r"
fi

# ------------------------------------------------------------------ telemetry + adversary
rec command telemetry "$(q "$here/telemetry.sh" "$out/telemetry.csv" --gpu "$gpu" --interval-ms 1000)"
"$here/telemetry.sh" "$out/telemetry.csv" --gpu "$gpu" --interval-ms 1000 > "$out/telemetry.log" 2>&1 &
tel_pid=$!

log "adversary start"
rec command adversary "$(qe adv_env adv_cmd)"
env "${adv_env[@]}" "${adv_cmd[@]}" > "$out/adversary.log" 2>&1 &
adv_pid=$!

if [[ $mech == SOLO ]]; then
  wait "$adv_pid"; r=$?; adv_pid=""
  rec exit adversary "$r"
  alive "$tel_pid" || rec note telemetry "telemetry was not running at the end (see telemetry.log)"
  stop_proc "$tel_pid" telemetry 5; tel_pid=""
  reasons=()
  [[ $r -eq 0 ]] || reasons+=("adversary_rc$r")
  "$PY" -c 'import json,sys; json.load(open(sys.argv[1]))' "$out/adversary.json" 2>/dev/null \
    || reasons+=("no_adversary_json")
  if [[ ${#reasons[@]} -eq 0 ]]; then write_status ok; exit 0; fi
  st=$(IFS=,; echo "${reasons[*]}")
  write_status "invalid:$st"
  exit 1
fi

log "settle ${settle_s}s"
waitsleep "$settle_s"
alive "$adv_pid" || fail "adversary_exited_during_settle"
if [[ $use_mps -eq 1 ]]; then
  run_logged mps_status "$here/mps.sh" status --dir "$mpsdir"
  rec setting mps_status "$last_out"
  # Every adversary (idle too) creates a context, so an MPS server must exist by now.
  if ! grep -qE '^server_list=[0-9]' <<< "$last_out"; then
    fail "mps_not_active"
  fi
fi

# ------------------------------------------------------------------ driver
log "driver start: $slots slots"
rec command driver "$(qe drv_env drv_cmd)"
env "${drv_env[@]}" "${drv_cmd[@]}" > "$out/driver.log" 2>&1 &
drv_pid=$!
wait "$drv_pid"; drv_rc=$?; drv_pid=""
rec exit driver "$drv_rc"
log "driver exited rc=$drv_rc"

adv_alive=0; alive "$adv_pid" && adv_alive=1
rec setting adversary_alive_at_driver_end "$adv_alive"
stop_proc "$adv_pid" adversary 30; adv_pid=""
tel_alive=0; alive "$tel_pid" && tel_alive=1
[[ $tel_alive -eq 1 ]] || rec note telemetry "telemetry was not running at the end (see telemetry.log)"
stop_proc "$tel_pid" telemetry 5; tel_pid=""

# ------------------------------------------------------------------ validity
reasons=()
[[ $drv_rc -eq 0 ]] || reasons+=("driver_rc$drv_rc")
if [[ -f $out/meta.json ]]; then
  ovf=$("$PY" - "$out/meta.json" <<'PY'
import json, sys
def find(o, k):
    if isinstance(o, dict):
        if k in o:
            return o[k]
        for v in o.values():
            r = find(v, k)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v, k)
            if r is not None:
                return r
    return None
try:
    v = find(json.load(open(sys.argv[1])), "ring_overflows")
except Exception:
    print("unparseable"); sys.exit()
print("missing" if v is None else v)
PY
)
  rec setting ring_overflows "$ovf"
  case $ovf in
    0) ;;
    unparseable) reasons+=("meta_json_unparseable") ;;
    missing) reasons+=("ring_overflows_missing") ;;
    *) reasons+=("ring_overflows=$ovf") ;;
  esac
else
  reasons+=("no_meta_json")
fi
[[ $adv_alive -eq 1 ]] || reasons+=("adversary_exited_early")
"$PY" -c 'import json,sys; json.load(open(sys.argv[1]))' "$out/adversary.json" 2>/dev/null \
  || reasons+=("no_adversary_json")

if [[ ${#reasons[@]} -eq 0 ]]; then
  write_status ok
  final=0
else
  st=$(IFS=,; echo "${reasons[*]}")
  write_status "invalid:$st"
  final=1
fi

# Per-run summary (non-fatal; another component owns analysis/).
if [[ -f $root/analysis/summarize.py && -f $out/slots.bin ]]; then
  run_logged summarize env -C "$root" "$PY" -m analysis.summarize "$out" \
    || rec note summarize "analysis.summarize failed (rc $?), see setup.log"
fi
exit $final
