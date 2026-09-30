#!/usr/bin/env bash
# gpu_lock.sh lock [--gpu N] [--gc MHZ] [--mc MHZ] | unlock [--gpu N] | status [--gpu N]
# Persistence mode + fixed graphics/memory clocks so slot timings do not depend on boost state.
# lock:   nvidia-smi -pm 1; -lgc GC,GC; -lmc MC,MC. Default MC = highest supported memory clock;
#         default GC = the default applications graphics clock if the driver reports one, else the
#         highest supported graphics clock <= 80% of the maximum (a sustainable, non-boost value).
# unlock: nvidia-smi -rgc; -rmc (persistence mode is left on).
# status: prints key=value lines with the clocks actually in effect now.
# Anything not permitted/supported prints "NOT SUPPORTED: <what>" and the script exits 3 after
# doing what it can (callers record it as a finding). Needs root (uses sudo -n if available).
set -uo pipefail

usage() { echo "usage: $0 lock [--gpu N] [--gc MHZ] [--mc MHZ] | unlock [--gpu N] | status [--gpu N]" >&2; exit 2; }

[[ $# -ge 1 ]] || usage
cmd=$1; shift
gpu=0; gc=""; mc=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) gpu=${2:?}; shift 2 ;;
    --gc) gc=${2:?}; shift 2 ;;
    --mc) mc=${2:?}; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NOT SUPPORTED: nvidia-smi not found"
  exit 3
fi

unsupported=0

# Run a privileged nvidia-smi command; on failure print NOT SUPPORTED with its message.
as_root() {
  local what=$1; shift
  local outp rc
  if [[ $(id -u) -eq 0 ]]; then
    outp=$("$@" 2>&1); rc=$?
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    outp=$(sudo -n "$@" 2>&1); rc=$?
  else
    outp="not root and passwordless sudo unavailable"; rc=4
  fi
  echo "cmd: $*"
  if [[ $rc -ne 0 ]]; then
    echo "NOT SUPPORTED: $what (rc=$rc: $(echo "$outp" | tr '\n' ' ' | cut -c1-200))"
    unsupported=1
    return 1
  fi
  return 0
}

query() {  # query FIELD -> value or empty
  nvidia-smi -i "$gpu" --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' '
}

is_int() { [[ $1 =~ ^[0-9]+$ ]]; }

print_status() {
  local f v
  echo "gpu=$gpu"
  echo "name=$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
  for f in persistence_mode clocks.sm clocks.mem clocks.max.sm clocks.max.mem \
           clocks.applications.graphics clocks.default_applications.graphics power.limit; do
    v=$(query "$f")
    echo "${f//./_}=${v:-N/A}"
  done
}

choose_clocks() {
  local pairs target best g
  pairs=$(nvidia-smi -i "$gpu" --query-supported-clocks=mem,gr --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  if [[ -z $mc ]]; then
    mc=$(echo "$pairs" | cut -d, -f1 | grep -E '^[0-9]+$' | sort -n | tail -n1)
  fi
  if [[ -z $gc ]]; then
    local grs
    grs=$(echo "$pairs" | awk -F, -v m="$mc" '$1==m {print $2}' | grep -E '^[0-9]+$' | sort -n)
    [[ -n $grs ]] || grs=$(echo "$pairs" | cut -d, -f2 | grep -E '^[0-9]+$' | sort -n -u)
    target=$(query clocks.default_applications.graphics)
    if ! is_int "$target"; then
      local mx
      mx=$(echo "$grs" | tail -n1)
      is_int "$mx" && target=$(( mx * 8 / 10 ))
    fi
    best=""
    if is_int "$target"; then
      for g in $grs; do
        (( g <= target )) && best=$g
      done
      [[ -n $best ]] || best=$(echo "$grs" | head -n1)
    fi
    gc=$best
  fi
}

case $cmd in
  lock)
    as_root "persistence mode (nvidia-smi -pm 1)" nvidia-smi -i "$gpu" -pm 1
    choose_clocks
    echo "gc_requested=${gc:-none}"
    echo "mc_requested=${mc:-none}"
    if is_int "$gc"; then
      as_root "graphics clock lock (-lgc $gc,$gc)" nvidia-smi -i "$gpu" -lgc "$gc,$gc" \
        && echo "gc_locked=$gc"
    else
      echo "NOT SUPPORTED: graphics clock lock (no supported clock list)"; unsupported=1
    fi
    if is_int "$mc"; then
      as_root "memory clock lock (-lmc $mc,$mc)" nvidia-smi -i "$gpu" -lmc "$mc,$mc" \
        && echo "mc_locked=$mc"
    else
      echo "NOT SUPPORTED: memory clock lock (no supported clock list)"; unsupported=1
    fi
    print_status
    ;;
  unlock)
    as_root "graphics clock reset (-rgc)" nvidia-smi -i "$gpu" -rgc && echo "gc_reset=1"
    as_root "memory clock reset (-rmc)" nvidia-smi -i "$gpu" -rmc && echo "mc_reset=1"
    print_status
    ;;
  status)
    print_status
    ;;
  *) usage ;;
esac

[[ $unsupported -eq 0 ]] || exit 3
exit 0
