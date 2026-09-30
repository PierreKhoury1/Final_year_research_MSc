#!/usr/bin/env bash
# check_rt.sh [--core 4] [--threshold-us 20] [--loops 100000] [--interval-us 500] [--prio 99] [--out FILE]
# Scheduling-latency check on the isolated core before touching the GPU:
#   cyclictest -t1 -p99 -i500 -l100000 -a4 -m -q
# Parses the max latency and prints PASS/FAIL against the threshold (plan: max < 20 us).
# Exit 0 PASS, 1 FAIL, 2 error (cyclictest missing, not permitted, unparseable output).
# 100000 loops at 500 us take 50 s. Run as root (SCHED_FIFO + mlockall).
set -uo pipefail

core=4; thr=20; loops=100000; interval=500; prio=99; out=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --core) core=${2:?}; shift 2 ;;
    --threshold-us) thr=${2:?}; shift 2 ;;
    --loops) loops=${2:?}; shift 2 ;;
    --interval-us) interval=${2:?}; shift 2 ;;
    --prio) prio=${2:?}; shift 2 ;;
    --out) out=${2:?}; shift 2 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! command -v cyclictest >/dev/null 2>&1; then
  echo "cyclictest not found: sudo apt install rt-tests" >&2
  exit 2
fi

pre=()
if [[ $(id -u) -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then pre=(sudo); else echo "warning: not root; cyclictest may fail" >&2; fi
fi

iso=$(cat /sys/devices/system/cpu/isolated 2>/dev/null || true)
echo "kernel: $(uname -r)   isolated cpus: ${iso:-none}   testing core $core"
echo "running: cyclictest -t1 -p$prio -i$interval -l$loops -a$core -m -q  (~$(( loops * interval / 1000000 )) s)"
res=$("${pre[@]}" cyclictest -t1 -p"$prio" -i"$interval" -l"$loops" -a"$core" -m -q 2>&1)
rc=$?
[[ -n $out ]] && printf '%s\n' "$res" > "$out"
printf '%s\n' "$res"
if [[ $rc -ne 0 ]]; then
  echo "ERROR: cyclictest exited $rc" >&2
  exit 2
fi

# Summary line: "T: 0 ( 1234) P:99 I:500 C: 100000 Min: 1 Act: 2 Avg: 2 Max: 11"
max=$(printf '%s\n' "$res" | sed -n 's/.*Max:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | sort -n | tail -n1)
if [[ -z $max ]]; then
  echo "ERROR: could not parse Max from cyclictest output" >&2
  exit 2
fi
if (( max < thr )); then
  echo "PASS: max latency ${max} us < ${thr} us on core $core"
  exit 0
fi
echo "FAIL: max latency ${max} us >= ${thr} us on core $core"
exit 1
