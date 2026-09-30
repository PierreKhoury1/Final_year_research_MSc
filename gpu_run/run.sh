#!/usr/bin/env bash
# Go/no-go run on a rented NVIDIA GPU. Everything it produces lands in one folder (out_<utc>/) and one
# tarball (results_<utc>.tgz) next to this script: run.log, gpu_info.txt, results.jsonl, raw/ start errors,
# pp_cuda_summary.json. Nothing here needs root, SSH or network beyond what the image already has.
#
# Usage:  bash run.sh                       (2000 targets per method, 2 ms apart, ~4 min in all)
#         TARGETS=5000 GAP_US=1000 bash run.sh
cd "$(dirname "$0")" || exit 1
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="out_$STAMP"; mkdir -p "$OUT/raw"
exec > >(tee "$OUT/run.log") 2>&1
TARGETS=${TARGETS:-2000}; GAP_US=${GAP_US:-2000}; HOST_CORE=${HOST_CORE:-1}; PP_SECS=${PP_SECS:-6}
echo "== lockstep go/no-go run $STAMP   targets=$TARGETS gap_us=$GAP_US host_core=$HOST_CORE"

echo "== machine"
{
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "hostname: $(hostname)"
  echo "kernel: $(uname -r)"
  echo "vast: CONTAINER_ID=${CONTAINER_ID:-?} VAST_CONTAINERLABEL=${VAST_CONTAINERLABEL:-?} PUBLIC_IPADDR=${PUBLIC_IPADDR:-?}"
  echo "cpus_visible: $(nproc)"
  grep -m1 'model name' /proc/cpuinfo
  echo "cpu_quota: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo unknown)"
  echo "rtprio_limit: $(ulimit -r)"
  nvidia-smi --query-gpu=name,driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,clocks.sm,pstate,compute_mode,persistence_mode --format=csv 2>&1
  nvcc --version 2>&1 | tail -1
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1
} | tee "$OUT/gpu_info.txt"

echo "== build"
nvcc -O2 -std=c++17 -o lockstep lockstep.cu -lpthread || { echo "BUILD FAILED"; exit 1; }
nvcc -O2 -o pingpong_cuda ../code/pingpong_cuda.cu -lpthread || echo "ping-pong build failed; continuing without it"

RES="$OUT/results.jsonl"; : > "$RES"
export LOCKSTEP_RAW="$PWD/$OUT/raw"
run_cond() {
  echo "== $1 (load=$2)"
  ./lockstep "$1" "$2" "$TARGETS" "$GAP_US" "$HOST_CORE" >> "$RES" || echo "lockstep exited with status $? in condition '$1'"
}
run_cond "idle" none
run_cond "AI job, same process" stream
echo "== starting the PyTorch load in a separate process"
python3 load_torch.py 900 & LOAD=$!
sleep 8
run_cond "AI job, other process" none
kill "$LOAD" 2>/dev/null; wait "$LOAD" 2>/dev/null

if [ -x ./pingpong_cuda ]; then
  echo "== clock ping-pong, $PP_SECS s per phase: does the GPU clock drift against the CPU's? (globaltimer counts ns, so 1000 MHz is nominal)"
  ./pingpong_cuda "$PP_SECS" "$HOST_CORE" 1 pp_cuda.bin && python3 - "$OUT" <<'PY'
import json, sys
sys.path.insert(0, "../code")
from ana2 import analyse
r = analyse("pp_cuda.bin")
json.dump(r, open(sys.argv[1] + "/pp_cuda_summary.json", "w"), indent=1)
print(f"GPU clock {r['gpu_mhz']:.6f} MHz vs TSC {r['tsc_mhz']:.3f} MHz; rate wander std {r['wander_ppm_std']:.4f} ppm, range {r['wander_ppm_range']}; quad {r['quad_ns_per_s2']:.4f} ns/s^2")
for p, s in r["phases"].items():
    print(f"  {p:9s} n={s['n']:8d} best {s['p1']/1e3:7.2f} us  median {s['p50']/1e3:8.2f} us  p99 {s['p99']/1e3:9.1f} us  max {s['max']/1e6:6.1f} ms")
PY
fi

echo "== summary"
python3 - "$RES" <<'PY'
import json, sys
def f(v, w=9, d=2):
    return f"{v:{w}.{d}f}" if isinstance(v, (int, float)) else f"{'n/a':>{w}}"
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    g = r["globaltimer_step_ns"]
    print(f"\n{r['condition']}  |  {r['gpu']} ({r['sm']} SMs, driver {r['driver']})  |  clock bound ±{f(r['clock_bound_us'],0,2)} us "
          f"(after run ±{f(r['clock_bound_after_us'],0,2)}, rate change {f(r['rate_change_ppm'],0,4)} ppm)  |  globaltimer step {g['median']} ns  |  pinned={r['pinned']} fifo={r['sched_fifo']}")
    print(f"  {'method':26s} {'median':>9s} {'p99':>9s} {'p99.9':>9s} {'worst':>9s} {'>100us':>7s}   (us late vs target)")
    for k, m in r["methods"].items():
        note = "  TIMEOUT" if m.get("timeout") else ""
        print(f"  {k:26s} {f(m['median_us'])} {f(m['p99_us'])} {f(m['p999_us'])} {f(m['max_us'])} {f(m['over_100us_pct'],6,1)}%   n={m['n']}{note}")
PY

tar czf "results_$STAMP.tgz" "$OUT"
echo "== done: $PWD/results_$STAMP.tgz  (raw ping-pong data stays in $PWD/pp_cuda.bin, not in the tarball)"
