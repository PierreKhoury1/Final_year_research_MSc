#!/usr/bin/env bash
# 9 Oct 2026, 8x A100 box. Runs inside the rented container; results in /root/res, /root/res.tgz + /root/DONE at the end.
#  1. build the CUDA 12.6 probe (one quick ktrace run on GPU 0) and check the ktrace phases in its real SASS
#  2. MULTI: the same ktrace kernel on every GPU at once, one process per GPU, each synced to the host clock
#     (CLOCK_MONOTONIC_RAW) by its own clock sync before and after; all processes start together (TB_START_AT_NS)
#     and launch on one shared 2 ms grid of that clock (TB_ALIGN_US), so the kernels of all GPUs run at the same time
#  3. REORDER (12.6): the six experiments of the reorder profile, one per GPU, in parallel; Tier B edits on GPU 6
#  4. REORDER (12.2): the same six with the CUDA 12.2 compiler (E7: same source, another ptxas)
# Each step continues on failure. No secret is read or printed.
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
T126=/usr/local/cuda-12.6; [ -x $T126/bin/nvcc ] || T126=/usr/local/cuda
export PATH=$T126/bin:$PATH CUDA_HOME=$T126
R=/root/res; mkdir -p $R
exec > >(tee -a $R/onbox.log) 2>&1
step() { echo; echo "===== $(date -u +%H:%M:%S) $*"; }

step "machine"
nvidia-smi; nvidia-smi topo -m | tee $R/topo.txt; nvidia-smi -q > $R/nvidia-smi-q.txt 2>&1
nvidia-smi --query-gpu=index,pci.bus_id,name,driver_version,clocks.sm,clocks.max.sm,persistence_mode,uuid --format=csv | tee $R/gpus.csv
nvidia-smi nvlink -s > $R/nvlink.txt 2>&1; head -20 $R/nvlink.txt
which nvcc; nvcc --version | tail -2; nproc; grep -m1 "model name" /proc/cpuinfo; uname -a
python3 -c "import os; print('affinity', len(os.sched_getaffinity(0)), 'cpus')" 2>/dev/null

step "python + tickbound"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3 python3-venv python3-pip >/dev/null; python3 --version
python3 -m venv /root/venv && . /root/venv/bin/activate
mkdir -p /root/tb && tar -xzf /root/tickbound.tgz -C /root/tb
pip install -q /root/tb && tickbound --version
tickbound doctor | tee $R/doctor.txt

# CUDA 12.2 compiler for step 4, installed in the background (explicit paths everywhere: installing it may repoint
# /usr/local/cuda, so nothing below uses that link)
( apt-get install -y -qq cuda-nvcc-12-2 cuda-cudart-dev-12-2 cuda-cuobjdump-12-2 cuda-nvdisasm-12-2 cuda-crt-12-2 >/root/apt122.log 2>&1 \
  || apt-get install -y -qq cuda-nvcc-12-2 cuda-cudart-dev-12-2 cuda-cuobjdump-12-2 cuda-nvdisasm-12-2 >>/root/apt122.log 2>&1; \
  echo "apt 12.2 exit $?" >> /root/apt122.log ) &
APT=$!

# two NUMA-local cores per GPU (sysfs local_cpulist of the GPU's PCI device, inside this container's CPU set);
# GPUs on one node take consecutive pairs after the node's first two cores
python3 - > $R/cores.txt <<'PY'
import os, subprocess
rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"], text=True).splitlines()
avail = sorted(os.sched_getaffinity(0)); taken = set(); out = []
for row in [r for r in rows if r.strip()]:
    g, bus = [x.strip() for x in row.split(",")]
    dom, rest = bus.lower().split(":", 1)
    local = []
    try:
        for part in open("/sys/bus/pci/devices/%04x:%s/local_cpulist" % (int(dom, 16), rest)).read().strip().split(","):
            a, _, b = part.partition("-"); local += range(int(a), int(b or a) + 1)
    except Exception:
        pass
    local = [c for c in local if c in avail]
    # skip the first two cores of the node (and of the container), never reuse a core
    pool = [c for c in local[2:] if c not in taken] + [c for c in avail[2:] if c not in taken and c not in local]
    if len(pool) < 2:
        pool = [c for c in avail if c not in taken] or avail
    pick = pool[:2] if len(pool) >= 2 else [pool[0], pool[0]]
    taken.update(pick)
    out.append("%s %d %d %s" % (g, pick[0], pick[1], "local" if pick[0] in local else "not-local"))
print(chr(10).join(out))
PY
cat $R/cores.txt

step "1. build the 12.6 probe + ktrace SASS check (GPU 0)"
read g0 c0 k0 _ < $R/cores.txt
timeout 1200 tickbound run --strategy ktrace --gpu 0 --ffma 256 --blocks 1 --reps 3 --core $c0 --clock-core $k0 --out $R/build126/k; echo "warm-up run exit $?"
BIN126=$(ls -t ~/.cache/tickbound/bin/tickbound_probe_* 2>/dev/null | head -1); echo "binary 12.6: $BIN126"
cuobjdump -sass "$BIN126" > $R/probe126.sass 2>/dev/null
tickbound sass $R/probe126.sass --phases --json $R/ktrace_phases126.json | tee $R/ktrace_phases126.txt | tail -16

step "2. MULTI: ktrace on every GPU at once (start together, shared 2 ms launch grid)"
mkdir -p $R/multi
START=$(python3 -c "import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) + 20 * 10**9)")
echo "TB_START_AT_NS=$START (CLOCK_MONOTONIC_RAW)" | tee $R/multi/start.txt
while read g c k _; do
  TB_START_AT_NS=$START TB_ALIGN_US=2000 timeout 600 tickbound run --strategy ktrace --gpu $g --ffma 256 --blocks 1,sm \
      --reps 40 --idle-us 200 --core $c --clock-core $k --out $R/multi/gpu$g > $R/multi/gpu$g.log 2>&1 &
done < $R/cores.txt
wait $(jobs -p | grep -v "^$APT\$") 2>/dev/null
for f in $R/multi/gpu*.log; do echo "--- $f"; tail -3 $f; done
for g in $(cut -d' ' -f1 $R/cores.txt); do
  timeout 300 tickbound analyze $R/multi/gpu$g --json $R/multi/gpu$g.analysis.json | cut -c1-300
done

step "3. REORDER 12.6: six experiments in parallel (GPUs 0-5) + Tier B edits (GPU 6)"
mkdir -p $R/reorder126
EXPS=(load_use:64 war:64 bar:32 fence:64 pipe:16 stall:64)
i=0
while read g c k _; do
  [ $i -lt 6 ] || break
  e=${EXPS[$i]%%:*}; n=${EXPS[$i]##*:}
  timeout 1500 tickbound run --strategy reorder --exp $e --reps $n --gpu $g --core $c --clock-core $k --out $R/reorder126/reorder_$e > $R/reorder126/$e.log 2>&1 &
  i=$((i + 1))
done < $R/cores.txt
( CUDA_VISIBLE_DEVICES=6 NVCC=$T126/bin/nvcc timeout 1500 bash /root/tb/devtools/cuasm/run_edits.sh /root/tb/cuasm_edits/sm_80 $R/cuasm_sm80 200 > $R/cuasm_sm80.log 2>&1; echo "run_edits exit $?" >> $R/cuasm_sm80.log ) &
wait $(jobs -p | grep -v "^$APT\$") 2>/dev/null
for f in $R/reorder126/*.log $R/cuasm_sm80.log; do echo "--- $f"; tail -4 $f; done
tickbound sass "$BIN126" --reorder --json $R/reorder126/sass_reorder.json > $R/reorder126/sass_reorder.txt 2>&1; echo "gate 12.6 exit $?"
grep -E "^ +FAIL|EXCLUDED|reorder SASS gate" $R/reorder126/sass_reorder.txt | head -20
for e in load_use war bar fence pipe stall; do
  timeout 300 tickbound analyze $R/reorder126/reorder_$e --json $R/reorder126/reorder_$e.analysis.json | cut -c1-400
done
S=$R/cuasm_sm80/summary.json
[ -f "$S" ] && python3 - "$S" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
for e in s["edits"]:
    print(f"  {e['id']:26s} {e['group']:9s} legal={e['legal_per_checker']!s:5s} match={e.get('match')!s:5s} "
          f"mismatching={e.get('mismatching_launches')}/{e.get('launches_compared')} p50={e.get('cycles_p50')} "
          f"err={(e.get('load_error') or e.get('launch_error') or '')[:80]}")
PY

step "4. REORDER 12.2 (E7): wait for the compiler, build once, six experiments in parallel"
wait $APT 2>/dev/null; tail -2 /root/apt122.log
T122=/usr/local/cuda-12.2
if [ -x $T122/bin/nvcc ]; then
  (
    export PATH=$T122/bin:$PATH CUDA_HOME=$T122
    nvcc --version | tail -1
    mkdir -p $R/reorder122
    read g7 c7 k7 _ < <(sed -n 8p $R/cores.txt)
    timeout 1200 tickbound run --strategy reorder --exp pipe --reps 1 --gpu ${g7:-7} --core ${c7:-30} --clock-core ${k7:-31} --out $R/reorder122/warm > $R/reorder122/warm.log 2>&1; echo "12.2 build/warm exit $?"
    BIN122=$(ls -t ~/.cache/tickbound/bin/tickbound_probe_* 2>/dev/null | head -1); echo "binary 12.2: $BIN122"
    i=0
    while read g c k _; do
      [ $i -lt 6 ] || break
      e=${EXPS[$i]%%:*}; n=${EXPS[$i]##*:}
      timeout 1500 tickbound run --strategy reorder --exp $e --reps $n --gpu $g --core $c --clock-core $k --out $R/reorder122/reorder_$e > $R/reorder122/$e.log 2>&1 &
      i=$((i + 1))
    done < $R/cores.txt
    wait
    for f in $R/reorder122/*.log; do echo "--- $f"; tail -3 $f; done
    tickbound sass "$BIN122" --reorder --json $R/reorder122/sass_reorder.json > $R/reorder122/sass_reorder.txt 2>&1; echo "gate 12.2 exit $?"
    grep -E "^ +FAIL|EXCLUDED|reorder SASS gate" $R/reorder122/sass_reorder.txt | head -20
    for e in load_use war bar fence pipe stall; do
      timeout 300 tickbound analyze $R/reorder122/reorder_$e --json $R/reorder122/reorder_$e.analysis.json | cut -c1-400
    done
  )
else
  echo "E7 NOT MEASURED: no $T122/bin/nvcc (apt failed)"
fi

step "pack"
cd /root && tar -czf /root/res.tgz res && ls -la /root/res.tgz && touch /root/DONE
echo "=====ONBOX-A100x8-DONE====="
