#!/usr/bin/env bash
# 9 Oct 2026, 1x H100 box. Runs inside the rented container; results in /root/res, /root/res.tgz + /root/DONE.
#  1. REORDER with CUDA 12.6 (characterize --profile reorder: build, SASS gate, six experiments)
#  2. REORDER with CUDA 12.2 (E7: same source, another ptxas)
#  3. ktrace profile with 12.6 (H100 %globaltimer step census on a second H100 host)
# Each step continues on failure. No secret is read or printed.
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
T126=/usr/local/cuda-12.6; [ -x $T126/bin/nvcc ] || T126=/usr/local/cuda
export PATH=$T126/bin:$PATH CUDA_HOME=$T126
R=/root/res; mkdir -p $R
exec > >(tee -a $R/onbox.log) 2>&1
step() { echo; echo "===== $(date -u +%H:%M:%S) $*"; }

step "machine"
nvidia-smi; nvidia-smi -q > $R/nvidia-smi-q.txt 2>&1
nvidia-smi --query-gpu=index,pci.bus_id,name,driver_version,clocks.sm,clocks.max.sm,persistence_mode,uuid --format=csv | tee $R/gpus.csv
which nvcc; nvcc --version | tail -2; nproc; grep -m1 "model name" /proc/cpuinfo; uname -a

step "python + tickbound"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3 python3-venv python3-pip >/dev/null; python3 --version
python3 -m venv /root/venv && . /root/venv/bin/activate
mkdir -p /root/tb && tar -xzf /root/tickbound.tgz -C /root/tb
pip install -q /root/tb && tickbound --version
tickbound doctor | tee $R/doctor.txt

( apt-get install -y -qq cuda-nvcc-12-2 cuda-cudart-dev-12-2 cuda-cuobjdump-12-2 cuda-nvdisasm-12-2 cuda-crt-12-2 >/root/apt122.log 2>&1 \
  || apt-get install -y -qq cuda-nvcc-12-2 cuda-cudart-dev-12-2 cuda-cuobjdump-12-2 cuda-nvdisasm-12-2 >>/root/apt122.log 2>&1; \
  echo "apt 12.2 exit $?" >> /root/apt122.log ) &
APT=$!

probe_path() {
    python - <<'PY'
import os
from tickbound import cli
g = cli.gpus()
print(os.path.join(cli.cache_dir(), "bin", "tickbound_probe_" + cli.probe_tag(g[0]["sm"])) if g else "")
PY
}
reorder_profile() {   # $1 = output directory; uses the nvcc on PATH
    local out=$1 bin
    mkdir -p "$out"
    which nvcc cuobjdump; nvcc --version | tail -1
    timeout 2400 tickbound characterize --profile reorder --out "$out"; echo "characterize exit $?"
    sed -n 1,24p "$out/report.md" 2>/dev/null | cut -c1-400
    bin=$(probe_path); echo "binary: $bin"
    if [ -n "$bin" ] && [ -f "$bin" ]; then
        cuobjdump -sass "$bin" 2>/dev/null | gzip > "$out/probe.sass.gz"
        tickbound sass "$bin" --reorder --json "$out/sass_reorder.json" > "$out/sass_reorder.txt" 2>&1
        echo "sass --reorder exit $? (0 = PASS, 1 = a FAIL)"; grep -E "^ +FAIL|EXCLUDED|reorder SASS gate" "$out/sass_reorder.txt" | head -40
    else
        echo "no probe binary found for this toolkit"
    fi
}

step "1. REORDER, CUDA 12.6"
reorder_profile $R/reorder126

step "2. REORDER, CUDA 12.2 (E7)"
wait $APT 2>/dev/null; tail -2 /root/apt122.log
if [ -x /usr/local/cuda-12.2/bin/nvcc ]; then
    ( export PATH=/usr/local/cuda-12.2/bin:$PATH CUDA_HOME=/usr/local/cuda-12.2; reorder_profile $R/reorder122 )
else
    echo "E7 NOT MEASURED: no /usr/local/cuda-12.2/bin/nvcc (apt failed)"
fi

step "3. ktrace profile, CUDA 12.6 (timer step census on this H100 host)"
timeout 1500 tickbound characterize --profile ktrace --out $R/ktrace126; echo "ktrace exit $?"

step "pack"
cd /root && tar -czf /root/res.tgz res && ls -la /root/res.tgz && touch /root/DONE
echo "=====ONBOX-H100-DONE====="
