#!/usr/bin/env bash
# Runs the lockstep test in three conditions and writes results.jsonl.
set -euo pipefail
cd "$(dirname "$0")"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv | tee gpu_info.txt
nvcc -O2 -std=c++17 -o lockstep lockstep.cu -lpthread
: > results.jsonl

echo "== idle"
./lockstep idle none >> results.jsonl

echo "== AI load, same process (low-priority stream)"
./lockstep "AI job, same process" stream >> results.jsonl

echo "== AI load, separate process (PyTorch matmuls, GPU time-slicing)"
python3 load_torch.py 40 & LOAD=$!
sleep 5
./lockstep "AI job, other process" none >> results.jsonl
kill $LOAD 2>/dev/null || true; wait $LOAD 2>/dev/null || true

python3 - <<'EOF'
import json
for line in open("results.jsonl"):
    r = json.loads(line)
    print(f"\n{r['condition']}  |  {r['gpu']}  |  clock bound ±{r['clock_bound_us']:.2f} us  |  globaltimer step {r['globaltimer_step_ns']['median']} ns")
    print(f"  {'method':26s} {'median':>9s} {'p99':>9s} {'worst':>9s}   (us late vs target)")
    for k, m in r["methods"].items():
        print(f"  {k:26s} {m['median_us']:9.2f} {m['p99_us']:9.2f} {m['max_us']:9.2f}   n={m['n']}")
EOF
