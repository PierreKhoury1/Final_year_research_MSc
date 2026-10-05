#!/bin/bash
# Rented-GPU run of tools/pcieclock: GPU %globaltimer <-> host clock over PCIe, classic vs tick-edge bounds.
# Usage: onstart_pcieclock.sh [CONFIG [BRANCH [REPO]]]   (CONFIG is ignored; vast.py passes it)
set -Eeuo pipefail
log() { echo "[slotbench $(date -u +%H:%M:%S)] $*"; }
W=/workspace/pcieclock; OUT=$W/out; mkdir -p "$OUT"; cd "$W"
BRANCH=${2:-${SB_BRANCH:-main}}; REPO=${3:-${SB_REPO:-https://github.com/PierreKhoury1/Final_year_research_MSc}}

emit() {   # archive OUT and print it as (multi-part) base64 blocks, then DONE; then stop the container
    local status=$1 archive=$W/results.tar.gz sha n i=0 part
    tar -czf "$archive" -C "$W" out
    sha=$(sha256sum "$archive" | cut -d' ' -f1)
    rm -rf "$W/parts"; mkdir -p "$W/parts"
    split -b 800k -d -a 2 --additional-suffix=.bin "$archive" "$W/parts/p"
    n=$(ls "$W/parts" | wc -l)
    echo "=====SLOTBENCH-PARTS pcieclock/results $n $sha====="
    for part in "$W"/parts/p*.bin; do
        i=$((i + 1))
        echo "=====SLOTBENCH-BEGIN pcieclock/results.part$(printf %02d "$i") $(sha256sum "$part" | cut -d' ' -f1)====="
        base64 -w 76 "$part"
        echo "=====SLOTBENCH-END pcieclock/results.part$(printf %02d "$i")====="
        if (( i < n )); then sleep "${SB_PART_GAP_S:-50}"; fi
    done
    echo "=====SLOTBENCH-PARTS pcieclock/results $n $sha====="
    echo "=====SLOTBENCH-DONE status=$status====="
    sleep "${SB_POST_DONE_GRACE_S:-900}"; echo "=====SLOTBENCH-SELF-STOP====="; kill -TERM 1; sleep 10; kill -KILL 1
}
trap 'echo "=====SLOTBENCH-ERROR $LINENO $BASH_COMMAND====="; emit error' ERR

log "apt"
apt-get -o Acquire::Retries=3 update -qq > "$OUT/apt.log" 2>&1
apt-get install -y -qq --no-install-recommends git python3 ca-certificates >> "$OUT/apt.log" 2>&1
log "clone $BRANCH"
git clone -q --depth 1 --branch "$BRANCH" "$REPO" "$W/sb"
git -C "$W/sb" rev-parse HEAD > "$OUT/commit.txt"
{ date -u; nvidia-smi; nvidia-smi -q | grep -iE "persistence|link width|link gen|clocks" ; nproc; lscpu | head -20; } > "$OUT/header.txt" 2>&1 || true
log "ptm check"
apt-get install -y -qq --no-install-recommends pciutils >> "$OUT/apt.log" 2>&1 || true
python3 "$W/sb/slotbench/tools/ptm_check.py" > "$OUT/ptm.json" 2> "$OUT/ptm.err" || true
lspci -vvv -d 10de: > "$OUT/lspci_nvidia.txt" 2>&1 || true
lspci -tv > "$OUT/lspci_tree.txt" 2>&1 || true
python3 - "$OUT/ptm.json" <<'PY' || true
import json, sys
r = json.load(open(sys.argv[1]))
for g in r["gpus"]:
    print("[slotbench] PTM", g["gpu"], [(d["bdf"], d.get("ext_config_readable"), d.get("ptm_present"), d.get("ptm")) for d in g["path"]])
PY
grep -iE "PTM|Precision Time" "$OUT/lspci_nvidia.txt" | head -5 | sed 's/^/[slotbench] lspci: /' || true
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
log "build sm_$cc"
nvcc -O2 -std=c++17 -arch=sm_$cc -I"$W/sb/slotbench/common" -o "$W/pcieclock" "$W/sb/slotbench/tools/pcieclock.cu" -lpthread > "$OUT/build.log" 2>&1
NP=$(nproc); CORE=$(( NP > 4 ? 2 : 0 )); CCORE=$(( NP > 4 ? 3 : 1 ))
for run in 1 2 3; do
    log "run $run (cores $CORE/$CCORE)"
    "$W/pcieclock" --rounds "${SB_PC_ROUNDS:-10}" --per-phase "${SB_PC_PER_PHASE:-2000}" --core "$CORE" --clock-core "$CCORE" \
        --out "$OUT/run$run" > "$OUT/run$run.log" 2>&1
    python3 "$W/sb/slotbench/analysis/pcieclock.py" "$OUT/run$run" --json "$OUT/run$run.analysis.json" > /dev/null 2>&1 \
        || log "analysis run $run failed"
    python3 - "$OUT/run$run.analysis.json" <<'PY' || true
import json, sys
r = json.load(open(sys.argv[1]))
c, e, t = r["classic"], r["edge"], r["edge_trim2"]
print(f"[slotbench] classic eps {c['eps_ns']:.0f} ns (+half tick {c['eps_with_tick_ns']:.0f}) | edge strict {e['bound_ns']:.0f} ns "
      f"feasible={e['feasible']} | edge trim2 {t['bound_ns']:.0f} ns | rate {e['rate_ppm']:.2f} ppm | tick {r['tick_ns']} ns")
PY
    sleep 20
done
emit ok
