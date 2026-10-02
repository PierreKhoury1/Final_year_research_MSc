#!/bin/bash
set -Eeuo pipefail
export W=/workspace/cuphy-lockstep
export S="$W/acar" LOGS="$W/logs-build-repair" OUT="$W/out-build-repair"
mkdir -p "$LOGS" "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$LOGS/build.rc"' EXIT
git -C "$W/sb" pull --ff-only
source "$W/sb/slotbench/cloud/debug_cuphy_lockstep.sh"
refresh_adapter > "$LOGS/adapter.log" 2>&1
cmake -S "$W/wrapper" -B "$W/build" -GNinja -DACAR_SRC="$S" \
  -DCMAKE_TOOLCHAIN_FILE="$S/cuPHY/cmake/toolchains/x86-64" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80-real -DBUILD_DOCS=OFF -DENABLE_TESTS=OFF \
  -DNVINFER:FILEPATH=/usr/lib/x86_64-linux-gnu/libnvinfer.so.10 > "$LOGS/configure.log" 2>&1
timeout --signal=TERM --kill-after=20 1200 cmake --build "$W/build" --target cuphy_ex_pusch_rx_multi_pipe -- -j16 > "$LOGS/build.log" 2>&1
make -C "$W/sb/slotbench" SM=80 CUDA_HOME=/usr/local/cuda bin/adversary > "$LOGS/adversary.log" 2>&1
