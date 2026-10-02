#!/bin/bash
set -Eeuo pipefail
W=/workspace/cuphy-lockstep
S="$W/acar"
source "$W/host-controls/probe/recommended_cpus.env"
export SB_CUPHY_CLOCK_EVIDENCE="$W/host-controls/clock_control.json"
export SB_CUPHY_REPEAT_TIMEOUT_S=1200 SB_CUPHY_REPEAT_SEED=20261003
export SB_CUPHY_LOCKSTEP_SLOTS=5000 SB_CUPHY_LOCKSTEP_WARMUP=1000
date -u +%FT%TZ
PUSCH="$W/build/cuPHY/examples/pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe"
[[ ! -e $W/primary-pusch ]]
sha256sum "$PUSCH"
cp -p "$PUSCH" "$W/primary-pusch"
git -C "$W/sb" pull --ff-only
git -C "$W/sb" rev-parse HEAD
cp "$W/sb/slotbench/cuphy/cuphy_lockstep.cu" "$S/cuPHY/examples/pusch_rx_multi_pipe/cuphy_lockstep.cu"
timeout --signal=TERM --kill-after=20 300 cmake --build "$W/build" --target cuphy_ex_pusch_rx_multi_pipe -- -j8 > "$W/activity_build.log" 2>&1
sha256sum "$PUSCH"
exec timeout --signal=TERM --kill-after=30 1300 bash "$W/sb/slotbench/cloud/control_cuphy_activity.sh"
