#!/bin/bash
set -Eeuo pipefail
W=/workspace/cuphy-lockstep
source "$W/host-controls/probe/recommended_cpus.env"
export SB_CUPHY_CLOCK_EVIDENCE="$W/host-controls/clock_control.json"
export SB_CUPHY_REPEAT_TIMEOUT_S=1800
export SB_CUPHY_REPEATS=6 SB_CUPHY_REPEAT_SEED=20261002
export SB_CUPHY_LOCKSTEP_SLOTS=5000 SB_CUPHY_LOCKSTEP_WARMUP=1000
date -u +%FT%TZ
git -C "$W/sb" rev-parse HEAD
sha256sum "$W/tv/GPU_test_input/TVnr_7304_PUSCH_gNB_CUPHY_s0p0.h5"
printf 'PHY CPU=%s adversary CPU=%s telemetry CPU=%s background=%s\n' "$SB_CUPHY_CPU" "$SB_ADVERSARY_CPU" "$SB_TELEMETRY_CPU" "$SB_BACKGROUND_CPUS"
exec timeout --signal=TERM --kill-after=30 1900 bash "$W/sb/slotbench/cloud/repeat_cuphy_lockstep.sh"
