#!/bin/bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export W=/workspace/cuphy-lockstep
export S="$W/acar"
source "$W/sb/slotbench/cloud/onstart_cuphy_lockstep.sh"
export ACAR_COMMIT
export -f tv
cd "$W"
rc=0
timeout --signal=TERM --kill-after=20 2400 bash -e -o pipefail -c tv > "$W/logs/tv-parallel.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$W/logs/tv-parallel.rc"
exit "$rc"
