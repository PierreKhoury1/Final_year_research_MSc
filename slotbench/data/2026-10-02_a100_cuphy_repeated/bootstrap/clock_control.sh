#!/bin/bash
set -u
out=/workspace/cuphy-lockstep/host-controls
mkdir -p "$out"
date -u +%FT%TZ > "$out/started_utc.txt"
nvidia-smi -q -x > "$out/gpu_before.xml"
nvidia-smi -q -d SUPPORTED_CLOCKS,CLOCK,POWER > "$out/supported_clocks.txt"
nvidia-smi -lgc 1110,1110 > "$out/sm_lock.log" 2>&1
sm_rc=$?
nvidia-smi -lmc 1215,1215 > "$out/memory_lock.log" 2>&1
mem_rc=$?
printf '{"requested_sm_mhz":1110,"requested_memory_mhz":1215,"sm_command_rc":%d,"memory_command_rc":%d,"power_limit_changed":false}\n' "$sm_rc" "$mem_rc" > "$out/clock_control.json"
nvidia-smi -q -x > "$out/gpu_after.xml"
cat "$out/clock_control.json" "$out/sm_lock.log" "$out/memory_lock.log"
