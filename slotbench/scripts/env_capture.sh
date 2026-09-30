#!/usr/bin/env bash
# env_capture.sh OUT.txt
# Snapshot of the machine a run happened on: GPU (nvidia-smi -q), driver, kernel, cmdline, CPU,
# isolated cpus, governor, RT kernel check, NUMA, container detection and selected environment
# variables (vast.ai CONTAINER_ID etc.). Variables whose name contains KEY/TOKEN/SECRET/PASS/AUTH/
# CRED are never written. Missing tools are noted, never fatal.
set -uo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 OUT.txt" >&2; exit 2; }
out=$1

section() { printf '\n===== %s =====\n' "$1"; }
run() {  # run CMD... : print command and its output, or a note if the tool is missing
  if command -v "$1" >/dev/null 2>&1; then
    echo "\$ $*"
    "$@" 2>&1
  else
    echo "($1 not available)"
  fi
}

container_kind() {
  local k=""
  [[ -f /.dockerenv ]] && k+="docker(/.dockerenv) "
  [[ -f /run/.containerenv ]] && k+="podman(/run/.containerenv) "
  grep -qaE 'docker|kubepods|containerd|lxc' /proc/1/cgroup 2>/dev/null && k+="cgroup "
  if command -v systemd-detect-virt >/dev/null 2>&1; then
    local v; v=$(systemd-detect-virt -c 2>/dev/null) && [[ $v != none ]] && k+="systemd-detect-virt:$v "
  fi
  [[ -n ${CONTAINER_ID:-} || -n ${VAST_CONTAINERLABEL:-} ]] && k+="vast.ai "
  echo "${k:-none}"
}

{
  section "capture"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname 2>/dev/null)"
  echo "user=$(id -un 2>/dev/null) uid=$(id -u)"

  section "kernel"
  echo "uname=$(uname -a)"
  echo "cmdline=$(cat /proc/cmdline 2>/dev/null)"
  rt=no
  if uname -v | grep -q PREEMPT_RT || [[ $(cat /sys/kernel/realtime 2>/dev/null) == 1 ]]; then rt=yes; fi
  echo "preempt_rt=$rt"
  echo "isolated_cpus=$(cat /sys/devices/system/cpu/isolated 2>/dev/null)"
  echo "nohz_full=$(cat /sys/devices/system/cpu/nohz_full 2>/dev/null)"
  if [[ -f /etc/os-release ]]; then grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release; fi

  section "cpu frequency governors"
  for g in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
    [[ -r $g ]] && echo "$(basename "$(dirname "$(dirname "$g")")")=$(cat "$g") driver=$(cat "$(dirname "$g")/scaling_driver" 2>/dev/null)"
  done
  echo "rt_runtime_us=$(cat /proc/sys/kernel/sched_rt_runtime_us 2>/dev/null)"

  section "container"
  echo "container=$(container_kind)"

  section "environment (filtered)"
  env | grep -E '^(CONTAINER_ID|VAST_|CUDA_|NVIDIA_|MIG_|GPU|PUBLIC_IPADDR|SB_|OMP_|LD_LIBRARY_PATH|PATH)[A-Za-z0-9_]*=' \
      | grep -viE '^[^=]*(KEY|TOKEN|SECRET|PASS|AUTH|CRED)[^=]*=' | sort

  section "lscpu"
  run lscpu

  section "numactl -H"
  run numactl -H

  section "nvidia driver"
  cat /proc/driver/nvidia/version 2>/dev/null || echo "(/proc/driver/nvidia/version not available)"
  run nvidia-smi --query-gpu=index,name,uuid,driver_version,pci.bus_id,persistence_mode,compute_mode,mig.mode.current --format=csv
  run nvidia-smi compute-policy -l
  run nvcc --version

  section "nvidia-smi -q"
  run nvidia-smi -q
} > "$out" 2>&1

echo "environment written to $out"
