#!/usr/bin/env bash
# mps.sh start [--gpu N] [--dir DIR] | stop [--dir DIR] [--all] | status [--dir DIR]
# CUDA MPS control daemon for mechanism M2. DIR (default /tmp/slotbench-mps) holds the pipe
# directory (DIR/pipe) and log directory (DIR/log, or --log-dir). Clients must see the same
# CUDA_MPS_PIPE_DIRECTORY; "start" prints the export lines to eval:
#   eval "$(scripts/mps.sh start --gpu 0)"
# stop sends "quit" to the daemon on DIR (and, with --all, to the NVIDIA default /tmp/nvidia-mps)
# and waits for the control/server processes to go away. Absence of MPS is tolerated:
# start exits 3 (NOT SUPPORTED), stop/status exit 0.
# Keep DIR short: the pipe is a Unix socket and the path is limited to about 100 characters.
set -uo pipefail

usage() { echo "usage: $0 start [--gpu N] [--dir DIR] [--log-dir DIR] | stop [--dir DIR] [--all] | status [--dir DIR]" >&2; exit 2; }

[[ $# -ge 1 ]] || usage
cmd=$1; shift
gpu=0; dir=/tmp/slotbench-mps; logdir=""; all=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) gpu=${2:?}; shift 2 ;;
    --dir) dir=${2:?}; shift 2 ;;
    --log-dir) logdir=${2:?}; shift 2 ;;
    --all) all=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
pipe=$dir/pipe
[[ -n $logdir ]] || logdir=$dir/log

have_mps() { command -v nvidia-cuda-mps-control >/dev/null 2>&1; }

# PIDs of MPS control daemons/servers. Match the full command line: process names (comm) are cut
# to 15 characters ("nvidia-cuda-mps"), so pgrep -x on the full name would never match.
mps_procs() { pgrep -f '(^|/)nvidia-cuda-mps-(control|server)( |$)' || true; }

quit_daemon() {  # quit_daemon PIPE_DIR
  local p=$1
  [[ -S $p/control || -e $p/control ]] || return 0
  echo "stopping MPS daemon (pipe $p)" >&2
  echo quit | CUDA_MPS_PIPE_DIRECTORY=$p nvidia-cuda-mps-control >/dev/null 2>&1 || true
}

case $cmd in
  start)
    if ! have_mps; then
      echo "NOT SUPPORTED: nvidia-cuda-mps-control not found" >&2
      exit 3
    fi
    mkdir -p "$pipe" "$logdir"
    if [[ -e $pipe/control ]] && echo get_server_list | CUDA_MPS_PIPE_DIRECTORY=$pipe \
         timeout 5 nvidia-cuda-mps-control >/dev/null 2>&1; then
      echo "MPS daemon already running on $pipe" >&2
    else
      # stdio to a file: a daemon holding our stdout would block callers that capture it.
      if ! CUDA_VISIBLE_DEVICES=$gpu CUDA_MPS_PIPE_DIRECTORY=$pipe CUDA_MPS_LOG_DIRECTORY=$logdir \
           nvidia-cuda-mps-control -d </dev/null >>"$logdir/start.log" 2>&1; then
        echo "NOT SUPPORTED: nvidia-cuda-mps-control -d failed (see $logdir/control.log)" >&2
        exit 3
      fi
      for _ in $(seq 50); do [[ -e $pipe/control ]] && break; sleep 0.1; done
      if [[ ! -e $pipe/control ]]; then
        echo "NOT SUPPORTED: MPS control pipe did not appear in $pipe" >&2
        exit 3
      fi
    fi
    echo "export CUDA_MPS_PIPE_DIRECTORY=$pipe"
    echo "export CUDA_MPS_LOG_DIRECTORY=$logdir"
    ;;
  stop)
    if ! have_mps; then
      echo "MPS not installed; nothing to stop" >&2
      exit 0
    fi
    quit_daemon "$pipe"
    [[ $all -eq 1 ]] && quit_daemon /tmp/nvidia-mps
    for _ in $(seq 100); do
      [[ -z $(mps_procs) ]] && break
      sleep 0.1
    done
    if [[ -n $(mps_procs) ]]; then
      echo "warning: MPS processes still running: $(mps_procs | tr '\n' ' ')" >&2
      exit 1
    fi
    ;;
  status)
    if ! have_mps; then
      echo "mps_installed=0"
      exit 0
    fi
    echo "mps_installed=1"
    echo "pipe_dir=$pipe"
    echo "processes=$(mps_procs | tr '\n' ' ')"
    if [[ -e $pipe/control ]]; then
      echo "server_list=$(echo get_server_list | CUDA_MPS_PIPE_DIRECTORY=$pipe timeout 5 nvidia-cuda-mps-control 2>/dev/null | tr '\n' ' ')"
    else
      echo "server_list="
    fi
    ;;
  *) usage ;;
esac
exit 0
