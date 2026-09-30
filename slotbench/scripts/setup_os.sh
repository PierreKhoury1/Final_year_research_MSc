#!/usr/bin/env bash
# setup_os.sh [--apply] [--cores 4,5] [--iface IFACE] [--ptp-units] [--rt] [--grub-file FILE]
# One-time home-lab OS preparation for Ubuntu 24.04. DRY-RUN BY DEFAULT: prints every change it
# would make. --apply performs them (root only, refused in containers/vast.ai and on non-Ubuntu):
#  - GRUB_CMDLINE_LINUX += isolcpus=C nohz_full=C rcu_nocbs=C intel_pstate=disable
#    processor.max_cstate=1 idle=poll (idempotent: existing copies of these keys are replaced),
#    after backing up /etc/default/grub with a timestamp; then update-grub. Never reboots.
#  - apt install rt-tests numactl (cyclictest for scripts/check_rt.sh)
#  - --rt: install a PREEMPT_RT kernel (linux-image-realtime if apt has it, else Ubuntu Pro
#    "pro enable realtime-kernel", which needs an attached Pro subscription - free for personal use)
#  - --iface: install linuxptp + ethtool, check hardware timestamping, print ptp4l/phc2sys commands;
#    --ptp-units also writes (does not enable) systemd units for them.
# Also prints the BIOS checklist and the GPU steps (scripts/gpu_lock.sh).
# --grub-file FILE (testing): with --apply, edits only FILE (backup next to it); every other step,
# including update-grub, apt and systemd, stays dry-run.
set -euo pipefail

apply=0; cores="4,5"; iface=""; ptp_units=0; rt=0; grub=/etc/default/grub; grub_custom=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --apply) apply=1; shift ;;
    --cores) cores=${2:?}; shift 2 ;;
    --iface) iface=${2:?}; shift 2 ;;
    --ptp-units) ptp_units=1; shift ;;
    --rt) rt=1; shift ;;
    --grub-file) grub=${2:?}; grub_custom=1; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ $cores =~ ^[0-9]+([,-][0-9]+)*$ ]] || { echo "bad --cores '$cores' (e.g. 4,5 or 4-5)" >&2; exit 2; }

# sys_apply: whether system-wide commands really run (never with --grub-file).
sys_apply=$apply
[[ $grub_custom -eq 1 ]] && sys_apply=0

say() { printf '%s\n' "$*"; }
# do_cmd CMD...: print in dry-run, print and execute when applying system changes.
do_cmd() {
  if [[ $sys_apply -eq 1 ]]; then say "+ $*"; "$@"; else say "[dry-run] would run: $*"; fi
}

in_container() {
  [[ -f /.dockerenv || -f /run/.containerenv ]] && return 0
  [[ -n ${CONTAINER_ID:-} || -n ${VAST_CONTAINERLABEL:-} ]] && return 0
  grep -qaE 'docker|kubepods|containerd|lxc' /proc/1/cgroup 2>/dev/null && return 0
  if command -v systemd-detect-virt >/dev/null 2>&1; then
    local v; v=$(systemd-detect-virt -c 2>/dev/null || true)
    [[ -n $v && $v != none ]] && return 0
  fi
  return 1
}

# shellcheck disable=SC1091
os_id=$( (. /etc/os-release 2>/dev/null && echo "${ID:-unknown} ${VERSION_ID:-}") || echo unknown)
mode=DRY-RUN
[[ $sys_apply -eq 1 ]] && mode=APPLY
[[ $apply -eq 1 && $grub_custom -eq 1 ]] && mode="APPLY to $grub only"
say "== slotbench OS setup ($mode) on: $os_id, kernel $(uname -r)"

if [[ $sys_apply -eq 1 ]]; then
  if in_container; then
    say "REFUSED: this looks like a container (docker/vast.ai). Host kernel settings cannot be changed here."
    exit 1
  fi
  if [[ ${os_id%% *} != ubuntu ]]; then
    say "REFUSED: not Ubuntu ($os_id). Apply the steps below by hand."
    exit 1
  fi
  if [[ $(id -u) -ne 0 ]]; then
    say "REFUSED: --apply needs root (sudo $0 --apply ...)"
    exit 1
  fi
fi

# ---------------------------------------------------------------- kernel command line
params="isolcpus=$cores nohz_full=$cores rcu_nocbs=$cores intel_pstate=disable processor.max_cstate=1 idle=poll"
say ""
say "== 1. kernel command line ($grub)"
if [[ ! -f $grub ]]; then
  say "warning: $grub not found; add to the kernel command line by hand: $params"
else
  # New GRUB_CMDLINE_LINUX line: drop existing copies of our keys, append ours (last assignment wins).
  newline=$(python3 - "$grub" "$params" <<'PY'
import re, shlex, sys
path, params = sys.argv[1], sys.argv[2].split()
keys = {p.split("=", 1)[0] for p in params}
text = open(path).read()
found = re.findall(r'^GRUB_CMDLINE_LINUX=(.*)$', text, re.M)
cur = ""
if found:
    try:
        cur = " ".join(shlex.split(found[-1]))
    except ValueError:
        cur = found[-1].strip().strip('"')
kept = [t for t in cur.split() if t.split("=", 1)[0] not in keys]
print('GRUB_CMDLINE_LINUX="' + " ".join(kept + params) + '"')
PY
)
  oldline=$(grep -E '^GRUB_CMDLINE_LINUX=' "$grub" | tail -n1 || true)
  say "current: ${oldline:-<none>}"
  say "new:     $newline"
  if [[ $oldline == "$newline" ]]; then
    say "already set; nothing to do"
  elif [[ $apply -eq 1 ]]; then
    bak="$grub.slotbench.$(date +%Y%m%d-%H%M%S).bak"
    cp -p "$grub" "$bak"
    say "backup: $bak"
    NEWLINE=$newline python3 - "$grub" <<'PY'
import os, re, sys
path, new = sys.argv[1], os.environ["NEWLINE"]
lines = open(path).read().split("\n")
idx = [i for i, l in enumerate(lines) if re.match(r'^GRUB_CMDLINE_LINUX=', l)]
if idx:
    lines[idx[-1]] = new
else:
    if lines and lines[-1] == "":
        lines.insert(len(lines) - 1, new)
    else:
        lines.append(new)
open(path, "w").write("\n".join(lines))
PY
    say "wrote $grub"
    do_cmd update-grub
  else
    say "[dry-run] would back up $grub and replace the line above, then run update-grub"
  fi
  if [[ $grub_custom -eq 0 ]]; then
    for f in /etc/default/grub.d/*.cfg; do
      if [[ -f $f ]] && grep -qE '^GRUB_CMDLINE_LINUX=' "$f"; then
        say "warning: $f also sets GRUB_CMDLINE_LINUX and is read after $grub; check it by hand"
      fi
    done
  fi
  grep -q GenuineIntel /proc/cpuinfo 2>/dev/null || \
    say "note: CPU is not Intel; intel_pstate=disable has no effect (set the governor to performance instead)"
fi

# ---------------------------------------------------------------- packages
say ""
say "== 2. packages"
do_cmd apt-get update
do_cmd apt-get install -y rt-tests numactl

if [[ $rt -eq 1 ]]; then
  say ""
  say "== 3. PREEMPT_RT kernel"
  if apt-cache show linux-image-realtime >/dev/null 2>&1; then
    do_cmd apt-get install -y linux-image-realtime
  elif command -v pro >/dev/null 2>&1; then
    say "linux-image-realtime not in apt; using Ubuntu Pro (needs 'sudo pro attach <token>' first)"
    do_cmd pro enable realtime-kernel --assume-yes
  else
    say "no RT kernel source found: install ubuntu-advantage-tools, 'pro attach', then 'pro enable realtime-kernel'"
  fi
fi

if [[ -n $iface ]]; then
  say ""
  say "== 4. PTP on $iface"
  do_cmd apt-get install -y linuxptp ethtool
  if command -v ethtool >/dev/null 2>&1; then
    ethtool -T "$iface" 2>&1 | sed 's/^/  /' || true
    say "  (need 'hardware-transmit', 'hardware-receive' and a 'PTP Hardware Clock' index above)"
  fi
  say "run (as root, in two terminals or as the units below):"
  say "  ptp4l -i $iface -m -s"
  say "  phc2sys -s $iface -c CLOCK_REALTIME -O 0 -m"
  if [[ $ptp_units -eq 1 ]]; then
    unit_ptp="[Unit]
Description=slotbench ptp4l on $iface
After=network-online.target
[Service]
ExecStart=/usr/sbin/ptp4l -i $iface -m -s
Restart=on-failure
[Install]
WantedBy=multi-user.target"
    unit_phc="[Unit]
Description=slotbench phc2sys from $iface
After=slotbench-ptp4l.service
Requires=slotbench-ptp4l.service
[Service]
ExecStart=/usr/sbin/phc2sys -s $iface -c CLOCK_REALTIME -O 0 -m
Restart=on-failure
[Install]
WantedBy=multi-user.target"
    if [[ $sys_apply -eq 1 ]]; then
      printf '%s\n' "$unit_ptp" > /etc/systemd/system/slotbench-ptp4l.service
      printf '%s\n' "$unit_phc" > /etc/systemd/system/slotbench-phc2sys.service
      do_cmd systemctl daemon-reload
      say "units written; enable with: systemctl enable --now slotbench-ptp4l slotbench-phc2sys"
    else
      say "[dry-run] would write /etc/systemd/system/slotbench-ptp4l.service:"
      printf '%s\n' "$unit_ptp" | sed 's/^/    /'
      say "[dry-run] would write /etc/systemd/system/slotbench-phc2sys.service:"
      printf '%s\n' "$unit_phc" | sed 's/^/    /'
    fi
  fi
fi

# ---------------------------------------------------------------- checklists
cat <<TXT

== BIOS checklist (by hand, in firmware setup)
  [ ] C-states disabled (or limited to C1)
  [ ] SMT / Hyper-Threading disabled
  [ ] SpeedStep / EIST (AMD: Cool'n'Quiet) disabled; Turbo either off or recorded
  [ ] PCIe slot running at x16 (check LnkSta in lspci -vv); Resizable BAR as the vendor recommends
  [ ] fixed fan curve / no "eco" power profile

== after reboot
  cat /proc/cmdline                         # should contain: $params
  cat /sys/devices/system/cpu/isolated      # should print: $cores
  uname -v | grep PREEMPT_RT                # if --rt was used
  scripts/check_rt.sh --core ${cores%%[,-]*}          # want max < 20 us before touching the GPU
  sudo scripts/gpu_lock.sh lock --gpu 0     # persistence mode + fixed clocks
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  # keep IRQs off the isolated cores: IRQBALANCE_BANNED_CPULIST=$cores in /etc/default/irqbalance
  # turn off Wi-Fi, browsers, automatic updates (systemctl stop unattended-upgrades) during runs
TXT
say ""
if [[ $sys_apply -eq 1 ]]; then
  say "Done. REBOOT to use the new kernel command line (this script never reboots)."
else
  say "Dry run: no system changes were made. Re-run with --apply (as root) to make them."
fi
