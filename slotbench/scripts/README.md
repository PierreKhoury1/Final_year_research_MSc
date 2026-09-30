# slotbench scripts: machine setup and the run procedure

These scripts turn the run procedure into commands, so nothing is done by hand between runs.
Everything below is relative to `slotbench/`. The contract (flags, file formats, mechanisms) is
`DESIGN.md`. This file is the step-by-step home-lab procedure.

| Script | What it does |
|---|---|
| `setup_os.sh` | One-time OS preparation. Dry-run by default; `--apply` edits GRUB (with a backup), installs packages and prints the BIOS checklist. It never reboots. |
| `check_rt.sh` | Runs cyclictest on the isolated core and prints PASS/FAIL against a max latency (default 20 us). |
| `gpu_lock.sh` | `lock` / `unlock` / `status`: persistence mode and fixed graphics and memory clocks. Prints `NOT SUPPORTED: ...` and exits 3 when the card or driver refuses. |
| `telemetry.sh` | Logs nvidia-smi every second (temperature, clocks, power, utilisation, throttle reasons) to CSV until killed. |
| `mps.sh` | `start` / `stop` / `status` of the CUDA MPS control daemon on a private pipe directory. |
| `env_capture.sh` | Machine snapshot (`nvidia-smi -q`, kernel, cmdline, isolated CPUs, governors, container, filtered env) written to `env.txt`. |
| `run_one.sh` | One matrix cell, start to finish (procedure below). |
| `run_matrix.py` | Expands a `configs/*.toml` matrix and runs every cell with `run_one.sh`. Resumable, with ETA. |

## 1. Bench parts list (home lab)

- Desktop with an RTX 3060 12 GB in a CPU-attached x16 slot, and a CPU with at least 6 physical
  cores (2 isolated for the driver and collector, the rest for the OS and the adversary).
- 32 GB RAM. Pinned buffers are locked with `mlockall`.
- NVMe SSD with at least 20 GB free. The full matrix writes about 5 GB of `slots.bin`.
- A wired Ethernet NIC with hardware timestamping (Intel i210/i225/i226) if PTP is used.
  Check with `ethtool -T <iface>`. Wi-Fi stays off.
- Ubuntu 24.04 LTS on bare metal. A VM or container cannot isolate cores.

## 2. OS setup (once)

```bash
scripts/setup_os.sh --cores 4,5 --rt --iface enp3s0          # read what it would do
sudo scripts/setup_os.sh --apply --cores 4,5 --rt --iface enp3s0 --ptp-units
```

`--apply` adds these to `GRUB_CMDLINE_LINUX` in `/etc/default/grub`:
`isolcpus=4,5 nohz_full=4,5 rcu_nocbs=4,5 intel_pstate=disable processor.max_cstate=1 idle=poll`.
It first backs the file up as `grub.slotbench.<time>.bak`. Running it again does not duplicate
parameters. It then runs `update-grub` and installs `rt-tests` and `numactl`.

- `--rt` installs a PREEMPT_RT kernel. It uses `linux-image-realtime` if apt has it. Otherwise it
  runs `pro enable realtime-kernel`, which needs `sudo pro attach <token>` first (Ubuntu Pro is
  free for personal use).
- `--iface` installs linuxptp.
- The script refuses `--apply` inside a container (docker, vast.ai) and on non-Ubuntu systems.

Then set the BIOS options from the printed checklist: C-states off, SMT off, SpeedStep/EIST off,
PCIe x16. Reboot.

## 3. Verification (before touching the GPU)

```bash
cat /proc/cmdline; cat /sys/devices/system/cpu/isolated      # 4,5
uname -v | grep PREEMPT_RT
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sudo scripts/check_rt.sh --core 4 --threshold-us 20           # cyclictest -t1 -p99 -i500 -l100000 -a4
```

Do not continue until `check_rt.sh` prints PASS (max < 20 us). If it fails:
- look for SMIs (BIOS power management);
- move IRQs off cores 4 and 5 (`IRQBALANCE_BANNED_CPULIST=4,5` in `/etc/default/irqbalance`);
- stop desktop services;
- turn off Wi-Fi.

## 4. GPU clocks

```bash
sudo scripts/gpu_lock.sh lock --gpu 0            # -pm 1, -lgc GC,GC, -lmc MC,MC
scripts/gpu_lock.sh status
nvidia-smi --query-supported-clocks=mem,gr --format=csv   # to choose --gc/--mc yourself
```

The default memory clock is the highest supported one. The default graphics clock is the driver's
default applications clock, or the highest supported clock at or below 80% of the maximum.

Choose a graphics clock the card holds under the sgemm adversary without power throttling. Check
the last column of `telemetry.csv` from the quick run.

GeForce drivers may refuse `-lmc` or `-lgc`. The script then prints `NOT SUPPORTED: ...` and
exits 3. `run_one.sh` records this in `run.json` and carries on. Treat it as a finding, not a
failure.

`run_one.sh --lock-clocks 1` locks before each run and unlocks afterwards. M5 unlocks instead.

## 5. PTP (only needed for the NIC clock work in DESIGN.md section 6)

```bash
sudo ptp4l -i enp3s0 -m -s                                  # or: systemctl enable --now slotbench-ptp4l
sudo phc2sys -s enp3s0 -c CLOCK_REALTIME -O 0 -m            # or: systemctl enable --now slotbench-phc2sys
```

## 6. Build and tune

```bash
make
bin/slot_driver --selftest --out /tmp/st            # GPU decoder check, must exit 0
bin/slot_driver --tune-us 200 --out /tmp/tune       # prints e.g. "--ldpc-cb 50 --ldpc-iters 20"
```

Paste the printed flags into `sizes = "..."` in the config.

## 7. Run: quick config first, then the full matrix

```bash
python3 scripts/run_matrix.py configs/quick.toml --dry-run    # check the commands
python3 scripts/run_matrix.py configs/quick.toml              # ~6 min: M0,M1,M4 x sgemm x d0,d100
cat runs/quick/*/status                                        # all "ok"?
python3 scripts/run_matrix.py configs/home_lab.toml --only 'SOLO_*,M0_*,M1_*'   # first night
python3 scripts/run_matrix.py configs/home_lab.toml           # the rest; ok cells are skipped
```

Run the matrix as root, or with passwordless sudo, so clock locking and the time-slice work. The
driver's `--fifo 99` also needs root or an rtprio limit. A failure there is logged in
`meta.json` (`fifo_ok`), not fatal.

- **Order:** cells run in a fixed order: `SOLO_<W>` (adversary alone, D=100, no driver) first,
  then mechanisms x workloads x duties x repeats.
- **D=0 cells:** D=0 means an idle adversary, so it runs once per mechanism, as `<M>_idle_d0_r<rep>`.
- **Output:** each cell writes `runs/<config>/<cell>/`. `matrix.log` (timestamped) and
  `matrix.json` (plan plus per-cell outcome) sit next to the cells.
- **Stopping:** Ctrl-C stops the current cell cleanly. GPU settings are restored and the cell is
  marked `invalid:interrupted`. Rerunning the same command resumes.
- **Repeats on another day:**
  `python3 scripts/run_matrix.py configs/home_lab.toml --repeat-from 2 --only 'M1_sgemm_d100_*'`

### What run_one.sh does for each cell

1. If the cell directory already holds a previous attempt, it moves it to
   `old_attempts/<time>/`. Stale files can never validate a new run.
2. Writes `env.txt` (`env_capture.sh`).
3. Resets MPS. It stops daemons. If an MPS daemon is still alive afterwards the run becomes
   `invalid:mps_daemon_still_running`, because it would silently put the run in MPS mode.
4. Clocks: `gpu_lock.sh lock` (or `unlock` for M5). `gpu_lock.sh status` is recorded.
5. Mechanism setup, from the DESIGN.md section 7 table:

   | Mechanism | Setup |
   |---|---|
   | M0 | priorities default/default |
   | M1 | driver high, adversary low |
   | M2 | MPS daemon on a private pipe directory; adversary gets `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50`; driver high |
   | M3 | `nvidia-smi compute-policy --set-timeslice=1`, restored to 0 afterwards |
   | M4 | M1 plus `--mode streams` |
   | M5 | M1 with clocks unlocked |
   | M6 | `CUDA_VISIBLE_DEVICES=$MIG_DRIVER_UUID` / `$MIG_ADV_UUID` |

   When a mechanism is not supported the cell exits 3 with `invalid:not_supported:<what>`. This
   happens for M3 on GeForce cards and for M6 without MIG. It is a result.
6. Warm-up: the same adversary (workload, duty, priority, MPS environment) runs for `--warmup-s`
   and writes `warmup_adversary.json`.
7. Starts telemetry (`telemetry.csv`), then the adversary (`adversary.json`,
   `adversary_timeline.csv`), then waits `--settle-s`.
8. Runs the driver for N slots (`slots.bin`, `meta.json`, `calib_*.csv`, `driver.log`).
9. Stops the adversary (SIGTERM, then SIGKILL after 30 s) and the telemetry. An EXIT trap then
   restores time-slice, MPS and clocks, so this also happens on failure or Ctrl-C.
10. Validity is written to `status`. The run is `ok` only if:
    - the driver exited 0;
    - `meta.json` exists and has `ring_overflows == 0`;
    - the adversary was still running when the driver finished;
    - `adversary.json` exists.

    Otherwise it is `invalid:<reasons>`. The analysis (`validate.py`) adds further checks:
    - throttling;
    - recorded == requested slots;
    - stamp mismatches.
11. Records every command, exit code, applied setting and unsupported item in `run.json`, then
    runs `python3 -m analysis.summarize DIR` if it is available (non-fatal).

## 8. Disk and time estimates

| Config | Cells | Per cell | Total | slots.bin |
|---|---|---|---|---|
| quick | 6 | 10 s warm-up + 5 s settle + 10 s slots + ~30 s | ~6 min | 6 x 1.3 MB |
| home_lab | 81 | 300 + 30 + 500 s + ~30 s = ~14 min | ~19 h | 1e6 x 64 B = 64 MB per run, ~5.2 GB |
| cloud | 38 | 30 + 10 + 50 + ~30 s | ~1.3 h | 6.4 MB per run |

`run_matrix.py --dry-run` prints the estimate for any config or `--only` selection.

## 9. Known traps

- **`cudaDeviceSynchronize` spinning.** With `cudaDeviceScheduleSpin` the waiting thread burns a
  whole core. That is why the driver sits on an isolated core and waits on an event or the flag
  (`--wait`). Never put a synchronize on a non-isolated core in the timed loop.
- **Pinned allocation on the hot path.** `cudaHostAlloc` or `cudaMallocHost` inside the loop takes
  milliseconds and serialises with other contexts. All allocation happens at init, and so does
  `cublasSetWorkspace`. Any real workload wrapper you add must follow the same rule.
- **Sionna/TensorFlow overhead.** A TF-based PHY adds Python and TF launch overhead of hundreds of
  microseconds per slot. That is why the slot workload is native CUDA. Do not compare against
  TF-based timings.
- **GeForce caps.** RTX cards may refuse:
  - memory clock locking;
  - `compute-policy --set-timeslice`;
  - MIG (never available).

  MPS works but is less isolated than on datacenter parts. Record every "NOT SUPPORTED" as a
  finding. `run.json` does this automatically.
- **Background noise.** Turn off Wi-Fi, browsers, desktop search and indexers, and automatic
  updates (`sudo systemctl stop unattended-upgrades packagekit`). Stop snap refreshes. Do not
  use the desktop during a run: the display shares the GPU, so run headless or on the iGPU if
  possible.
- **MPS ownership.** The MPS daemon and its clients must run as the same user. `run_one.sh`
  starts its own daemon on `/tmp/sb-mps-<pid>`. A daemon started by someone else on the default
  pipe is stopped by the reset. One on an unknown pipe makes the cell invalid.
- **Thermal state.** The warm-up exists because a cold GPU boosts higher at first. Check the
  throttle-reason column in `telemetry.csv`.
- **Containers (vast.ai).** None of these work reliably in a container:
  - isolcpus;
  - PREEMPT_RT;
  - SCHED_FIFO;
  - clock locking.

  `cloud.toml` therefore leaves cores and FIFO off and attempts clock locking, which is recorded.
