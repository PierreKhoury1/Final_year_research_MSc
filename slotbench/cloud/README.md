# slotbench on vast.ai

`cloud/vast.py` rents one GPU on vast.ai and runs a slotbench config on it. It collects the results from the
container log and then destroys the instance. The script uses only the Python standard library (Python >= 3.10);
nothing needs to be pip-installed. `cloud/onstart.sh` is the script that runs inside the rented container.

## Setup

1. Create an account at https://cloud.vast.ai and add credit. Prepaid credit is itself a spending cap.
2. Create an API key (Account -> Keys) and put it in the environment only:
   ```bash
   read -rs VAST_API_KEY && export VAST_API_KEY    # paste, Enter; not echoed and not stored in shell history
   ```
   Never paste the key into chats, issues or commits, and never write it into a file inside the repo.
   `vast.py` reads it only from `VAST_API_KEY` and sends it as an `Authorization: Bearer` header.
   It never prints, logs or saves the key, and redacts it from every error message.
3. The instance clones the **public** repo at the branch you choose (default `claude/optimistic-ptolemy-r42xrh`,
   flag `--branch`). Push your changes before you launch: the instance only sees what is on GitHub.

## Commands

```bash
python3 cloud/vast.py offers --gpu "RTX 3060"                  # no key needed
python3 cloud/vast.py offers --gpu "A100 SXM4" --max-dph 2 --verified
python3 cloud/vast.py run --gpu "RTX 3060" --config cloud.toml --max-hours 3 --max-cost 5
python3 cloud/vast.py launch --gpu "RTX 3060" --dry-run        # print the exact create payload, do nothing
python3 cloud/vast.py launch --offer 20610591                  # launch a specific offer
python3 cloud/vast.py status ID | logs ID --tail 200 | collect ID --out results/ID | destroy ID
```

GPU names are vast.ai's exact `gpu_name` strings: `RTX 3060`, `A100 PCIE`, `A100 SXM4`, `H100 PCIE`, `H100 SXM`.
Offer filters: `--max-dph` (total $/h including the 40 GB disk), `--min-reliability 0.98`, `--cuda 12.4`
(minimum `cuda_max_good` of the host driver), `--verified`, `--limit`.

### What `run` does

1. It searches for offers and picks the cheapest on-demand single-GPU offer that passes every filter. The filters
   are applied on the server and then again locally.
2. It creates the instance (`PUT /api/v0/asks/<offer>/`) from `nvidia/cuda:12.6.3-devel-ubuntu24.04` with SSH
   access. It passes `SB_BRANCH`, `SB_CONFIG`, `SB_REPO`, `SB_TUNE_US` and `SB_SM` in the environment. The
   onstart field is a short command: it downloads `slotbench/cloud/onstart.sh` from raw.githubusercontent.com at
   that branch and runs it. The command sends the script's output to the container log (PID 1 stdout) and to
   `/root/slotbench.log`. Each instance gets a unique label, and its state is saved to
   `cloud/.state/<id>.json`, which never contains the key.
   A failed create is never retried blindly, because a retry could rent a second GPU. If the create request has
   an unclear outcome, `run` searches your instances for its unique label.
3. Every `--poll-s` (60 s) it reads the instance status and the tail of the log. It prints the elapsed time,
   the cost so far and the last log line.
4. It stops when one of these happens:
   - `=====SLOTBENCH-DONE=====` appears.
   - An error marker appears. `run` then waits up to `--error-grace-s` for the results that the script still
     prints.
   - The instance exits, or it is still not running after `--max-load-min`.
   - The **time cap** (`--max-hours`) or the **cost cap** (`--max-cost`) would be crossed by the next poll. Cost is
     computed locally as `dph_total x elapsed hours` since creation.
5. It collects: it fetches the whole log (`PUT /instances/request_logs/<id>/`, then GETs the returned
   `result_url`) and extracts the result blocks into `results/<id>/` (see below).
6. It **destroys the instance in a `finally` block**. This also happens on Ctrl-C, SIGTERM and exceptions. Pass
   `--keep` to skip the destroy. If the destroy fails, `run` prints the instance id in a loud warning so that you
   can destroy it by hand. At the end it prints the estimated spend. The vast.ai billing page has the exact
   figure; storage and bandwidth are billed separately in small amounts.

### What the instance does (`onstart.sh`)

The steps run in this order:
1. Prints a header with the date, `nvidia-smi -L`, the driver, the CPU, the kernel and the container id.
2. `apt-get install`s whatever is missing (git, build-essential, python3, numpy, scipy, matplotlib and pandas
   from apt). It retries while apt is locked.
3. Runs `git clone --depth 1 -b $SB_BRANCH`.
4. Builds with `make SM=<compute capability>`. The compute capability comes from
   `nvidia-smi --query-gpu=compute_cap`; if that query fails, the offer's `compute_cap` is used. The script also
   runs `make test`.
5. Runs `slot_driver --selftest`. A failure is reported and the run continues.
6. Runs `slot_driver --tune-us 200`. The `--ldpc-cb N --ldpc-iters M` flags it prints are merged into the
   `sizes` value of a copy of the config. The copy keeps the same file name, so run directories are still
   `runs/<config>/`.
7. Runs `python3 scripts/run_matrix.py <config copy> --out-root runs`. The matrix output goes to a log file.
   Meanwhile a background emitter prints each **finished** cell's results, so a time or cost cap still returns
   everything that finished before it.
8. Runs `python3 -m analysis.summarize` on run directories that have no summary yet. Then it prints a compact
   table (cell, p50, p99.99, max, miss rate, status), the remaining result blocks and a `_logs/<config>` block
   (header, apt/build/selftest/tune/matrix logs, the config used, matrix.json/log).
9. Prints `=====SLOTBENCH-DONE=====` (or `=====SLOTBENCH-DONE status=<why>=====` if something failed) and then
   runs `sleep infinity`, so the controller can collect before it destroys.

Any failing step prints `=====SLOTBENCH-ERROR <line> <command> (exit N)=====` and the tail of the apt/build
log. The script still emits whatever results exist. If the container restarts, the script resumes: the clone
and completed matrix cells are reused, and `run_matrix.py` is resumable.

### How results arrive

Each block in the log looks like this:
```
=====SLOTBENCH-BEGIN cloud/M1_sgemm_d50_r0 <sha256 of the tar.gz>=====
<base64 of a tar.gz, 76 columns>
=====SLOTBENCH-END cloud/M1_sgemm_d50_r0=====
```
Each run block holds that run's `summary.json`, `meta.json`, `run.json`, `adversary.json`, `telemetry.csv`,
`status` and `env.txt`.

`collect` handles the blocks as follows:
- It base64-decodes each block, checks the sha256 and untars it into `results/<id>/` (for example
  `results/<id>/cloud/M1_sgemm_d50_r0/summary.json`). Extraction refuses absolute paths, `..` and links.
- A corrupted or truncated block is reported by name. If the same block was sent again intact, the intact copy
  wins.
- It saves the raw log as `instance.log` and a machine-readable report as `collect.json`.

Then run the analysis on the collected results:
```bash
python3 -m analysis.report results/<id> --out report_<id>
```
The analysis works from `summary.json` alone. Its log-binned histogram is enough to redraw the CCDF.

**Raw `slots.bin` files stay on the instance** at `/root/sb/repo/slotbench/runs/`. They are 64 B per slot, so
6.4 MB per 100k-slot run. To keep them, run with `--keep`, add your SSH public key in the vast.ai console, and
copy them before destroying:
```bash
python3 cloud/vast.py status ID            # shows ssh_host / ssh_port
scp -P PORT -r root@HOST:/root/sb/repo/slotbench/runs ./runs_ID
python3 cloud/vast.py destroy ID
```

## Costs seen (live offers query, 2026-09-30, 40 GB disk included)

| GPU | cheapest on-demand $/h (reliability >= 0.98) | typical |
|---|---|---|
| RTX 3060 | 0.044 (unverified host, CN) | 0.058-0.064 |
| A100 SXM4 | 0.674 (verified, SI) | 0.68-0.71 |
| H100 SXM | 1.96 (verified, DE) | 2.7-3.9 |

`configs/cloud.toml` runs about 38 cells of about 100 s each (100k slots plus 30 s warm-up and 10 s settle).
Add image pull, apt and build time, and a run takes roughly 1.5 h. That is about $0.10 on an RTX 3060, about
$1 on an A100 and $3-5 on an H100. Prices move; check with `offers` first.

## Caveats (record them; the noise is a finding, not a failure)

- **Containers are not a real-time host.** There is no `isolcpus`, no `nohz_full` and no PREEMPT_RT, and the
  host kernel and its other tenants are unknown. SCHED_FIFO and mlock may be refused. The driver logs
  `fifo_ok`/`mlock_ok`. Other tenants' CPU load shows up as wake-up overshoot and launch jitter.
- **GPU clock locking** (`nvidia-smi -lgc/-lmc`) and **time-slice changes** (`compute-policy`) usually need host
  privileges and are normally refused. The run scripts record them as "NOT SUPPORTED" in `run.json`. M3 and
  clock-locked runs are therefore rarely possible in the cloud.
- **MPS** may or may not start inside a container, depending on the host's driver and container setup.
  `run.json` says whether it was applied.
- **MIG** (M6) needs an A100/H100 with MIG enabled by the host, which is not possible in a normal vast.ai
  container. Use a bare-metal or VM provider for MIG.
- `gpu_frac` < 1 in an offer seems to mean the machine has several GPUs (vast.ai does not document it clearly).
  Only your GPU is visible, but other tenants may share the host's PCIe root, CPU and memory bandwidth.
- The logs API returns the Docker log. Very large logs may be truncated by the provider. The emitter keeps the
  blocks small (summaries only), and `/root/slotbench.log` on the instance has everything.
- `--runtype args` runs the command as PID 1 without SSH. This is an alternative if a host's SSH image wrapper
  misbehaves, but `--keep`/scp is then impossible.
- The instance clones the branch when it boots, so results correspond to the commit printed in the log
  (`commit <sha>`), not to your working tree.

## Tests

```bash
python3 -m pytest -q cloud/tests          # no network; mocks urllib and the API
```
