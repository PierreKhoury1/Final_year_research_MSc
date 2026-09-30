# Runbook: the NVIDIA go/no-go run on vast.ai

Self-contained instructions for a person or an agent with shell access and a vast.ai API key.
Everything below is copy-pasteable. Read `README.md` section *Running the NVIDIA test* first for what
the run is for and what we expect.

## 0. Fixed values

| Variable | Value | Meaning |
|---|---|---|
| `REPO` | `PierreKhoury1/Final_year_research_MSc` | public GitHub repo, no credentials needed to fetch |
| `BRANCH` | `claude/modest-ride-bc8yxa` | the branch with the working runner; do not use `main` |
| `IMAGE` | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` | Docker image for the instance; `devel` is required because it has `nvcc` |
| `ONSTART` | `bash -c "curl -fsSL https://raw.githubusercontent.com/PierreKhoury1/Final_year_research_MSc/claude/modest-ride-bc8yxa/gpu_run/onstart.sh \| bash"` | on-start command for the instance (one line, remove the backslash before the pipe) |
| GPU | `A100 PCIE` or `A100 SXM4` (fallback `A100X`); any single NVIDIA GPU works for a smoke test | compute capability 8.0 is what Aerial's cuPHY targets |
| Disk | 20 GB | image plus build |
| CPUs | at least 8 effective cores | the timing thread needs a core of its own |
| Reliability | above 0.98 | vast host reliability score |
| Price ceiling | 2.00 $/h (override with `--max-dph`) | the run needs ~10 min including image load |
| Results markers | `LOCKSTEP_RESULTS_BEGIN` … `LOCKSTEP_RESULTS_END` | in the instance log; everything between them is the result |
| Failure markers | `LOCKSTEP_FETCH_FAILED`, `BUILD FAILED` | in the instance log |

## 1. Environment variables

| Variable | Required | Default | Used by |
|---|---|---|---|
| `VAST_API_KEY` | yes | none | `vast_run.py`, passed to every `vastai` call. Create it in the vast.ai console under Account → API Keys. Never paste it into a chat or commit it. |
| `TARGETS` | no | `2000` | `run.sh`: starts per method per condition (2000 × 2 ms = 4 s per method) |
| `GAP_US` | no | `2000` | `run.sh`: spacing between targets in µs |
| `HOST_CORE` | no | `1` | `run.sh`: CPU core the timing thread is pinned to |
| `PP_SECS` | no | `6` | `run.sh`: seconds per phase of the clock ping-pong (five phases) |
| `LOCKSTEP_RAW` | set by `run.sh` | `out_<utc>/raw` | `lockstep`: directory for one-error-per-line raw files |
| `BRANCH`, `REPO` | no | values above | `onstart.sh`: which branch to fetch on the instance |

## 2. Driver flags (`gpu_run/vast_run.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--search` | off | list matching offers and exit without renting |
| `--gpu NAME` | A100 variants | exact vast `gpu_name`, e.g. `"A100 PCIE"`, `"RTX 4090"` |
| `--max-dph X` | `2.0` | refuse offers above X $/hour |
| `--min-cpus N` | `8` | minimum effective CPU cores |
| `--offer ID` | cheapest match | rent this offer id instead |
| `--collect INSTANCE_ID` | off | skip renting; read results from an instance that is already running |
| `--keep` | off | do not destroy the instance afterwards (it bills until destroyed) |
| `--timeout-min M` | `30` | give up waiting after M minutes (the instance is still destroyed unless `--keep`) |

## 3. Run it (driver path, no SSH)

```bash
git clone -b claude/modest-ride-bc8yxa https://github.com/PierreKhoury1/Final_year_research_MSc.git
cd Final_year_research_MSc
pip install vastai
export VAST_API_KEY=...            # or have it in the environment already
python3 gpu_run/vast_run.py --search            # 1. look at offers; nothing is rented
python3 gpu_run/vast_run.py                     # 2. rent cheapest fit, run, collect, destroy
```

Step 2 prints the offer it picks, the instance id, status changes every 30 s, then the results table,
then `destroyed instance <id>: ... still listed: False`. Expect 8–15 minutes in all. Results are saved in
`gpu_run/vast_results/<instance_id>/` as `run.log`, `gpu_info.txt`, `results.jsonl`,
`pp_cuda_summary.json`, `results.tgz`.

If `--search` shows nothing: `python3 gpu_run/vast_run.py --search --max-dph 3 --min-cpus 4`, or
`--gpu "RTX 4090"` for a cheap smoke test (results are then not A100 results; say so in the write-up).

## 4. Run it by hand (vast.ai web console)

1. Search: 1× A100 PCIE or SXM4, on-demand, ≥ 8 CPUs, reliability > 0.98, disk 20 GB.
2. Template/image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`. Launch mode: SSH (any mode works).
3. On-start script: paste the `ONSTART` line from section 0.
4. Rent. After ~10 minutes open the instance logs and copy everything between the two markers.
5. `python3 gpu_run/vast_run.py --collect <instance_id>` saves and parses it, or paste the block into
   `gpu_run/vast_results/<instance_id>/run.log` by hand.
6. Destroy the instance in the console. A stopped instance still bills for storage.

## 5. Run it on any CUDA machine you already have a shell on

```bash
git clone -b claude/modest-ride-bc8yxa https://github.com/PierreKhoury1/Final_year_research_MSc.git
bash Final_year_research_MSc/gpu_run/run.sh
```
Produces `gpu_run/out_<utc>/` and `gpu_run/results_<utc>.tgz`.

## 6. What to check when it is done

- `gpu_info.txt`: GPU name, driver, PCIe gen and width, CPU model, `cpu_quota`, `rtprio_limit`.
- `results.jsonl`: three lines (idle; AI job same process; AI job other process). In each, look at
  `globaltimer_step_ns.median`, `clock_bound_us`, `sched_fifo`, `pinned`, and per method
  `median_us`, `p99_us`, `p999_us`, `max_us`, `over_100us_pct`, `n`, `timeout`.
- `pp_cuda_summary.json`: `gpu_mhz` (nominal 1000, the timer counts ns), `wander_ppm_std`, per-phase
  `p1`/`p50`/`p99` bracket widths in ns.
- Compare against the prediction table in `README.md`. Record which predictions held.
- Confirm the instance is gone: `vastai show instances --api-key "$VAST_API_KEY"` prints no lockstep instance,
  and https://cloud.vast.ai/instances/ shows none.

## 7. Failure handling

| Symptom | Action |
|---|---|
| `LOCKSTEP_FETCH_FAILED` in the log | the instance could not reach GitHub; destroy, pick another host |
| `BUILD FAILED` | copy the compiler output from the log into an issue or to the person who owns the branch; destroy the instance; do not edit `lockstep.cu` blind on a paid instance |
| a method shows `"timeout": true` | expected under separate-process load if the resident block was preempted for over 5 s; report it, do not re-run |
| `sched_fifo: false` | the container refused real-time priority; results are still valid, note it |
| driver times out after 30 min with the instance `loading` | image pull is slow on that host; `--collect <id>` later, or destroy and pick a host with higher `inet_down` |
| `still listed: True` after destroy | destroy by hand in the console immediately |

## 8. Hand the results back

```bash
git add gpu_run/vast_results/<instance_id>
git commit -m "A100 go/no-go run <instance_id>: results"
git push -u origin claude/modest-ride-bc8yxa
```
Do not commit `pp_cuda.bin` or anything from the scratch `out_*` directory other than what
`vast_run.py` already saved. Then report: GPU and driver, the three summary tables, the clock bound,
the timer step, whether each prediction in the README held, and the instance id with confirmation it is destroyed.
