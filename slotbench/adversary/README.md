# adversary: co-located GPU workloads

`bin/adversary` is the "other tenant" that shares the GPU with the 500 us slot driver
(DESIGN.md section 5). It runs in its own process (MPS needs separate processes) on its own
CUDA stream at the requested priority, with cuBLAS bound to that stream.

## Workloads

| Workload | Stands for | Proxy | Unit | Throughput field |
|---|---|---|---|---|
| `sgemm` (W1) | training / big batched compute | `cublasSgemm` N x N x N, back to back (FP32, default math: no TF32) | one GEMM | `tflops` = 2 N^3 units / active s |
| `llm` (W2) | LLM token generation (decode) | per token, for each of L layers: fp16 GEMVs (`cublasGemmEx`, fp16 in/out, fp32 compute, n = 1) for qkv (3h x h), o (h x h), MLP up (4h x h) and down (h x 4h), with an elementwise kernel after each (attention mix, residual, SiLU, residual). 12 h^2 L params, h = sqrt(params / 12L) rounded to 64 | one token | `gb_per_s` = weight bytes per token x tokens / active s |
| `vision` (W3) | camera / perception inference | YOLOv8n-like frame at batch 1, 640x640: 30 conv-as-GEMM `cublasSgemm` calls (backbone Conv s2 stages, C2f 1x1/3x3 convs, SPPF 1x1s, one detection conv per scale; K = k*k*Cin, channels 16..256, spatial 320^2 .. 20^2) each followed by a SiLU kernel = 60 kernels | one frame | `fps` = frames / active s |
| `idle` | nothing, but a second CUDA context exists | context + stream + 1 MiB buffer, then sleep | none | - |

The W2 defaults (1e9 params, 16 layers) give h = 2304 and 2.04 GB of weights, so each token streams
the whole weight set from DRAM like a real decode step. If free memory is short the parameter count
is scaled down (80% of free memory minus 256 MiB) and `sizes.llm_scaled_down` is set. The W3 proxy
skips im2col and the neck (the data flow between layers is fake); the kernel shapes and count are
what matter for preemption and scheduling. `--vision-scale` scales the spatial side only.

Why proxies instead of the real models: no model downloads or framework installs on the rented
or home machine, bit-for-bit the same kernels on every GPU, deterministic unit sizes, and exact
work accounting (FLOP or bytes per unit). The real workloads are available as optional wrappers
(below) to check that the proxies look like the real thing.

## Duty cycle

Period P (`--period-ms`), duty D. Period p starts at t_begin + p*P (integer ns, no drift). While
`now - period_start < D% * P` the loop issues one whole unit and calls `cudaStreamSynchronize`, so
the active time is real GPU time, not queueing; then it sleeps to the period end with
`clock_nanosleep` (`sb::sleep_until_raw`, in <= 50 ms chunks so signals stay responsive). Only
completed units are counted. The last unit of a window can overrun the window (a 4096^3 SGEMM is
about 10-15 ms on an RTX 3060), so the measured `active_fraction` is slightly above D/100; if a unit
runs past later period boundaries the loop resumes with the period containing "now" and counts
the skipped boundaries in `overrun_periods`. D = 100 never sleeps; D = 0 and `idle` only sleep.
`--seconds` is checked before each unit, so the run can end up to one unit late.

The CUDA context uses `cudaDeviceScheduleBlockingSync` so the adversary does not spin a CPU core
that the slot driver might share. One warm-up unit (not counted) runs before the clock starts;
all device buffers and the 32 MiB cuBLAS workspace are allocated before it.

## Flags

`--workload sgemm|llm|vision|idle` (sgemm), `--duty D` (100), `--period-ms F` (100),
`--seconds F` (0 = until SIGINT/SIGTERM), `--prio high|default|low` (default; high = greatest =
numerically lowest of `cudaDeviceGetStreamPriorityRange`, low = least, default = 0), `--gpu N` (0),
`--out FILE` (adversary.json), `--timeline FILE` (per-second CSV `t_s,units,units_per_s`, flushed
every row), `--size N` (4096), `--llm-params F` (1e9), `--llm-layers N` (16), `--vision-scale F`
(1.0), `--print-config` (resolved config + derived shapes, no GPU needed). Unknown flag: usage, exit 2.
One config line at start and one summary line at exit go to stderr.

## Summary JSON (`--out`, written atomically: temp file, fsync, rename)

Always written, also on SIGINT/SIGTERM and on CUDA errors (then `ok: false`, exit code 1).

- `ok`, `exit_reason` (`seconds`, `sigint`, `sigterm`, `error: ...`)
- `workload`, `duty`, `period_ms`, `prio`, `prio_value`, `prio_range {least, greatest}`
- `seconds_requested`, `seconds_total`, `seconds_active` (sum of per-unit issue-to-sync times),
  `active_fraction`, `periods`, `overrun_periods`
- `units`, `warmup_units`, `units_per_s` (over total time: use this for relative throughput vs the
  SOLO run), `units_per_s_active` (over active time), `unit_ms_mean/min/max`
- `tflops` (sgemm), `gb_per_s` (llm), `fps` (vision), all over active time, `null` otherwise
- `flop_per_unit`, `bytes_per_unit` (llm weight bytes per token), `kernels_per_unit`
- `sizes`: `size_requested/actual`, `llm_params_requested/actual`, `llm_layers`, `llm_hidden`,
  `llm_weight_bytes`, `llm_scaled_down`, `vision_scale`, `vision_convs`, `device_bytes_allocated`
- `gpu`, `gpu_name`, `gpu_uuid` (GPU-xxxxxxxx-...), `free_mem_before`, `total_mem`,
  `cuda_runtime_version`, `cuda_driver_version`, `mps_active_thread_percentage`,
  `cuda_visible_devices`, `host`, `pid`, `start_time`, `end_time` (ISO 8601 UTC), `timeline`

## Real workloads (optional)

`real/dutycycle.py --duty D --period-ms P [--seconds S] -- CMD ARGS...` runs any program in its own
process group and alternates SIGCONT (D% of each period) / SIGSTOP; SIGINT/SIGTERM are forwarded
(after a SIGCONT), and the exit status is the child's. `python3 real/dutycycle.py --selftest`
checks the duty cycle on a busy loop (CPU share within 10% of D) and the signal handling.
SIGSTOP only freezes the CPU side: kernels already queued keep running, so the GPU duty cycle of a
real program is smeared by its queue depth (the built-in proxies sync per unit and do not have
this problem).

- `real/llama.sh [--duty D] [--seconds S] [--tokens N]`: builds llama.cpp with `GGML_CUDA=ON` into
  `$LLAMA_DIR` (default `~/llama.cpp`), downloads `$MODEL_URL` (default Qwen2.5-1.5B-Instruct
  Q4_K_M GGUF), loops `llama-bench -p 0 -n N -ngl 99` (decode only).
- `real/yolo.sh [--duty D] [--seconds S]`: venv + `pip install ultralytics`, exports YOLOv8n to a
  TensorRT FP16 engine (`YOLO_FORMAT=pt` for plain PyTorch), predicts a fixed synthetic image in a loop.

## Validating the proxies with Nsight Systems

Compare kernel count per unit and the kernel duration histogram of proxy and real workload on the
same GPU; the preemption behaviour the slot driver sees depends mostly on those two.

```
nsys profile -t cuda -o proxy_llm  ./bin/adversary --workload llm --seconds 10
nsys profile -t cuda -o real_llm   adversary/real/llama.sh --seconds 30
nsys stats -r cuda_gpu_kern_sum proxy_llm.nsys-rep     # per-kernel count, total/avg/min/max ns
nsys stats -r cuda_gpu_trace -f csv -o proxy_llm proxy_llm.nsys-rep   # every kernel: histogram it
```

Kernels per unit = kernel count / `units` (the proxy's `kernels_per_unit` excludes kernels cuBLAS
adds internally, e.g. split-K reductions). For llama.cpp divide by tokens generated; for YOLO by
frames. Report both histograms (log-binned duration, like the slot latency plots) in the
dissertation: if the real workload has longer maximum kernels than the proxy, the proxy
underestimates the worst-case blocking of the slot and the difference should be stated.
