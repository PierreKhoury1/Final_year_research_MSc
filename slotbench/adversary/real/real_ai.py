"""Real AI workloads as the co-located adversary, with the same CLI and summary JSON as bin/adversary.

  real-llm     Qwen2.5-1.5B-Instruct (fp16, Hugging Face transformers) generating text one token at a
               time with a KV cache, as an LLM server does per request; unit = token.
  real-vision  YOLOv8n (Ultralytics, PyTorch) detecting objects in a 640x640 frame; unit = frame
               (pre-processing, inference and NMS, as a camera analytics service runs it).

Duty cycle as DESIGN.md section 5: within each --period-ms, issue units until duty% of the period
has passed (synchronising the GPU after each unit), then sleep to the period end.

Usage: python3 real_ai.py --workload real-llm --duty 100 [--seconds S] [--prio low] [--gpu 0]
                          [--out adversary.json] [--timeline t.csv] [--period-ms 100]
       python3 real_ai.py --workload real-llm --prefetch     (download weights, then exit)
Needs: torch, transformers (real-llm), ultralytics (real-vision). Models download on first use.
"""
import argparse
import json
import os
import signal
import socket
import sys
import time

LLM_MODEL = os.environ.get("SB_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_CONTEXT = 512          # tokens kept before the conversation restarts
YOLO_MODEL = os.environ.get("SB_YOLO_MODEL", "yolov8n.pt")
PROMPT = ("You are a network assistant at a 5G base station. Summarise the current cell load, list "
          "the busiest users and suggest how to schedule them over the next few slots.")

stop = False


def _on_signal(_sig, _frm):
    global stop
    stop = True


def iso_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Llm:
    unit = "token"

    def __init__(self, torch, dev):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float16).to(dev).eval()
        ids = self.tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                           return_tensors="pt")
        self.prompt = ids.to(dev)
        self.params = sum(p.numel() for p in self.model.parameters())
        self.reset()

    def reset(self):
        with self.torch.inference_mode():
            out = self.model(self.prompt, use_cache=True)
        self.past = out.past_key_values
        self.next = out.logits[:, -1:].argmax(-1)
        self.n = self.prompt.shape[1]

    def step(self):
        if self.n >= LLM_CONTEXT:
            self.reset()  # a new request: prefill again
        with self.torch.inference_mode():
            out = self.model(self.next, past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        self.next = out.logits[:, -1:].argmax(-1)
        self.n += 1

    def describe(self):
        return {"model": LLM_MODEL, "params": self.params, "dtype": "float16", "context_tokens": LLM_CONTEXT}


class Vision:
    unit = "frame"

    def __init__(self, torch, dev):
        import numpy as np
        from ultralytics import YOLO
        self.model = YOLO(YOLO_MODEL)
        self.dev = dev
        rng = np.random.default_rng(1)
        self.frame = (rng.random((640, 640, 3)) * 255).astype("uint8")  # fixed frame: same work every unit

    def step(self):
        self.model.predict(self.frame, imgsz=640, device=self.dev, half=True, verbose=False)

    def describe(self):
        return {"model": YOLO_MODEL, "imgsz": 640, "half": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workload", choices=["real-llm", "real-vision"], required=True)
    ap.add_argument("--duty", type=float, default=100)
    ap.add_argument("--period-ms", type=float, default=100)
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--prio", choices=["high", "default", "low"], default="default")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="adversary.json")
    ap.add_argument("--timeline", default="")
    ap.add_argument("--prefetch", action="store_true", help="download/load the model, run 3 units, exit")
    a = ap.parse_args()
    if not 0 <= a.duty <= 100 or a.period_ms <= 0:
        ap.error("duty must be 0..100 and period-ms > 0")
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    import torch
    torch.cuda.set_device(a.gpu)
    dev = f"cuda:{a.gpu}"
    lo, hi = torch.cuda.Stream.priority_range()  # (least, greatest); greatest is numerically lowest
    prio_value = {"high": hi, "default": 0, "low": lo}[a.prio]
    stream = torch.cuda.Stream(device=dev, priority=prio_value)
    summary = {"workload": a.workload, "duty": a.duty, "period_ms": a.period_ms, "prio": a.prio,
               "prio_value": prio_value, "prio_range": [lo, hi], "gpu": a.gpu,
               "gpu_name": torch.cuda.get_device_name(a.gpu), "pid": os.getpid(), "host": socket.gethostname(),
               "torch": torch.__version__, "cuda_runtime_version": torch.version.cuda,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
               "mps_active_thread_percentage": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", ""),
               "seconds_requested": a.seconds, "start_time": iso_now(), "ok": False, "real_model": True}
    t_start = time.monotonic()
    units = active = 0.0
    unit_ms = []
    tl = open(a.timeline, "w") if a.timeline else None
    if tl:
        tl.write("t_s,units,units_per_s\n")
    try:
        with torch.cuda.stream(stream):
            w = (Llm if a.workload == "real-llm" else Vision)(torch, dev)
            for _ in range(3):  # warm-up, not counted
                w.step()
            torch.cuda.synchronize()
        summary.update(unit=w.unit, **w.describe())
        if a.prefetch:
            summary.update(ok=True, exit_reason="prefetch")
            return 0
        period = a.period_ms / 1000.0
        t0 = time.monotonic()
        k = 0
        last_tl, last_units = t0, 0
        while not stop and (a.seconds <= 0 or time.monotonic() - t0 < a.seconds):
            p0 = t0 + k * period
            while (not stop and a.duty > 0 and time.monotonic() - p0 < a.duty / 100.0 * period
                   and (a.seconds <= 0 or time.monotonic() - t0 < a.seconds)):
                u0 = time.monotonic()
                with torch.cuda.stream(stream):
                    w.step()
                stream.synchronize()
                u1 = time.monotonic()
                units += 1
                active += u1 - u0
                unit_ms.append((u1 - u0) * 1000.0)
            k += 1
            now = time.monotonic()
            if tl and now - last_tl >= 1.0:
                tl.write(f"{now - t0:.3f},{int(units)},{(units - last_units) / (now - last_tl):.3f}\n")
                tl.flush()
                last_tl, last_units = now, units
            nxt = t0 + k * period
            while not stop and time.monotonic() < nxt:
                time.sleep(min(0.05, max(0.0, nxt - time.monotonic())))
        total = time.monotonic() - t0
        summary.update(ok=True, exit_reason="signal" if stop else "seconds", seconds_total=total,
                       seconds_active=active, units=int(units),
                       units_per_s=units / total if total > 0 else 0.0,
                       units_per_s_active=units / active if active > 0 else 0.0,
                       active_fraction=active / total if total > 0 else 0.0,
                       unit_ms_mean=sum(unit_ms) / len(unit_ms) if unit_ms else None,
                       unit_ms_max=max(unit_ms) if unit_ms else None)
        if a.workload == "real-llm":
            summary["tokens_per_s"] = summary["units_per_s"]
        else:
            summary["fps"] = summary["units_per_s"]
        return 0
    except Exception as e:  # always leave a summary behind
        summary.update(ok=False, exit_reason=f"error: {type(e).__name__}: {e}")
        return 1
    finally:
        if tl:
            tl.close()
        summary["end_time"] = iso_now()
        summary.setdefault("seconds_total", time.monotonic() - t_start)
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, a.out)
        print(f"real_ai: {a.workload} {summary.get('exit_reason')} units={summary.get('units')} "
              f"units/s={summary.get('units_per_s')}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
