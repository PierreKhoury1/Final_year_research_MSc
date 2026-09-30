#!/usr/bin/env bash
# Optional W3 with a real vision model: Ultralytics YOLOv8n exported to TensorRT, predicting a fixed
# synthetic 640x640 image at max FPS, duty-cycled by dutycycle.py. Optional: the harness itself only
# needs the built-in proxy (bin/adversary --workload vision). No sudo; creates a Python venv.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YOLO_VENV="${YOLO_VENV:-$HOME/yolo-venv}"
YOLO_DIR="${YOLO_DIR:-$HOME/yolo-work}"
YOLO_MODEL="${YOLO_MODEL:-yolov8n.pt}"
YOLO_FORMAT="${YOLO_FORMAT:-engine}"   # engine = TensorRT FP16; pt = plain PyTorch CUDA fallback
ULTRALYTICS_SPEC="${ULTRALYTICS_SPEC:-ultralytics}"

DUTY=100
PERIOD_MS=100
SECONDS_RUN=0
GPU=""
SETUP_ONLY=0

usage() {
    cat <<EOF
usage: $(basename "$0") [--duty D] [--period-ms P] [--seconds S] [--gpu N] [--setup-only]

Real-workload W3 (optional): YOLOv8n at batch 1, 640x640, TensorRT FP16 engine, predicting one
synthetic image in a loop, duty-cycled with SIGSTOP/SIGCONT. Prints FPS every 5 s.
First run: python3 -m venv \$YOLO_VENV, pip install ultralytics, export the engine (downloads
yolov8n.pt from the Ultralytics release; TensorRT is pip-installed by the exporter).

  --duty D        percent of each period the process runs (default 100)
  --period-ms P   duty-cycle period (default 100)
  --seconds S     stop after S seconds (default 0 = until Ctrl-C / SIGTERM)
  --gpu N         CUDA device index for Ultralytics (default 0)
  --setup-only    install + export, then exit

Environment: YOLO_VENV ($YOLO_VENV), YOLO_DIR ($YOLO_DIR), YOLO_MODEL, YOLO_FORMAT (engine|pt),
             ULTRALYTICS_SPEC (pip spec, e.g. ultralytics==8.3.0 to pin).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duty) DUTY="$2"; shift 2 ;;
        --period-ms) PERIOD_MS="$2"; shift 2 ;;
        --seconds) SECONDS_RUN="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
DEVICE="${GPU:-0}"

if [[ ! -x "$YOLO_VENV/bin/python" ]]; then
    echo "yolo.sh: creating venv $YOLO_VENV" >&2
    python3 -m venv "$YOLO_VENV"
fi
PY="$YOLO_VENV/bin/python"
if ! "$PY" -c "import ultralytics" 2>/dev/null; then
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install "$ULTRALYTICS_SPEC"
fi

mkdir -p "$YOLO_DIR"
cd "$YOLO_DIR"
stem="${YOLO_MODEL%.pt}"
if [[ "$YOLO_FORMAT" == "engine" ]]; then
    WEIGHTS="$YOLO_DIR/$stem.engine"
    if [[ ! -s "$WEIGHTS" ]]; then
        echo "yolo.sh: exporting $YOLO_MODEL to TensorRT (FP16, 640, batch 1, device $DEVICE)" >&2
        "$PY" - "$YOLO_MODEL" "$DEVICE" <<'PYEOF'
import sys
from ultralytics import YOLO
YOLO(sys.argv[1]).export(format="engine", half=True, imgsz=640, batch=1, device=int(sys.argv[2]))
PYEOF
    fi
else
    WEIGHTS="$YOLO_MODEL"
fi
echo "yolo.sh: weights=$WEIGHTS" >&2
if [[ "$SETUP_ONLY" == 1 ]]; then exit 0; fi

cat > "$YOLO_DIR/yolo_loop.py" <<'PYEOF'
# Predict one fixed synthetic image forever at batch 1; print FPS every 5 s (flushed).
import sys, time
import numpy as np
from ultralytics import YOLO
model = YOLO(sys.argv[1], task="detect")
img = np.random.default_rng(0).integers(0, 256, (640, 640, 3), dtype=np.uint8)
dev = int(sys.argv[2])
for _ in range(10):
    model.predict(img, imgsz=640, device=dev, verbose=False)
n, t = 0, time.monotonic()
while True:
    model.predict(img, imgsz=640, device=dev, verbose=False)
    n += 1
    now = time.monotonic()
    if now - t >= 5.0:
        print("yolo fps=%.1f frames=%d" % (n / (now - t), n), flush=True)
        n, t = 0, now
PYEOF

echo "yolo.sh: duty=$DUTY period_ms=$PERIOD_MS seconds=$SECONDS_RUN" >&2
exec python3 "$HERE/dutycycle.py" --duty "$DUTY" --period-ms "$PERIOD_MS" --seconds "$SECONDS_RUN" -- \
    "$PY" "$YOLO_DIR/yolo_loop.py" "$WEIGHTS" "$DEVICE"
