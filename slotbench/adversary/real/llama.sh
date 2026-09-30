#!/usr/bin/env bash
# Optional W2 with a real LLM: llama.cpp token generation, duty-cycled by dutycycle.py.
# Builds llama.cpp with CUDA into $LLAMA_DIR if missing, downloads a small GGUF model once, then
# loops llama-bench text generation (-p 0 -n N: decode only, like the W2 proxy) under SIGSTOP/SIGCONT.
# No sudo: needs git, cmake >= 3.14, a C++ compiler, curl and the CUDA toolkit (nvcc) on PATH.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_REF="${LLAMA_REF:-}"   # optional tag/commit to pin, e.g. b6000; empty = default branch
MODEL_URL="${MODEL_URL:-https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
MODEL_DIR="${MODEL_DIR:-$LLAMA_DIR/models}"
MODEL="${MODEL:-$MODEL_DIR/$(basename "$MODEL_URL")}"

DUTY=100
PERIOD_MS=100
SECONDS_RUN=0
GPU=""
TOKENS=256
SETUP_ONLY=0

usage() {
    cat <<EOF
usage: $(basename "$0") [--duty D] [--period-ms P] [--seconds S] [--gpu N] [--tokens N] [--setup-only]

Real-workload W2: llama.cpp generation (llama-bench -p 0 -n TOKENS -ngl 99, looped) duty-cycled
with SIGSTOP/SIGCONT. First run clones and builds llama.cpp (GGML_CUDA=ON) and downloads the model.

  --duty D        percent of each period the process runs (default 100)
  --period-ms P   duty-cycle period (default 100)
  --seconds S     stop after S seconds (default 0 = until Ctrl-C / SIGTERM)
  --gpu N         sets CUDA_VISIBLE_DEVICES=N (default: inherit)
  --tokens N      tokens generated per llama-bench repetition (default 256)
  --setup-only    build + download, then exit

Environment: LLAMA_DIR ($LLAMA_DIR), LLAMA_REPO, LLAMA_REF, MODEL_URL, MODEL_DIR, MODEL,
             CUDA_ARCH (e.g. 86; default: cmake detects the local GPU).
Example:     $(basename "$0") --duty 50 --seconds 600 > llama.log
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duty) DUTY="$2"; shift 2 ;;
        --period-ms) PERIOD_MS="$2"; shift 2 ;;
        --seconds) SECONDS_RUN="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for tool in git cmake curl; do
    command -v "$tool" >/dev/null || { echo "llama.sh: missing '$tool' on PATH" >&2; exit 1; }
done

BENCH="$LLAMA_DIR/build/bin/llama-bench"
if [[ ! -x "$BENCH" ]]; then
    if [[ ! -d "$LLAMA_DIR/.git" ]]; then
        echo "llama.sh: cloning $LLAMA_REPO into $LLAMA_DIR" >&2
        git clone --depth 1 "$LLAMA_REPO" "$LLAMA_DIR"
    fi
    if [[ -n "$LLAMA_REF" ]]; then
        git -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_REF"
        git -C "$LLAMA_DIR" checkout --detach FETCH_HEAD
    fi
    command -v nvcc >/dev/null || echo "llama.sh: warning: nvcc not on PATH (try PATH=/usr/local/cuda/bin:\$PATH)" >&2
    cmake_args=(-S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF)
    if [[ -n "${CUDA_ARCH:-}" ]]; then cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"); fi
    cmake "${cmake_args[@]}"
    cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)" --target llama-bench
fi

if [[ ! -s "$MODEL" ]]; then
    mkdir -p "$(dirname "$MODEL")"
    echo "llama.sh: downloading $MODEL_URL" >&2
    curl -fL --retry 3 -o "$MODEL.part" "$MODEL_URL"
    mv "$MODEL.part" "$MODEL"
fi

echo "llama.sh: bench=$BENCH model=$MODEL" >&2
if [[ "$SETUP_ONLY" == 1 ]]; then exit 0; fi

if [[ -n "$GPU" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
echo "llama.sh: duty=$DUTY period_ms=$PERIOD_MS seconds=$SECONDS_RUN tokens=$TOKENS" >&2
# The loop runs inside dutycycle.py's process group, so SIGSTOP freezes bash and llama-bench together.
# llama-bench prints tokens/s per repetition (tg column) in markdown; the log is the throughput record.
# shellcheck disable=SC2016  # $0..$2 are expanded by the inner bash, on purpose
exec python3 "$HERE/dutycycle.py" --duty "$DUTY" --period-ms "$PERIOD_MS" --seconds "$SECONDS_RUN" -- \
    bash -c 'while :; do "$0" -m "$1" -ngl 99 -p 0 -n "$2" -r 5; done' "$BENCH" "$MODEL" "$TOKENS"
