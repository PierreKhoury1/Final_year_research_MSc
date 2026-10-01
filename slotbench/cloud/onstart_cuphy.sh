#!/bin/bash
# vast.ai onstart for NVIDIA Aerial cuPHY: build cuPHY's PUSCH example (no cuPHY-CP/DPDK/DOCA), generate a
# 100 MHz 4-layer 256-QAM PUSCH test vector with the free MATLAB Runtime, build slotbench's adversary, then
# time cuPHY's PUSCH receiver alone and next to co-located AI work under no isolation, MPS and green contexts.
# Results stream back as =====SLOTBENCH-BEGIN cuphy/<name> <sha256>===== blocks (cloud/vast.py collect).
#
# Usage: onstart_cuphy.sh [CONFIG [BRANCH [REPO]]]  (CONFIG unused; kept for cloud/vast.py)
# Image: nvidia/cuda:13.3.0-devel-ubuntu24.04, host driver >= 580, disk >= 100 GB, A100/H100 (ARCHS env).
# Build recipe: aerial-cuda-accelerated-ran cuPHY-CP/container (pinned deps), see slotbench/docs/cuphy.md.
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
SB_BRANCH="${2:-${SB_BRANCH:-claude/optimistic-ptolemy-r42xrh}}"
SB_REPO="${3:-${SB_REPO:-https://github.com/PierreKhoury1/Final_year_research_MSc}}"
ACAR_COMMIT="${ACAR_COMMIT:-main}"
W=/workspace; LOGS=$W/logs; OUT=$W/out
mkdir -p "$W" "$LOGS" "$OUT"
cd "$W"

log() { echo "[cuphy $(date -u +%H:%M:%S)] $*"; }
emit_block() {  # emit_block NAME ROOT FILES... (same format as onstart.sh)
    local name="$1" root="$2" tmp sha
    shift 2
    tmp="$(mktemp)"
    tar -czf "$tmp" -C "$root" "$@" 2>/dev/null || { log "tar failed for $name"; rm -f "$tmp"; return 0; }
    sha="$(sha256sum "$tmp" | cut -d' ' -f1)"
    echo "=====SLOTBENCH-BEGIN $name $sha====="; base64 -w 76 "$tmp"; echo "=====SLOTBENCH-END $name====="
    rm -f "$tmp"
}
finish() {  # emit logs + results, DONE marker, keep the container alive for the controller
    local status="$1" f
    trap - ERR
    for f in "$LOGS"/*; do [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 300000 ] && { tail -c 300000 "$f" > "$f.t"; mv "$f.t" "$f"; }; done
    emit_block "_logs/cuphy" "$W" logs || true
    [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && emit_block "cuphy/results" "$W" out || true
    if [ "$status" = ok ]; then echo "=====SLOTBENCH-DONE====="; else echo "=====SLOTBENCH-DONE status=$status====="; fi
    sleep infinity
}
on_err() { echo "=====SLOTBENCH-ERROR $1 $2====="; tail -n 30 "$LOGS/current.log" 2>/dev/null | sed 's/^/  | /'; finish error; }
trap 'on_err $LINENO "$BASH_COMMAND"' ERR
step() { local name="$1"; shift; log "$name"; : > "$LOGS/current.log"; "$@" >>"$LOGS/current.log" 2>&1; cat "$LOGS/current.log" >> "$LOGS/$name.log"; }

{ date -u; nvidia-smi; nproc; free -g; df -h /; } > "$LOGS/header.txt" 2>&1 || true
head -20 "$LOGS/header.txt"
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCHS="${ARCHS:-${GPU_CC}-real}"
JOBS="${JOBS:-$(( $(nproc) > 8 ? $(nproc) / 2 : 4 ))}"
SMS=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
log "GPU $SMS cc $GPU_CC archs $ARCHS jobs $JOBS"

# ---- 1. apt deps + TensorRT (libcuphy links it unconditionally) ----
TRT=10.14.1.48-1+cuda13.0
step apt bash -c "apt-get update -y && apt-get install -y --no-install-recommends git git-lfs cmake ninja-build \
  pkg-config libhdf5-dev hdf5-tools libyaml-dev python3-pip python3-venv numactl wget unzip ca-certificates curl \
  libnvinfer10=$TRT libnvinfer-headers-dev=$TRT libnvinfer-dev=$TRT && rm -f /usr/lib/x86_64-linux-gnu/libnvinfer_static.a"

# ---- 2. sources: Aerial repo, pinned header/C++ deps, MathDx ----
step clone bash -c "rm -rf $W/acar && git init -q $W/acar && cd $W/acar && git remote add origin \
  https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git && GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 origin $ACAR_COMMIT \
  && GIT_LFS_SKIP_SMUDGE=1 git checkout -q FETCH_HEAD && git rev-parse HEAD"
S=$W/acar
gcmake() { local n=$1 u=$2 c=$3; shift 3
    rm -rf "$W/deps/$n"; git init -q "$W/deps/$n"; git -C "$W/deps/$n" fetch -q --depth 1 "$u" "$c"
    git -C "$W/deps/$n" checkout -q FETCH_HEAD
    if [ "$n" = fmtlog ]; then (cd "$W/deps/$n" && git apply "$S/cuPHY-CP/container/patches/fmtlog.patch" \
        && cp fmtlog.h fmtlog-inl.h /usr/local/include/); fi
    cmake -S "$W/deps/$n" -B "$W/deps/$n/b" -GNinja -DCMAKE_BUILD_TYPE=Release "$@"
    cmake --build "$W/deps/$n/b"; cmake --install "$W/deps/$n/b"; }
deps() {
    gcmake fmt       https://github.com/fmtlib/fmt.git          e69e5f977d458f2650bb346dadf2ad30c5320281 -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DFMT_TEST=OFF -DFMT_DOC=OFF
    gcmake fmtlog    https://github.com/MengRao/fmtlog.git      acd521b1a64480354136a745c511358da1ec7dc5 -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    gcmake gsl-lite  https://github.com/gsl-lite/gsl-lite.git   56dab5ce071c4ca17d3e0dbbda9a94bd5a1cbca1
    gcmake wise_enum https://github.com/quicknir/wise_enum.git  34ac79f7ea2658a148359ce82508cc9301e31dd3
    gcmake CLI11     https://github.com/CLIUtils/CLI11.git      4160d259d961cd393fd8d67590a8c7d210207348 -DCLI11_BUILD_TESTS=OFF -DCLI11_BUILD_EXAMPLES=OFF
    gcmake yaml-cpp  https://github.com/jbeder/yaml-cpp.git     f7320141120f720aecc4c32be25586e7da9eb978 -DYAML_CPP_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    ldconfig
    wget -q https://developer.download.nvidia.com/compute/cuFFTDx/redist/cuFFTDx/cuda13/nvidia-mathdx-26.03.0-cuda13.tar.gz
    tar xzf nvidia-mathdx-26.03.0-cuda13.tar.gz -C /usr/local --strip-components=1
    rm nvidia-mathdx-26.03.0-cuda13.tar.gz
}
step deps deps

# ---- 3. cuPHY PUSCH example via an out-of-tree wrapper project ----
mkdir -p "$W/wrapper"
cat > "$W/wrapper/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
project(cuphy_only LANGUAGES C CXX ASM CUDA)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}/cccl)
set(ENV{cuBB_SDK} ${ACAR_SRC})
set(ENABLE_CUMAC OFF CACHE BOOL "" FORCE)
set(NVIPC_FMTLOG_ENABLE ON)
add_definitions(-DNVIPC_FMTLOG_ENABLE)
add_subdirectory(${ACAR_SRC}/cuPHY cuPHY)
EOF
step cuphy_build bash -c "cmake -S $W/wrapper -B $W/build -GNinja -DACAR_SRC=$S \
  -DCMAKE_TOOLCHAIN_FILE=$S/cuPHY/cmake/toolchains/x86-64 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES='$ARCHS' -DBUILD_DOCS=OFF -DENABLE_TESTS=OFF \
  && cmake --build $W/build --target cuphy_ex_pusch_rx_multi_pipe -- -j$JOBS"
PUSCH=$(find "$W/build" -name cuphy_ex_pusch_rx_multi_pipe -type f -perm -u+x | head -1)
[ -x "$PUSCH" ]
log "built $PUSCH"

# ---- 4. slotbench adversary (same co-located workloads as the slotbench runs) ----
step slotbench bash -c "git clone -q --depth 1 -b $SB_BRANCH $SB_REPO $W/sb && cd $W/sb/slotbench \
  && make SM=$GPU_CC CUDA_HOME=/usr/local/cuda bin/adversary"
ADV=$W/sb/slotbench/bin/adversary

# ---- 5. PUSCH test vector: TC 7304 = 273 PRB (100 MHz @ 30 kHz), 4 layers, 4 Rx, 256-QAM MCS 27 ----
tv() {
    apt-get install -y --no-install-recommends default-jre libxfont2 x11-xkb-utils xkb-data libxcomposite1 libnss3 \
        libxrandr-dev libatk1.0-0 libatk-bridge2.0-0 libx11-xcb-dev libxcb-dri3-0 libxcursor-dev libxdamage-dev \
        libxi-dev libdrm-dev libgbm-dev libasound-dev libcups2-dev libxtst-dev
    wget -q https://ssd.mathworks.com/supportfiles/downloads/R2026a/Release/4/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2026a_Update_4_glnxa64.zip -O mcr.zip
    rm -rf mcr && mkdir mcr && (cd mcr && unzip -q ../mcr.zip && ./install -mode silent -agreeToLicense yes)
    rm -rf mcr mcr.zip
    local whl=aerial_mcore-0.20261.508652.508652-py3-none-any.whl
    mkdir -p "$S/5GModel/aerial_mcore/aerial_pkg/dist"
    (cd "$S" && git lfs install --local && git lfs pull --include="5GModel/aerial_mcore/aerial_pkg/dist/*.whl") \
        || wget -q -O "$S/5GModel/aerial_mcore/aerial_pkg/dist/$whl" \
            "https://media.githubusercontent.com/media/NVIDIA/aerial-cuda-accelerated-ran/main/5GModel/aerial_mcore/aerial_pkg/dist/$whl"
    python3 -m venv "$W/venv"
    "$W/venv/bin/pip" install -q numpy pyyaml h5py "$S"/5GModel/aerial_mcore/aerial_pkg/dist/aerial_mcore-*.whl
    mkdir -p "$W/tv" && cd "$W/tv"
    # shellcheck disable=SC1091
    source "$S/5GModel/aerial_mcore/scripts/setup.sh"
    "$W/venv/bin/python" -c "import aerial_mcore as M, matlab; e = M.initialize(); print(e.testCompGenTV_pusch(matlab.double([7304]), 'genTV', nargout=4))"
    find "$W/tv" -name '*.h5' -exec ls -la {} \;
}
step tv tv
cd "$W"
TV=$(find "$W/tv" -name '*7304*PUSCH*CUPHY*.h5' | head -1)
[ -f "$TV" ]
log "test vector $TV"

# ---- 6. experiments ----
SM_COUNT=$(python3 -c 'import ctypes; c = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so"); n = ctypes.c_int(); c.cudaDeviceGetAttribute(ctypes.byref(n), 16, 0); print(n.value)' || echo 108)  # 16 = cudaDevAttrMultiProcessorCount
HALF_SMS=$(( ${SM_COUNT:-108} / 2 ))
ITERS="${ITERS:-3000}"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log
mkdir -p /tmp/mps /tmp/mps-log

run_case() {  # run_case NAME ADV_WORKLOAD(none|sgemm|llm) MODE(none|mps|gc) [extra pusch flags]
    local name="$1" w="$2" mode="$3" adv_pid="" adv_env=() px=()
    shift 3
    [ "$mode" = gc ] && px+=(--G "$HALF_SMS")
    [ "$mode" = mps ] && adv_env+=(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50)
    if [ "$w" != none ]; then
        env "${adv_env[@]}" "$ADV" --workload "$w" --duty 100 --out "$OUT/$name.adversary.json" > "$OUT/$name.adversary.log" 2>&1 &
        adv_pid=$!
        sleep 15
    fi
    log "case $name: workload $w, isolation $mode"
    timeout 1800 "$PUSCH" -i "$TV" -m 1 -r "$ITERS" "${px[@]}" "$@" --timing-json "$OUT/$name.timing.json" \
        > "$OUT/$name.pusch.txt" 2>&1 || log "case $name: pusch exit $?"
    if [ -n "$adv_pid" ]; then kill -TERM "$adv_pid" 2>/dev/null || true; wait "$adv_pid" 2>/dev/null || true; fi
    grep -iE "mean|p99|max|throughput|crc|bler" "$OUT/$name.pusch.txt" | head -12 | sed "s/^/  [$name] /" || true
}
log "SMs $SM_COUNT, green context size $HALF_SMS, $ITERS iterations per case"
run_case alone       none  none
run_case alone_gc    none  gc
run_case sgemm       sgemm none
run_case llm         llm   none
run_case sgemm_gc    sgemm gc
run_case llm_gc      llm   gc
if nvidia-cuda-mps-control -d >> "$LOGS/mps.log" 2>&1; then
    sleep 2
    run_case alone_mps none  mps
    run_case sgemm_mps sgemm mps
    run_case llm_mps   llm   mps
    echo quit | nvidia-cuda-mps-control >> "$LOGS/mps.log" 2>&1 || true
else
    log "MPS daemon did not start (NOT SUPPORTED in this container)"
fi
cp "$LOGS/header.txt" "$OUT/" || true
echo "$TV" > "$OUT/test_vector.txt"
finish ok
