#!/usr/bin/env bash
# slotbench cloud run, executed as root inside a rented vast.ai container (DESIGN.md section 9).
# Usage: onstart.sh [CONFIG [BRANCH [REPO]]]   (env SB_CONFIG/SB_BRANCH/SB_REPO/SB_TUNE_US/SB_SM as fallbacks)
# Installs build deps, clones the repo, builds, runs selftest + tune + the config's matrix, then prints each
# run's small files as base64 tar.gz between =====SLOTBENCH-BEGIN <name> <sha256>===== and
# =====SLOTBENCH-END <name>===== markers, then =====SLOTBENCH-DONE===== and sleeps (the controller destroys).
# Results are emitted incrementally while the matrix runs, so a time/cost cap still yields finished cells.
# Raw slots.bin stays under /root/sb/repo/slotbench/runs (scp it with --keep).
set -Eeuo pipefail

SB_CONFIG="${1:-${SB_CONFIG:-cloud.toml}}"
SB_BRANCH="${2:-${SB_BRANCH:-claude/optimistic-ptolemy-r42xrh}}"
SB_REPO="${3:-${SB_REPO:-https://github.com/PierreKhoury1/Final_year_research_MSc}}"
SB_TUNE_US="${SB_TUNE_US:-200}"
SB_SM="${SB_SM:-}"
WORK="${SB_WORK:-/root/sb}"   # overridable for local tests
REPO_DIR="$WORK/repo"
SB_DIR="$REPO_DIR/slotbench"
LOGS="$WORK/logs"
EMITTED="$WORK/emitted"
CFG_STEM="${SB_CONFIG%.toml}"
EMITTER_PID=""
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$WORK" "$LOGS" "$EMITTED"

log() { echo "[slotbench $(date -u +%H:%M:%S)] $*"; }

# ---- result emission -------------------------------------------------------------------------
# emit_block NAME ROOT PATH...: tar.gz PATHs (relative to ROOT), print base64 wrapped at 76 columns.
emit_block() {
    local name="$1" root="$2" tmp sha
    shift 2
    tmp="$(mktemp)"
    if ! tar -czf "$tmp" -C "$root" "$@" 2>/dev/null; then
        log "tar failed for $name"
        rm -f "$tmp"
        return 0
    fi
    sha="$(sha256sum "$tmp" | cut -d' ' -f1)"
    { echo "=====SLOTBENCH-BEGIN $name $sha====="; base64 -w 76 "$tmp"; echo "=====SLOTBENCH-END $name====="; } > "$tmp.txt"
    cat "$tmp.txt"   # one writer at a time: the matrix output goes to a file, only emitters print
    rm -f "$tmp" "$tmp.txt"
}

SMALL_FILES=(summary.json meta.json run.json adversary.json telemetry.csv status env.txt)

# emit_run RUNDIR: emit the small files of one run directory (name = <config>/<cell>).
emit_run() {
    local d="$1" rel files=() f
    rel="${d#"$SB_DIR/runs/"}"
    for f in "${SMALL_FILES[@]}"; do
        if [ -f "$d/$f" ]; then files+=("$rel/$f"); fi
    done
    if [ ${#files[@]} -eq 0 ]; then return 0; fi
    emit_block "$rel" "$SB_DIR/runs" "${files[@]}"
    touch "$EMITTED/$(echo "$rel" | tr '/' '_')"
}

run_dirs() { find "$SB_DIR/runs" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort || true; }

emitted() { [ -e "$EMITTED/$(echo "${1#"$SB_DIR/runs/"}" | tr '/' '_')" ]; }

# A cell is finished when its status file exists and its summary (or, without one, the status) is >= 60 s
# old: run_one.sh writes status and then summarizes.
cell_finished() {
    local d="$1" now
    now=$(date +%s)
    [ -f "$d/status" ] || return 1
    if [ -f "$d/summary.json" ]; then
        [ $((now - $(stat -c %Y "$d/summary.json"))) -ge 10 ]
    else
        [ $((now - $(stat -c %Y "$d/status"))) -ge 60 ]
    fi
}

# Background loop; stops (between blocks, never mid-block) when $WORK/emitter.stop appears.
emitter_loop() {
    trap - ERR
    set +e
    local d
    while true; do
        for d in $(run_dirs); do
            [ -e "$WORK/emitter.stop" ] && exit 0
            emitted "$d" && continue
            if cell_finished "$d"; then
                emit_run "$d"
                log "cell done: ${d#"$SB_DIR/runs/"} ($(cat "$d/status" 2>/dev/null))"
            fi
        done
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            [ -e "$WORK/emitter.stop" ] && exit 0
            sleep 2
        done
    done
}

stop_emitter() {
    if [ -n "$EMITTER_PID" ]; then
        touch "$WORK/emitter.stop"
        wait "$EMITTER_PID" 2>/dev/null || true
        EMITTER_PID=""
        rm -f "$WORK/emitter.stop"
    fi
}

# compact human table from summary.json files (key names looked up leniently)
print_table() {
    [ -d "$SB_DIR/runs" ] || return 0
    python3 - "$SB_DIR/runs" <<'PY' || true
import json, os, sys
root = sys.argv[1]
def find(d, names):
    if not isinstance(d, dict):
        return None
    for n in names:
        if n in d and not isinstance(d[n], (dict, list)):
            return d[n]
    for k in ("latency", "latency_us", "stats", "latency_stats"):
        if isinstance(d.get(k), dict):
            v = find(d[k], names)
            if v is not None:
                return v
    return None
def fmt(v, p="%.1f"):
    try:
        return p % float(v)
    except (TypeError, ValueError):
        return "?"
rows = []
for cfg in sorted(os.listdir(root)):
    for cell in sorted(os.listdir(os.path.join(root, cfg)) if os.path.isdir(os.path.join(root, cfg)) else []):
        d = os.path.join(root, cfg, cell)
        if not os.path.isdir(d):
            continue
        st = open(os.path.join(d, "status")).read().strip() if os.path.exists(os.path.join(d, "status")) else "-"
        s = {}
        try:
            s = json.load(open(os.path.join(d, "summary.json")))
        except (OSError, ValueError):
            pass
        rows.append((f"{cfg}/{cell}", fmt(find(s, ["p50_us", "p50"])), fmt(find(s, ["p99.99_us", "p9999_us", "p99_99_us", "p99.99", "p9999", "p99_99"])),
                     fmt(find(s, ["max_us", "max"])), fmt(find(s, ["miss_rate"]), "%.3g"), st[:30]))
print("%-40s %10s %10s %10s %10s  %s" % ("cell", "p50_us", "p99.99_us", "max_us", "miss_rate", "status"))
for r in rows:
    print("%-40s %10s %10s %10s %10s  %s" % r)
PY
}

# emit everything not yet emitted, plus the controller-side logs; then the DONE marker
finish() {
    local status="$1" d f extra=()
    stop_emitter
    if [ -d "$SB_DIR/runs" ] && [ -f "$SB_DIR/analysis/summarize.py" ]; then
        for d in $(run_dirs); do
            if [ -f "$d/slots.bin" ] && [ ! -f "$d/summary.json" ]; then
                (cd "$SB_DIR" && timeout 1800 python3 -m analysis.summarize "$d" >>"$LOGS/summarize.log" 2>&1) \
                    || log "summarize failed for $d (see logs block)"
            fi
        done
    fi
    echo "===== slotbench summary table ====="
    print_table
    for d in $(run_dirs); do
        emitted "$d" || emit_run "$d"
    done
    for f in "$SB_DIR/runs/$CFG_STEM/matrix.json" "$SB_DIR/runs/$CFG_STEM/matrix.log" "$WORK/header.txt"; do
        if [ -f "$f" ]; then cp "$f" "$LOGS/" || true; fi
    done
    for f in "$LOGS"/*; do
        if [ -f "$f" ]; then extra+=("logs/$(basename "$f")"); fi
    done
    # truncate big logs (build output) to their tails so the block stays small
    for f in "${extra[@]}"; do
        if [ "$(stat -c %s "$WORK/$f")" -gt 200000 ]; then
            tail -c 200000 "$WORK/$f" > "$WORK/$f.tmp" && mv "$WORK/$f.tmp" "$WORK/$f"
        fi
    done
    if [ ${#extra[@]} -gt 0 ]; then emit_block "_logs/$CFG_STEM" "$WORK" "${extra[@]}"; fi
    if [ "$status" = ok ]; then
        echo "=====SLOTBENCH-DONE====="
        touch "$WORK/DONE"
    else
        echo "=====SLOTBENCH-DONE status=$status====="
    fi
}

on_error() {
    local rc=$? line="$1" cmd="$2"
    trap - ERR
    set +e
    # inside a subshell / command substitution: just fail it, the main shell reports
    if [ "$BASHPID" != "$$" ]; then exit "$rc"; fi
    echo "=====SLOTBENCH-ERROR $line ${cmd//=/-} (exit $rc)====="
    for f in "$LOGS/apt.log" "$LOGS/build.log"; do
        if [ -f "$f" ]; then echo "--- tail $f"; tail -n 40 "$f"; fi
    done
    finish error
    exec sleep infinity
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
trap 'stop_emitter' EXIT

# ---- 0. container restart: results already complete -> re-emit and idle ------------------------
if [ -f "$WORK/DONE" ]; then
    log "previous run already complete; re-emitting results"
    rm -rf "$EMITTED"; mkdir -p "$EMITTED"
    finish ok
    exec sleep infinity
fi

# ---- 1. header ---------------------------------------------------------------------------------
{
    echo "===== slotbench cloud run ====="
    echo "date_utc:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "config:       $SB_CONFIG   branch: $SB_BRANCH   repo: $SB_REPO"
    echo "container_id: ${CONTAINER_ID:-${VAST_CONTAINERLABEL:-?}}   hostname: $(hostname)"
    echo "gpus:";  nvidia-smi -L 2>&1 | sed 's/^/  /' || true
    echo "driver:       $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
    echo "cpu:          $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//' || true)  (nproc $(nproc))"
    echo "kernel:       $(uname -r)"
    # shellcheck disable=SC1091
    echo "os:           $( (. /etc/os-release && echo "$PRETTY_NAME") 2>/dev/null || true)"
} | tee "$WORK/header.txt"

# ---- 2. packages (only what is missing; retry while another apt holds the lock) ------------------
PKGS=(git build-essential python3 python3-venv python3-numpy python3-scipy python3-matplotlib python3-pandas ca-certificates)
MISSING=()
for p in "${PKGS[@]}"; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || MISSING+=("$p")
done
if [ ${#MISSING[@]} -gt 0 ]; then
    log "apt-get install ${MISSING[*]}"
    ok=0
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        if apt-get -o DPkg::Lock::Timeout=120 update >>"$LOGS/apt.log" 2>&1 &&
           apt-get -o DPkg::Lock::Timeout=120 install -y --no-install-recommends "${MISSING[@]}" >>"$LOGS/apt.log" 2>&1; then
            ok=1; break
        fi
        log "apt attempt $attempt failed; retrying in 15 s"
        sleep 15
    done
    [ "$ok" = 1 ]
else
    log "all packages present"
fi

# ---- 3. clone ----------------------------------------------------------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
    log "git clone -b $SB_BRANCH $SB_REPO"
    git clone --depth 1 -b "$SB_BRANCH" "$SB_REPO" "$REPO_DIR" >>"$LOGS/git.log" 2>&1
else
    log "repo already present (container restart); resuming"
fi
cd "$SB_DIR"
log "commit $(git -C "$REPO_DIR" rev-parse HEAD)"
[ -f "configs/$SB_CONFIG" ]

# ---- 4. build for this GPU ---------------------------------------------------------------------
CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .' || true)"
if [[ "$CC_RAW" =~ ^[0-9]+$ ]]; then SM="$CC_RAW"; else SM="$SB_SM"; fi
[ -n "$SM" ] || { log "cannot determine compute capability (nvidia-smi and SB_SM empty)"; false; }
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
log "make SM=$SM CUDA_HOME=$CUDA_HOME (log: $LOGS/build.log)"
make -j"$(nproc)" SM="$SM" CUDA_HOME="$CUDA_HOME" >>"$LOGS/build.log" 2>&1
if make test >>"$LOGS/host_tests.log" 2>&1; then log "host tests: PASS"; else log "host tests: FAIL (continuing)"; fi

# ---- 5. selftest (failure reported, run continues) ------------------------------------------------
log "slot_driver --selftest"
# (the ERR trap fires on any failing command even under set +e, so failures are absorbed with ||)
SELFTEST_RC=0
./bin/slot_driver --selftest 2>&1 | tee "$LOGS/selftest.txt" || SELFTEST_RC=$?
log "selftest exit code $SELFTEST_RC ($([ "$SELFTEST_RC" = 0 ] && echo PASS || echo FAIL))"

# ---- 5b. debug configs: race detector over the selftest (small slot) ------------------------------
if [[ "$SB_CONFIG" == debug* ]]; then
    SAN="$(command -v compute-sanitizer || echo /usr/local/cuda/bin/compute-sanitizer)"
    log "racecheck: $SAN --tool racecheck slot_driver --selftest --ldpc-cb 1"
    timeout 1800 "$SAN" --tool racecheck --racecheck-report hazard --print-limit 50 \
        ./bin/slot_driver --selftest --ldpc-cb 1 >"$LOGS/racecheck.txt" 2>&1 || true
    grep -E "RACECHECK SUMMARY|hazard|Error|at 0x|in .*k_ldpc" "$LOGS/racecheck.txt" | head -40 || true
fi

# ---- 6. tune the workload size to the idle target -------------------------------------------------
log "slot_driver --tune-us $SB_TUNE_US"
TUNE_RC=0
./bin/slot_driver --tune-us "$SB_TUNE_US" 2>&1 | tee "$LOGS/tune.txt" || TUNE_RC=$?
SIZES="$(grep -oE -- '--subcarriers [0-9]+ --ldpc-cb [0-9]+ --ldpc-iters [0-9]+' "$LOGS/tune.txt" | tail -1 || true)"
mkdir -p "$WORK/configs"
CFG="$WORK/configs/$SB_CONFIG"   # same file name, so run dirs are still runs/<config stem>/
if [ "$TUNE_RC" = 0 ] && [ -n "$SIZES" ]; then
    log "tuned sizes: $SIZES"
    python3 - "configs/$SB_CONFIG" "$CFG" "$SIZES" <<'PY'
import re, sys
src, dst, tuned = sys.argv[1:4]
lines = open(src).read().splitlines()

def merged(old):
    # keep other flags already in sizes, replace --subcarriers/--ldpc-cb/--ldpc-iters with the tuned values
    rest = re.sub(r"--(subcarriers|ldpc-cb|ldpc-iters)\s+\S+", "", old or "")
    return " ".join((rest.split() + tuned.split()))

def line(old):
    return f'sizes = "{merged(old)}"   # set by cloud/onstart.sh from --tune-us'

out, section, done = [], None, False
for ln in lines:
    m = re.match(r"\s*\[([^\]]+)\]\s*(#.*)?$", ln)
    if m:
        if section == "run" and not done:
            out.append(line(""))
            done = True
        section = m.group(1).strip()
        out.append(ln)
        continue
    sm = re.match(r'\s*sizes\s*=\s*"([^"]*)"', ln)
    if section == "run" and sm:
        out.append(line(sm.group(1)))
        done = True
        continue
    out.append(ln)
if not done:
    if section == "run":
        out.append(line(""))
    else:
        out += ["", "[run]", line("")]
open(dst, "w").write("\n".join(out) + "\n")
PY
else
    log "tune failed or printed no flags (rc $TUNE_RC); using the config's sizes unchanged"
    cp "configs/$SB_CONFIG" "$CFG"
fi
cp "$CFG" "$LOGS/config_used.toml"

# ---- 6b. real AI models (configs using real-llm / real-vision): torch venv + weights -------------
if grep -qE '"real-(llm|vision)"' "$CFG"; then
    log "real AI workloads: creating torch venv (log: $LOGS/ai_env.log)"
    AI_VENV="$WORK/aienv"
    if { python3 -m venv "$AI_VENV" && "$AI_VENV/bin/pip" install -q --upgrade pip \
         && "$AI_VENV/bin/pip" install -q torch transformers accelerate ultralytics; } >>"$LOGS/ai_env.log" 2>&1; then
        export SB_AI_PYTHON="$AI_VENV/bin/python"
        for w in real-llm real-vision; do
            if grep -q "\"$w\"" "$CFG"; then
                if "$SB_AI_PYTHON" adversary/real/real_ai.py --workload "$w" --prefetch \
                        --out "$LOGS/prefetch_$w.json" >>"$LOGS/ai_env.log" 2>&1; then
                    log "prefetch $w: ok"
                else
                    log "prefetch $w: FAILED (see ai_env.log)"
                fi
            fi
        done
    else
        log "torch venv install FAILED (see ai_env.log); real-* cells will fail"
    fi
fi

# ---- 7. matrix (output to a file; the emitter prints finished cells meanwhile) --------------------
log "run_matrix $CFG (progress: $LOGS/matrix.out)"
emitter_loop &
EMITTER_PID=$!
MATRIX_RC=0
python3 scripts/run_matrix.py "$CFG" --out-root "$SB_DIR/runs" >"$LOGS/matrix.out" 2>&1 || MATRIX_RC=$?
stop_emitter
log "run_matrix exit code $MATRIX_RC"
tail -n 20 "$LOGS/matrix.out" || true

# ---- 8. summaries, table, blocks, DONE ------------------------------------------------------------
if [ "$MATRIX_RC" = 0 ]; then finish ok; else finish "matrix_rc_$MATRIX_RC"; fi
exec sleep infinity
