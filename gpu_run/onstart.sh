#!/usr/bin/env bash
# vast.ai on-start command. Fetches this branch, runs gpu_run/run.sh, then prints every result into the
# instance log between LOCKSTEP_RESULTS_BEGIN / LOCKSTEP_RESULTS_END, so the results can be collected with
# the vast API ("vastai logs <id>") and no SSH is needed. Nothing is uploaded anywhere else.
#
# On-start command to give vast:
#   bash -c "curl -fsSL https://raw.githubusercontent.com/PierreKhoury1/Final_year_research_MSc/claude/modest-ride-bc8yxa/gpu_run/onstart.sh | bash"
set -u
BRANCH=${BRANCH:-claude/modest-ride-bc8yxa}
REPO=${REPO:-PierreKhoury1/Final_year_research_MSc}
cd /root 2>/dev/null || cd ~
echo "LOCKSTEP_ONSTART_BEGIN $(date -u +%FT%TZ)"
rm -rf lockstep_work && mkdir lockstep_work && cd lockstep_work
URL="https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz"
if command -v curl >/dev/null; then curl -fsSL "$URL" | tar xz
elif command -v wget >/dev/null; then wget -qO- "$URL" | tar xz
else python3 -c "import urllib.request, tarfile, io, sys; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(sys.argv[1]).read()), mode='r:gz').extractall()" "$URL"
fi
DIR=$(ls -d */ 2>/dev/null | head -1)
cd "$DIR/gpu_run" 2>/dev/null || { echo "LOCKSTEP_FETCH_FAILED $URL"; echo "LOCKSTEP_RESULTS_END"; exit 1; }
bash run.sh
OUT=$(ls -d out_* 2>/dev/null | tail -1)
echo "LOCKSTEP_RESULTS_BEGIN"
echo "--- gpu_info.txt"; cat "$OUT/gpu_info.txt"
echo "--- results.jsonl"; cat "$OUT/results.jsonl"
echo "--- pp_cuda_summary.json (without the residual trace)"
python3 -c "import json, sys; d = json.load(open(sys.argv[1])); d.pop('residual', None); print(json.dumps(d))" "$OUT/pp_cuda_summary.json" 2>/dev/null || echo "(none)"
echo "--- results tarball, base64"
base64 -w 76 "$(ls results_*.tgz | tail -1)"
echo "LOCKSTEP_RESULTS_END"
