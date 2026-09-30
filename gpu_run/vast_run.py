#!/usr/bin/env python3
"""Rent one NVIDIA GPU on vast.ai, run gpu_run/onstart.sh on it, collect the results, destroy it.

No SSH is used: the on-start script prints the results into the instance log, and this script reads the
log back through the vast API. Needs the vast CLI (pip install vastai) and the API key in VAST_API_KEY.

    python3 vast_run.py --search                 # only list candidate offers, rent nothing
    python3 vast_run.py                          # rent the cheapest fit, run, collect, destroy
    python3 vast_run.py --gpu "RTX 4090" --max-dph 0.6
    python3 vast_run.py --collect <instance_id>  # fetch results from an instance already running

Results land in gpu_run/vast_results/<instance_id>/: run.log, gpu_info.txt, results.jsonl, the tarball.
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time

REPO = "PierreKhoury1/Final_year_research_MSc"
BRANCH = "claude/modest-ride-bc8yxa"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"   # "devel" = has nvcc
ONSTART = f'bash -c "curl -fsSL https://raw.githubusercontent.com/{REPO}/{BRANCH}/gpu_run/onstart.sh | bash"'
HERE = os.path.dirname(os.path.abspath(__file__))


def vast(*args, raw=True, check=True):
    key = os.environ.get("VAST_API_KEY")
    if not key:
        sys.exit("VAST_API_KEY is not set")
    cmd = ["vastai", *args, "--api-key", key] + (["--raw"] if raw else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        sys.exit(f"vastai {' '.join(args)} failed ({p.returncode}):\n{p.stdout}\n{p.stderr}")
    if not raw:
        return p.stdout
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        sys.exit(f"vastai {' '.join(args)} returned non-JSON:\n{p.stdout[:2000]}\n{p.stderr[:2000]}")


def search(gpu, max_dph, min_cpus):
    names = [gpu] if gpu else ["A100 PCIE", "A100 SXM4", "A100X"]
    q = (f'gpu_name in {json.dumps(names)} num_gpus=1 reliability>0.98 cpu_cores_effective>={min_cpus} '
         f'dph<={max_dph} inet_down>200 rentable=true verified=true')
    offers = vast("search", "offers", q, "-o", "dph")
    if isinstance(offers, dict):
        offers = offers.get("offers", [])
    return offers


def show(offers, n=8):
    print(f"{'id':>10} {'gpu':14} {'$/h':>6} {'cpus':>5} {'reliab':>7} {'down Mb/s':>10} {'cuda':>5}  location")
    for o in offers[:n]:
        print(f"{o['id']:>10} {o.get('gpu_name',''):14} {o.get('dph_total',0):6.3f} {o.get('cpu_cores_effective',0):5.1f} "
              f"{o.get('reliability2', o.get('reliability', 0)):7.4f} {o.get('inet_down',0):10.0f} {o.get('cuda_max_good',0):5}  {o.get('geolocation','')}")


def instance(iid):
    lst = vast("show", "instances")
    if isinstance(lst, dict):
        lst = lst.get("instances", [])
    for i in lst:
        if i.get("id") == iid:
            return i
    return None


def logs(iid, tail=30000):
    return vast("logs", str(iid), "--tail", str(tail), "--full", raw=False, check=False)


def collect(iid, text):
    m = re.search(r"LOCKSTEP_RESULTS_BEGIN\n(.*?)\nLOCKSTEP_RESULTS_END", text, re.S)
    if not m:
        return False
    block = m.group(1)
    out = os.path.join(HERE, "vast_results", str(iid))
    os.makedirs(out, exist_ok=True)
    sections = re.split(r"^--- (.+)$", block, flags=re.M)
    for name, body in zip(sections[1::2], sections[2::2]):
        body = body.strip("\n")
        if name.startswith("results tarball"):
            try:
                with open(os.path.join(out, "results.tgz"), "wb") as f:
                    f.write(base64.b64decode("".join(body.split())))
            except Exception as e:
                print(f"tarball decode failed: {e}")
        else:
            with open(os.path.join(out, name.split(" ")[0]), "w") as f:
                f.write(body + "\n")
    with open(os.path.join(out, "run.log"), "w") as f:
        f.write(text)
    print(f"results saved to {out}")
    res = os.path.join(out, "results.jsonl")
    if os.path.exists(res):
        for line in open(res):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            print(f"\n{r['condition']} | {r['gpu']} | clock bound ±{r['clock_bound_us']} us | globaltimer step {r['globaltimer_step_ns']['median']} ns | fifo={r['sched_fifo']}")
            for k, mth in r["methods"].items():
                fmt = lambda v: f"{v:9.2f}" if isinstance(v, (int, float)) else f"{'n/a':>9}"
                print(f"  {k:26s} median {fmt(mth['median_us'])} p99 {fmt(mth['p99_us'])} p99.9 {fmt(mth['p999_us'])} worst {fmt(mth['max_us'])} us  n={mth['n']}")
    return True


def destroy(iid):
    r = vast("destroy", "instance", str(iid))
    time.sleep(5)
    still = instance(iid)
    print(f"destroyed instance {iid}: {r}; still listed: {bool(still)}")
    if still:
        print("WARNING: instance still listed. Check https://cloud.vast.ai/instances/ and destroy it by hand, it bills until then.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="list candidate offers and exit")
    ap.add_argument("--gpu", default=None, help='exact vast gpu_name, e.g. "A100 PCIE", "RTX 4090"')
    ap.add_argument("--max-dph", type=float, default=2.0, help="refuse offers above this $/hour")
    ap.add_argument("--min-cpus", type=int, default=8)
    ap.add_argument("--offer", type=int, default=None, help="rent this offer id instead of the cheapest fit")
    ap.add_argument("--collect", type=int, default=None, help="instance id to collect from (skips renting)")
    ap.add_argument("--keep", action="store_true", help="do not destroy the instance afterwards")
    ap.add_argument("--timeout-min", type=float, default=30)
    a = ap.parse_args()

    if a.collect:
        iid = a.collect
    else:
        offers = search(a.gpu, a.max_dph, a.min_cpus)
        if not offers:
            sys.exit("no offers match; loosen --max-dph / --min-cpus or pick another --gpu")
        show(offers)
        if a.search:
            return
        pick = next((o for o in offers if o["id"] == a.offer), offers[0]) if a.offer else offers[0]
        print(f"\nrenting offer {pick['id']}: {pick.get('gpu_name')} at ${pick.get('dph_total'):.3f}/h")
        r = vast("create", "instance", str(pick["id"]), "--image", IMAGE, "--disk", "20", "--ssh",
                 "--label", "lockstep-go-nogo", "--cancel-unavail", "--onstart-cmd", ONSTART)
        if not r.get("success"):
            sys.exit(f"create failed: {r}")
        iid = r["new_contract"]
        print(f"instance {iid} created; the run itself takes about 5 minutes once the image has loaded")

    t_end = time.time() + a.timeout_min * 60
    done = False
    last = ""
    try:
        while time.time() < t_end:
            time.sleep(30)
            inst = instance(iid)
            status = (inst or {}).get("actual_status"), (inst or {}).get("status_msg", "")[:80]
            if status != last:
                print(f"[{time.strftime('%H:%M:%S')}] {status}")
                last = status
            if inst and inst.get("actual_status") == "running":
                text = logs(iid)
                if "LOCKSTEP_RESULTS_END" in text:
                    done = collect(iid, text)
                    break
                if "LOCKSTEP_FETCH_FAILED" in text or "BUILD FAILED" in text:
                    print("the run failed on the instance; last log lines:\n" + "\n".join(text.splitlines()[-40:]))
                    break
        if not done and time.time() >= t_end:
            print("timed out waiting; last log lines:\n" + "\n".join(logs(iid).splitlines()[-40:]))
    finally:
        if a.keep:
            print(f"keeping instance {iid} (it bills until destroyed)")
        else:
            destroy(iid)


if __name__ == "__main__":
    main()
