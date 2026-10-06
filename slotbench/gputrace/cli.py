#!/usr/bin/env python3
"""gputrace: characterise a GPU and the flow of execution on it, from one command.

  gputrace characterize [--profile quick|full|sharing|instr|multi] [--out DIR] [--gpu N] [--mps auto|on|off]
      detect GPUs, build the probes for this architecture, run the profile, analyse, verify brackets in SASS,
      export every run to the trace format, build the timeline viewer, write report.md and summary.json
  gputrace run --strategy S [args...] [--out PREFIX]      one strategy (passes through to the binary)
  gputrace analyze PREFIX [--json OUT]                     analyse one run
  gputrace export PREFIX [--out FILE]                      Chrome/Perfetto trace of one run
  gputrace viewer OUT.html PREFIX[:Name] ...               the interactive timeline with these runs embedded
  gputrace sass [BINARY]                                   verify the instruction brackets in the compiled SASS

Every host-vs-GPU number carries the run's hard clock bound (tick-edge sync before and after each run).
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # slotbench/
sys.path.insert(0, ROOT)
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
BIN_DIR = os.path.join(ROOT, "bin")

# profile -> list of (name, args). Names are the run prefixes under --out.
PROFILES = {
    "quick": [
        ("launch", "--strategy launch --iters 1000"),
        ("launch_idle2000", "--strategy launch --iters 300 --idle-us 2000"),
        ("launch_spin2000", "--strategy launch --iters 300 --idle-us 2000 --idle-spin 1"),
        ("notify", "--strategy notify --iters 600 --dur-us 20"),
        ("dispatch", "--strategy dispatch --dur-us 200 --reps 3"),
        ("concurrency_a2000_prio", "--strategy concurrency --dur-us 2000 --dur-b-us 200 --offset-us 100 --reps 3 --priority 1"),
        ("copy", "--strategy copy --iters 300"),
        ("memory", "--strategy memory --reps 2 --hops 65536 --ws 16k,64k,128k,256k,1m,4m,16m,24m,48m,128m"),
        ("instr", "--strategy instr --reps 64"),
        ("timeslice", "--strategy timeslice --seconds 6 --gap-us 20 --hog-dur-us 5000"),
    ],
    "full": [
        ("launch", "--strategy launch --iters 2000"),
        ("launch_idle100", "--strategy launch --iters 2000 --idle-us 100"),
        ("launch_idle2000", "--strategy launch --iters 1000 --idle-us 2000"),
        ("launch_idle50000", "--strategy launch --iters 200 --idle-us 50000"),
        ("launch_spin2000", "--strategy launch --iters 1000 --idle-us 2000 --idle-spin 1"),
        ("launch_depth8", "--strategy launch --iters 500 --depth 8"),
        ("launch_graph", "--strategy launch --iters 2000 --graph 1"),
        ("launch_graph8", "--strategy launch --iters 500 --depth 8 --graph 1"),
        ("notify", "--strategy notify --iters 2000 --dur-us 20"),
        ("dispatch", "--strategy dispatch --dur-us 200 --reps 5"),
        ("dispatch_t64", "--strategy dispatch --dur-us 200 --reps 3 --threads 64 --blocks 1,sm,2sm,8sm"),
        ("dispatch_busy", "--strategy dispatch --dur-us 200 --reps 3 --spin-mode 1 --spin-param 32 --blocks sm,2sm,8sm,32sm"),
        ("concurrency_a200", "--strategy concurrency --dur-us 200 --dur-b-us 200 --offset-us 100 --reps 5"),
        ("concurrency_a2000", "--strategy concurrency --dur-us 2000 --dur-b-us 200 --offset-us 100 --reps 5"),
        ("concurrency_a2000_prio", "--strategy concurrency --dur-us 2000 --dur-b-us 200 --offset-us 100 --reps 5 --priority 1"),
        ("coexist_half", "--strategy concurrency --blocks-a hsm --dur-us 50000 --dur-b-us 100 --offset-us 1000 --reps 3 --blocks-b 1"),
        ("clocks", "--strategy clocks --seconds 4 --sample-us 100"),
        ("ramp_idle50000", "--strategy ramp --idle-us 50000 --dur-us 3000 --sample-us 20 --reps 40"),
        ("copy", "--strategy copy --iters 1000"),
        ("memory", "--strategy memory --reps 3 --hops 65536"),
        ("memory_cotenant", "--strategy memory --reps 3 --hops 65536 --cotenant 1 --cotenant-ms 50"),
        ("instr", "--strategy instr --reps 256"),
        ("instr_cotenant", "--strategy instr --reps 256 --cotenant 1 --cotenant-ms 50"),
        ("timeslice", "--strategy timeslice --seconds 8 --gap-us 20 --hog-dur-us 5000"),
        ("memory_coproc", "--strategy memory --reps 3 --hops 65536 --cotenant 2 --cotenant-ms 50 --seconds 300"),
    ],
    "sharing": [
        ("memory", "--strategy memory --reps 3 --hops 65536"),
        ("memory_cotenant", "--strategy memory --reps 3 --hops 65536 --cotenant 1 --cotenant-ms 50"),
        ("memory_coproc", "--strategy memory --reps 3 --hops 65536 --cotenant 2 --cotenant-ms 50 --seconds 300"),
        ("instr", "--strategy instr --reps 256"),
        ("instr_cotenant", "--strategy instr --reps 256 --cotenant 1 --cotenant-ms 50"),
        ("concurrency_a2000_prio", "--strategy concurrency --dur-us 2000 --dur-b-us 200 --offset-us 100 --reps 5 --priority 1"),
        ("timeslice", "--strategy timeslice --seconds 8 --gap-us 20 --hog-dur-us 5000"),
    ],
    "multi": [
        ("gpus", "--strategy gpus --reps 3 --sync-rounds 3 --sync-per-phase 1000"),
    ],
    "instr": [
        ("instr", "--strategy instr --reps 256"),
        ("instr_cotenant", "--strategy instr --reps 256 --cotenant 1 --cotenant-ms 50"),
    ],
}
# runs that need the MPS daemon: the MPS variant of a run is added when the profile has the base run (name before
# _mps50) and MPS is available (--mps auto) or requested (--mps on)
MPS_RUNS = [
    ("timeslice_mps50", "--strategy timeslice --seconds 8 --gap-us 20 --hog-dur-us 5000 --hog-mps-pct 50"),
    ("memory_mps50", "--strategy memory --reps 3 --hops 65536 --cotenant 2 --cotenant-ms 50 --seconds 300 --hog-mps-pct 50"),
    ("instr_mps50", "--strategy instr --reps 256 --cotenant 2 --cotenant-ms 50 --seconds 300 --hog-mps-pct 50"),
]
NCCL_RUN = ("nccl", "--iters 200 --sizes 8,4096,65536,1048576,16777216,134217728")


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)


def gpus():
    r = sh(["nvidia-smi", "--query-gpu=index,name,compute_cap,driver_version,persistence_mode", "--format=csv,noheader"])
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.strip().splitlines():
        idx, name, cc, drv, pm = [x.strip() for x in line.split(",")]
        out.append(dict(index=int(idx), name=name, cc=cc, sm=cc.replace(".", ""), driver=drv, persistence=pm))
    return out


def build(sm, nccl=False, force=False):
    """Compile the probes for this architecture if the binary is older than the sources."""
    os.makedirs(BIN_DIR, exist_ok=True)
    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
    srcs = [os.path.join(HERE, f) for f in ("gputrace.cu", "gputrace.cuh", "gputrace.h", "clocksync.cuh", "instr.cuh")]
    out = os.path.join(BIN_DIR, f"gputrace_sm{sm}")
    if force or not os.path.exists(out) or max(os.path.getmtime(s) for s in srcs) > os.path.getmtime(out):
        print(f"building {os.path.basename(out)} ...", flush=True)
        r = sh([nvcc, "-O2", "-lineinfo", "-std=c++17", f"-arch=sm_{sm}", "-I" + os.path.join(ROOT, "common"), "-I" + HERE,
                "-o", out, os.path.join(HERE, "gputrace.cu"), "-lpthread"])
        if r.returncode != 0:
            sys.exit("build failed:\n" + r.stderr[-3000:])
    outn = None
    if nccl and os.path.exists("/usr/include/nccl.h"):
        outn = os.path.join(BIN_DIR, f"gputrace_nccl_sm{sm}")
        if force or not os.path.exists(outn) or os.path.getmtime(os.path.join(HERE, "gputrace_nccl.cu")) > os.path.getmtime(outn):
            print(f"building {os.path.basename(outn)} ...", flush=True)
            r = sh([nvcc, "-O2", "-lineinfo", "-std=c++17", f"-arch=sm_{sm}", "-I" + os.path.join(ROOT, "common"), "-I" + HERE,
                    "-o", outn, os.path.join(HERE, "gputrace_nccl.cu"), "-lnccl", "-lpthread"])
            if r.returncode != 0:
                print("gputrace_nccl build failed (NCCL runs skipped):", r.stderr[-800:])
                outn = None
    return out, outn


class Mps:
    """Start a private MPS daemon for the sharing runs and stop it afterwards."""
    def __init__(self, workdir):
        self.pipe = os.path.join(workdir, "mps-pipe"); self.log = os.path.join(workdir, "mps-log"); self.on = False
    def available(self):
        return shutil.which("nvidia-cuda-mps-control") is not None and sh("pgrep -f 'nvidia-cuda-mps-(control|server)'").returncode != 0
    def start(self):
        os.makedirs(self.pipe, exist_ok=True); os.makedirs(self.log, exist_ok=True)
        env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=self.pipe, CUDA_MPS_LOG_DIRECTORY=self.log)
        r = subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, capture_output=True, text=True, timeout=30)
        self.on = r.returncode == 0 and os.path.exists(os.path.join(self.pipe, "control"))
        return self.on
    def env(self):
        return dict(CUDA_MPS_PIPE_DIRECTORY=self.pipe, CUDA_MPS_LOG_DIRECTORY=self.log) if self.on else {}
    def stop(self):
        if self.on:
            subprocess.run("echo quit | nvidia-cuda-mps-control", shell=True, env=dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=self.pipe),
                           capture_output=True, timeout=30)
            self.on = False


def run_one(binary, name, args, out_dir, gpu, core, clock_core, extra_env=None, log=print):
    prefix = os.path.join(out_dir, name)
    cmd = [binary, "--out", prefix, "--gpu", str(gpu), "--core", str(core), "--clock-core", str(clock_core)] + args.split()
    if os.path.basename(binary).startswith("gputrace_nccl"):
        cmd = [binary, "--out", prefix, "--core", str(core), "--clock-core", str(clock_core)] + args.split()
    env = dict(os.environ, **(extra_env or {}))
    t0 = time.time()
    with open(prefix + ".log", "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    if r.returncode != 0:
        tail = open(prefix + ".log").read()[-400:]
        log(f"  {name}: FAILED ({dt:.0f} s): {tail.strip().splitlines()[-1] if tail.strip() else ''}")
        return None
    from analysis.gputrace import analyse, one_line
    try:
        res = analyse(prefix)
        json.dump(res, open(prefix + ".analysis.json", "w"), indent=1, default=float)
        log(f"  {name} ({dt:.0f} s): {one_line(res)}")
        return res
    except Exception as e:   # keep going: the raw run is on disk
        log(f"  {name}: analysis failed: {e}")
        return None


def write_report(out_dir, info, results, sass, viewer_path):
    from analysis.gputrace import one_line
    lines = [f"# gputrace characterisation: {info['gpu']['name']} ({info['host']}, {info['date']})", ""]
    lines += [f"GPU {info['gpu']['index']}: {info['gpu']['name']}, sm_{info['gpu']['sm']}, driver {info['gpu']['driver']}, "
              f"persistence {info['gpu']['persistence']}; {info['n_gpus']} GPU(s) on the host; profile `{info['profile']}`; "
              f"MPS {'used' if info['mps'] else 'not used'}; {info['minutes']:.1f} min of runs.", ""]
    bounds = [r["clock"].get("bound_ns") for r in results.values() if r and r.get("clock", {}).get("bound_ns")]
    if bounds:
        lines += [f"Host<->GPU clock bound (tick-edge sync before and after every run): {min(bounds):.0f}-{max(bounds):.0f} ns, "
                  f"feasible in {sum(1 for r in results.values() if r and r['clock'].get('feasible'))} of {len(bounds)} runs.", ""]
    lines += ["## Results, one line per run", ""]
    for name, r in results.items():
        lines.append(f"- `{name}`: " + (one_line(r) if r else "failed"))
    lines += [""]
    def res(name):   # analysis result, or None when the run failed or lacked clock files
        r = results.get(name)
        return r.get("result") if r else None
    if res("instr") and res("instr").get("table"):
        lines += ["## Instruction table (cycles per instruction = slope over chain length; alone" +
                  (" / same-SM co-tenant" if res("instr_cotenant") else "") + (" / MPS 50 %" if res("instr_mps50") else "") + ")", "",
                  "| instruction | ws | " + " | ".join(n for n in ("alone", "co-tenant", "MPS 50 %") if n == "alone" or res({"co-tenant": "instr_cotenant", "MPS 50 %": "instr_mps50"}[n])) + " |",
                  "|---|---|" + "---|" * sum(1 for n in ("instr", "instr_cotenant", "instr_mps50") if res(n))]
        cols = [n for n in ("instr", "instr_cotenant", "instr_mps50") if res(n)]
        for t in res("instr")["table"]:
            ws = f"{t['ws'] >> 10} KB" if 0 < t["ws"] < (1 << 20) else (f"{t['ws'] >> 20} MB" if t["ws"] else "")
            cells = []
            for c in cols:
                x = [y for y in res(c).get("table", []) if (y["kind"], y["ws"]) == (t["kind"], t["ws"])]
                cells.append(f"{x[0]['latency_cycles']:.1f}" if x else "-")
            lines.append(f"| {t['kind']} | {ws} | " + " | ".join(cells) + " |")
        lines += [""]
    if sass:
        ok = sum(1 for v in sass.values() if v["verified"])
        bad = sorted(set(v["kind"] for v in sass.values() if not v["verified"]))
        lines += [f"SASS verification of the instruction brackets: {ok}/{len(sass)} kernels have exactly N target opcodes between the clock reads"
                  + (f"; not verified: {', '.join(bad)}" if bad else "") + ".", ""]
    mem = [n for n in results if n.startswith("memory") and res(n) and res(n).get("modifiers")]
    if "memory" in mem:
        lines += ["## Memory hierarchy (cycles per dependent load, .ca, p50)", "", "| working set | " +
                  " | ".join(n.replace("memory_", "") or "alone" for n in mem) + " |", "|---|" + "---|" * len(mem)]
        curves = {n: {c["ws"]: c for c in res(n)["modifiers"].get(".ca", {}).get("curve", [])} for n in mem}
        for ws in sorted(curves["memory"]):
            lab = f"{ws >> 10} KB" if ws < (1 << 20) else f"{ws >> 20} MB"
            lines.append(f"| {lab} | " + " | ".join(f"{curves[n][ws]['cycles']['p50']:.0f}" if ws in curves[n] else "-" for n in curves) + " |")
        lines += [""]
    lines += ["## Files", "", f"- `summary.json`: every run's analysis", f"- `{os.path.basename(viewer_path)}`: interactive timeline (open in a browser)",
              "- `*.trace.json`: per-run Chrome trace format (https://ui.perfetto.dev)", "- `*.gpu.bin`, `*.host.bin`, `*.json`: raw records", ""]
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines))


def cmd_characterize(a):
    gs = gpus()
    if not gs:
        sys.exit("no NVIDIA GPU found (nvidia-smi failed)")
    g = next((x for x in gs if x["index"] == a.gpu), gs[0])
    out_dir = a.out or f"gputrace_{g['name'].replace(' ', '_')}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(out_dir, exist_ok=True)
    binary, nccl_bin = build(g["sm"], nccl=(len(gs) > 1 and a.profile in ("multi", "full")), force=a.rebuild)
    ncpu = os.cpu_count() or 2
    core, clock_core = (2, 3) if ncpu > 4 else (0, 1 if ncpu > 1 else 0)
    print(f"GPU {g['index']}: {g['name']} (sm_{g['sm']}), {len(gs)} GPU(s), profile {a.profile}, output {out_dir}/")
    runs = list(PROFILES[a.profile])
    if len(gs) > 1 and a.profile in ("full", "multi") and ("gpus", PROFILES["multi"][0][1]) not in runs:
        runs += PROFILES["multi"]
    mps = Mps(out_dir)
    mps_runs = [(n, x) for n, x in MPS_RUNS if any(n.rsplit("_mps", 1)[0] == b for b, _ in runs)]
    use_mps = bool(mps_runs) and (a.mps == "on" or (a.mps == "auto" and mps.available()))
    t0 = time.time()
    results = {}
    for name, args in runs:
        results[name] = run_one(binary, name, args, out_dir, g["index"], core, clock_core)
    if nccl_bin and len(gs) > 1:
        results[NCCL_RUN[0]] = run_one(nccl_bin, NCCL_RUN[0], NCCL_RUN[1], out_dir, g["index"], core, clock_core)
    if use_mps:
        if mps.start():
            print("MPS daemon started for the partitioned runs")
            try:
                for name, args in mps_runs:
                    results[name] = run_one(binary, name, args, out_dir, g["index"], core, clock_core, extra_env=mps.env())
            finally:
                mps.stop()
        else:
            print("MPS daemon could not be started; partitioned runs skipped")
    minutes = (time.time() - t0) / 60
    # SASS verification of the instruction brackets
    sass = None
    try:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import sass_check
        cuobjdump = os.path.join(CUDA_HOME, "bin", "cuobjdump")
        text = sh([cuobjdump, "-sass", binary]).stdout
        sass = sass_check.check(sass_check.parse(text)) if text else None
        if sass:
            json.dump(sass, open(os.path.join(out_dir, "sass_check.json"), "w"), indent=1)
    except Exception as e:
        print("sass check skipped:", e)
    # traces and viewer
    from analysis.gputrace_export import export
    specs = []
    for name, r in results.items():
        if not r:
            continue
        try:
            tr = export(os.path.join(out_dir, name), max_kernels=200)
            path = os.path.join(out_dir, name + ".trace.json")
            json.dump(tr, open(path, "w"), separators=(",", ":"))
            specs.append((path, f"{g['name']} · {name}", one_line_safe(r)))   # every run goes into the viewer (first 8)
        except Exception as e:
            print(f"  export {name} skipped: {e}")
    viewer = os.path.join(out_dir, "timeline.html")
    try:
        from analysis.gputrace_compact import compact
        ds = [compact(p, n, note) for p, n, note in specs[:8]]
        t = open(os.path.join(ROOT, "tools", "timeline", "template.html")).read()
        open(viewer, "w").write(t.replace("/*DATA*/[]", json.dumps(ds, separators=(",", ":"))))
    except Exception as e:
        print("viewer skipped:", e)
    info = dict(gpu=g, n_gpus=len(gs), host=os.uname().nodename, date=f"{datetime.datetime.now():%Y-%m-%d %H:%M}", profile=a.profile,
                mps=bool(use_mps and any(n in results for n, _ in MPS_RUNS)), minutes=minutes)
    json.dump(dict(info=info, results=results), open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=float)
    write_report(out_dir, info, results, sass, viewer)
    print(f"\n{out_dir}/report.md written; {sum(1 for r in results.values() if r)}/{len(results)} runs analysed in {minutes:.1f} min")


def one_line_safe(r):
    from analysis.gputrace import one_line
    try:
        return one_line(r)
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(prog="gputrace", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("characterize"); c.add_argument("--profile", default="quick", choices=sorted(PROFILES))
    c.add_argument("--out"); c.add_argument("--gpu", type=int, default=0); c.add_argument("--mps", default="auto", choices=["auto", "on", "off"])
    c.add_argument("--rebuild", action="store_true")
    r = sub.add_parser("run"); r.add_argument("args", nargs=argparse.REMAINDER)
    an = sub.add_parser("analyze"); an.add_argument("prefix"); an.add_argument("--json")
    ex = sub.add_parser("export"); ex.add_argument("prefix"); ex.add_argument("--out")
    vw = sub.add_parser("viewer"); vw.add_argument("out"); vw.add_argument("runs", nargs="+")
    sa = sub.add_parser("sass"); sa.add_argument("binary", nargs="?")
    a = ap.parse_args()
    if a.cmd == "characterize":
        cmd_characterize(a)
    elif a.cmd == "run":
        gs = gpus(); sm = gs[0]["sm"] if gs else "80"
        binary, _ = build(sm)
        sys.exit(subprocess.call([binary] + a.args))
    elif a.cmd == "analyze":
        from analysis.gputrace import analyse, one_line
        res = analyse(a.prefix)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1, default=float)
        print(one_line(res))
    elif a.cmd == "export":
        from analysis.gputrace_export import export
        tr = export(a.prefix); path = a.out or a.prefix + ".trace.json"
        json.dump(tr, open(path, "w"), separators=(",", ":")); print(path)
    elif a.cmd == "viewer":
        from analysis.gputrace_compact import compact
        from analysis.gputrace_export import export
        ds = []
        for spec in a.runs:
            prefix, _, name = spec.partition(":")
            tr = export(prefix, max_kernels=0)
            tmp = prefix + ".trace.json"; json.dump(tr, open(tmp, "w"), separators=(",", ":"))
            ds.append(compact(tmp, name or os.path.basename(prefix)))
        t = open(os.path.join(ROOT, "tools", "timeline", "template.html")).read()
        open(a.out, "w").write(t.replace("/*DATA*/[]", json.dumps(ds, separators=(",", ":")))); print(a.out, len(ds), "runs")
    elif a.cmd == "sass":
        sys.path.insert(0, os.path.join(ROOT, "tools")); import sass_check
        binary = a.binary or sorted(os.path.join(BIN_DIR, f) for f in os.listdir(BIN_DIR) if f.startswith("gputrace_sm"))[-1]
        sys.argv = ["sass_check", binary]; sass_check.main()


if __name__ == "__main__":
    main()
