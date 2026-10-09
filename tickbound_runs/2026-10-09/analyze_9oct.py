"""Local analysis of the 9 Oct 2026 boxes (after babysit.sh fetched out_a100x8/res.tgz and out_h100/res.tgz).

  A. the multi-GPU instruction-level timeline (multi_fig.py) from out_a100x8/res/multi
  B. reorder E1-E6 per box and toolkit: the reorder SASS gate (re-run with the fixed gate, which accepts sm_90's
     MEMBAR.ALL.CTA + scoped MEMBAR pair as one fence; the original gate JSON is kept as *.orig.json) and each
     experiment's one-line analysis
  C. Tier B CuAssembler edits (A100, CUDA 12.6): legal per checker, output match, cycles
  D. %globaltimer step census (timer_edge_steps_ns of every run's meta, the 8 Oct method)

Usage: python analyze_9oct.py BOX_DIR   (writes BOX_DIR/summary_9oct.json and BOX_DIR/figs/)
"""
import collections, glob, gzip, io, json, os, shutil, subprocess, sys

BOX = os.path.abspath(sys.argv[1])
SRC = os.path.join(BOX, "stage", "tb", "src")
sys.path.insert(0, SRC)
from tickbound.tools import sass_reorder   # noqa: E402

ENV = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8")
OUT = {}


def tb(*args):
    r = subprocess.run([sys.executable, "-m", "tickbound", *args], capture_output=True, text=True, env=ENV, cwd=BOX,
                       encoding="utf-8", errors="replace")
    return (r.stdout + r.stderr).strip()


def extract(name):
    d = os.path.join(BOX, f"out_{name}")
    t = os.path.join(d, "res.tgz")
    if os.path.exists(t) and not os.path.isdir(os.path.join(d, "res")):
        import tarfile
        with tarfile.open(t) as tf:
            tf.extractall(d, filter="data")
    return os.path.join(d, "res")


A100, H100 = extract("a100x8"), extract("h100")

# A. multi-GPU figure
print("== A. multi-GPU timeline")
m = os.path.join(A100, "multi")
if os.path.isdir(m):
    r = subprocess.run([sys.executable, "-I", os.path.join(BOX, "multi_fig.py"), SRC, m, os.path.join(BOX, "figs"), "2000"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print((r.stdout + r.stderr).strip()[-4000:])
else:
    print("no multi directory")

# B. reorder
print("\n== B. reorder E1-E6")
OUT["reorder"] = {}
for box, res in (("A100", A100), ("H100", H100)):
    for v in ("126", "122"):
        d = os.path.join(res, f"reorder{v}")
        if not os.path.isdir(d):
            print(f"{box} CUDA 12.{v[2]}: not run"); continue
        sass = os.path.join(d, "probe.sass.gz")
        gate = None
        if os.path.exists(sass):
            text = gzip.open(sass, "rt", encoding="utf-8", errors="replace").read()
            gate = sass_reorder.gate(text)
            bad = sass_reorder.report(gate) if False else None
        # the gate file each run's analysis reads: characterize -> DIR/reorder_sass.json, run -> PREFIX.reorder_sass.json
        targets = glob.glob(os.path.join(d, "*.reorder_sass.json")) + glob.glob(os.path.join(d, "reorder_sass.json"))
        if gate is not None:
            for t in targets:
                o = t[:-5] + ".orig.json"
                if not os.path.exists(o):
                    shutil.copy(t, o)
                json.dump(gate, open(t, "w"), indent=1, default=str)
        st = gate.get("status") if gate else None
        fails = [k for k, x in (gate or {}).get("results", {}).items() if isinstance(x, dict) and x.get("status") == "FAIL"] if gate else []
        print(f"-- {box} CUDA 12.{v[2]}: gate {st}" + (f" (re-run from probe.sass.gz, {len(targets)} gate file(s) rewritten)" if gate else " (original)"))
        rows = {}
        for p in sorted(glob.glob(os.path.join(d, "reorder_*.json"))):
            if p.endswith((".analysis.json", ".reorder_sass.json", ".orig.json")) or ".reorder_sass" in p:
                continue
            prefix = p[:-5]
            line = tb("analyze", prefix, "--json", prefix + ".analysis.json")
            rows[os.path.basename(prefix)] = line
            print(f"  {os.path.basename(prefix)}: {line[:1200]}")
        OUT["reorder"][f"{box}_12.{v[2]}"] = dict(gate=st, rows=rows)

# C. Tier B
print("\n== C. Tier B CuAssembler edits (A100, CUDA 12.6)")
s = os.path.join(A100, "cuasm_sm80", "summary.json")
if os.path.exists(s):
    S = json.load(open(s))
    OUT["tierB"] = S
    agree = 0
    for e in S["edits"]:
        legal, match = e.get("legal_per_checker"), e.get("match")
        agree += (legal == match)
        print(f"  {e['id']:28s} {e['group']:10s} legal={legal!s:5s} match={match!s:5s} "
              f"mismatching={e.get('mismatching_launches')}/{e.get('launches_compared')} p50={e.get('cycles_p50')} "
              f"{(e.get('load_error') or e.get('launch_error') or '')[:80]}")
    print(f"  checker verdict == hardware outcome for {agree}/{len(S['edits'])} edits")
else:
    print("no summary.json")

# D. timer step census
print("\n== D. %globaltimer step census (timer_edge_steps_ns over every run's meta)")
OUT["census"] = {}
for box, res in (("A100", A100), ("H100", H100)):
    c = collections.Counter(); n = 0
    for f in glob.glob(os.path.join(res, "**", "*.json"), recursive=True):
        b = os.path.basename(f)
        if b.endswith((".analysis.json", ".trace.json", ".orig.json")) or "sass" in b or "phases" in b or b == "summary.json":
            continue
        try:
            mm = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        st = mm.get("timer_edge_steps_ns") if isinstance(mm, dict) else None
        if st:
            n += 1; c.update(st if isinstance(st, list) else [st])
    OUT["census"][box] = dict(runs=n, steps=dict(sorted(c.items())))
    print(f"  {box}: {n} runs, steps {dict(sorted(c.items()))}")

json.dump(OUT, open(os.path.join(BOX, "summary_9oct.json"), "w"), indent=1, default=str)
print("\nwrote", os.path.join(BOX, "summary_9oct.json"))
