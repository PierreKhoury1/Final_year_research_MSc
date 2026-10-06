"""Build data.json for the instruction-table page from the two final CLI runs: fits (from the analysis) plus
per-(kind, ws, N) quantiles and a log-binned histogram of the raw bracket samples (shared samples only for the
same-SM co-tenant, as the analysis does)."""
import json, os, sys, numpy as np
sys.path.insert(0, "/home/user/Final_year_research_MSc/slotbench")
from analysis.gputrace import Run, KINDS, GPU_DT
S = sys.argv[1] if len(sys.argv) > 1 else "runs"
GPUS = [("A100 SXM4", f"{S}/cli_a100d/out/out"), ("RTX 3060", f"{S}/cli_3060c/out/out")]
COND = [("alone", "instr"), ("cotenant", "instr_cotenant"), ("mps", "instr_mps50")]
EDGES = np.logspace(np.log10(100), np.log10(20000), 61)   # cycles, log bins

def samples(prefix, shared_only):
    run = Run(prefix)
    enter = run.events("LAUNCH_ENTER"); ws_of = {int(k): int(a) for k, a in zip(enter["kernel_id"], enter["a"])}
    r = run.recs[(run.recs["tag"] < 13) & (run.recs["clk_begin"] == 0) & (run.recs["n_iters"] > 0)]
    if shared_only:
        co = run.recs[run.recs["tag"] == 20]
        by = {}
        for c in co: by.setdefault(int(c["smid"]), []).append((int(c["g_begin"]), int(c["g_end"])))
        keep = []
        for rec in r:
            iv = by.get(int(rec["smid"]), []); b, e = int(rec["g_begin"]), int(rec["g_end"])
            keep.append(any(cb <= b and e <= ce for cb, ce in iv))
        r = r[np.array(keep, dtype=bool)]
    out = {}
    for rec in r:
        key = (int(rec["tag"]), ws_of.get(int(rec["kernel_id"]), 0), int(rec["n_iters"]))
        out.setdefault(key, []).append(float(rec["clk_end"]))
    return out, run

data = dict(edges=[round(float(x), 1) for x in EDGES], gpus=[])
for gname, d in GPUS:
    g = dict(name=gname, conds={})
    for cname, run in COND:
        an = json.load(open(f"{d}/{run}.analysis.json")); res = an["result"]
        smp, r = samples(f"{d}/{run}", cname == "cotenant")
        rows = []
        for t in res["table"]:
            kind_id = [k for k, v in KINDS.items() if v == t["kind"]][0]
            perN = {}
            for N in sorted(int(n) for n in t["per_N_median"]):
                v = np.array(smp.get((kind_id, t["ws"], N), []))
                if v.size == 0: continue
                perN[N] = dict(n=int(v.size), p10=float(np.percentile(v, 10)), p25=float(np.percentile(v, 25)), p50=float(np.median(v)),
                               p75=float(np.percentile(v, 75)), p90=float(np.percentile(v, 90)), p99=float(np.percentile(v, 99)),
                               hist=[int(x) for x in np.histogram(v, bins=EDGES)[0]])
            rows.append(dict(kind=t["kind"], ws=t["ws"], latency=t["latency_cycles"], overhead=t["bracket_overhead_cycles"],
                             latency_p10=t["latency_cycles_p10"], ns_at_nmax=t["ns_per_instr_at_Nmax"], ghz=t["sm_ghz"], perN=perN))
        g["conds"][cname] = dict(run=run, bound_ns=an["clock"]["bound_ns"], samples=res["samples"], shared=res["samples_shared"],
                                fit_on=res["fit_on"], sms=int(r.meta.get("sms", 0)), tick_ns=an["clock"]["tick_ns"], rows=rows)
    data["gpus"].append(g)
json.dump(data, open(os.path.join(os.path.dirname(__file__), "data.json"), "w"), separators=(",", ":"))
print("ok", os.path.getsize(os.path.join(os.path.dirname(__file__), "data.json")))

# ---- intra-kernel traces (N = 1 and N = max per row and condition) and co-tenant coverage on the probe's SM
import re
def traces(prefix, cot):
    run = Run(prefix)
    enter = run.events("LAUNCH_ENTER"); ws_of = {int(k): int(a) for k, a in zip(enter["kernel_id"], enter["a"])}
    r = run.recs[(run.recs["tag"] < 13) & (run.recs["clk_begin"] == 0) & (run.recs["n_iters"] > 0)]
    co = run.recs[run.recs["tag"] == 20]
    by = {}
    for c in co: by.setdefault(int(c["smid"]), []).append((int(c["g_begin"]), int(c["g_end"])))
    out = {}
    for kid in sorted(set(int(k) for k in r["kernel_id"])):
        p = r[r["kernel_id"] == kid]; p = p[np.argsort(p["g_begin"])]
        key = (int(p["tag"][0]), ws_of.get(kid, 0), int(p["n_iters"][0]))
        t0 = int(p["g_begin"][0]); sm = int(p["smid"][0])
        iv = by.get(sm, [])
        pts = [[round((int(b) - t0) / 1000, 2), int(c), int(any(cb <= int(b) and int(e) <= ce for cb, ce in iv)) if cot else 1]
               for b, e, c in zip(p["g_begin"], p["g_end"], p["clk_end"])]
        t1 = int(p["g_end"][-1])
        cov = sorted(set((max(0, (cb - t0) / 1000), min((t1 - t0) / 1000, (ce - t0) / 1000)) for cb, ce in iv if ce > t0 and cb < t1))
        out[key] = dict(sm=sm, span_us=round((t1 - t0) / 1000, 2), pts=pts, cov=[[round(a, 2), round(b, 2)] for a, b in cov])
    return out
for g, (gname, d) in zip(data["gpus"], GPUS):
    for cname, run in COND:
        tr = traces(f"{d}/{run}", cname == "cotenant")
        for row in g["conds"][cname]["rows"]:
            kind_id = [k for k, v in KINDS.items() if v == row["kind"]][0]
            Ns = sorted(int(n) for n in row["perN"]); 
            row["trace"] = {str(N): tr[(kind_id, row["ws"], N)] for N in (Ns[0], Ns[-1]) if (kind_id, row["ws"], N) in tr}
json.dump(data, open(os.path.join(os.path.dirname(__file__), "data.json"), "w"), separators=(",", ":"))
print("with traces", os.path.getsize(os.path.join(os.path.dirname(__file__), "data.json")))
