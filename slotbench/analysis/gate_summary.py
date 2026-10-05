#!/usr/bin/env python3
"""Time-aware GPU gating campaign: busy-time calibration and summary (no GPU needed).

  gate_summary.py --busy OUT/calib_alone [--margin-us 30] [--period-us 500]
      prints the reserved busy time per slot: p99.9 of completion latency (from the slot target) over
      slots launched on time (host launch stalls excluded), plus the margin. Exits 2 if that leaves less
      than 100 us of the period free, or if there are too few on-time slots.

  gate_summary.py OUT [--units-dir DIR] [--json FILE]
      one row per arm (sharing x gate mode x GEMM size, pooled over repeats). Cases enter the pool only
      if cuPHY passed its checks AND (for contended cases) the tenant finished OK with the timetable loaded.
      5G side, re-scored from the raw records: misses split by cause at 500/300/250 us budgets. A miss is
      host-caused if its slot's launch call returned > 50 us after the target (or was skipped behind such a slot), else GPU-caused.
      Overlap: fraction of slots whose GPU execution [g0, g1] (mapped to host time with the run's two-point
      clock fit) intersects a tenant GEMM as the host saw it (issue -> completion; a superset of its GPU time,
      so overlap is an upper bound). AI side: GEMMs/s inside the slot window divided by the same GEMM size
      running alone under the same sharing setting (ai_alone_<sharing>_n<N>.json).
"""
import argparse
import bisect
import glob
import json
import math
import os
import re
import struct
import sys

RECORD = struct.Struct("<Qq6Q")   # slot, t_target, g_target, g_launch, g_launch_done, g0, g1, flags
UNIT = struct.Struct("<qq")       # tenant unit: t_start, t_done (host CLOCK_MONOTONIC_RAW ns)
LATE_US = 50.0
BUDGETS = (500, 300, 250)
CASE_RE = re.compile(r"^(proc|mps\d*)_(observe|gated)_n(\d+)_r(\d+)$")
ALONE_RE = re.compile(r"^alone_(proc|mps)_r(\d+)$")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


class Case:
    """cuPHY raw records of one case, with GPU times mapped to the host clock."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.meta = load_json(prefix + ".json")
        with open(prefix + ".bin", "rb") as f:
            raw = f.read()
        pre, post = self.meta["clock_fit_pre"], self.meta["clock_fit_post"]
        self.ha = int(pre["t_ref"]) + round(float(pre["b_ns"]))
        hb = int(post["t_ref"]) + round(float(post["b_ns"]))
        self.ga, gb = int(pre["g_ref"]), int(post["g_ref"])
        if hb - self.ha <= 1_000_000 or gb <= self.ga:
            raise ValueError("invalid clock anchors")
        self.rate = (gb - self.ga) / (hb - self.ha)   # GPU ns per host ns
        self.rows = []
        for slot, t, g_t, g_l, g_ld, g0, g1, fl in RECORD.iter_unpack(raw):
            if fl & 1:
                self.rows.append(dict(skip=True))
                continue
            g_true = self.ga + round((t - self.ha) * self.rate)   # integer, as in cuphy_lockstep.cu
            # "late" = the launch call returned > LATE_US after the target (stalls inside cudaGraphLaunch count)
            self.rows.append(dict(skip=False, late_us=(g_ld - g_t) / 1e3, lat_us=(g1 - g_true) / 1e3,
                                  exec_us=(g1 - g0) / 1e3, h0=self.host(g0), h1=self.host(g1)))

    def host(self, g):
        return self.ha + (g - self.ga) / self.rate

    def misses(self, budget_us):
        host = gpu = 0
        last = "gpu"
        for r in self.rows:
            if r["skip"]:   # blocked by the previous slot: inherits that slot's cause
                host += last == "host"
                gpu += last == "gpu"
                continue
            last = "host" if r["late_us"] > LATE_US else "gpu"
            if r["lat_us"] > budget_us:
                host += last == "host"
                gpu += last == "gpu"
        return host, gpu

    def overlap(self, units):
        """(slots whose GPU execution intersects a tenant unit, executed slots)."""
        starts = [u[0] for u in units]
        hit = n = 0
        for r in self.rows:
            if r["skip"]:
                continue
            n += 1
            i = bisect.bisect_right(starts, r["h1"])   # units starting before the slot ends
            j = i - 1
            while j >= 0 and units[j][0] >= r["h0"] - 50_000_000:   # units are short; bound the scan
                if units[j][1] > r["h0"]:
                    hit += 1
                    break
                j -= 1
        return hit, n


def busy_us(prefix, margin_us, period_us):
    c = Case(prefix)
    on_time = [r["lat_us"] for r in c.rows if not r["skip"] and r["late_us"] <= LATE_US]
    if len(on_time) < 1000:
        raise SystemExit(f"busy: only {len(on_time)} on-time slots in {prefix}")
    b = int(math.ceil(pct(on_time, 0.999) + margin_us))
    if b > period_us - 100:
        raise SystemExit(f"busy: {b} us leaves less than 100 us of the {period_us} us period")
    return b


def read_units(path):
    with open(path, "rb") as f:
        raw = f.read()
    return sorted(UNIT.iter_unpack(raw[: len(raw) // UNIT.size * UNIT.size]))


def summarise(out, units_dir=None):
    problems = []
    ai_alone = {}
    for p in glob.glob(os.path.join(out, "ai_alone_*_n*.json")):
        m = re.search(r"ai_alone_(proc|mps\d*)_n(\d+)\.json$", p)
        d = load_json(p)
        if m and d and d.get("ok") and d.get("units_per_s"):
            ai_alone[(m.group(1), int(m.group(2)))] = d["units_per_s"]
        elif m:
            problems.append(f"{os.path.basename(p)}: tenant-alone run missing or not ok")
    arms, runs = {}, []
    for jp in sorted(glob.glob(os.path.join(out, "*.json"))):
        name = os.path.basename(jp)[:-5]
        m, a = CASE_RE.match(name), ALONE_RE.match(name)
        if not (m or a):
            continue
        prefix = jp[:-5]
        d = load_json(jp)
        chk = (open(prefix + ".check.log").read().strip().splitlines() or ["{}"])[-1] \
            if os.path.exists(prefix + ".check.log") else "{}"
        try:
            chk_ok = json.loads(chk).get("valid") is True
        except ValueError:
            chk_ok = False
        run = load_json(prefix + ".run.json") or {}
        if not d or d.get("ok") is not True or not chk_ok or run.get("exit_code") != 0 \
                or os.path.exists(prefix + ".skipped") or not os.path.exists(prefix + ".bin"):
            problems.append(f"{name}: cuPHY case invalid (ok={d and d.get('ok')}, check={chk_ok}, "
                            f"exit={run.get('exit_code')}); excluded")
            continue
        gate = None
        if m:
            adv = load_json(prefix + ".adversary.json") or {}
            gate = adv.get("gate") or {}
            if adv.get("ok") is not True or gate.get("timetable_loaded") is not True:
                problems.append(f"{name}: tenant not ok or timetable not loaded; excluded")
                continue
            key = (m.group(1), m.group(2), int(m.group(3)))
            rep = int(m.group(4))
        else:
            key, rep = (a.group(1), "alone", 0), int(a.group(2))
        c = Case(prefix)
        row = dict(name=name, sharing=key[0], gate=key[1], n=key[2], rep=rep, slots=len(c.rows),
                   exec=[r["exec_us"] for r in c.rows if not r["skip"]],
                   lat_on_time=[r["lat_us"] for r in c.rows if not r["skip"] and r["late_us"] <= LATE_US])
        for b in BUDGETS:
            row[f"host{b}"], row[f"gpu{b}"] = c.misses(b)
        if gate is not None:
            row["ai_ups"] = gate["units_per_s_in_window"]
            row["units"] = gate["units_in_window"]
            row["fits"] = gate.get("fits_at_load")
            row["hist"] = gate.get("slots_by_units_started")
            row["est_end"] = gate.get("est_unit_us_window_end")
            row["window_violations"] = gate["overrun_units"]
            if key[1] == "gated" and gate["units_in_window"] == 0:
                problems.append(f"{name}: gate never opened (0 GEMMs in the window; fits_at_load={row['fits']})")
            up = None
            if units_dir and os.path.exists(os.path.join(units_dir, name + ".units.bin")):
                up, row["overlap_source"] = os.path.join(units_dir, name + ".units.bin"), "full"
            elif os.path.exists(prefix + ".units.head.bin"):
                up, row["overlap_source"] = prefix + ".units.head.bin", "first 2000 units only"
            if up:
                units = read_units(up)
                if row["overlap_source"] != "full" and units:   # only score slots the partial log covers
                    lim = units[-1][1]
                    c.rows = [r for r in c.rows if not r["skip"] and r["h1"] <= lim]
                row["overlap_hit"], row["overlap_n"] = c.overlap(units)
        runs.append(row)
        arms.setdefault(key, []).append(row)
    table = []
    for key, rs in sorted(arms.items()):
        slots = sum(r["slots"] for r in rs)
        t = dict(sharing=key[0], gate=key[1], n=key[2], runs=len(rs), slots=slots,
                 exec_p50=pct([x for r in rs for x in r["exec"]], 0.5),
                 exec_p99=pct([x for r in rs for x in r["exec"]], 0.99),
                 lat_p99_on_time=pct([x for r in rs for x in r["lat_on_time"]], 0.99))
        for b in BUDGETS:
            t[f"host_miss{b}"] = sum(r[f"host{b}"] for r in rs) / slots
            t[f"gpu_miss{b}"] = sum(r[f"gpu{b}"] for r in rs) / slots
            t[f"gpu_miss{b}_per_run"] = [round(100 * r[f"gpu{b}"] / r["slots"], 3) for r in rs]
        if key[1] != "alone":
            ups = sum(r["ai_ups"] for r in rs) / len(rs)
            base = ai_alone.get((key[0], key[2]))
            t.update(ai_units_per_s=ups, ai_alone_units_per_s=base, ai_kept=ups / base if base else None,
                     ai_units=sum(r["units"] for r in rs), window_violations=sum(r["window_violations"] for r in rs),
                     fits=all(r.get("fits") for r in rs), hist=[r.get("hist") for r in rs],
                     est_end_us=[r.get("est_end") for r in rs])
            ov = [r for r in rs if "overlap_n" in r]
            if ov:
                t["overlap_slots"] = sum(r["overlap_hit"] for r in ov) / max(1, sum(r["overlap_n"] for r in ov))
                t["overlap_source"] = sorted({r["overlap_source"] for r in ov})
            if base is None:
                problems.append(f"{key}: no tenant-alone baseline for this sharing mode and size")
        table.append(t)
    # paired gated - observe differences per (sharing, n, repeat)
    pairs = []
    byname = {(r["sharing"], r["gate"], r["n"], r["rep"]): r for r in runs}
    for (sh, g, n, rep), r in sorted(byname.items()):
        o = byname.get((sh, "observe", n, rep))
        if g == "gated" and o:
            pairs.append(dict(sharing=sh, n=n, rep=rep,
                              d_gpu_miss500=100 * (r["gpu500"] / r["slots"] - o["gpu500"] / o["slots"]),
                              d_gpu_miss250=100 * (r["gpu250"] / r["slots"] - o["gpu250"] / o["slots"]),
                              ai_ratio=r["ai_ups"] / o["ai_ups"] if o["ai_ups"] else None))
    busy = open(os.path.join(out, "busy_us.txt")).read().strip() if os.path.exists(os.path.join(out, "busy_us.txt")) else None
    return dict(busy_us=busy, arms=table, pairs=pairs, problems=problems, ai_alone={f"{k[0]}_n{k[1]}": v for k, v in ai_alone.items()})


def f(v, spec, scale=1.0, dash="-"):
    return dash if v is None else format(v * scale, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--busy")
    ap.add_argument("--margin-us", type=float, default=30.0)
    ap.add_argument("--period-us", type=float, default=500.0)
    ap.add_argument("--units-dir")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.busy:
        print(busy_us(a.busy, a.margin_us, a.period_us))
        return 0
    if not a.out:
        ap.error("OUT or --busy is required")
    s = summarise(a.out, a.units_dir)
    print(f"busy_us reserved per slot: {s['busy_us']}   (misses: % of slots; host = launch call returned >{LATE_US:.0f} us after target)")
    print(f"{'sharing':5} {'gate':7} {'n':>5} {'runs':>4} {'slots':>6} | {'GPU miss 500':>12} {'250':>7} | "
          f"{'host 500':>8} | {'exec p50/p99':>13} | {'overlap':>7} | {'AI/s':>7} {'kept':>6} | gate")
    for t in s["arms"]:
        gate_txt = "" if t["gate"] == "alone" else \
            f"fits={int(t['fits'])} units={t['ai_units']}"
        print(f"{t['sharing']:5} {t['gate']:7} {t['n']:5d} {t['runs']:4d} {t['slots']:6d} | "
              f"{f(t['gpu_miss500'], '11.2f', 100)}% {f(t['gpu_miss250'], '6.2f', 100)}% | "
              f"{f(t['host_miss500'], '7.2f', 100)}% | {f(t['exec_p50'], '6.1f')}/{f(t['exec_p99'], '6.1f')} | "
              f"{f(t.get('overlap_slots'), '6.1f', 100)}% | {f(t.get('ai_units_per_s'), '7.0f')} "
              f"{f(t.get('ai_kept'), '5.1f', 100)}% | {gate_txt}")
    if s["pairs"]:
        print("\npaired gated - observe (same sharing, size, repeat): GPU-caused miss change and AI throughput ratio")
        for p in s["pairs"]:
            print(f"  {p['sharing']:5} n={p['n']:5d} r{p['rep']}: d_miss500={p['d_gpu_miss500']:+7.2f} pp  "
                  f"d_miss250={p['d_gpu_miss250']:+7.2f} pp  AI gated/observe={f(p['ai_ratio'], '.2f')}")
    for p in s["problems"]:
        print("PROBLEM:", p)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(s, fh, indent=2)
    return 1 if s["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
