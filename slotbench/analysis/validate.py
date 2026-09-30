"""Run validation (DESIGN.md section 8): python3 -m analysis.validate RUNS_ROOT [--out FILE] [options].

Every check is named and returns pass / fail / warn / skip with a reason. A run is invalid if any check
fails; warn is reported but does not invalidate. Works from slots.bin or from summary.json alone.
Writes validation.json (default RUNS_ROOT/validation.json), prints a table, exits 1 if any run is invalid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

from . import sbio, stats

# nvidia-smi clocks_event_reasons bits (NVML nvmlClocksEventReason*).
REASON_BITS = {0x1: "GpuIdle", 0x2: "ApplicationsClocksSetting", 0x4: "SwPowerCap", 0x8: "HwSlowdown",
               0x10: "SyncBoost", 0x20: "SwThermalSlowdown", 0x40: "HwThermalSlowdown",
               0x80: "HwPowerBrakeSlowdown", 0x100: "DisplayClockSetting"}
BENIGN_MASK = 0x1 | 0x2

DEFAULTS = {
    "idle_p9999_us": 300.0,          # D=0 cells: p99.99 must be below this
    "throttle_fail_mask": 0x4 | 0x8 | 0x20 | 0x40 | 0x80,   # power/thermal throttling -> fail
    "throttle_mode": "fail",         # "fail" or "warn" for throttle_fail_mask bits; other non-benign bits warn
    "clock_tol_mhz": 15.0,           # locked-clock tolerance
    "clock_off_frac": 0.05,          # fraction of telemetry samples allowed off the locked clock
    "cv_warn": 0.10,                 # repeatability: warn when CV(p99.99) across reps exceeds this
}


def _c(name, status, reason):
    return {"name": name, "status": status, "reason": reason}


def reasons_text(mask: int) -> str:
    names = [n for b, n in REASON_BITS.items() if mask & b]
    rest = mask & ~sum(REASON_BITS)
    if rest:
        names.append(hex(rest))
    return "|".join(names) or "none"


def check_throttle(info, cfg) -> dict:
    t = sbio.read_telemetry(info.file("telemetry.csv"))
    if not t or "reasons" not in t:
        return _c("throttle", "skip", "no telemetry.csv with a clocks_event_reasons column")
    vals = [v for v in t["reasons"] if isinstance(v, int)]
    if not vals:
        return _c("throttle", "skip", "no parseable throttle-reason samples")
    bad = [v & ~BENIGN_MASK for v in vals]
    fail_mask = int(cfg["throttle_fail_mask"])
    n_fail = sum(1 for v in bad if v & fail_mask)
    n_other = sum(1 for v in bad if v and not v & fail_mask)
    union = 0
    for v in bad:
        union |= v
    msg = f"{n_fail}/{len(vals)} samples power/thermal, {n_other} other; active: {reasons_text(union)}"
    if n_fail:
        return _c("throttle", "fail" if cfg["throttle_mode"] == "fail" else "warn", msg)
    if n_other:
        return _c("throttle", "warn", msg)
    return _c("throttle", "pass", f"{len(vals)} samples, no non-idle reasons")


def _lock_request(runj):
    """(requested: bool|None, target_mhz: float|None, not_supported: bool) from run.json (lenient)."""
    if not isinstance(runj, dict):
        return None, None, False
    req = sbio.find_key(runj, ["clocks_locked", "lock_clocks", "clock_lock", "gpu_lock", "locked_clocks"])
    target = sbio.find_key(runj, ["gc_mhz", "locked_gc_mhz", "lock_gc_mhz", "sm_clock_mhz",
                                  "graphics_clock_mhz", "locked_sm_mhz"])
    ns = False
    if isinstance(req, dict):
        ns = any(isinstance(v, str) and "NOT SUPPORTED" in v.upper() for _, v in sbio.iter_items(req))
        target = target if target is not None else sbio.find_key(req, ["gc", "sm", "mhz"])
        applied = sbio.find_key(req, ["applied", "ok", "locked"])
        req = bool(applied) if applied is not None else True
    elif isinstance(req, str):
        ns = "NOT SUPPORTED" in req.upper()
        req = req.strip().lower() not in ("0", "false", "no", "off", "")
    elif req is not None:
        req = bool(req)
    for _, v in sbio.iter_items(runj):
        if isinstance(v, str) and "NOT SUPPORTED" in v.upper() and ("clock" in v.lower() or "-lgc" in v):
            ns = True
    try:
        target = float(target) if target is not None else None
    except (TypeError, ValueError):
        target = None
    return req, target, ns


def check_clocks(info, cfg) -> dict:
    runj = sbio.load_json(info.file("run.json"))
    if info.mechanism == "M5":
        return _c("clocks_locked", "skip", "M5 runs with clocks unlocked by design")
    req, target, ns = _lock_request(runj)
    if ns:
        return _c("clocks_locked", "skip", "clock locking NOT SUPPORTED on this machine (recorded in run.json)")
    if not req:
        return _c("clocks_locked", "skip", "clock locking not requested" if req is False else
                  "run.json does not say whether clocks were locked")
    t = sbio.read_telemetry(info.file("telemetry.csv"))
    sm = [v for v in (t or {}).get("sm_mhz", []) if isinstance(v, (int, float))]
    if not sm:
        return _c("clocks_locked", "skip", "no SM clock samples in telemetry.csv")
    sm = np.asarray(sm, float)
    tol = float(cfg["clock_tol_mhz"])
    ref = target if target is not None else float(np.median(sm))
    off = float(np.mean(np.abs(sm - ref) > tol))
    msg = (f"SM clock min/median/max {sm.min():.0f}/{np.median(sm):.0f}/{sm.max():.0f} MHz vs "
           f"{'requested ' + format(target, '.0f') if target is not None else 'median'}; {off:.1%} samples off by >{tol:.0f}")
    return _c("clocks_locked", "fail" if off > cfg["clock_off_frac"] else "pass", msg)


def check_status_file(info) -> dict:
    try:
        with open(info.file("status")) as f:
            st = f.read().strip()
    except OSError:
        return _c("status_file", "skip", "no status file")
    if st == "ok":
        return _c("status_file", "pass", "ok")
    return _c("status_file", "fail", st or "empty status file")


def check_run(info: sbio.RunInfo, summary: dict | None, cfg: dict | None = None) -> dict:
    """All per-run checks. summary is a summary dict (from summary.json or built in memory)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    checks = [check_status_file(info)]
    if info.kind == "solo":
        adv = (summary or {}).get("adversary")
        checks.append(_c("adversary_present", "pass" if adv else "fail",
                         "adversary.json present" if adv else "adversary.json missing"))
        checks.append(check_throttle(info, cfg))
        return _finish(checks)

    if not summary or summary.get("source") != "slots.bin":
        checks.append(_c("slots_present", "fail", "no slots.bin and no usable summary.json"))
        return _finish(checks)
    c = summary["counts"]
    mc = c.get("meta") or {}
    checks.append(_c("not_crashed", "fail" if c.get("crashed") else "pass",
                     ("header n_records == 0 (driver did not close cleanly)" if c.get("header_crashed")
                      else "meta.json missing") if c.get("crashed")
                     else f"clean close, exit_reason={c.get('exit_reason')}"))
    flags = summary.get("file_flags") or []
    if "partial_trailing_record" in flags or "header_count_mismatch" in flags:
        checks.append(_c("file_integrity", "fail", ", ".join(flags)))
    else:
        checks.append(_c("file_integrity", "pass", "whole records, header count matches"))
    for key, name in (("ring_overflows", "ring_overflows"), ("stamp_mismatches", "stamp_mismatches")):
        v = mc.get(key)
        if v is None:
            checks.append(_c(name, "fail", f"{key} not found in meta.json"))
        else:
            checks.append(_c(name, "pass" if int(v) == 0 else "fail", f"{key} = {v}"))
    req = c.get("requested")
    if req is None:
        checks.append(_c("recorded_eq_requested", "fail", "requested slot count not found in meta.json config"))
    else:
        checks.append(_c("recorded_eq_requested", "pass" if c["recorded"] == req else "fail",
                         f"recorded {c['recorded']} / requested {req}"))
    checks.append(_c("slot_order", "pass" if c.get("slot_monotonic") else "fail",
                     "slot indices strictly increasing" if c.get("slot_monotonic")
                     else f"slot indices not increasing ({c.get('slot_duplicates')} duplicates)"))
    ms = mc.get("skipped")
    if ms is None:
        checks.append(_c("skipped_consistent", "skip", "meta.json has no skipped counter"))
    else:
        ok = int(ms) == c["skipped_from_gaps"]
        checks.append(_c("skipped_consistent", "pass" if ok else "warn",
                         f"meta skipped {ms}, gaps in slot indices {c['skipped_from_gaps']}"))
    if info.duty == 0:
        m = summary["miss"]
        p = (summary.get("latency_us") or {}).get("p99_99")
        lim = float(cfg["idle_p9999_us"])
        ok = m["misses"] == 0 and p is not None and p < lim
        checks.append(_c("idle_baseline", "pass" if ok else "fail",
                         f"D=0: misses {m['misses']} (need 0), p99.99 {p if p is None else round(p, 1)} us (need < {lim:g})"))
    if (info.duty or 0) > 0:
        checks.append(_c("adversary_present", "pass" if summary.get("adversary") else "warn",
                         "adversary.json present" if summary.get("adversary") else "adversary.json missing"))
    checks.append(check_throttle(info, cfg))
    checks.append(check_clocks(info, cfg))
    return _finish(checks)


def _finish(checks):
    return {"valid": not any(c["status"] == "fail" for c in checks),
            "failed": [c["name"] for c in checks if c["status"] == "fail"],
            "warned": [c["name"] for c in checks if c["status"] == "warn"],
            "checks": checks}


def get_summary(info: sbio.RunInfo) -> dict | None:
    """summary.json if it is current, else build one in memory from the raw files (not written)."""
    from . import summarize
    if not summarize.needs_summary(info):
        s = sbio.load_json(info.file("summary.json"))
        if s is not None:
            return s
    try:
        return summarize.build_summary(info.path, with_validation=False)
    except (OSError, ValueError):
        return None


def repeatability(summaries: dict, cfg=None) -> list:
    """Across reps of each cell: CV of latency p99.99 (reported; warn above cfg cv_warn). Callers pass
    valid runs only."""
    cfg = {**DEFAULTS, **(cfg or {})}
    by_cell = defaultdict(list)
    for info, s in summaries.values():
        if info.kind == "cell" and s and s.get("latency_us"):
            by_cell[info.cell].append((info.rep, s["latency_us"].get("p99_99")))
    out = []
    for cell in sorted(by_cell):
        reps = sorted(by_cell[cell])
        if len(reps) < 2:
            continue
        cv = stats.coefficient_of_variation([v for _, v in reps])
        st = "skip" if cv is None else ("warn" if cv > cfg["cv_warn"] else "pass")
        out.append({"cell": cell, "reps": [r for r, _ in reps], "p99_99_us": [v for _, v in reps],
                    "cv": cv, "status": st})
    return out


def validate_tree(root: str, cfg=None) -> dict:
    runs = sbio.discover_runs(root)
    summaries, results = {}, {}
    for info in runs:
        s = get_summary(info) if info.kind == "cell" else (sbio.load_json(info.file("summary.json")) or
                                                           {"adversary": sbio.load_json(info.file("adversary.json"))})
        summaries[info.path] = (info, s)
        r = check_run(info, s, cfg)
        r["run"] = os.path.relpath(info.path, os.path.abspath(root)) if info.path != os.path.abspath(root) else info.name
        r["cell"] = info.cell
        results[r["run"]] = r
        if not r["valid"]:
            summaries.pop(info.path)
    return {"config": {**DEFAULTS, **(cfg or {})}, "runs": results,
            "n_runs": len(results), "n_invalid": sum(1 for r in results.values() if not r["valid"]),
            "repeatability": repeatability(summaries, cfg)}


def print_table(v: dict, out=None):
    out = out or sys.stdout
    w = max([len(k) for k in v["runs"]] + [3])
    print(f"{'run':<{w}}  valid  failed / warned", file=out)
    for k in sorted(v["runs"]):
        r = v["runs"][k]
        extra = ",".join(r["failed"]) + (" / " + ",".join(r["warned"]) if r["warned"] else "")
        print(f"{k:<{w}}  {'yes' if r['valid'] else 'NO ':<5}  {extra}", file=out)
        for c in r["checks"]:
            if c["status"] in ("fail", "warn"):
                print(f"{'':<{w}}    {c['status'].upper()} {c['name']}: {c['reason']}", file=out)
    for rp in v["repeatability"]:
        cv = "n/a" if rp["cv"] is None else f"{rp['cv']:.3f}"
        print(f"repeatability {rp['cell']}: reps {rp['reps']} CV(p99.99) = {cv} [{rp['status']}]", file=out)
    print(f"{v['n_runs']} runs, {v['n_invalid']} invalid", file=out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate slotbench runs (DESIGN.md section 8).")
    ap.add_argument("root")
    ap.add_argument("--out", help="validation.json path (default ROOT/validation.json)")
    ap.add_argument("--idle-p9999-us", type=float, default=DEFAULTS["idle_p9999_us"])
    ap.add_argument("--throttle", choices=["fail", "warn"], default=DEFAULTS["throttle_mode"])
    ap.add_argument("--clock-tol-mhz", type=float, default=DEFAULTS["clock_tol_mhz"])
    ap.add_argument("--cv-warn", type=float, default=DEFAULTS["cv_warn"])
    a = ap.parse_args(argv)
    cfg = {"idle_p9999_us": a.idle_p9999_us, "throttle_mode": a.throttle, "clock_tol_mhz": a.clock_tol_mhz,
           "cv_warn": a.cv_warn}
    v = validate_tree(a.root, cfg)
    out = a.out or os.path.join(a.root if os.path.isdir(a.root) else ".", "validation.json")
    with open(out, "w") as f:
        json.dump(v, f, sort_keys=True, indent=1)
        f.write("\n")
    print_table(v)
    print(f"wrote {out}")
    return 1 if v["n_invalid"] else 0


if __name__ == "__main__":
    sys.exit(main())
