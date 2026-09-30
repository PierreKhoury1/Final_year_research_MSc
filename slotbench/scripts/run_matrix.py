#!/usr/bin/env python3
"""run_matrix.py: expand a TOML experiment matrix into cells and run each with run_one.sh.

  python3 scripts/run_matrix.py CONFIG.toml [--dry-run] [--only GLOB[,GLOB...]] [--out-root DIR]
                                [--repeat-from N] [--bin DIR]

Cell order (fixed): SOLO_<W> cells first (if [matrix] solo = true), then
mechanisms x workloads x duties x repeats. D=0 means the adversary is idle, so it is run once per
mechanism and repeat as workload "idle", named <M>_idle_d0_r<rep>, at the position of its first
occurrence. Other cells are <M>_<W>_d<D>_r<rep>. Repeats are numbered from --repeat-from (default 1).

Output: <out_root>/<config>/<cell>/ (run_one.sh), <out_root>/<config>/matrix.log (timestamped
events, appended) and matrix.json (the expanded plan and per-cell outcome, rewritten after each cell).
Resumable: cells whose status file says "ok" are skipped. Ctrl-C (or SIGTERM) lets the current
cell stop cleanly (run_one.sh restores GPU settings), records it as interrupted and exits 130.
--dry-run prints the run_one.sh commands plus the resolved driver/adversary commands; no GPU needed,
nothing is written. Stdlib only (Python >= 3.11 for tomllib).
Exit: 0 all cells ok, 1 some cell not ok, 2 config error, 130 interrupted.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN_ONE = os.path.join(HERE, "run_one.sh")

MECHANISMS = {"M0", "M1", "M2", "M3", "M4", "M5", "M6"}
WORKLOADS = {"sgemm", "llm", "vision"}
# Per-cell settings a [overrides.<M>] table (or [overrides.SOLO]) may change.
OVERRIDABLE = {"slots", "warmup_s", "settle_s", "driver_flags", "adversary_flags", "solo_seconds",
               "lock_clocks"}

RUN_DEFAULTS = {
    "slots": 1000000,
    "warmup_s": 300,
    "settle_s": 30,
    "repeats": 1,
    "out_root": "runs",
    "driver_core": -1,
    "collector_core": -1,
    "fifo": 0,
    "lock_clocks": False,
    "sizes": "",
    "driver_flags": "",
    "adversary_flags": "",
    "solo_seconds": 120,
    "gpu": 0,
    "cell_overhead_s": 30,
}


class ConfigError(Exception):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def load_config(path: str) -> dict:
    """Read and validate a matrix TOML; returns {"run": {...}, "matrix": {...}, "overrides": {...}}."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    unknown = set(cfg) - {"run", "matrix", "overrides"}
    if unknown:
        raise ConfigError(f"unknown top-level tables: {sorted(unknown)}")
    run = dict(RUN_DEFAULTS)
    for k, v in cfg.get("run", {}).items():
        if k not in RUN_DEFAULTS:
            raise ConfigError(f"unknown [run] key: {k}")
        run[k] = v
    m = cfg.get("matrix", {})
    for k in m:
        if k not in {"mechanisms", "workloads", "duties", "solo"}:
            raise ConfigError(f"unknown [matrix] key: {k}")
    matrix = {
        "mechanisms": list(m.get("mechanisms", [])),
        "workloads": list(m.get("workloads", [])),
        "duties": list(m.get("duties", [])),
        "solo": bool(m.get("solo", False)),
    }
    for x in matrix["mechanisms"]:
        if x not in MECHANISMS:
            raise ConfigError(f"unknown mechanism {x!r}")
    for x in matrix["workloads"]:
        if x not in WORKLOADS:
            raise ConfigError(f"unknown workload {x!r}")
    for d in matrix["duties"]:
        if not isinstance(d, int) or not 0 <= d <= 100:
            raise ConfigError(f"duty must be an integer 0..100, got {d!r}")
    if not isinstance(run["repeats"], int) or run["repeats"] < 1:
        raise ConfigError("repeats must be an integer >= 1")
    overrides = cfg.get("overrides", {})
    for mech, tbl in overrides.items():
        if mech not in MECHANISMS | {"SOLO"}:
            raise ConfigError(f"[overrides.{mech}]: unknown mechanism")
        bad = set(tbl) - OVERRIDABLE
        if bad:
            raise ConfigError(f"[overrides.{mech}]: keys not overridable: {sorted(bad)}")
    return {"run": run, "matrix": matrix, "overrides": overrides}


def expand(cfg: dict, repeat_from: int = 1) -> list[dict]:
    """Cells in execution order: SOLO first, then mechanisms x workloads x duties x repeats (D=0 deduped)."""
    run, matrix = cfg["run"], cfg["matrix"]
    reps = range(repeat_from, repeat_from + run["repeats"])
    cells: list[dict] = []
    seen: set[str] = set()

    def add(name, mech, workload, duty, rep):
        if name in seen:
            return
        seen.add(name)
        cells.append({"name": name, "mechanism": mech, "workload": workload, "duty": duty, "rep": rep})

    if matrix["solo"]:
        for w in matrix["workloads"]:
            add(f"SOLO_{w}", "SOLO", w, 100, None)
    for mech in matrix["mechanisms"]:
        for w in matrix["workloads"]:
            for d in matrix["duties"]:
                for r in reps:
                    if d == 0:
                        add(f"{mech}_idle_d0_r{r}", mech, "idle", 0, r)
                    else:
                        add(f"{mech}_{w}_d{d}_r{r}", mech, w, d, r)
    return cells


def cell_settings(cfg: dict, cell: dict) -> dict:
    s = dict(cfg["run"])
    s.update(cfg["overrides"].get(cell["mechanism"], {}))
    return s


def period_us(flags: str) -> float:
    m = re.search(r"--period-us[ =]([0-9.]+)", flags)
    return float(m.group(1)) if m else 500.0


def estimate_s(cfg: dict, cell: dict) -> float:
    """Wall-time estimate of one cell: warm-up + settle + slots x period + fixed overhead."""
    s = cell_settings(cfg, cell)
    base = float(s["warmup_s"]) + float(s["cell_overhead_s"])
    if cell["mechanism"] == "SOLO":
        return base + float(s["solo_seconds"])
    dflags = f'{s["sizes"]} {s["driver_flags"]}'
    return base + float(s["settle_s"]) + s["slots"] * period_us(dflags) * 1e-6


def build_command(cfg: dict, cell: dict, out_dir: str, bin_dir: str | None, run_one: str) -> list[str]:
    s = cell_settings(cfg, cell)
    cmd = [run_one, "--out", out_dir, "--mechanism", cell["mechanism"], "--workload", cell["workload"],
           "--duty", str(cell["duty"]), "--slots", str(s["slots"]), "--warmup-s", str(s["warmup_s"]),
           "--settle-s", str(s["settle_s"]), "--gpu", str(s["gpu"]),
           "--lock-clocks", "1" if s["lock_clocks"] else "0", "--cell", cell["name"]]
    if cell["rep"] is not None:
        cmd += ["--rep", str(cell["rep"])]
    if cell["mechanism"] == "SOLO":
        cmd += ["--solo-seconds", str(s["solo_seconds"])]
    if s["driver_core"] >= 0:
        cmd += ["--driver-core", str(s["driver_core"])]
    if s["collector_core"] >= 0:
        cmd += ["--collector-core", str(s["collector_core"])]
    if s["fifo"] > 0:
        cmd += ["--fifo", str(s["fifo"])]
    dflags = " ".join(x for x in (s["sizes"], s["driver_flags"]) if x)
    if dflags:
        cmd += ["--driver-flags", dflags]
    if s["adversary_flags"]:
        cmd += ["--adversary-flags", s["adversary_flags"]]
    if bin_dir:
        cmd += ["--bin", os.path.abspath(bin_dir)]
    return cmd


def read_status(cell_dir: str) -> str | None:
    try:
        with open(os.path.join(cell_dir, "status")) as f:
            return f.read().strip()
    except OSError:
        return None


def fmt_dur(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


class Matrix:
    def __init__(self, args):
        self.args = args
        self.config_path = os.path.abspath(args.config)
        self.cfg = load_config(args.config)
        self.name = os.path.splitext(os.path.basename(args.config))[0]
        out_root = args.out_root or self.cfg["run"]["out_root"]
        self.dir = os.path.join(os.path.abspath(out_root), self.name)
        self.cells = expand(self.cfg, args.repeat_from)
        if args.only:
            pats = [p for p in args.only.split(",") if p]
            self.cells = [c for c in self.cells if any(fnmatch.fnmatchcase(c["name"], p) for p in pats)]
        self.run_one = args.run_one or RUN_ONE
        self.stop = False
        self.child: subprocess.Popen | None = None
        self.outcomes: dict = {}

    def cell_dir(self, cell) -> str:
        return os.path.join(self.dir, cell["name"])

    def command(self, cell) -> list[str]:
        return build_command(self.cfg, cell, self.cell_dir(cell), self.args.bin, self.run_one)

    # ---------------------------------------------------------------- dry run
    def dry_run(self) -> int:
        total = sum(estimate_s(self.cfg, c) for c in self.cells)
        print(f"# config {self.config_path}: {len(self.cells)} cells, estimated {fmt_dur(total)}"
              f" (including cells already ok)")
        for i, c in enumerate(self.cells, 1):
            cmd = self.command(c)
            st = read_status(self.cell_dir(c))
            tag = " (status ok: would skip)" if st == "ok" else ""
            print(f"\n# [{i}/{len(self.cells)}] {c['name']} ~{fmt_dur(estimate_s(self.cfg, c))}{tag}")
            print(shlex.join(cmd))
            plan = subprocess.run(cmd + ["--print-plan"], capture_output=True, text=True)
            for line in (plan.stdout + plan.stderr).splitlines():
                print(f"#   {line}")
        return 0

    # ---------------------------------------------------------------- real run
    def log(self, msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        print(line, flush=True)
        with open(os.path.join(self.dir, "matrix.log"), "a") as f:
            f.write(line + "\n")

    def write_json(self) -> None:
        plan = [dict(c, dir=self.cell_dir(c), command=shlex.join(self.command(c)),
                     estimate_s=round(estimate_s(self.cfg, c), 1)) for c in self.cells]
        doc = {"config": self.name, "config_path": self.config_path, "matrix_dir": self.dir,
               "updated": now_iso(), "run": self.cfg["run"], "matrix": self.cfg["matrix"],
               "overrides": self.cfg["overrides"], "only": self.args.only,
               "repeat_from": self.args.repeat_from, "plan": plan, "outcomes": self.outcomes}
        tmp = os.path.join(self.dir, "matrix.json.tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
        os.replace(tmp, os.path.join(self.dir, "matrix.json"))

    def on_signal(self, signum, _frame):
        if self.stop:
            return
        print(f"\n*** signal {signum}: stopping the current cell cleanly "
              "(rerun to resume) ***", file=sys.stderr, flush=True)
        self.stop = True
        # run_one.sh runs in its own session (terminal Ctrl-C does not reach it), so it gets
        # exactly one SIGINT, from here; its EXIT trap stops driver/adversary and restores settings.
        if self.child and self.child.poll() is None:
            try:
                self.child.send_signal(signal.SIGINT)
            except OSError:
                pass

    def run(self) -> int:
        os.makedirs(self.dir, exist_ok=True)
        try:
            with open(os.path.join(self.dir, "matrix.json")) as f:
                self.outcomes = json.load(f).get("outcomes", {})
        except (OSError, ValueError):
            self.outcomes = {}
        signal.signal(signal.SIGINT, self.on_signal)
        signal.signal(signal.SIGTERM, self.on_signal)

        todo = [c for c in self.cells if read_status(self.cell_dir(c)) != "ok"]
        skipped = len(self.cells) - len(todo)
        est_total = sum(estimate_s(self.cfg, c) for c in todo)
        self.log(f"start {self.config_path}: {len(self.cells)} cells, {skipped} already ok, "
                 f"{len(todo)} to run, estimated {fmt_dur(est_total)}")
        self.write_json()

        done_est = 0.0
        done_actual = 0.0
        n_bad = 0
        for i, c in enumerate(self.cells, 1):
            if self.stop:
                break
            name = c["name"]
            if read_status(self.cell_dir(c)) == "ok":
                self.log(f"[{i}/{len(self.cells)}] {name}: status ok, skipped")
                self.outcomes.setdefault(name, {"status": "ok"})["skipped_at"] = now_iso()
                continue
            remaining_est = est_total - done_est
            scale = done_actual / done_est if done_est > 0 else 1.0
            self.log(f"[{i}/{len(self.cells)}] {name}: start (~{fmt_dur(estimate_s(self.cfg, c))}, "
                     f"ETA for the rest {fmt_dur(remaining_est * scale)})")
            cmd = self.command(c)
            self.log("  " + shlex.join(cmd))
            started = time.monotonic()
            t0 = now_iso()
            self.child = subprocess.Popen(cmd, start_new_session=True)
            while True:
                try:
                    rc = self.child.wait()
                    break
                except InterruptedError:
                    continue
            self.child = None
            dur = time.monotonic() - started
            done_est += estimate_s(self.cfg, c)
            done_actual += dur
            st = read_status(self.cell_dir(c)) or f"no_status(rc={rc})"
            if self.stop and st != "ok":
                st = st if st.startswith("invalid") else "invalid:interrupted"
            self.outcomes[name] = {"status": st, "returncode": rc, "started": t0, "ended": now_iso(),
                                   "duration_s": round(dur, 1)}
            if st != "ok":
                n_bad += 1
            self.log(f"[{i}/{len(self.cells)}] {name}: {st} (rc {rc}, {fmt_dur(dur)})")
            self.write_json()

        self.write_json()
        if self.stop:
            self.log("interrupted: rerun the same command to resume (ok cells are skipped)")
            return 130
        n_ok = sum(1 for c in self.cells if read_status(self.cell_dir(c)) == "ok")
        self.log(f"finished: {n_ok}/{len(self.cells)} cells ok, {n_bad} not ok this pass")
        return 0 if n_ok == len(self.cells) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    ap.add_argument("--only", help="comma-separated fnmatch globs on cell names, e.g. 'M1_*,SOLO_*'")
    ap.add_argument("--out-root", help="override [run] out_root (relative to the current directory)")
    ap.add_argument("--repeat-from", type=int, default=1, help="first repeat index (default 1)")
    ap.add_argument("--bin", help="directory with slot_driver and adversary (default slotbench/bin)")
    ap.add_argument("--run-one", help=argparse.SUPPRESS)  # testing: replacement for run_one.sh
    args = ap.parse_args(argv)
    try:
        m = Matrix(args)
    except (ConfigError, tomllib.TOMLDecodeError, OSError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if not m.cells:
        print("no cells selected", file=sys.stderr)
        return 2
    if args.dry_run:
        try:
            return m.dry_run()
        except BrokenPipeError:  # e.g. piped into head
            sys.stderr.close()
            return 0
    return m.run()


if __name__ == "__main__":
    sys.exit(main())
