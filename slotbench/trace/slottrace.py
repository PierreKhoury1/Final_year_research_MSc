#!/usr/bin/env python3
"""slottrace kernel layer: capture scheduler and interrupt events on the launcher's CPU(s) with ftrace,
on the CLOCK_MONOTONIC_RAW axis (trace_clock mono_raw) that the slot records also use.

  slottrace.py start --cpus 2[,3] [--buffer-mb 64] [--instance slottrace] [--extra-events ...]
  slottrace.py stop  --out RUN/trace          # writes RUN/trace.dat (trace-cmd), RUN/trace.txt, RUN/trace.meta.json
  slottrace.py run   --cpus 2 --out RUN/trace -- COMMAND ...   # start, run COMMAND, stop
  slottrace.py parse RUN/trace.txt            # print a summary of the parsed events

The tracer uses a private ftrace instance (instances/<name>) so it does not disturb other tracing, does not
overwrite events when the buffer fills (overwrite=0; drops are counted and reported), and restricts recording
to the given CPUs (tracing_cpumask). Events: sched_switch, sched_waking, sched_wakeup, sched_migrate_task,
irq_handler_entry/exit, softirq_entry/exit, and (if present) hrtimer_expire_entry. Extraction uses
`trace-cmd extract` for nanosecond timestamps; without trace-cmd the instance's text buffer (microsecond
resolution) is copied instead and the meta file says so.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TRACEFS_CANDIDATES = ("/sys/kernel/tracing", "/sys/kernel/debug/tracing")
DEFAULT_EVENTS = (
    "sched/sched_switch", "sched/sched_waking", "sched/sched_wakeup", "sched/sched_migrate_task",
    "irq/irq_handler_entry", "irq/irq_handler_exit", "irq/softirq_entry", "irq/softirq_exit",
    "timer/hrtimer_expire_entry",
)


def tracefs_root() -> Path:
    for c in TRACEFS_CANDIDATES:
        if os.path.exists(os.path.join(c, "trace_clock")):
            return Path(c)
    # try to mount it (needs CAP_SYS_ADMIN)
    root = Path(TRACEFS_CANDIDATES[0])
    subprocess.run(["mount", "-t", "tracefs", "nodev", str(root)], check=False, capture_output=True)
    if (root / "trace_clock").exists():
        return root
    raise SystemExit("slottrace: tracefs is not available (needs root / CAP_SYS_ADMIN and a kernel with ftrace)")


def write(path: Path, value: str) -> None:
    with open(path, "w") as f:
        f.write(value)


def cpumask(cpus: list[int]) -> str:
    mask = 0
    for c in cpus:
        mask |= 1 << c
    # tracing_cpumask takes comma-separated 32-bit hex words, most significant first
    words = []
    while True:
        words.append(f"{mask & 0xffffffff:08x}")
        mask >>= 32
        if not mask:
            break
    return ",".join(reversed(words))


def parse_cpus(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return sorted(set(out))


def start(cpus: list[int], buffer_mb: int, name: str, extra: list[str]) -> dict:
    root = tracefs_root()
    inst = root / "instances" / name
    if inst.exists():
        write(inst / "tracing_on", "0")
        os.rmdir(inst)
    os.mkdir(inst)
    clocks = (inst / "trace_clock").read_text()
    if "mono_raw" not in clocks:
        raise SystemExit("slottrace: trace_clock mono_raw not supported by this kernel")
    write(inst / "trace_clock", "mono_raw")
    write(inst / "buffer_size_kb", str(buffer_mb * 1024))
    write(inst / "options" / "overwrite", "0")
    write(inst / "tracing_cpumask", cpumask(cpus))
    enabled, missing = [], []
    for ev in list(DEFAULT_EVENTS) + list(extra):
        p = inst / "events" / ev / "enable"
        if p.exists():
            write(p, "1")
            enabled.append(ev)
        else:
            missing.append(ev)
    write(inst / "trace", "")
    write(inst / "tracing_on", "1")
    meta = {"instance": name, "tracefs": str(root), "cpus": cpus, "buffer_mb": buffer_mb, "clock": "mono_raw",
            "events": enabled, "events_missing": missing, "kernel": os.uname().release,
            "start_mono_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
            "cpu_times_start": cpu_times(), "user_hz": os.sysconf("SC_CLK_TCK")}
    return meta


def cpu_times() -> dict:
    """Per-CPU /proc/stat counters (USER_HZ ticks): user nice system idle iowait irq softirq steal."""
    out = {}
    names = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3:4].isdigit():
            f = line.split()
            out[f[0][3:]] = dict(zip(names, (int(x) for x in f[1:9])))
    return out


def per_cpu_stats(inst: Path, cpus: list[int]) -> dict:
    out = {}
    for c in cpus:
        p = inst / "per_cpu" / f"cpu{c}" / "stats"
        if p.exists():
            d = {}
            for line in p.read_text().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    d[k.strip()] = v.strip()
            out[str(c)] = d
    return out


def stop(out_prefix: str, name: str, meta: dict | None = None) -> dict:
    root = tracefs_root()
    inst = root / "instances" / name
    if not inst.exists():
        raise SystemExit(f"slottrace: instance {name} is not running")
    write(inst / "tracing_on", "0")
    meta = dict(meta or {})
    cpus = meta.get("cpus") or [i for i in range(os.cpu_count() or 1)]
    meta["stop_mono_raw_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    meta["per_cpu_stats"] = per_cpu_stats(inst, cpus)
    end = cpu_times()
    hz = meta.get("user_hz") or os.sysconf("SC_CLK_TCK")
    if "cpu_times_start" in meta:
        # hypervisor steal and interrupt time on the traced CPUs over the capture (ms; USER_HZ resolution)
        meta["cpu_time_delta_ms"] = {str(c): {k: (end[str(c)][k] - meta["cpu_times_start"][str(c)][k]) * 1000 / hz
                                              for k in ("steal", "irq", "softirq", "idle")}
                                     for c in cpus if str(c) in end and str(c) in meta["cpu_times_start"]}
    dropped = 0
    for st in meta["per_cpu_stats"].values():
        for key in ("overrun", "dropped events"):
            try:
                dropped += int(st.get(key, "0"))
            except ValueError:
                pass
    meta["events_lost"] = dropped
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    txt = out.with_suffix(".txt")
    if shutil.which("trace-cmd"):
        dat = out.with_suffix(".dat")
        r = subprocess.run(["trace-cmd", "extract", "-B", name, "-o", str(dat)], capture_output=True, text=True)
        if r.returncode != 0 or not dat.exists():
            raise SystemExit(f"slottrace: trace-cmd extract failed: {r.stderr.strip()[:300]}")
        with open(txt, "w") as f:
            r = subprocess.run(["trace-cmd", "report", "-t", "-i", str(dat)], stdout=f, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            raise SystemExit(f"slottrace: trace-cmd report failed: {r.stderr.strip()[:300]}")
        meta["resolution"] = "ns (trace-cmd)"
        meta["dat"] = str(dat)
    else:
        shutil.copyfile(inst / "trace", txt)
        meta["resolution"] = "us (tracefs text)"
    meta["txt"] = str(txt)
    write(inst / "trace", "")
    os.rmdir(inst)
    with open(out.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# ---------------------------------------------------------------- parsing
LINE = re.compile(r"^\s*(?:\S+:\s+)?(?P<comm>.*?)-(?P<pid>\d+)\s+\[(?P<cpu>\d+)\]\s+(?:\S+\s+)?"
                  r"(?P<sec>\d+)\.(?P<frac>\d+):\s+(?P<ev>\w+):\s*(?P<rest>.*)$")
KV = re.compile(r"(\w+)=(\S+)")
SWITCH_COMPACT = re.compile(r"^(.*):(\d+) \[(-?\d+)\] (\S+) ==> (.*):(\d+) \[(-?\d+)\]\s*$")
WAKEUP_COMPACT = re.compile(r"^(.*):(\d+) \[(-?\d+)\](?: success=\d+)? CPU:(\d+)\s*$")


def parse_ts(sec: str, frac: str) -> int:
    frac = (frac + "000000000")[:9]
    return int(sec) * 1_000_000_000 + int(frac)


def parse_line(line: str) -> dict | None:
    m = LINE.match(line)
    if not m:
        return None
    ev = m.group("ev")
    rest = m.group("rest")
    d = {"t": parse_ts(m.group("sec"), m.group("frac")), "cpu": int(m.group("cpu")), "ev": ev,
         "comm": m.group("comm").strip(), "pid": int(m.group("pid"))}  # the task that was running when it fired
    if ev == "sched_switch" and "prev_comm=" not in rest:
        # trace-cmd's compact form: prev_comm:prev_pid [prev_prio] prev_state ==> next_comm:next_pid [next_prio]
        mc = SWITCH_COMPACT.match(rest)
        if mc:
            d.update(prev_comm=mc.group(1), prev_pid=int(mc.group(2)), prev_prio=int(mc.group(3)),
                     prev_state=mc.group(4), next_comm=mc.group(5), next_pid=int(mc.group(6)),
                     next_prio=int(mc.group(7)))
        return d
    if ev == "sched_wakeup" and "comm=" not in rest:
        mw = WAKEUP_COMPACT.match(rest)
        if mw:
            d.update(target_comm=mw.group(1), target_pid=int(mw.group(2)), prio=int(mw.group(3)),
                     target_cpu=int(mw.group(4)))
        return d
    if ev == "sched_switch":
        left, _, right = rest.partition("==>")
        for k, v in KV.findall(left):
            d[k] = v
        for k, v in KV.findall(right):
            d[k] = v
        # prev_comm/next_comm may contain spaces; recover them from the raw text
        mp = re.search(r"prev_comm=(.*?) prev_pid=", left)
        mn = re.search(r"next_comm=(.*?) next_pid=", right)
        if mp:
            d["prev_comm"] = mp.group(1)
        if mn:
            d["next_comm"] = mn.group(1)
        for k in ("prev_pid", "next_pid", "prev_prio", "next_prio"):
            if k in d:
                d[k] = int(d[k])
    else:
        kv = dict(KV.findall(rest))
        mc = re.search(r"comm=(.*?) pid=", rest)
        if mc:
            d["target_comm"] = mc.group(1)
        rename = {"pid": "target_pid"}
        for k in ("pid", "prio", "target_cpu", "irq", "vec", "orig_cpu", "dest_cpu"):
            if k in kv:
                try:
                    d[rename.get(k, k)] = int(kv[k])
                except ValueError:
                    pass
        for k in ("function", "hrtimer", "now", "ret"):
            if k in kv:
                d[k] = kv[k]
        mi = re.search(r"name=(\S+)", rest)
        if mi:
            d["irq_name"] = mi.group(1)
        ma = re.search(r"action=(\w+)", rest)
        if ma:
            d["action"] = ma.group(1)
    return d


def load_events(txt_path: str) -> list[dict]:
    evs = []
    with open(txt_path, errors="replace") as f:
        for line in f:
            if line.startswith(("#", "cpus=", "version")) or not line.strip():
                continue
            d = parse_line(line)
            if d:
                evs.append(d)
    evs.sort(key=lambda e: e["t"])
    return evs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--cpus", required=True)
    s.add_argument("--buffer-mb", type=int, default=64)
    s.add_argument("--instance", default="slottrace")
    s.add_argument("--extra-events", nargs="*", default=[])
    s.add_argument("--meta-out", default=None)
    t = sub.add_parser("stop")
    t.add_argument("--out", required=True)
    t.add_argument("--instance", default="slottrace")
    t.add_argument("--meta-in", default=None)
    r = sub.add_parser("run")
    r.add_argument("--cpus", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--buffer-mb", type=int, default=64)
    r.add_argument("--instance", default="slottrace")
    r.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("parse")
    p.add_argument("txt")
    a = ap.parse_args(argv)
    if a.cmd == "start":
        meta = start(parse_cpus(a.cpus), a.buffer_mb, a.instance, a.extra_events)
        if a.meta_out:
            Path(a.meta_out).write_text(json.dumps(meta))
        print(json.dumps(meta))
    elif a.cmd == "stop":
        meta = json.loads(Path(a.meta_in).read_text()) if a.meta_in else None
        print(json.dumps(stop(a.out, a.instance, meta), indent=2))
    elif a.cmd == "run":
        cmd = a.command[1:] if a.command and a.command[0] == "--" else a.command
        if not cmd:
            raise SystemExit("slottrace run: no command")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        meta = start(parse_cpus(a.cpus), a.buffer_mb, a.instance, [])
        try:
            rc = subprocess.call(cmd)
        finally:
            meta = stop(a.out, a.instance, meta)
        meta["command"] = cmd
        meta["command_rc"] = rc
        Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        print(f"slottrace: command rc={rc}, events lost={meta['events_lost']}, resolution {meta['resolution']}, {meta['txt']}")
        return rc
    elif a.cmd == "parse":
        evs = load_events(a.txt)
        from collections import Counter
        c = Counter(e["ev"] for e in evs)
        print(f"{len(evs)} events, {dict(c)}")
        if evs:
            print(f"span {(evs[-1]['t'] - evs[0]['t']) / 1e9:.3f} s, first {evs[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
