"""Tests for the slottrace parser and attribution (synthetic traces with known causes)."""
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import analyze  # noqa: E402
import slottrace  # noqa: E402

TID, CPU = 500, 2


def line(t_ns, comm, pid, ev, rest, cpu=CPU):
    return f"slottrace:  {comm}-{pid}  [{cpu:03d}]  {t_ns // 10**9}.{t_ns % 10**9:09d}: {ev}:  {rest}"


def test_parse_compact_and_kv_forms():
    a = slottrace.parse_line(line(5_000_000_123, "kworker/2:2", 72, "sched_switch",
                                  "kworker/2:2:72 [120] R+ ==> mock slots:500 [49]"))
    assert a["prev_comm"] == "kworker/2:2" and a["prev_pid"] == 72 and a["prev_state"] == "R+"
    assert a["next_comm"] == "mock slots" and a["next_pid"] == 500 and a["next_prio"] == 49
    assert a["t"] == 5_000_000_123
    b = slottrace.parse_line(line(5_000_000_200, "<idle>", 0, "sched_wakeup", "mock slots:500 [49] CPU:002"))
    assert b["target_pid"] == 500 and b["target_cpu"] == 2 and b["pid"] == 0
    c = slottrace.parse_line(line(5_000_000_300, "<idle>", 0, "sched_waking", "comm=Bun Pool 1 pid=126 prio=120 target_cpu=001"))
    assert c["target_comm"] == "Bun Pool 1" and c["target_pid"] == 126 and c["pid"] == 0
    d = slottrace.parse_line("slottrace: python3-1 [002] 1.000002000: softirq_entry: vec=7 [action=SCHED]")
    assert d["vec"] == 7 and d["action"] == "SCHED"
    us = slottrace.parse_line("python3-1 [002] d..2. 1.000002: sched_switch: prev_comm=a b prev_pid=1 prev_prio=120 "
                              "prev_state=S ==> next_comm=c prev next_pid=2 next_prio=120")
    assert us["t"] == 1_000_002_000 and us["prev_comm"] == "a b" and us["next_comm"] == "c prev"


def make_run(tmp, slots, mode="cpu", spin_us=200.0):
    """slots: list of (T, wake, launch, ret, start, end, flags) in host ns; identity clock mapping."""
    lsrec = struct.Struct("<Qq6Q")
    host = struct.Struct("<QqqqqiI")
    raw, hraw = os.path.join(tmp, "r.bin"), os.path.join(tmp, "h.bin")
    with open(raw, "wb") as f, open(hraw, "wb") as h:
        for k, (T, w, l, r, s, e, fl) in enumerate(slots):
            f.write(lsrec.pack(k, T, T, l, r, s, e, fl))
            h.write(host.pack(k, T, w, l, r, CPU, fl))
    t0, t1 = slots[0][0], slots[-1][0]
    run = {"mode": mode, "spin_us": spin_us, "deadline_us": 300, "launcher_tid": TID, "launcher_cpu": CPU,
           "clock_fit_pre": {"t_ref": t0, "b_ns": 0, "g_ref": t0}, "clock_fit_post": {"t_ref": t1, "b_ns": 0, "g_ref": t1}}
    rp = os.path.join(tmp, "run.json")
    with open(rp, "w") as f:
        json.dump(run, f)
    return rp, raw, hraw


def on_time(T):
    return (T, T - 195_000, T + 100, T + 8_000, T + 13_000, T + 153_000, 0)


def late_host(T, delay):
    return (T, T - 195_000, T + delay, T + delay + 8_000, T + delay + 13_000, T + delay + 153_000, 0)


def run_case(events_text, slots, launcher_rt=True):
    with tempfile.TemporaryDirectory() as tmp:
        rp, raw, hraw = make_run(tmp, slots)
        tp = os.path.join(tmp, "trace.txt")
        with open(tp, "w") as f:
            f.write("\n".join(events_text) + "\n")
        run = json.load(open(rp))
        recs = analyze.load_records(raw, hraw)
        kv = analyze.build_kernel_view(slottrace.load_events(tp), TID, CPU)
        rows, _ = analyze.analyze(run, recs, kv, None, [], None)
        return rows


def scenario(other_comm, other_prio, lp, state="R"):
    """20 on-time slots, then slot 20 delayed by 400 us because the launcher was off-CPU in [T-150us, T+390us]."""
    P = 500_000
    base = 10 * 10**9
    slots = [on_time(base + k * P) for k in range(20)]
    T = base + 20 * P
    slots.append(late_host(T, 400_000))
    slots += [on_time(base + k * P) for k in range(21, 30)]
    lname = "launcher"
    ev = [line(T - 150_000, lname, TID, "sched_switch", f"{lname}:{TID} [{lp}] {state} ==> {other_comm}:900 [{other_prio}]")]
    if not state.startswith("R"):
        ev.append(line(T - 140_000, other_comm, 900, "sched_waking", f"comm={lname} pid={TID} prio={lp} target_cpu=002"))
    ev.append(line(T + 390_000, other_comm, 900, "sched_switch", f"{other_comm}:900 [{other_prio}] R ==> {lname}:{TID} [{lp}]"))
    return run_case(ev, slots), 20


def test_preempt_rt():
    rows, k = scenario("rtjob", 30, 49)
    assert rows[k]["late"] and rows[k]["cause"] == "preempt_rt", rows[k]


def test_rt_throttle_lower_priority_ran():
    rows, k = scenario("swapper/2", 120, 49)
    assert rows[k]["cause"] == "rt_throttle", rows[k]


def test_cfs_contention():
    rows, k = scenario("stress", 120, 120)
    assert rows[k]["cause"] == "cpu_contention", rows[k]
    assert rows[k]["detail"]["ran_instead"] == "stress"


def test_woken_then_waited_counts_as_runnable():
    rows, k = scenario("stress", 120, 120, state="S")
    assert rows[k]["cause"] == "cpu_contention", rows[k]


def test_invisible_when_trace_is_silent():
    P = 500_000
    base = 10 * 10**9
    slots = [on_time(base + k * P) for k in range(20)] + [late_host(base + 20 * P, 400_000)]
    rows = run_case([line(base, "x", 1, "softirq_entry", "vec=1 [action=TIMER]")], slots)
    assert rows[20]["cause"] == "invisible", rows[20]


def test_skipped_inherits_root_cause():
    P = 500_000
    base = 10 * 10**9
    slots = [on_time(base + k * P) for k in range(20)]
    T = base + 20 * P
    slots.append(late_host(T, 400_000))
    slots.append((base + 21 * P, 0, 0, 0, 0, 0, 1))  # skipped
    lp = 49
    ev = [line(T - 150_000, "launcher", TID, "sched_switch", f"launcher:{TID} [{lp}] R ==> rtjob:900 [30]"),
          line(T + 390_000, "rtjob", 900, "sched_switch", f"rtjob:900 [30] R ==> launcher:{TID} [{lp}]")]
    rows = run_case(ev, slots)
    assert rows[21]["cause"] == "carryover:preempt_rt"


def test_gpu_timeslice_from_probe():
    P = 500_000
    base = 10 * 10**9
    slots = [on_time(base + k * P) for k in range(20)]
    T = base + 20 * P
    slots.append((T, T - 195_000, T + 100, T + 8_000, T + 13_000, T + 2_300_000, 0))  # exec 2.3 ms
    with tempfile.TemporaryDirectory() as tmp:
        rp, raw, hraw = make_run(tmp, slots)
        run = json.load(open(rp))
        recs = analyze.load_records(raw, hraw)
        probe = {"gaps": [(T + 50_000, T + 2_200_000)]}
        rows, _ = analyze.analyze(run, recs, None, probe, [], None)
    assert rows[20]["cause"] == "gpu_timeslice", rows[20]


def test_confusion_and_ground_truth():
    rows = [{"late": True, "cause": "cpu_contention", "fault": "cfs_hog", "t_target": 1},
            {"late": True, "cause": "invisible", "fault": "cfs_hog", "t_target": 2},
            {"late": True, "cause": "carryover:cpu_contention", "fault": "cfs_hog", "skipped": True, "t_target": 3}]
    s = analyze.summarize(rows, [(0, 10, "cfs_hog")], {})
    assert s["per_fault"]["cfs_hog"]["correct"] == 2 and s["per_fault"]["cfs_hog"]["late_slots"] == 3
