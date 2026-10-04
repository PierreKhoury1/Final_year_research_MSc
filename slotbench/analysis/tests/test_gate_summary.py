"""Synthetic checks for analysis/gate_summary.py (identity host<->GPU clock mapping)."""
import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gate_summary as gs  # noqa: E402

P = 500_000
T0 = 10 * 10**9


def write_case(d, name, slots, ok=True):
    """slots: list of (launch_late_ns, exec_start_offset_ns, exec_ns, flags) relative to each target."""
    with open(os.path.join(d, name + ".bin"), "wb") as f:
        for k, (late, start, ex, fl) in enumerate(slots):
            t = T0 + k * P
            f.write(gs.RECORD.pack(k, t, t, t + late, t + late + 5000, t + start, t + start + ex, fl))
    fit = lambda t: {"ok": True, "t_ref": t, "b_ns": 0, "g_ref": t}
    json.dump({"ok": ok, "clock_fit_pre": fit(T0 - 10**8), "clock_fit_post": fit(T0 + len(slots) * P + 10**8)},
              open(os.path.join(d, name + ".json"), "w"))
    open(os.path.join(d, name + ".check.log"), "w").write('{"valid": true}\n')
    json.dump({"exit_code": 0}, open(os.path.join(d, name + ".run.json"), "w"))


def on_time(ex=180_000):
    return (1000, 20_000, ex, 0)


def test_busy_uses_on_time_slots_and_refuses_impossible(tmp_path):
    slots = [on_time() for _ in range(2000)]
    slots[100] = (11_000_000, 11_010_000, 180_000, 0)   # one 11 ms host stall: must not set busy
    write_case(tmp_path, "calib", slots)
    assert gs.busy_us(str(tmp_path / "calib"), 30, 500) == 230   # 200 us latency + 30
    write_case(tmp_path, "slow", [on_time(ex=400_000) for _ in range(2000)])
    with pytest.raises(SystemExit):
        gs.busy_us(str(tmp_path / "slow"), 30, 500)


def test_miss_split_host_vs_gpu_and_skip_inheritance(tmp_path):
    slots = [on_time() for _ in range(10)]
    slots[2] = (1000, 20_000, 600_000, 0)        # on time, GPU too slow -> GPU miss at 500
    slots[5] = (900_000, 910_000, 180_000, 0)    # launched 900 us late -> host miss
    slots[6] = (0, 0, 0, 1)                      # skipped behind slot 5 -> host
    write_case(tmp_path, "c", slots)
    c = gs.Case(str(tmp_path / "c"))
    assert c.misses(500) == (2, 1)


def test_overlap_counts_slots_touched_by_tenant_units(tmp_path):
    write_case(tmp_path, "c", [on_time() for _ in range(4)])
    c = gs.Case(str(tmp_path / "c"))
    units = sorted([(T0 + 250_000, T0 + 400_000),            # in the gap of slot 0: no overlap
                    (T0 + P + 100_000, T0 + P + 150_000),    # inside slot 1's execution [20, 200] us
                    (T0 + 3 * P - 10_000, T0 + 3 * P + 30_000)])  # runs into slot 3's start (20 us)
    assert c.overlap(units) == (2, 4)


def test_summary_excludes_invalid_tenant_and_flags_closed_gate(tmp_path):
    for name in ("alone_proc_r1", "proc_gated_n768_r1", "proc_observe_n768_r1"):
        write_case(tmp_path, name, [on_time() for _ in range(50)])
    gate = {"timetable_loaded": True, "units_per_s_in_window": 0, "units_in_window": 0, "overrun_units": 0,
            "fits_at_load": False}
    json.dump({"ok": True, "gate": gate}, open(tmp_path / "proc_gated_n768_r1.adversary.json", "w"))
    json.dump({"ok": False, "gate": gate}, open(tmp_path / "proc_observe_n768_r1.adversary.json", "w"))
    json.dump({"ok": True, "units_per_s": 1000}, open(tmp_path / "ai_alone_proc_n768.json", "w"))
    s = gs.summarise(str(tmp_path))
    arms = {(a["sharing"], a["gate"], a["n"]) for a in s["arms"]}
    assert ("proc", "observe", 768) not in arms          # tenant failed: excluded from 5G pool too
    assert any("gate never opened" in p for p in s["problems"])
    assert any("proc_observe_n768_r1: tenant not ok" in p for p in s["problems"])
