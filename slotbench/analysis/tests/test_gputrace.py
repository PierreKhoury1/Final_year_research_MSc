"""gputrace analysis on synthetic runs: a known host<->GPU clock relation, records and events generated from
it; the analysis must recover the latencies within the bound it reports."""
import json
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis import gputrace as gt  # noqa: E402

A, B = 1.0 + 3e-6, 123_456_789_000.0   # host = A*g + B
TICK = 32
rng = np.random.default_rng(7)


def host_of(g):
    return A * g + B


def write_clock(prefix, tag, g_start, n=300):
    """up: host(E) <= h (h = host(E) + delay); down: host(E) >= t (t = host(E) - delay). Also classic brackets."""
    up, down, classic = [], [], []
    for i in range(n):
        E = g_start + i * 3000
        E -= E % TICK
        up.append((int(host_of(E) + rng.uniform(200, 900)), E))
        down.append((int(host_of(E) - rng.uniform(200, 900)), E))
        t0 = int(host_of(E) - rng.uniform(300, 1200))
        classic.append((t0, int(host_of(E) + rng.uniform(300, 1200)), E))
    for name, rows, ncol in (("up", up, 2), ("down", down, 2), ("classic", classic, 3)):
        with open(f"{prefix}.{tag}.{name}.bin", "wb") as f:
            for r in rows:
                f.write(struct.pack("<" + "q" * ncol, *r))


def write_run(prefix, strategy, recs, events, meta_extra=None):
    np.asarray(recs, dtype=gt.GPU_DT).tofile(prefix + ".gpu.bin")
    np.asarray(events, dtype=gt.HOST_DT).tofile(prefix + ".host.bin")
    meta = dict(strategy=strategy, gpu="synthetic", sms=4, timer_edge_steps_ns=[TICK] * 63, gpu_recs_dropped=0)
    meta.update(meta_extra or {})
    json.dump(meta, open(prefix + ".json", "w"))
    write_clock(prefix, "pre", 10_000_000)
    write_clock(prefix, "post", 10_000_000 + 60_000_000_000)


def rec(g0, g1, smid=0, kid=1, block=0, tag=0, gap=0, ghz=1.5):
    return (g0, g1, 1000, 1000 + int((g1 - g0) * ghz), gap, smid, kid, block, tag, 10, 0)


def test_clock_fit_recovers_model(tmp_path):
    p = str(tmp_path / "x")
    write_run(p, "launch", [rec(20_000_000, 20_001_000)], [(int(host_of(20_000_000)) - 5000, 1, 1, 1, 32)])
    run = gt.Run(p)
    assert run.clock["method"] == "edge" and run.clock["feasible"]
    assert run.clock["bound_ns"] < 1000
    g = 10_000_000 + 30_000_000_000
    assert abs(float(run.host_of(g)) - host_of(g)) <= run.clock["bound_ns"] + 1


def test_launch_latency(tmp_path):
    p = str(tmp_path / "l")
    recs, evs = [], []
    g = 20_000_000
    for kid in range(1, 51):
        g += 1_000_000
        t_enter = int(host_of(g)) - 7_000   # launched 7 us before the block started
        evs += [(t_enter, gt.EV["LAUNCH_ENTER"], kid, 1, 32), (t_enter + 3000, gt.EV["LAUNCH_RETURN"], kid, 0, 0),
                (t_enter + 5000, gt.EV["SYNC_ENTER"], kid, 0, 0), (int(host_of(g + 1000)) + 4000, gt.EV["SYNC_RETURN"], kid, 0, 0)]
        recs.append(rec(g, g + 1000, kid=kid))
    write_run(p, "launch", recs, evs, dict(depth=1))
    res = gt.analyse(p)["result"]
    b = res["bound_ns"]
    assert abs(res["launch_to_start_ns"]["p50"] - 7000) <= b + TICK
    assert abs(res["call_ns"]["p50"] - 3000) < 1
    assert abs(res["end_to_sync_return_ns"]["p50"] - 4000) <= b + TICK
    assert abs(res["sm_clock_ghz"]["p50"] - 1.5) < 0.01


def test_notify(tmp_path):
    p = str(tmp_path / "n")
    recs, evs = [], []
    g = 20_000_000
    for kid in range(1, 31):
        tag = (kid - 1) % 3
        g += 100_000
        t_enter = int(host_of(g)) - 5000
        evs += [(t_enter, gt.EV["LAUNCH_ENTER"], kid, 1, 32), (t_enter + 2000, gt.EV["LAUNCH_RETURN"], kid, 0, 0)]
        end = g + 20_000
        recs.append(rec(g, end, kid=kid, tag=tag))
        typ, lat = {0: ("FLAG_SEEN", 1500), 1: ("EVENT_SEEN", 3000), 2: ("SYNC_RETURN", 9000)}[tag]
        evs.append((int(host_of(end)) + lat, gt.EV[typ], kid, end + 100 if tag == 0 else 0, 0))
    write_run(p, "notify", recs, evs)
    r = gt.analyse(p)["result"]
    b = r["bound_ns"]
    assert abs(r["flag_poll"]["p50"] - 1500) <= b + TICK
    assert abs(r["event_query"]["p50"] - 3000) <= b + TICK
    assert abs(r["stream_sync"]["p50"] - 9000) <= b + TICK
    assert abs(r["flag_write_after_end_ns"]["p50"] - 100) < 1


def test_dispatch_waves_and_sm_fill(tmp_path):
    p = str(tmp_path / "d")
    # 4 SMs, 2 concurrent blocks per SM, 16 blocks of 200 us: 2 waves; dispatch 1 us apart in wave 1
    g = 20_000_000
    recs = []
    for blk in range(16):
        wave, slot = divmod(blk, 8)
        start = g + wave * 200_000 + slot * 1000
        recs.append(rec(start, start + 200_000, smid=slot % 4, kid=1, block=blk))
    evs = [(int(host_of(g)) - 6000, gt.EV["LAUNCH_ENTER"], 1, 16, 256), (int(host_of(g)) - 4000, gt.EV["LAUNCH_RETURN"], 1, 0, 0)]
    write_run(p, "dispatch", recs, evs)
    k = gt.analyse(p)["result"]["kernels"][0]
    assert k["blocks"] == 16 and k["sms_used"] == 4 and k["max_blocks_per_sm_concurrent"] == 2
    assert k["waves"] == 2 and k["first_wave_span_ns"] == 7000
    assert abs(k["first_wave_rate_blocks_per_us"] - 1.0) < 1e-9
    assert abs(k["launch_to_first_block_ns"] - 6000) <= gt.Run(p).clock["bound_ns"] + TICK
    assert k["blocks_with_gap_gt_5us"] == 0 and k["max_gap_ns"]["p100"] == 0


def test_concurrency_and_preemption_flag(tmp_path):
    p = str(tmp_path / "c")
    g = 20_000_000
    recs = [rec(g, g + 2_000_000, smid=s, kid=1, block=s, tag=0, gap=(8000 if s == 1 else 300)) for s in range(4)]
    tb = int(host_of(g + 500_000))
    recs += [rec(g + 530_000, g + 730_000, smid=s, kid=2, block=s, tag=1) for s in range(4)]
    evs = [(int(host_of(g)) - 5000, gt.EV["LAUNCH_ENTER"], 1, 4, 256), (int(host_of(g)) - 3000, gt.EV["LAUNCH_RETURN"], 1, 0, 0),
           (tb, gt.EV["LAUNCH_ENTER"], 2, 4, 256), (tb + 2000, gt.EV["LAUNCH_RETURN"], 2, 0, 0)]
    write_run(p, "concurrency", recs, evs, dict(priority=1))
    r = gt.analyse(p)["result"]["reps"][0]
    assert abs(r["b_launch_to_first_block_ns"] - 30_000) <= gt.Run(p).clock["bound_ns"] + TICK
    assert r["b_blocks_started_before_a_end"] == 4 and r["a_blocks_running_when_b_started"] == 4
    assert r["a_blocks_with_gap_gt_5us"] == 1 and r["a_max_gap_ns"] == 8000


def test_timeslice_gaps(tmp_path):
    p = str(tmp_path / "t")
    g = 20_000_000
    on = int(host_of(g + 1_000_000))
    recs = [rec(g + 200_000, g + 230_000, kid=1, tag=2, block=1)]   # one 30 us gap alone
    # with hog: gaps of 2 ms every 4 ms
    for i in range(10):
        s = g + 2_000_000 + i * 4_000_000
        recs.append(rec(s, s + 2_000_000, kid=1, tag=2, block=2 + i))
    recs.append((g, g + 44_000_000, 0, 0, 2_000_000, 0, 1, 11, 3, 123456, 0))
    evs = [(int(host_of(g)) - 5000, gt.EV["LAUNCH_ENTER"], 1, 1, 32), (on, gt.EV["MARK"], 1, 1, 0),
           (int(host_of(g + 44_000_000)) + 100, gt.EV["SYNC_RETURN"], 1, 0, 0), (int(host_of(g + 44_000_000)) + 200, gt.EV["MARK"], 1, 2, 0)]
    write_run(p, "timeslice", recs, evs, dict(gap_us=20))
    r = gt.analyse(p)["result"]
    assert r["n_gaps"] == 11 and r["gaps_alone"]["n"] == 1 and r["gaps_with_hog"]["n"] == 10
    assert abs(r["gaps_with_hog"]["p50"] - 2_000_000) < 1
    assert abs(r["our_quantum_ns"]["p50"] - 2_000_000) < 1
    assert 0.4 < r["fraction_not_running_with_hog"] < 0.5


def test_one_line_does_not_crash(tmp_path):
    p = str(tmp_path / "o")
    write_run(p, "launch", [rec(20_000_000, 20_001_000)], [(int(host_of(20_000_000)) - 5000, 1, 1, 1, 32), (int(host_of(20_000_000)) - 4000, 2, 1, 0, 0)])
    assert "launch:" in gt.one_line(gt.analyse(p))


def test_gpus_offsets(tmp_path):
    p = str(tmp_path / "g")
    # GPU 1's timer reads 5 ms behind GPU 0's at the same host instant, with the same rate
    write_run(p, "gpus", [rec(20_000_000, 20_001_000)], [(int(host_of(20_000_000)) - 5000, 7, 0, 200, 0)], dict(n_gpus=2, reps=2))
    for r in range(2):
        write_clock(p, f"gpu0.r{r}", 10_000_000 + r * 30_000_000_000)
        up, down, classic = [], [], []
        for i in range(300):
            E = 10_000_000 + r * 30_000_000_000 + i * 3000
            E -= E % TICK
            E1 = E - 5_000_000   # GPU1 reading at the instant GPU0 reads E
            up.append((int(host_of(E) + rng.uniform(200, 900)), E1))
            down.append((int(host_of(E) - rng.uniform(200, 900)), E1))
            classic.append((int(host_of(E) - rng.uniform(300, 1200)), int(host_of(E) + rng.uniform(300, 1200)), E1))
        for name, rows, ncol in (("up", up, 2), ("down", down, 2), ("classic", classic, 3)):
            with open(f"{p}.gpu1.r{r}.{name}.bin", "wb") as f:
                for row in rows:
                    f.write(struct.pack("<" + "q" * ncol, *row))
    res = gt.analyse(p)["result"]
    g1 = res["gpus"][1]
    assert g1["feasible"] and abs(g1["timer_minus_gpu0_ns"] + 5_000_000) <= g1["offset_bound_ns"] + TICK
    assert abs(g1["rate_minus_gpu0_ppm"]) < 0.5
    assert "gpus:" in gt.one_line(gt.analyse(p))
