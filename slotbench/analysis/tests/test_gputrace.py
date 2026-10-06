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


def test_nccl_spans_skew_and_causality(tmp_path):
    p = str(tmp_path / "n")
    off1 = -5_000_000   # GPU 1's timer reads 5 ms behind GPU 0's at the same host instant
    recs, evs = [], []
    g0 = 20_000_000
    kid = 0
    for size, dur in ((8, 20_000), (1 << 20, 60_000)):
        for i in range(20):
            kid += 1
            base = g0 + kid * 1_000_000
            t_enter = int(host_of(base))
            # GPU 0: stamp ends at base, collective dur, after-stamp begins base+dur+2000
            # GPU 1 starts 3 us later (host enqueue order) and finishes 1 us after GPU 0
            for d, start_off, end_off in ((0, 0, 0), (1, 3_000, 1_000)):
                o = 0 if d == 0 else off1
                b_end = base + start_off + o
                a_beg = base + dur + 2_000 + end_off + o
                recs.append((b_end - 1000, b_end, 0, 0, 0, 0, kid, i, 1, 0, d))
                recs.append((a_beg, a_beg + 1000, 0, 0, 0, 0, kid, i, 2, 0, d))
            evs += [(t_enter, gt.EV["LAUNCH_ENTER"], kid, size, 2), (t_enter + 4000, gt.EV["LAUNCH_RETURN"], kid, size, 2),
                    (int(host_of(base + dur + 4000)) + 3000, gt.EV["SYNC_RETURN"], kid, 0, 0)]
    np.asarray(recs, dtype=gt.GPU_DT).tofile(p + ".gpu.bin")
    np.asarray(evs, dtype=gt.HOST_DT).tofile(p + ".host.bin")
    json.dump(dict(strategy="nccl", gpu="synthetic", sms=4, n_gpus=2, reps=2, gpu_recs_dropped=0), open(p + ".json", "w"))
    for r in range(2):
        start = 10_000_000 + r * 60_000_000_000
        write_clock(p, f"gpu0.r{r}", start)
        up, down, classic = [], [], []
        for i in range(300):
            E = start + i * 3000
            E -= E % TICK
            up.append((int(host_of(E) + rng.uniform(200, 900)), E + off1))
            down.append((int(host_of(E) - rng.uniform(200, 900)), E + off1))
            classic.append((int(host_of(E) - rng.uniform(300, 1200)), int(host_of(E) + rng.uniform(300, 1200)), E + off1))
        for name, rows, ncol in (("up", up, 2), ("down", down, 2), ("classic", classic, 3)):
            with open(f"{p}.gpu1.r{r}.{name}.bin", "wb") as f:
                for row in rows:
                    f.write(struct.pack("<" + "q" * ncol, *row))
    res = gt.analyse(p)
    r = res["result"]
    assert "error" not in r, r
    b = r["skew_bound_ns"][1]
    s8, s1m = r["sizes"][8], r["sizes"][1 << 20]
    assert s8["n"] == 20 and s1m["n"] == 20
    assert abs(s8["span_ns"][0]["p50"] - 22_000) < 1 and abs(s8["span_ns"][1]["p50"] - 20_000) < 1
    assert abs(s8["start_skew_ns"][1]["p50"] - 3_000) <= b + TICK
    assert abs(s1m["end_skew_ns"][1]["p50"] - 1_000) <= b + TICK
    assert s8["causal_violations"] == 0 and s8["causal_slack_min_ns"] > 0
    assert abs(s1m["algbw_GBps"] - (1 << 20) / 61_000) < 1
    assert abs(s1m["call_ns"]["p50"] - 4000) < 1
    assert "nccl x2" in gt.one_line(res)


def test_export_chrome_trace(tmp_path):
    from analysis.gputrace_export import export
    p = str(tmp_path / "e")
    g = 20_000_000
    recs = [rec(g + 1000 * i, g + 1000 * i + 200_000, smid=i % 4, kid=1 + i // 4, block=i % 4) for i in range(8)]
    evs = []
    for k in (1, 2):
        t = int(host_of(g)) - 5000 + (k - 1) * 4000
        evs += [(t, gt.EV["LAUNCH_ENTER"], k, 4, 256), (t + 2000, gt.EV["LAUNCH_RETURN"], k, 0, 0),
                (t + 2500, gt.EV["SYNC_ENTER"], k, 0, 0), (int(host_of(g + 210_000)), gt.EV["SYNC_RETURN"], k, 0, 0)]
    write_run(p, "dispatch", recs, evs)
    tr = export(p)
    xs = [e for e in tr["traceEvents"] if e["ph"] == "X"]
    gpu_x = [e for e in xs if e["pid"] >= 10]
    host_x = [e for e in xs if e["pid"] == 1]
    assert len(gpu_x) == 8 and len(host_x) == 4
    # a block's drawn start is within the bound of its true host time
    run = gt.Run(p)
    first_host = min(e["ts"] for e in xs)
    blk = min(gpu_x, key=lambda e: e["ts"])
    true_us = (host_of(g) - (int(host_of(g)) - 5000)) / 1000.0
    assert abs((blk["ts"] - first_host) - true_us) * 1000 <= run.clock["bound_ns"] + TICK
    json.dumps(tr)


def test_memory_curve_and_knees(tmp_path):
    p = str(tmp_path / "m")
    recs, evs = [], []
    g = 20_000_000
    kid = 0
    ghz = 1.4
    for mod in (0, 1):
        for ws, cyc in ((16384, 40), (1 << 20, 220), (16 << 20, 220), (128 << 20, 540)):
            kid += 1
            t = int(host_of(g + kid * 1_000_000))
            evs += [(t, gt.EV["LAUNCH_ENTER"], kid, ws, mod), (t + 2000, gt.EV["LAUNCH_RETURN"], kid, 0, 0)]
            for b in range(5):
                g0 = g + kid * 1_000_000 + b * 50_000
                ns = cyc * 64 / ghz
                recs.append((g0, g0 + int(ns), 1000, 1000 + cyc * 64, 0, 3, kid, 0, mod, 64, 0))
    write_run(p, "memory", recs, evs, dict(cotenant=0, batch=64))
    r = gt.analyse(p)["result"]
    ca = r["modifiers"][".ca"]
    assert [c["ws"] for c in ca["curve"]] == [16384, 1 << 20, 16 << 20, 128 << 20]
    assert abs(ca["curve"][0]["cycles"]["p50"] - 40) < 1e-6 and abs(ca["curve"][0]["ghz"] - ghz) < 0.01
    assert ca["knees"] == [1 << 20, 128 << 20]
    assert "memory(" in gt.one_line(gt.analyse(p))
