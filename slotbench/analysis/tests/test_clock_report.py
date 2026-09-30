"""Tests for analysis.clock_report: Allan deviation against hand-computed values, drift recovery from a
synthetic clockcal run (linear drift + white noise + a temperature step), PHC maths, outputs produced."""
import json
import math

import numpy as np
import pytest

from analysis import clock_report as cr

HEADER = ("t_host_ns,n,n_kept,offset_ns,rate_ppm,residual_rms_ns,residual_max_ns,min_width_ns,"
          "median_width_ns,eps_ns,gpu_temp_c,sm_clock_mhz\n")


def test_adev_hand_example():
    # x = i^2: second differences are 2*m^2, so sigma^2 = (2 m^2)^2 / (2 m^2) = 2 m^2
    x = np.array([0, 1, 4, 9, 16], float)
    taus, adev, counts = cr.overlapping_adev(x, 1.0)
    assert list(taus) == [1.0, 2.0]
    assert adev[0] == pytest.approx(math.sqrt(2.0))   # (3 terms of 2^2) / (2 * 1 * 3)
    assert adev[1] == pytest.approx(math.sqrt(8.0))   # (16 - 8 + 0)^2 / (2 * 4 * 1)
    assert list(counts) == [3, 1]


def test_adev_tau0_scaling_and_gaps():
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    _, a1, _ = cr.overlapping_adev(x, 1.0, [1])
    _, a2, _ = cr.overlapping_adev(x, 2.0, [1])
    assert a1[0] == pytest.approx(math.sqrt(2.0))    # 4 terms of (+-2)^2: 16 / (2 * 1 * 4)
    assert a2[0] == pytest.approx(a1[0] / 2)
    xg = x.copy()
    xg[2] = np.nan                                    # kills triples (0,1,2), (1,2,3), (2,3,4)
    _, a3, k3 = cr.overlapping_adev(xg, 1.0, [1])
    assert k3[0] == 1 and a3[0] == pytest.approx(a1[0])


def test_linear_drift_and_grid():
    t = np.arange(10, dtype=float) * 60
    drift, c, r = cr.linear_drift(t, 2500.0 * t + 7.0)   # 2500 ns/s = 2.5 ppm
    assert drift == pytest.approx(2.5) and c == pytest.approx(7.0) and np.allclose(r, 0, atol=1e-6)
    g = cr.to_grid(np.array([0.0, 60.2, 180.1]), np.array([1.0, 2.0, 3.0]), 60.0)
    assert np.isnan(g[2]) and list(g[[0, 1, 3]]) == [1.0, 2.0, 3.0]


def _synthetic_run(d, drift_ppm=2.5, n=240, interval=60.0, noise_ns=20.0, seed=3):
    rng = np.random.default_rng(seed)
    t0 = 123_456_789_000_000
    t = np.arange(n) * interval
    off = drift_ppm * 1e3 * t + rng.normal(0, noise_ns, n)
    off -= off[0]
    temp = np.where(np.arange(n) < n // 2, 45.0, 72.0)          # thermal load switched on mid-run
    rate = drift_ppm + rng.normal(0, 3.0, n)                     # short-window fits are noisy
    rows = []
    for i in range(n):
        tt = t0 + int(round(t[i] * 1e9))
        if i == 17:   # one failed window: no fit, empty cells
            rows.append(f"{tt},10000,0,,,,,,,,{temp[i]:.0f},1777\n")
            continue
        rows.append(f"{tt},10000,100,{off[i]:.3f},{rate[i]:.6f},35.1,90.2,900,1500,480.0,{temp[i]:.0f},1777\n")
    with open(d / "clock_windows.csv", "w") as f:
        f.write("# clockcal synthetic\n# offset_ns: see clockcal\n" + HEADER + "".join(rows))
    with open(d / "meta.json", "w") as f:
        json.dump({"tool": "clockcal", "method": "pingpong", "config": {"interval_s": interval},
                   "gpu": {"name": "Synthetic"}, "global_fit": None}, f)


def test_report_recovers_drift_and_writes_outputs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _synthetic_run(run)
    # PHC: 1.7e18 ns epoch-like values, drift -0.8 ppm, 1 s interval; offsets must be done in int64
    m = 1_000_000_000_000 + np.arange(600, dtype=np.int64) * 1_000_000_000
    phc = 1_700_000_000_000_000_000 + (m - m[0]) + np.rint(-0.8e-6 * (m - m[0])).astype(np.int64)
    with open(run / "phc.csv", "w") as f:
        f.write("host_monoraw_ns,host_realtime_ns,phc_ns,method,width_ns\n")
        for a, b in zip(m, phc):
            f.write(f"{a},{a + 5},{b},extended,120\n")

    out = tmp_path / "out"
    rc = cr.main([str(run), "--out", str(out)])
    assert rc == 0
    s = json.loads((out / "clock_summary.json").read_text())
    assert s["n_windows"] == 240 and s["n_windows_fitted"] == 239
    assert s["drift_ppm"] == pytest.approx(2.5, abs=0.001)
    assert s["rate_between_windows_ppm"]["mean"] == pytest.approx(2.5, abs=0.05)
    assert s["rate_fit_ppm"]["std"] == pytest.approx(3.0, rel=0.25)
    assert s["wander_detrended_p2p_ns"] < 200 and s["wander_p2p_ns"] > 2.5e3 * 60 * 200
    assert s["gpu_temp_c"]["min"] == 45 and s["gpu_temp_c"]["max"] == 72
    assert s["tau0_s"] == 60.0 and s["adev"] and s["adev"][0]["tau_s"] == 60.0
    # white phase noise sigma: adev(tau0) ~ sqrt(6) sigma / tau0 (in s/s)
    assert s["adev"][0]["adev"] == pytest.approx(math.sqrt(6) * 20e-9 / 60, rel=0.3)
    assert s["phc"]["drift_ppm"] == pytest.approx(-0.8, abs=1e-4)
    assert s["phc"]["methods"] == {"extended": 600}
    for name in ("clock_offset.png", "clock_rate.png", "clock_residual.png", "clock_adev.png", "clock_phc.png"):
        assert (out / name).stat().st_size > 1000
        assert name in s["figures"]
    json.dumps(s, allow_nan=False)   # strict JSON


def test_no_temperature_no_phc(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "clock_windows.csv").write_text(
        HEADER + "1000000000,100,50,0,1.0,10,20,500,800,300,,\n"
                 "61000000000,100,50,60,1.0,10,20,500,800,300,,\n")
    s = cr.report(str(run))
    assert s["drift_ppm"] == pytest.approx(0.001)
    assert "gpu_temp_c" not in s and "phc" not in s and s["adev"] == []
    assert (run / "clock_summary.json").exists()
