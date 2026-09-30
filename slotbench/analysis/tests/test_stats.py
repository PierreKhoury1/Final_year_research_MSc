"""Statistics: quantile method, Clopper-Pearson, skipped-slot accounting, histogram CCDF, clock mapping."""
import numpy as np
import pytest

from analysis import sbio, stats, synth


def test_quantile_higher_tiny_arrays():
    x = np.array([4, 1, 3, 2])
    q = stats.quantiles(x)
    assert q["p50"] == 3  # sorted [1,2,3,4], ceil(0.5*3)=2 -> 3 (linear would give 2.5)
    assert q["p90"] == 4 and q["p99_99"] == 4 and q["max"] == 4 and q["min"] == 1
    assert stats.quantiles([7])["p50"] == 7 and stats.quantiles([7])["p99_99"] == 7
    assert stats.quantiles([])["p50"] is None
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 5, 10, 101, 1000, 10001):
        v = rng.lognormal(size=n)
        for _, qq in stats.QUANTILES:
            got = stats.quantile_sorted(np.sort(v), qq)
            assert got == np.quantile(v, qq, method="higher")
            assert got in v and got >= np.quantile(v, qq)  # observed, never below interpolation


def test_quantile_index_exact_for_1e6():
    # p99.99 of 1e6 samples: ceil(0.9999 * 999999) = 999900 exactly (float rounding must not move it)
    assert stats.quantile_index(1_000_000, 0.9999) == 999_900
    assert stats.quantile_index(10_001, 0.9999) == 9_999  # 0.9999 * 10000 is an exact integer
    assert stats.quantile_index(1001, 0.5) == 500


def test_clopper_pearson_known_values():
    lo, hi = stats.clopper_pearson(0, 1_000_000)
    assert lo == 0.0 and hi == pytest.approx(3.6889e-6, rel=1e-3)
    assert hi == pytest.approx(1 - 0.025 ** (1 / 1_000_000), rel=1e-9)
    lo, hi = stats.clopper_pearson(5, 100)  # standard table: 0.0164 - 0.1128
    assert lo == pytest.approx(0.01643, abs=5e-5) and hi == pytest.approx(0.11284, abs=5e-5)
    lo, hi = stats.clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** 0.1, rel=1e-9)
    assert stats.clopper_pearson(0, 0) == (0.0, 1.0)


def test_skipped_slot_accounting():
    slots = np.array([0, 1, 2, 5, 6, 10], dtype=np.uint64)  # gaps: 3,4 and 7,8,9 -> 5 skipped
    g = stats.slot_gaps(slots)
    assert g["skipped"] == 5 and g["monotonic"]
    lat = np.array([100, 600, 200, 700, 100, 100]) * 1000  # two late (strictly > 500 us)
    m = stats.miss_stats(lat, 500_000, g["skipped"])
    assert m["total_slots"] == 11 and m["misses"] == 7 and m["late"] == 2
    assert m["miss_rate"] == pytest.approx(7 / 11)
    assert stats.miss_stats([500_000], 500_000, 0)["misses"] == 0  # exactly at the deadline is not a miss
    bad = stats.slot_gaps(np.array([0, 1, 1, 3], dtype=np.uint64))
    assert not bad["monotonic"] and bad["duplicates"] == 1 and bad["skipped"] == 1


def test_hist_ccdf_matches_raw_at_edges():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.lognormal(np.log(200), 0.1, 50_000), rng.pareto(1.2, 500) * 300 + 250,
                        [0.5, 2e6]])  # one underflow, one overflow
    h = stats.log_histogram(x)
    assert h["underflow"] == 1 and h["overflow"] == 1 and sum(h["counts"]) + 2 == x.size
    assert len(h["edges_us"]) == 601 and h["edges_us"][0] == 1.0 and h["edges_us"][-1] == pytest.approx(1e6)
    skipped = 7
    e, y, n = stats.ccdf_from_hist(h, skipped)
    assert n == x.size + skipped
    # conservative CCDF at an edge == exact fraction >= edge; >= exact P(X > edge)
    exact_ge = np.array([(np.count_nonzero(x >= v) + skipped) / n for v in e])
    assert np.allclose(y, exact_ge)
    exact_gt = stats.ccdf_at(x, e, skipped)
    assert np.all(y >= exact_gt - 1e-15)
    # with continuous data no sample sits on an edge: equality
    assert np.allclose(y, exact_gt)


def test_ccdf_raw_definition():
    v, p, n = stats.ccdf_raw([1, 2, 2, 3], n_inf=1)
    assert list(v) == [1, 2, 3] and n == 5
    assert np.allclose(p, [4 / 5, 2 / 5, 1 / 5])  # P(X > v); +inf sample keeps the tail at 1/5
    assert np.allclose(stats.ccdf_at([1, 2, 2, 3], [0, 2, 3], 1), [1.0, 2 / 5, 1 / 5])


def test_merge_hists():
    a = stats.log_histogram([10, 20, 0.1])
    b = stats.log_histogram([10, 2e6])
    m = stats.merge_hists([a, b])
    assert sum(m["counts"]) == 3 and m["underflow"] == 1 and m["overflow"] == 1


def test_fit_clock_recovers_rate():
    rng = np.random.default_rng(2)
    n = 5000
    t0 = 10**12 + np.cumsum(rng.integers(20_000, 40_000, n))
    w = (800 + rng.exponential(400, n) + (rng.random(n) < 0.1) * rng.exponential(30_000, n)).astype(np.int64)
    rate = 37e-6
    true = lambda h: np.uint64(5 * 10**15) + ((h - 10**12) * (1 + rate)).astype(np.int64).astype(np.uint64)
    g = true(t0 + (rng.random(n) * w).astype(np.int64))
    f = stats.fit_clock(t0, t0 + w, g)
    assert f["ok"] and f["n_kept"] >= 50
    assert f["rate_ppm"] == pytest.approx(-rate / (1 + rate) * 1e6, abs=0.5)  # host per GPU ns
    # mapping error on fresh points is within eps
    h = t0[::97] + 400
    err = stats.host_minus(f, true(h), h)
    assert np.max(np.abs(err)) <= f["eps_ns"] + 1
    assert not stats.fit_clock([], [], [])["ok"]
    assert not stats.fit_clock([1], [2], [3])["ok"]


def test_queue_delay_recovered_from_synth(tmp_path):
    d = tmp_path / "M1_sgemm_d100_r0"
    synth.make_run(str(d), "M1", "sgemm", 100, n_slots=3000, seed=5)
    rng = np.random.default_rng(5)
    t_start = int(1_000_000_000_000 + rng.integers(0, 10**12))
    _, true_qd = synth.simulate_slots(rng, 3000, "M1", "sgemm", 100, t_start)
    rec = sbio.read_slots(str(d / "slots.bin")).records
    meta = sbio.load_json(str(d / "meta.json"))
    fits = sbio.find_fits(meta)
    qd, mask, how = stats.queue_delay_ns(rec["g0"], rec["t0"], fits)
    assert how == "two_point" and mask.all()
    eps = max(fits["pre"]["eps_ns"], fits["post"]["eps_ns"])
    assert np.max(np.abs(qd - true_qd)) <= eps + 2


def test_derived_metrics_hand_example():
    r = np.zeros(2, dtype=sbio.RECORD_DTYPE)
    r["slot"] = [0, 1]
    r["t_sched"] = [1_000_000, 1_500_000]
    r["t_wake"] = [951_000, 1_452_000]
    r["t0"] = [1_000_100, 1_500_300]
    r["t_launched"] = [1_004_100, 1_505_300]
    r["t1"] = [1_200_100, 1_700_000]
    r["g0"] = [5_000_000, 0]
    r["g1"] = [5_190_000, 0]
    fit = {"ok": True, "a": 1.0, "b_ns": 0.0, "g_ref": 5_000_000, "t_ref": 1_006_100}
    d = stats.derived_metrics(r, 50_000, {"pre": fit})
    assert list(d["latency"]) == [200_000, 199_700]
    assert list(d["latency_from_boundary"]) == [200_100, 200_000]
    assert list(d["wake_overshoot"]) == [1_000, 2_000]
    assert list(d["start_lateness"]) == [100, 300]
    assert list(d["launch_cost"]) == [4_000, 5_000]
    assert list(d["gpu_exec"]) == [190_000]  # zero stamps excluded
    assert d["queue_delay"].tolist() == [6_000.0] and d["queue_delay_method"] == "pre"


def test_downsample_keeps_spikes():
    y = np.full(100_000, 200.0)
    y[12_345] = 9_999.0
    idx, mx, med = stats.downsample_max(y, 1000)
    assert mx.max() == 9_999.0 and len(idx) == 1000 and np.all(med == 200.0)


def test_cv():
    assert stats.coefficient_of_variation([1.0]) is None
    assert stats.coefficient_of_variation([1.0, 1.0, 1.0]) == 0.0
    assert stats.coefficient_of_variation([1.0, 3.0]) == pytest.approx(np.std([1, 3], ddof=1) / 2)
