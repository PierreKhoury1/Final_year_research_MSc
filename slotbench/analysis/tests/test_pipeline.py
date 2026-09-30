"""End to end on the synthetic tree: summarize -> validate -> report, fault detection, summary-only trees."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from analysis import report, sbio, summarize, synth, validate

CELLS = [f"{m}_{w}_d{d}_r0" for m in ("M0", "M1", "M2") for w, d in
         [("idle", 0), ("sgemm", 50), ("sgemm", 100), ("llm", 50), ("llm", 100)]] + ["M3_idle_d0_r0"]


def _checks(v):
    return {c["name"]: c["status"] for c in v["checks"]}


def test_summary_contents(synth_tree):
    d = synth_tree / "synth" / "M0_sgemm_d100_r0"
    summarize.write_summary(str(d))
    s = json.loads((d / "summary.json").read_text())
    rec = sbio.read_slots(str(d / "slots.bin")).records
    lat = rec["t1"].astype(np.int64) - rec["t0"].astype(np.int64)
    gaps = int(np.sum(np.diff(rec["slot"].astype(np.int64)) - 1))
    assert s["miss"]["recorded"] == len(rec) and s["miss"]["skipped"] == gaps > 0
    assert s["miss"]["misses"] == int(np.sum(lat > 500_000)) + gaps
    assert s["latency_us"]["max"] == lat.max() / 1e3
    assert s["latency_us"]["p99_99"] == np.quantile(lat, 0.9999, method="higher") / 1e3
    assert s["latency_us"]["jitter"] == pytest.approx(s["latency_us"]["p99_99"] - s["latency_us"]["p50"])
    assert len(s["worst"]) == 200 and s["worst"][0]["latency_us"] == s["latency_us"]["max"]
    assert set(sbio.RECORD_DTYPE.names) <= set(s["worst"][0])
    assert sum(s["histogram"]["counts"]) + s["histogram"]["overflow"] + s["histogram"]["underflow"] == len(rec)
    assert s["queue_delay_us"]["method"] == "two_point"
    for k in ("p50", "p99", "max"):
        assert s["gpu_exec_us"][k] > 150 and s["wake_overshoot_us"][k] >= 0
    assert s["adversary"]["units_per_s"] > 0 and 0.9 < s["adversary_relative_throughput"] < 1.1
    assert s["validation"]["valid"]
    # deterministic output
    first = (d / "summary.json").read_text()
    summarize.write_summary(str(d))
    assert (d / "summary.json").read_text() == first


def test_report_end_to_end(synth_tree, tmp_path):
    out = tmp_path / "report"
    r = report.make_report(str(synth_tree), str(out), log=lambda *a: None)
    assert r["n_runs"] == len(CELLS) + 2 and r["n_invalid"] == 0
    md = (out / "report.md").read_text()
    for c in CELLS + ["SOLO_sgemm_r0", "SOLO_llm_r0"]:
        assert c in md, c
    for c in {c.rsplit("_r", 1)[0] for c in CELLS}:
        assert c in md, c
    assert "NOT SUPPORTED" in md and "synthetic" in md
    for f in ["heatmap_synth", "pareto_synth", "timeseries_synth_worst", "ccdf_synth_idle_d0",
              "ccdf_synth_sgemm_d100", "ccdf_synth_llm_d50"]:
        for ext in (".png", ".pdf"):
            assert (out / (f + ext)).stat().st_size > 1000, f + ext
        assert f"({f}.png)" in md
    df = pd.read_csv(out / "summary.csv")
    assert len(df) == len(CELLS) + 2 and set(report.RUN_COLS) == set(df.columns)
    cells = pd.read_csv(out / "cells.csv")
    assert len(cells) == len(CELLS)
    assert all((synth_tree / "synth" / c / "summary.json").exists() for c in CELLS)


def test_validate_flags_overflow_and_idle_misses(synth_tree, capsys):
    root = synth_tree / "synth"
    synth.make_run(str(root / "M1_sgemm_d50_r1"), "M1", "sgemm", 50, 1, 3000, 11, ring_overflows=3)
    synth.make_run(str(root / "M2_idle_d0_r1"), "M2", "idle", 0, 1, 3000, 12, extra_late=2)
    synth.make_run(str(root / "M0_llm_d50_r1"), "M0", "llm", 50, 1, 3000, 13, crash=True)
    synth.make_run(str(root / "M0_sgemm_d50_r1"), "M0", "sgemm", 50, 1, 3000, 14, throttle_reasons=0x20)
    synth.make_run(str(root / "M1_llm_d100_r1"), "M1", "llm", 100, 1, 3000, 15)  # valid second rep
    v = validate.validate_tree(str(synth_tree))
    runs = v["runs"]
    ov = runs["synth/M1_sgemm_d50_r1"]
    assert not ov["valid"] and _checks(ov)["ring_overflows"] == "fail"
    idle = runs["synth/M2_idle_d0_r1"]
    assert not idle["valid"] and _checks(idle)["idle_baseline"] == "fail"
    cr = runs["synth/M0_llm_d50_r1"]
    assert not cr["valid"] and _checks(cr)["not_crashed"] == "fail"
    th = runs["synth/M0_sgemm_d50_r1"]
    assert not th["valid"] and _checks(th)["throttle"] == "fail"
    assert runs["synth/M0_sgemm_d50_r0"]["valid"] and runs["synth/M2_idle_d0_r0"]["valid"]
    assert v["n_invalid"] == 4
    rep = {r["cell"]: r for r in v["repeatability"]}
    assert list(rep) == ["synth/M1_llm_d100"]  # invalid reps are left out
    assert rep["synth/M1_llm_d100"]["reps"] == [0, 1] and rep["synth/M1_llm_d100"]["cv"] is not None
    # CLI: table, validation.json, exit code 1
    assert validate.main([str(synth_tree)]) == 1
    assert (synth_tree / "validation.json").exists()
    assert "4 invalid" in capsys.readouterr().out
    # throttling can be downgraded to a warning
    v2 = validate.validate_tree(str(synth_tree), {"throttle_mode": "warn"})
    assert v2["runs"]["synth/M0_sgemm_d50_r1"]["valid"]


def test_validate_clean_tree_exit0(synth_tree):
    assert validate.main([str(synth_tree), "--out", str(synth_tree / "v.json")]) == 0


def test_validate_stamp_mismatch_and_short_run(synth_tree):
    d = synth_tree / "synth" / "M0_sgemm_d100_r0"
    meta = json.loads((d / "meta.json").read_text())
    meta["counts"]["stamp_mismatches"] = 2
    meta["config"]["slots"] = 10_000_000
    (d / "meta.json").write_text(json.dumps(meta))
    info = sbio.run_info(str(d))
    c = _checks(validate.check_run(info, summarize.build_summary(str(d), with_validation=False)))
    assert c["stamp_mismatches"] == "fail" and c["recorded_eq_requested"] == "fail"


def test_clock_lock_check(synth_tree):
    d = synth_tree / "synth" / "M1_llm_d50_r0"
    runj = json.loads((d / "run.json").read_text())
    runj["applied"]["gc_mhz"] = 1800
    (d / "run.json").write_text(json.dumps(runj))
    info = sbio.run_info(str(d))
    s = summarize.build_summary(str(d), with_validation=False)
    assert _checks(validate.check_run(info, s))["clocks_locked"] == "fail"
    runj["applied"]["lock_clocks"] = "NOT SUPPORTED: -lgc"
    (d / "run.json").write_text(json.dumps(runj))
    assert _checks(validate.check_run(info, s))["clocks_locked"] == "skip"


def test_summary_only_tree_still_reports(synth_tree, tmp_path):
    for info in sbio.discover_runs(str(synth_tree)):
        summarize.write_summary(info.path)
    for info in sbio.discover_runs(str(synth_tree)):
        for f in ("slots.bin", "calib_pre.csv", "calib_post.csv"):
            if info.has(f):
                os.remove(info.file(f))
    v = validate.validate_tree(str(synth_tree))
    assert v["n_invalid"] == 0
    out = tmp_path / "rep"
    r = report.make_report(str(synth_tree), str(out), log=lambda *a: None)
    assert r["n_runs"] == len(CELLS) + 2 and r["n_invalid"] == 0
    md = (out / "report.md").read_text()
    for c in CELLS:
        assert c in md
    assert (out / "ccdf_synth_sgemm_d100.png").exists() and (out / "heatmap_synth.png").exists()
    assert not (out / "timeseries_synth_worst.png").exists()  # needs raw records
    df = pd.read_csv(out / "summary.csv")
    cells = df[df.kind == "cell"]
    assert cells.source.eq("slots.bin").all() and cells.p99_99_us.notna().sum() == len(CELLS)


def test_partial_matrix_and_missing_files(synth_tree, tmp_path):
    root = synth_tree / "synth"
    # a cell whose driver never produced anything but the status file
    (root / "M4_sgemm_d50_r0").mkdir()
    (root / "M4_sgemm_d50_r0" / "status").write_text("invalid:driver exit 1\n")
    (root / "M4_sgemm_d50_r0" / "run.json").write_text(json.dumps({"not_supported": ["M4 skipped"]}))
    out = tmp_path / "rep"
    r = report.make_report(str(synth_tree), str(out), log=lambda *a: None)
    assert r["n_invalid"] == 1
    md = (out / "report.md").read_text()
    assert "M4_sgemm_d50_r0" in md and "driver exit 1" in md


def test_summarize_cli(synth_tree, capsys):
    d = synth_tree / "synth" / "M2_llm_d100_r0"
    assert summarize.main([str(d), str(synth_tree / "synth" / "SOLO_llm_r0")]) == 0
    out = capsys.readouterr().out
    assert "M2_llm_d100_r0: n=6000" in out and (d / "summary.json").exists()
