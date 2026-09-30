"""Readers: slots.bin round trip, crash/partial-record handling, header validation, lenient JSON/CSV."""
import numpy as np
import pytest

from analysis import sbio


def _records(n, t0=10**12):
    r = np.zeros(n, dtype=sbio.RECORD_DTYPE)
    r["slot"] = np.arange(n)
    r["t_sched"] = t0 + np.arange(n) * 500_000
    r["t_wake"] = r["t_sched"] - 47_000
    r["t0"] = r["t_sched"] + 200
    r["t_launched"] = r["t0"] + 4_000
    r["t1"] = r["t0"] + 200_000 + np.arange(n)
    r["g0"] = np.uint64(2**63 + 5) + np.arange(n, dtype=np.uint64)  # exercises the full u64 range
    r["g1"] = r["g0"] + np.uint64(190_000)
    return r


def test_dtype_layout_matches_design():
    assert sbio.RECORD_DTYPE.itemsize == 64 and sbio.HEADER_DTYPE.itemsize == 64
    assert [sbio.RECORD_DTYPE.fields[n][1] for n in sbio.RECORD_DTYPE.names] == [0, 8, 16, 24, 32, 40, 48, 56]
    assert sbio.HEADER_DTYPE.fields["n_records"][1] == 16
    assert sbio.HEADER_DTYPE.fields["period_ns"][1] == 24
    assert sbio.HEADER_DTYPE.fields["t_start_ns"][1] == 40


def test_round_trip(tmp_path):
    p = str(tmp_path / "slots.bin")
    r = _records(1000)
    sbio.write_slots(p, r, 500_000.0, 450_000.0, 10**12)
    f = sbio.read_slots(p)
    assert f.header["n_records"] == 1000 and f.header["deadline_ns"] == 450_000.0
    assert f.header["t_start_ns"] == 10**12 and not f.crashed and f.flags == []
    assert np.array_equal(f.records, r)


def test_round_trip_memmap(tmp_path):
    p = str(tmp_path / "slots.bin")
    r = _records(5000)
    sbio.write_slots(p, r, 500_000.0, 500_000.0, 0)
    f = sbio.read_slots(p, mmap_threshold=0)
    assert isinstance(f.records, np.memmap)
    assert np.array_equal(np.asarray(f.records), r)


def test_crashed_header_uses_file_size(tmp_path):
    p = str(tmp_path / "slots.bin")
    sbio.write_slots(p, _records(321), 500_000.0, 500_000.0, 0, n_records=0)
    f = sbio.read_slots(p)
    assert f.crashed and "crashed" in f.flags and len(f.records) == 321


def test_partial_trailing_record_ignored(tmp_path):
    p = str(tmp_path / "slots.bin")
    r = _records(10)
    sbio.write_slots(p, r, 500_000.0, 500_000.0, 0, n_records=0)
    with open(p, "ab") as fh:
        fh.write(b"\x01" * 37)
    f = sbio.read_slots(p)
    assert len(f.records) == 10 and f.partial_tail_bytes == 37
    assert "partial_trailing_record" in f.flags
    assert np.array_equal(f.records, r)


def test_header_count_mismatch_flagged(tmp_path):
    p = str(tmp_path / "slots.bin")
    sbio.write_slots(p, _records(10), 500_000.0, 500_000.0, 0, n_records=12)
    f = sbio.read_slots(p)
    assert f.header_count_mismatch and len(f.records) == 10


@pytest.mark.parametrize("field,value", [("magic", b"NOTSLOTB"), ("version", 2), ("record_size", 72)])
def test_bad_header_rejected(tmp_path, field, value):
    p = str(tmp_path / "slots.bin")
    sbio.write_slots(p, _records(2), 500_000.0, 500_000.0, 0)
    raw = bytearray(open(p, "rb").read())
    h = np.frombuffer(bytes(raw[:64]), dtype=sbio.HEADER_DTYPE).copy()
    h[field] = value
    raw[:64] = h.tobytes()
    open(p, "wb").write(bytes(raw))
    with pytest.raises(sbio.SlotFileError):
        sbio.read_slots(p)


def test_short_file_rejected(tmp_path):
    p = tmp_path / "slots.bin"
    p.write_bytes(b"SLOTBEN1")
    with pytest.raises(sbio.SlotFileError):
        sbio.read_slots(str(p))


def test_cell_names():
    assert sbio.parse_cell_name("M1_sgemm_d50_r2") == ("cell", "M1", "sgemm", 50, 2)
    assert sbio.parse_cell_name("M0_idle_d0_r0") == ("cell", "M0", "idle", 0, 0)
    assert sbio.parse_cell_name("SOLO_llm_r1") == ("solo", None, "llm", None, 1)
    assert sbio.parse_cell_name("matrix.log") is None
    assert sbio.parse_cell_name("M1_sgemm_d50") is None


def test_telemetry_lenient(tmp_path):
    p = tmp_path / "telemetry.csv"
    p.write_text(
        "timestamp, temperature.gpu, clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W], "
        "utilization.gpu [%], clocks_throttle_reasons.active\n"
        "2026/09/30 10:00:00.000, 50, 1500 MHz, 7501 MHz, 100.50 W, 99 %, 0x0000000000000004\n"
        "timestamp, temperature.gpu, clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W], "
        "utilization.gpu [%], clocks_throttle_reasons.active\n"
        "2026/09/30 10:00:01.000, 51, [N/A], 7501 MHz, [N/A], 98 %, 0x0000000000000000\n"
        "2026/09/30 10:00:02.000, 51, 1500 MHz\n")
    t = sbio.read_telemetry(str(p))
    assert t["sm_mhz"] == [1500.0, None, 1500.0]
    assert t["mem_mhz"] == [7501.0, 7501.0, None]
    assert t["power_w"] == [100.5, None, None]
    assert t["reasons"] == [4, 0, None]
    assert t["temp_c"] == [50.0, 51.0, 51.0]


def test_find_fits_and_counts():
    meta = {"config": {"slots": 100, "spin_us": 50},
            "counts": {"recorded": 100, "skipped_boundaries": 3, "stamp_mismatches": 0, "ring_overflows": 1},
            "clock": {"fit_post": {"a": 1.0, "g_ref": 5, "t_ref": 7, "b_ns": 0.0},
                      "fit_pre": {"a": 1.0, "g_ref": 1, "t_ref": 2, "b_ns": 0.0}}}
    f = sbio.find_fits(meta)
    assert f["pre"]["g_ref"] == 1 and f["post"]["g_ref"] == 5
    c = sbio.meta_counts(meta)
    assert c == {"recorded": 100, "skipped": 3, "stamp_mismatches": 0, "ring_overflows": 1}
    assert sbio.spin_ns(meta) == 50_000.0 and sbio.requested_slots(meta) == 100
    assert sbio.meta_counts(None)["ring_overflows"] is None


def test_discover(synth_tree_ro):
    runs = sbio.discover_runs(str(synth_tree_ro))
    names = [r.name for r in runs]
    assert names[:2] == ["SOLO_llm_r0", "SOLO_sgemm_r0"]
    assert "M1_sgemm_d100_r0" in names and "M3_idle_d0_r0" in names
    assert all(r.config == "synth" for r in runs)
    assert len(runs) == 2 + 3 * 5 + 1
