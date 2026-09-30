"""Readers for slotbench run directories (DESIGN.md sections 3, 4, 7).

slots.bin is a 64-byte header followed by 64-byte records. read_slots() validates the header, uses the
file size when the header says n_records == 0 (crashed run) and ignores a trailing partial record;
both conditions are reported as flags rather than silently hidden.

meta.json / run.json key names are not fixed by DESIGN.md, so lookups here are lenient (find_key,
find_fits); every name that is searched for is listed next to the function that searches it.
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"SLOTBEN1"
VERSION = 1
HEADER_SIZE = 64
RECORD_SIZE = 64

RECORD_DTYPE = np.dtype([("slot", "<u8"), ("t_sched", "<i8"), ("t_wake", "<i8"), ("t0", "<i8"),
                         ("t_launched", "<i8"), ("t1", "<i8"), ("g0", "<u8"), ("g1", "<u8")])
HEADER_DTYPE = np.dtype([("magic", "S8"), ("version", "<u4"), ("record_size", "<u4"), ("n_records", "<u8"),
                         ("period_ns", "<f8"), ("deadline_ns", "<f8"), ("t_start_ns", "<i8"),
                         ("reserved", "u1", (16,))])
assert RECORD_DTYPE.itemsize == RECORD_SIZE and HEADER_DTYPE.itemsize == HEADER_SIZE

# Cell names: <M>_<W>_d<D>_r<rep> (W may be "idle"), solo baselines SOLO_<W>_r<rep>.
CELL_RE = re.compile(r"^(?P<mechanism>M\d+)_(?P<workload>[A-Za-z0-9]+)_d(?P<duty>\d+)_r(?P<rep>\d+)$")
SOLO_RE = re.compile(r"^SOLO_(?P<workload>[A-Za-z0-9]+)_r(?P<rep>\d+)$")


class SlotFileError(ValueError):
    pass


@dataclass
class SlotFile:
    header: dict
    records: np.ndarray            # structured RECORD_DTYPE array (memory-mapped when large)
    crashed: bool = False          # header n_records == 0: count taken from the file size
    partial_tail_bytes: int = 0    # bytes of an incomplete trailing record that were ignored
    header_count_mismatch: bool = False  # header n_records disagrees with the file size
    flags: list = field(default_factory=list)


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise SlotFileError(f"{path}: file shorter than the {HEADER_SIZE}-byte header")
    h = np.frombuffer(raw, dtype=HEADER_DTYPE, count=1)[0]
    hdr = {"magic": bytes(h["magic"]), "version": int(h["version"]), "record_size": int(h["record_size"]),
           "n_records": int(h["n_records"]), "period_ns": float(h["period_ns"]),
           "deadline_ns": float(h["deadline_ns"]), "t_start_ns": int(h["t_start_ns"])}
    if hdr["magic"] != MAGIC:
        raise SlotFileError(f"{path}: bad magic {hdr['magic']!r}")
    if hdr["version"] != VERSION:
        raise SlotFileError(f"{path}: unsupported version {hdr['version']}")
    if hdr["record_size"] != RECORD_SIZE:
        raise SlotFileError(f"{path}: record_size {hdr['record_size']} != {RECORD_SIZE}")
    return hdr


def read_slots(path: str, mmap_threshold: int = 64 << 20) -> SlotFile:
    """Read slots.bin. Files larger than mmap_threshold bytes are memory-mapped (read-only)."""
    hdr = read_header(path)
    size = os.path.getsize(path)
    body = size - HEADER_SIZE
    n_file = body // RECORD_SIZE
    tail = body - n_file * RECORD_SIZE
    flags = []
    crashed = hdr["n_records"] == 0
    n = n_file
    mismatch = False
    if crashed:
        flags.append("crashed")
    elif hdr["n_records"] != n_file:
        # Clean close but sizes disagree: trust the smaller (never read beyond what the header claims
        # was written completely, never invent records that are not in the file).
        mismatch = True
        n = min(n_file, hdr["n_records"])
        flags.append("header_count_mismatch")
    if tail:
        flags.append("partial_trailing_record")
    if n == 0:
        rec = np.zeros(0, dtype=RECORD_DTYPE)
    elif size > mmap_threshold:
        rec = np.memmap(path, dtype=RECORD_DTYPE, mode="r", offset=HEADER_SIZE, shape=(n,))
    else:
        with open(path, "rb") as f:
            f.seek(HEADER_SIZE)
            rec = np.fromfile(f, dtype=RECORD_DTYPE, count=n)
    return SlotFile(hdr, rec, crashed, tail, mismatch, flags)


def write_slots(path: str, records: np.ndarray, period_ns: float, deadline_ns: float, t_start_ns: int,
                n_records: int | None = None) -> None:
    """Write a slots.bin (used by synth and tests). n_records=None writes len(records); 0 mimics a crash."""
    h = np.zeros(1, dtype=HEADER_DTYPE)
    h["magic"] = MAGIC
    h["version"] = VERSION
    h["record_size"] = RECORD_SIZE
    h["n_records"] = len(records) if n_records is None else n_records
    h["period_ns"] = period_ns
    h["deadline_ns"] = deadline_ns
    h["t_start_ns"] = t_start_ns
    with open(path, "wb") as f:
        f.write(h.tobytes())
        f.write(np.ascontiguousarray(records, dtype=RECORD_DTYPE).tobytes())


# ---------------------------------------------------------------- JSON helpers

def load_json(path: str):
    """Parsed JSON or None if the file is missing/unparseable (a crash can leave a truncated file)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def iter_items(obj, path=()):
    """Depth-first (key path, value) pairs over nested dicts/lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path + (k,), v
            yield from iter_items(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_items(v, path + (i,))


def find_key(obj, names, prefer=None, default=None):
    """First value whose key is one of `names` (in order of `names`). A match under the top-level key
    `prefer` (e.g. "config" or "counts") wins over matches elsewhere."""
    if not isinstance(obj, dict):
        return default
    if prefer and isinstance(obj.get(prefer), dict):
        for n in names:
            if n in obj[prefer]:
                return obj[prefer][n]
    for n in names:
        if n in obj:
            return obj[n]
    for n in names:
        for p, v in iter_items(obj):
            if p and p[-1] == n:
                return v
    return default


FIT_KEYS = {"a", "g_ref", "t_ref"}


def find_fits(meta) -> dict:
    """Clock fits in meta.json: any dict holding a, g_ref, t_ref (the ClockFit::json layout). The key path
    decides pre/post ("pre"/"post" substring); unlabelled fits are assigned in order of appearance."""
    out, unlabelled = {}, []
    for p, v in iter_items(meta or {}):
        if isinstance(v, dict) and FIT_KEYS <= set(v):
            tag = "/".join(str(x) for x in p).lower()
            if "post" in tag and "post" not in out:
                out["post"] = v
            elif "pre" in tag and "pre" not in out:
                out["pre"] = v
            else:
                unlabelled.append(v)
    for v in unlabelled:
        for k in ("pre", "post"):
            if k not in out:
                out[k] = v
                break
    return out


def meta_counts(meta) -> dict:
    """Driver counters from meta.json (None when absent). Names searched: recorded/n_recorded,
    skipped_boundaries/skipped/n_skipped, stamp_mismatches/stamp_mismatch, ring_overflows/overflows."""
    m = meta or {}
    g = lambda names: find_key(m, names, prefer="counts")
    return {"recorded": g(["recorded", "n_recorded", "recorded_slots"]),
            "skipped": g(["skipped_boundaries", "skipped", "n_skipped"]),
            "stamp_mismatches": g(["stamp_mismatches", "stamp_mismatch"]),
            "ring_overflows": g(["ring_overflows", "overflows", "ring_overflow"])}


def meta_config_value(meta, names, default=None):
    return find_key(meta or {}, names, prefer="config", default=default)


def spin_ns(meta) -> float | None:
    """--spin-us from meta config (spin_us or spin_ns)."""
    v = meta_config_value(meta, ["spin_ns"])
    if isinstance(v, (int, float)):
        return float(v)
    v = meta_config_value(meta, ["spin_us"])
    return float(v) * 1e3 if isinstance(v, (int, float)) else None


def requested_slots(meta):
    v = meta_config_value(meta, ["slots", "n_slots", "requested_slots"])
    return int(v) if isinstance(v, (int, float)) else None


def exit_reason(meta):
    v = find_key(meta or {}, ["exit_reason"])
    return v if isinstance(v, str) else None


# ---------------------------------------------------------------- CSV readers

def read_calib_csv(path: str):
    """calib_pre/post.csv (t0_ns,t1_ns,gpu_ns) -> (t0, t1, g) int arrays, or None."""
    try:
        with open(path) as f:
            rows = list(csv.reader(f))
    except OSError:
        return None
    t0, t1, g = [], [], []
    for r in rows:
        if len(r) < 3:
            continue
        try:
            a, b, c = int(r[0]), int(r[1]), int(r[2])
        except ValueError:
            continue  # header or junk
        t0.append(a)
        t1.append(b)
        g.append(c)
    return np.array(t0, np.int64), np.array(t1, np.int64), np.array(g, np.uint64)


_TELEM_MAP = [  # (canonical, predicate on normalised header name)
    ("timestamp", lambda h: h.startswith("timestamp")),
    ("temp_c", lambda h: h.startswith("temperature")),
    ("sm_mhz", lambda h: "clocks" in h and ".sm" in h),
    ("mem_mhz", lambda h: "clocks" in h and ".mem" in h),
    ("power_w", lambda h: h.startswith("power.draw")),
    ("util_pct", lambda h: h.startswith("utilization.gpu")),
    ("reasons", lambda h: "reasons" in h),
]


def _num(s):
    s = s.strip()
    if not s or "N/A" in s or "Not Supported" in s or "Unknown" in s:
        return None
    if s.lower().startswith("0x"):
        try:
            return int(s, 16)
        except ValueError:
            return None
    m = re.match(r"^[-+]?\d+(\.\d*)?([eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else None


def read_telemetry(path: str):
    """nvidia-smi --format=csv output -> dict of canonical column -> list (None for missing values).
    Header units ("[MHz]") are stripped, repeated header lines skipped, unknown columns kept by name.
    Returns None if the file is missing or has no header."""
    try:
        with open(path, newline="") as f:
            rows = [r for r in csv.reader(f, skipinitialspace=True) if r and any(c.strip() for c in r)]
    except OSError:
        return None
    if not rows:
        return None
    header = rows[0]
    names = []
    for h in header:
        hn = re.sub(r"\s*\[.*?\]\s*", "", h).strip().lower()
        canon = next((c for c, pred in _TELEM_MAP if pred(hn)), hn)
        names.append(canon)
    out = {n: [] for n in names}
    for r in rows[1:]:
        if r[0].strip().lower().startswith("timestamp"):
            continue  # restarted logger wrote a second header
        for i, n in enumerate(names):
            v = r[i] if i < len(r) else ""
            out[n].append(v.strip() if n == "timestamp" else _num(v))
    return out


# ---------------------------------------------------------------- discovery

@dataclass
class RunInfo:
    path: str
    name: str
    config: str               # parent directory name (runs/<config>/<cell>)
    kind: str                 # "cell" or "solo"
    mechanism: str | None
    workload: str
    duty: int | None
    rep: int

    @property
    def cell(self) -> str:
        """Identity of the cell across reps."""
        if self.kind == "solo":
            return f"{self.config}/SOLO_{self.workload}"
        return f"{self.config}/{self.mechanism}_{self.workload}_d{self.duty}"

    def file(self, name):
        return os.path.join(self.path, name)

    def has(self, name):
        return os.path.exists(self.file(name))


def parse_cell_name(name: str):
    """(kind, mechanism, workload, duty, rep) or None if the name is not a run cell."""
    m = CELL_RE.match(name)
    if m:
        return "cell", m["mechanism"], m["workload"], int(m["duty"]), int(m["rep"])
    m = SOLO_RE.match(name)
    if m:
        return "solo", None, m["workload"], None, int(m["rep"])
    return None


RUN_FILES = ("slots.bin", "summary.json", "meta.json", "adversary.json", "run.json", "status")


def run_info(path: str) -> RunInfo | None:
    path = os.path.abspath(path)
    name = os.path.basename(path.rstrip("/"))
    p = parse_cell_name(name)
    if p is None:
        return None
    kind, mech, w, d, rep = p
    return RunInfo(path, name, os.path.basename(os.path.dirname(path)), kind, mech, w, d, rep)


def discover_runs(root: str) -> list[RunInfo]:
    """All run dirs under root (root itself may be a run dir) holding at least one run file, sorted."""
    found = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        info = run_info(dirpath)
        if info and any(f in filenames for f in RUN_FILES):
            found.append(info)
            dirnames[:] = []
    found.sort(key=lambda r: (r.config, r.kind != "solo", r.mechanism or "", r.workload, r.duty or 0, r.rep))
    return found
