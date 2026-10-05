#!/usr/bin/env python3
"""Report PCIe Precision Time Measurement (PTM, extended capability 0x001F) for every NVIDIA GPU and each
bridge between it and the root complex, from /sys/bus/pci/devices/*/config. Prints JSON.
Extended config space (offset >= 0x100) is often hidden from unprivileged readers; that case is reported
as "ext_config_readable": false rather than as "no PTM"."""
import json
import os
import struct

PTM_ID = 0x001F


def ext_caps(cfg):
    caps, off, seen = [], 0x100, set()
    while 0x100 <= off < len(cfg) - 3 and off not in seen:
        seen.add(off)
        hdr = struct.unpack_from("<I", cfg, off)[0]
        if hdr in (0, 0xFFFFFFFF):
            break
        caps.append((hdr & 0xFFFF, (hdr >> 16) & 0xF, off))
        off = (hdr >> 20) & 0xFFC
        if off == 0:
            break
    return caps


def describe(bdf):
    path = f"/sys/bus/pci/devices/{bdf}"
    d = {"bdf": bdf}
    for f in ("vendor", "device", "class"):
        try:
            d[f] = open(f"{path}/{f}").read().strip()
        except OSError:
            pass
    try:
        cfg = open(f"{path}/config", "rb").read()
    except OSError as e:
        d["error"] = str(e)
        return d
    d["config_bytes"] = len(cfg)
    d["ext_config_readable"] = len(cfg) > 0x100 and any(cfg[0x100:0x104])
    caps = ext_caps(cfg) if d["ext_config_readable"] else []
    d["ext_cap_ids"] = [f"0x{c:04x}" for c, _, _ in caps]
    ptm = [o for c, _, o in caps if c == PTM_ID]
    d["ptm_present"] = bool(ptm) if d["ext_config_readable"] else None
    if ptm:
        cap, ctl = struct.unpack_from("<II", cfg, ptm[0] + 4)
        d["ptm"] = {"requester_capable": bool(cap & 1), "responder_capable": bool(cap & 2),
                    "root_capable": bool(cap & 4), "local_clock_granularity_ns": (cap >> 8) & 0xFF,
                    "enabled": bool(ctl & 1), "root_select": bool(ctl & 2)}
    return d


out = []
for bdf in sorted(os.listdir("/sys/bus/pci/devices")):
    try:
        if open(f"/sys/bus/pci/devices/{bdf}/vendor").read().strip() != "0x10de":
            continue
        if not open(f"/sys/bus/pci/devices/{bdf}/class").read().strip().startswith("0x03"):
            continue
    except OSError:
        continue
    real = os.path.realpath(f"/sys/bus/pci/devices/{bdf}")
    chain = [p for p in real.split("/") if len(p) == 12 and p.count(":") == 2]   # upstream bridges ... gpu
    out.append({"gpu": bdf, "path": [describe(p) for p in chain]})
print(json.dumps({"gpus": out, "uid": os.getuid()}, indent=1))
