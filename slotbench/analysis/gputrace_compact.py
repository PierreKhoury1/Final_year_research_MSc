#!/usr/bin/env python3
"""Compact a gputrace Chrome trace (gputrace_export.py output) for the timeline viewer (tools/timeline.html):
lanes [{p: process, t: thread, b: bound_ns}], labels [...], ev [[lane, ts_us, dur_us, label_idx, kernel], ...].
Usage: gputrace_compact.py TRACE.json NAME [--note TEXT] > out.json"""
import json, re, sys


def compact(path, name, note=""):
    tr = json.load(open(path))
    pname, tname = {}, {}
    for e in tr["traceEvents"]:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pname[e["pid"]] = e["args"]["name"]
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            tname[(e["pid"], e["tid"])] = e["args"]["name"]
    lanes, lane_of, labels, label_of, ev = [], {}, [], {}, []
    for e in sorted((e for e in tr["traceEvents"] if e.get("ph") in ("X", "i")), key=lambda e: (e["pid"], e.get("tid", 0), e["ts"])):
        key = (e["pid"], e.get("tid", 0))
        if key not in lane_of:
            m = re.search(r"±(\d+) ns", pname.get(e["pid"], ""))
            lane_of[key] = len(lanes)
            lanes.append(dict(p=re.sub(r"\s*\(.*\)$", "", pname.get(e["pid"], f"pid {e['pid']}")), t=tname.get(key, f"{key[1]}"),
                              b=int(m.group(1)) if m else 0, pid=e["pid"]))
        lab = re.sub(r" k\d+.*$", "", e["name"]) if e["pid"] == 1 else e["name"]
        lab = re.sub(r" b\d+$", "", lab).replace("hog ", "other process ")
        if lab not in label_of:
            label_of[lab] = len(labels)
            labels.append(lab)
        a = e.get("args", {})
        ev.append([lane_of[key], round(e["ts"], 3), round(e.get("dur", 0), 3), label_of[lab], a.get("kernel", 0), a.get("bytes", a.get("a", 0))])
    return dict(name=name, note=note, lanes=lanes, labels=labels, ev=ev)


if __name__ == "__main__":
    note = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else ""
    json.dump(compact(sys.argv[1], sys.argv[2], note), sys.stdout, separators=(",", ":"))
