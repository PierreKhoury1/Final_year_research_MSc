#!/usr/bin/env python3
"""Build the self-contained timeline viewer: embed compacted runs into template.html.
Usage: build.py OUT.html RUN.trace.json:"Name":"Note" [...]   (traces from analysis/gputrace_export.py)
The viewer can also open any .trace.json itself (Open a trace…)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from analysis.gputrace_compact import compact

out, specs = sys.argv[1], sys.argv[2:]
ds = []
for s in specs:
    path, name, note = (s.split(":", 2) + ["", ""])[:3]
    ds.append(compact(path, name or os.path.basename(path), note))
t = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")).read()
open(out, "w").write(t.replace("/*DATA*/[]", json.dumps(ds, separators=(",", ":"))))
print(out, len(ds), "runs")
