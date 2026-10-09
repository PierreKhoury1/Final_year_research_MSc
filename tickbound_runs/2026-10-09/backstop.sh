#!/bin/sh
# On-box backstop (9 Oct 2026): after $1 seconds, stop this instance through the vast API with the instance's own
# CONTAINER_API_KEY (GPU billing ends, the disk is kept), in case the local babysitter died with the session.
python3 - <<'PY'
import os
env = dict(l.split("=", 1) for l in open("/proc/1/environ", "rb").read().decode(errors="replace").split("\0") if "=" in l)
print("backstop armed:", bool(env.get("CONTAINER_API_KEY")) and bool(env.get("CONTAINER_ID")), flush=True)
PY
sleep "$1"
python3 - <<'PY'
import json, urllib.request
env = dict(l.split("=", 1) for l in open("/proc/1/environ", "rb").read().decode(errors="replace").split("\0") if "=" in l)
k, i = env.get("CONTAINER_API_KEY"), env.get("CONTAINER_ID")
if not (k and i):
    raise SystemExit("no CONTAINER_API_KEY / CONTAINER_ID")
req = urllib.request.Request(f"https://console.vast.ai/api/v0/instances/{i}/", data=json.dumps({"state": "stopped"}).encode(),
                             method="PUT", headers={"Authorization": "Bearer " + k, "Content-Type": "application/json"})
print(urllib.request.urlopen(req, timeout=60).read()[:300])
PY
