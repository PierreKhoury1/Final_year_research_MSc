#!/usr/bin/env python3
"""vast.ai controller for slotbench (DESIGN.md section 9). Standard library only.

Subcommands: offers, launch, status, logs, collect, destroy, run.
The API key is read from the environment variable VAST_API_KEY only. It is sent as a Bearer header,
never printed, logged, written to disk or put into a URL; every error message passes through redact().
The instance runs cloud/onstart.sh (fetched from raw.githubusercontent.com at the chosen branch), which
prints each run's small result files as base64 tar.gz blocks between markers in the container log;
`collect` extracts, verifies (sha256) and untars them.
"""
import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import random
import re
import shlex
import signal
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_LOG_TAIL = 20000  # vast.ai rejects request_logs with tail > 20000 lines

API_BASE = "https://console.vast.ai/api/v0"
KEY_ENV = "VAST_API_KEY"
DEFAULT_REPO = "https://github.com/PierreKhoury1/Final_year_research_MSc"
DEFAULT_BRANCH = "claude/optimistic-ptolemy-r42xrh"
DEFAULT_IMAGE = "nvidia/cuda:12.6.3-devel-ubuntu24.04"
DEFAULT_CONFIG = "cloud.toml"
SUBDIR = "slotbench"
HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, ".state")

BEGIN_RE = re.compile(r"=====SLOTBENCH-BEGIN (\S+)(?: ([0-9a-fA-F]{64}))?=====")
END_RE = re.compile(r"=====SLOTBENCH-END (\S+)=====")
DONE_RE = re.compile(r"=====SLOTBENCH-DONE(?: status=(\S+?))?=====")
ERROR_RE = re.compile(r"=====SLOTBENCH-ERROR .*?=====")
SAFE_ARG_RE = re.compile(r"^[A-Za-z0-9._/:@+-]+$")


class VastError(Exception):
    """Any controller failure; the message is already redacted."""

    def __init__(self, msg, status=None, ambiguous=False):
        super().__init__(redact(msg))
        self.status = status
        self.ambiguous = ambiguous  # request may have been applied server-side


# ---------------------------------------------------------------- secrets

_SECRETS = set()


def register_secret(s):
    """Remember a secret so redact() removes it from any text."""
    if s:
        _SECRETS.add(s)


def redact(text):
    """Remove every registered secret (and any Bearer token / api_key param) from text."""
    text = str(text)
    for s in sorted(_SECRETS, key=len, reverse=True):
        text = text.replace(s, "***")
    text = re.sub(r"(Bearer\s+)[^\s\"']+", r"\1***", text)
    text = re.sub(r"(api_key=)[^&\s\"']+", r"\1***", text)
    return text


def get_key(required=True):
    """API key from env VAST_API_KEY; exits with a clear message if it is required and missing."""
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        if required:
            sys.exit(f"error: this subcommand needs a vast.ai API key in the environment variable {KEY_ENV} "
                     f"(export {KEY_ENV}=...; never paste it into chats or commit it)")
        return None
    register_secret(key)
    return key


# ---------------------------------------------------------------- HTTP

def ssl_context():
    """Default context plus SSL_CERT_FILE / REQUESTS_CA_BUNDLE if set (corporate proxies)."""
    ctx = ssl.create_default_context()
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        path = os.environ.get(var)
        if path and os.path.isfile(path):
            try:
                ctx.load_verify_locations(cafile=path)
            except (ssl.SSLError, OSError):
                pass
    return ctx


def default_opener():
    """urllib opener honouring HTTPS_PROXY/NO_PROXY from the environment."""
    return urllib.request.build_opener(urllib.request.ProxyHandler(),
                                       urllib.request.HTTPSHandler(context=ssl_context()))


class Api:
    """Minimal vast.ai REST client: timeouts, retries with backoff on 429/5xx and network errors."""

    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, key=None, base=API_BASE, opener=None, sleep=time.sleep, retries=5, timeout=60):
        self.key = key
        if key:
            register_secret(key)
        self.base = base.rstrip("/")
        self.opener = opener or default_opener()
        self.sleep = sleep
        self.retries = retries
        self.timeout = timeout

    def _headers(self, auth):
        h = {"User-Agent": "slotbench-vast/1", "Accept": "application/json"}
        if auth:
            if not self.key:
                raise VastError(f"missing API key: set {KEY_ENV}")
            h["Authorization"] = "Bearer " + self.key
        return h

    def request(self, method, url, body=None, auth=True, idempotent=True, parse=True, timeout=None):
        """One API call. Non-idempotent calls (create) are retried only on 429 (not processed)."""
        if not url.startswith("http"):
            url = self.base + url
        data = None
        headers = self._headers(auth)
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        delay = 1.0
        last = None
        for attempt in range(self.retries):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with self.opener.open(req, timeout=timeout or self.timeout) as r:
                    raw = r.read()
                if not parse:
                    return raw.decode("utf-8", "replace")
                return json.loads(raw.decode("utf-8")) if raw.strip() else {}
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode("utf-8", "replace")[:500]
                except Exception:  # noqa: BLE001 - body is best effort
                    detail = ""
                last = VastError(f"{method} {url} -> HTTP {e.code}: {detail}", status=e.code,
                                 ambiguous=(not idempotent and e.code >= 500))
                retry = e.code == 429 or (idempotent and e.code in self.RETRY_STATUS)
                if not retry:
                    raise last from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                reason = getattr(e, "reason", e)
                last = VastError(f"{method} {url} -> network error: {reason}", ambiguous=not idempotent)
                if not idempotent:
                    raise last from None
            except ValueError as e:
                raise VastError(f"{method} {url} -> invalid JSON response: {e}") from None
            if attempt + 1 < self.retries:
                self.sleep(min(delay, 30.0) * (1 + 0.1 * random.random()))
                delay *= 2
        raise last

    # --- endpoints (see vastai SDK api/instances.py, api/offers.py)
    def search_offers(self, query):
        return self.request("POST", "/bundles/", body=query, auth=bool(self.key)).get("offers", [])

    def create_instance(self, offer_id, payload):
        return self.request("PUT", f"/asks/{int(offer_id)}/", body=payload, idempotent=False)

    def show_instance(self, iid):
        r = self.request("GET", f"/instances/{int(iid)}/?owner=me")
        return r.get("instances")

    def list_instances(self):
        return self.request("GET", "/instances/?owner=me").get("instances", []) or []

    def destroy_instance(self, iid):
        return self.request("DELETE", f"/instances/{int(iid)}/", body={})

    def logs(self, iid, tail=None, wait_s=60.0):
        """Request container logs; fetch the returned result_url (no auth header: it is a storage URL)."""
        body = {"tail": str(min(int(tail), MAX_LOG_TAIL))} if tail else {}
        r = self.request("PUT", f"/instances/request_logs/{int(iid)}/", body=body)
        url = r.get("result_url")
        if not url:
            raise VastError(f"logs: no result_url in response: {str(r)[:300]}")
        t_end = time.monotonic() + wait_s
        while True:
            self.sleep(1.0)
            try:
                return self.request("GET", url, auth=False, parse=False, timeout=60)
            except VastError as e:
                # the log file appears a few seconds after the request (403/404 until then)
                if e.status not in (403, 404) or time.monotonic() > t_end:
                    raise


# ---------------------------------------------------------------- offers

def build_offer_query(gpu, max_dph=None, min_rel=None, cuda=None, limit=10, verified=False, disk=40.0):
    """Server-side search query (POST /bundles/): cheapest rentable on-demand single-GPU offers."""
    q = {
        "gpu_name": {"eq": gpu},
        "num_gpus": {"eq": 1},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "external": {"eq": False},
        "order": [["dph_total", "asc"]],
        "type": "on-demand",
        "limit": int(max(limit * 3, limit)),  # over-fetch; client-side filters may drop some
        "allocated_storage": float(disk),     # dph_total then includes this much disk
    }
    if max_dph is not None:
        q["dph_total"] = {"lte": float(max_dph)}
    if min_rel is not None:
        q["reliability2"] = {"gte": float(min_rel)}
    if cuda is not None:
        q["cuda_max_good"] = {"gte": float(cuda)}
    if verified:
        q["verified"] = {"eq": True}
    return q


def filter_offers(offers, gpu, max_dph=None, min_rel=None, cuda=None, limit=10, verified=False):
    """Re-apply every filter locally (never trust the server alone) and sort by price, then reliability."""
    out = []
    for o in offers:
        if o.get("gpu_name") != gpu or int(o.get("num_gpus") or 0) != 1:
            continue
        if o.get("rentable") is False or o.get("rented") is True:
            continue
        dph = o.get("dph_total")
        if dph is None or (max_dph is not None and dph > max_dph):
            continue
        rel = o.get("reliability2", o.get("reliability"))
        if min_rel is not None and (rel is None or rel < min_rel):
            continue
        if cuda is not None and (o.get("cuda_max_good") or 0) < cuda:
            continue
        if verified and o.get("verification", "verified" if o.get("verified") else "") != "verified":
            continue
        out.append(o)
    out.sort(key=lambda o: (o["dph_total"], -(o.get("reliability2") or o.get("reliability") or 0), o.get("id", 0)))
    return out[:limit]


def format_offers(offers):
    """Fixed-width table of offers."""
    cols = [("id", 10), ("gpu", 12), ("$/h", 7), ("cuda", 5), ("driver", 11), ("rel", 6), ("cc", 4),
            ("cpu", 5), ("ram_GB", 6), ("inet_dn/up", 11), ("verif", 10), ("location", 20)]
    lines = ["  ".join(f"{n:<{w}}" for n, w in cols)]
    for o in offers:
        rel = o.get("reliability2", o.get("reliability")) or 0
        cc = o.get("compute_cap")
        vals = [
            str(o.get("id", "")), str(o.get("gpu_name", ""))[:12], f"{o.get('dph_total', 0):.3f}",
            f"{o.get('cuda_max_good', 0):.1f}", str(o.get("driver_version", ""))[:11], f"{rel:.3f}",
            str(cc // 10 if isinstance(cc, int) and cc > 99 else cc or "?"),
            f"{o.get('cpu_cores_effective', o.get('cpu_cores', 0)):.0f}",
            f"{(o.get('cpu_ram') or 0) / 1024:.0f}",
            f"{o.get('inet_down', 0):.0f}/{o.get('inet_up', 0):.0f}",
            str(o.get("verification", ""))[:10], str(o.get("geolocation") or "")[:20],
        ]
        lines.append("  ".join(f"{v:<{w}}" for v, (_, w) in zip(vals, cols)))
    return "\n".join(lines)


def find_offers(api, args):
    q = build_offer_query(args.gpu, args.max_dph, args.min_reliability, args.cuda, args.limit,
                          args.verified, getattr(args, "disk", 40.0))
    offers = api.search_offers(q)
    return filter_offers(offers, args.gpu, args.max_dph, args.min_reliability, args.cuda, args.limit,
                         args.verified)


# ---------------------------------------------------------------- launch payload

def repo_slug(repo):
    """https://github.com/OWNER/NAME(.git) -> OWNER/NAME."""
    m = re.match(r"^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", repo)
    if not m:
        raise VastError(f"--repo must look like https://github.com/OWNER/NAME, got {repo!r}")
    return f"{m.group(1)}/{m.group(2)}"


def check_safe(name, value):
    if not SAFE_ARG_RE.match(value or ""):
        raise VastError(f"{name} contains characters that are not allowed: {value!r}")
    return value


def onstart_command(repo, branch, config, runtype="ssh_proxy", script="onstart.sh"):
    """Short onstart: fetch cloud/<script> (default onstart.sh) at the branch, run it with CONFIG BRANCH REPO, and send its output
    to PID 1's stdout (the container log that the logs API returns) plus /root/slotbench.log."""
    check_safe("branch", branch)
    check_safe("config", config)
    check_safe("script", script)
    url = (f"https://raw.githubusercontent.com/{repo_slug(repo)}/refs/heads/{branch}/"
           f"{SUBDIR}/cloud/{script}")
    q = shlex.quote
    body = ("O=/proc/1/fd/1; [ -w $O ] || O=/dev/stdout; "
            "(command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates)) "
            ">/dev/null 2>&1; "
            f"if curl -fsSL --retry 5 {q(url)} -o /root/sb_onstart.sh; then "
            f"bash /root/sb_onstart.sh {q(config)} {q(branch)} {q(repo)} 2>&1 | tee -a /root/slotbench.log >$O; "
            "else echo '=====SLOTBENCH-ERROR 0 fetch-onstart=====' >$O; fi")
    if runtype == "args":
        return body + "; sleep infinity"  # PID 1: must not exit
    # ssh launch: run detached so the image's own startup (sshd) is never blocked by our long job
    return "{ " + body + "; } </dev/null >/dev/null 2>&1 &"


def build_create_payload(image, disk, env, onstart, label, runtype="ssh_proxy"):
    """Create-instance JSON body, mirroring vastai build_create_instance_payload. Never contains the key."""
    payload = {
        "client_id": "me",
        "image": image,
        "env": dict(env),
        "price": None,
        "disk": float(disk),
        "label": label,
        "extra": None,
        "image_login": None,
        "python_utf8": False,
        "lang_utf8": False,
        "use_jupyter_lab": False,
        "jupyter_dir": None,
        "force": False,
        "cancel_unavail": True,
        "template_hash_id": None,
        "user": None,
    }
    if runtype == "args":
        # args launch: the container's entrypoint is bash, our command is PID 1 (its stdout is the log)
        payload["onstart"] = "bash"
        payload["runtype"] = "args"
        payload["args"] = ["-c", onstart]
    else:
        payload["onstart"] = onstart
        payload["runtype"] = runtype
    return payload


def make_label(gpu):
    return "slotbench-" + re.sub(r"[^a-z0-9]+", "", gpu.lower()) + "-" + time.strftime("%Y%m%d%H%M%S") + \
        "-" + "%04x" % random.randrange(0x10000)


def save_state(iid, state):
    """cloud/.state/<id>.json (the key is never in it). The directory ignores itself in git."""
    os.makedirs(STATE_DIR, exist_ok=True)
    gi = os.path.join(STATE_DIR, ".gitignore")
    if not os.path.exists(gi):
        with open(gi, "w") as f:
            f.write("*\n")
    path = os.path.join(STATE_DIR, f"{int(iid)}.json")
    text = redact(json.dumps(state, indent=2, sort_keys=True))
    with open(path, "w") as f:
        f.write(text + "\n")
    return path


def load_state(iid):
    try:
        with open(os.path.join(STATE_DIR, f"{int(iid)}.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def launch(api, args, out=print):
    """Pick an offer, create the instance. Returns (instance_id or None for dry-run, state dict)."""
    config = os.path.basename(args.config)
    env = {
        "SB_REPO": args.repo, "SB_BRANCH": args.branch, "SB_CONFIG": config,
        "SB_TUNE_US": str(args.tune_us),
        # on-instance watchdog: stops the container (and GPU billing) even if this controller dies
        "SB_MAX_HOURS": str(round((getattr(args, "max_hours", None) or 6.0) + 0.5, 2)),
    }
    offer = None
    if args.offer:
        offer_id = int(args.offer)
    else:
        offers = find_offers(api, args)
        if not offers:
            raise VastError(f"no offer matches gpu={args.gpu!r} max_dph={args.max_dph} "
                            f"min_reliability={args.min_reliability} cuda>={args.cuda}")
        offer = offers[0]
        offer_id = int(offer["id"])
        cc = offer.get("compute_cap")
        if isinstance(cc, int) and cc > 0:
            env["SB_SM"] = str(cc // 10)  # 860 -> 86; fallback if nvidia-smi cannot report compute_cap
    label = args.label or make_label(args.gpu or f"offer{offer_id}")
    payload = build_create_payload(args.image, args.disk, env, onstart_command(args.repo, args.branch, config, args.runtype, getattr(args, 'script', 'onstart.sh')),
                                   label, args.runtype)
    state = {"offer_id": offer_id, "label": label, "payload": payload, "branch": args.branch,
             "config": config, "gpu": (offer or {}).get("gpu_name", args.gpu),
             "dph_total": (offer or {}).get("dph_total"), "offer": offer}
    if args.dry_run:
        out(f"# dry run: would PUT {API_BASE}/asks/{offer_id}/ with this JSON body (auth header not shown)")
        out(json.dumps(payload, indent=2, sort_keys=True))
        return None, state
    t_create = time.time()
    state["created_unix"] = t_create
    try:
        r = api.create_instance(offer_id, payload)
    except VastError as e:
        if e.ambiguous:
            # the create may have happened: look for our unique label before giving up
            iid = find_by_label(api, label)
            if iid:
                out(f"create reported an error but instance {iid} with label {label} exists; using it")
                r = {"success": True, "new_contract": iid}
            else:
                raise VastError(f"{e} (no instance with label {label} found; check the vast.ai console)") from None
        else:
            raise
    iid = r.get("new_contract")
    if not r.get("success", True) or not iid:
        raise VastError(f"create failed: {str(r)[:400]}")
    iid = int(iid)
    if state["dph_total"] is None:
        try:
            info = api.show_instance(iid) or {}
            state["dph_total"] = info.get("dph_total")
            state["gpu"] = info.get("gpu_name", state["gpu"])
        except VastError:
            pass
    state["instance_id"] = iid
    path = save_state(iid, state)
    out(f"instance {iid} created (offer {offer_id}, {state['gpu']}, ${state['dph_total'] or 0:.3f}/h, "
        f"label {label}); state in {path}")
    return iid, state


def find_by_label(api, label):
    try:
        for inst in api.list_instances():
            if inst.get("label") == label:
                return int(inst["id"])
    except VastError:
        pass
    return None


# ---------------------------------------------------------------- log parsing / collection

def scan_markers(text):
    """(done: bool, done_status: str|None, error_lines: [str])."""
    done, status = False, None
    for m in DONE_RE.finditer(text):
        done, status = True, (m.group(1) or "ok").strip()
    errors = [ln.strip() for ln in text.splitlines() if ERROR_RE.search(ln)]
    return done, status, errors


def extract_blocks(text):
    """Parse every BEGIN..END block. Returns list of dicts {name, sha256, data(bytes|None), error(str|None)}.
    Lines are matched leniently: markers anywhere in the line, and for payload lines only the last
    whitespace-separated token is used (tolerates timestamp prefixes)."""
    blocks = []
    cur = None
    for ln in text.splitlines():
        m = BEGIN_RE.search(ln)
        if m:
            if cur is not None:
                cur["error"] = "no END marker (a new BEGIN started)"
                blocks.append(cur)
            cur = {"name": m.group(1), "sha256": (m.group(2) or "").lower() or None, "lines": [], "error": None}
            continue
        m = END_RE.search(ln)
        if m and cur is not None:
            if m.group(1) != cur["name"]:
                cur["error"] = f"END name {m.group(1)!r} does not match BEGIN"
            blocks.append(cur)
            cur = None
            continue
        if cur is not None:
            tok = ln.strip().split()
            if tok:
                cur["lines"].append(tok[-1])
    if cur is not None:
        cur["error"] = "truncated: no END marker"
        blocks.append(cur)
    out = []
    for b in blocks:
        data = None
        err = b["error"]
        if err is None:
            try:
                data = base64.b64decode("".join(b["lines"]), validate=True)
            except (binascii.Error, ValueError) as e:
                err = f"base64 decode failed: {e}"
        if err is None and b["sha256"] and hashlib.sha256(data).hexdigest() != b["sha256"]:
            err = "sha256 mismatch"
            data = None
        if err is not None:
            data = None
        out.append({"name": b["name"], "sha256": b["sha256"], "data": data, "error": err})
    return out


def safe_untar(data, dest):
    """Extract a tar.gz held in memory into dest, refusing absolute paths, '..', links and devices."""
    dest_abs = os.path.realpath(dest)
    names = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            target = os.path.realpath(os.path.join(dest_abs, m.name))
            if m.name.startswith("/") or not (target == dest_abs or target.startswith(dest_abs + os.sep)):
                raise VastError(f"unsafe path in tar: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise VastError(f"refusing non-regular tar member: {m.name!r}")
        os.makedirs(dest_abs, exist_ok=True)
        for m in members:
            if hasattr(tarfile, "data_filter"):
                tf.extract(m, dest_abs, filter="data")
            else:
                tf.extract(m, dest_abs)
            if m.isfile():
                names.append(m.name)
    return names


def collect_text(text, out_dir, out=print):
    """Save the raw log, extract and untar every block (the last valid copy of each name wins).
    Returns a report dict."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "instance.log"), "w") as f:
        f.write(text)
    blocks = extract_blocks(text)
    good, bad = {}, []
    for b in blocks:
        if b["data"] is not None:
            good[b["name"]] = b
        else:
            bad.append(b)
    files = []
    for name, b in sorted(good.items()):
        try:
            files += safe_untar(b["data"], out_dir)
        except (VastError, tarfile.TarError, OSError, EOFError) as e:
            bad.append({"name": name, "error": f"untar failed: {e}"})
    # a corrupt copy that was later re-sent intact is not a failure
    bad = [b for b in bad if b["name"] not in good or b["error"].startswith("untar")]
    done, status, errors = scan_markers(text)
    report = {"blocks_ok": sorted(good), "blocks_bad": [{"name": b["name"], "error": b["error"]} for b in bad],
              "files": sorted(set(files)), "done": done, "done_status": status, "errors": errors,
              "out_dir": out_dir}
    with open(os.path.join(out_dir, "collect.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    out(f"collected {len(good)} block(s), {len(report['files'])} file(s) into {out_dir}")
    for b in report["blocks_bad"]:
        out(f"  BAD block {b['name']}: {b['error']}")
    out(f"  DONE marker: {'yes (' + status + ')' if done else 'NO'}")
    for e in errors:
        out(f"  error line: {e}")
    return report


# ---------------------------------------------------------------- run loop helpers

def cost_so_far(dph, elapsed_s):
    """Local cost estimate: dph_total * elapsed hours (storage is included in dph_total)."""
    return (dph or 0.0) * elapsed_s / 3600.0


def stop_reason(elapsed_s, dph, max_hours, max_cost, poll_s):
    """Why the run must stop now, or None. Stops one poll early so the cap is not overshot before collection."""
    horizon = elapsed_s + poll_s
    if max_hours is not None and horizon >= max_hours * 3600.0:
        return f"time cap {max_hours} h reached"
    if max_cost is not None and cost_so_far(dph, horizon) >= max_cost:
        return f"cost cap ${max_cost:.2f} reached"
    return None


class _Terminate(Exception):
    pass


def _on_sigterm(signum, frame):
    raise _Terminate(f"signal {signum}")


def run(api, args, out=print, now=time.time, sleep=time.sleep):
    """launch -> poll until DONE / error / cap -> collect -> destroy (finally, unless --keep). Returns exit code."""
    iid, state = None, {}
    t0 = now()
    rc = 1
    old = signal.signal(signal.SIGTERM, _on_sigterm) if hasattr(signal, "SIGTERM") else None
    try:
        iid, state = launch(api, args, out=out)
        if iid is None:
            return 0
        t0 = state.get("created_unix", t0)
        dph = state.get("dph_total") or 0.0
        out_dir = args.out or os.path.join("results", str(iid))
        error_seen_at = None
        running_seen = False
        reason = None
        while True:
            sleep(args.poll_s)
            elapsed = now() - t0
            info = None
            try:
                info = api.show_instance(iid)
                if info is None:
                    reason = "instance no longer exists (destroyed elsewhere?)"
            except VastError as e:
                out(f"  status error (will retry): {e}")
            if info:
                dph = info.get("dph_total") or dph
                st = info.get("actual_status")
                if st == "running":
                    running_seen = True
                if st in ("exited", "offline") or (info.get("intended_status") == "stopped"):
                    reason = f"instance is {st} ({(info.get('status_msg') or '')[:120]})"
                elif not running_seen and elapsed > args.max_load_min * 60:
                    reason = f"not running after {args.max_load_min} min (status {st})"
            text = ""
            if running_seen:
                try:
                    text = api.logs(iid, tail=args.poll_tail)
                except VastError as e:
                    out(f"  logs error (will retry): {e}")
            done, dstat, errors = scan_markers(text)
            last = next((ln for ln in reversed(text.splitlines()) if ln.strip() and "=====" not in ln), "")
            st_txt = (info or {}).get("actual_status", "?")
            out(f"[{elapsed / 60:6.1f} min  ${cost_so_far(dph, elapsed):.3f}] {st_txt}: {last[:110]}")
            if done:
                reason = f"DONE ({dstat})"
            elif errors and error_seen_at is None:
                error_seen_at = elapsed
                out(f"  error marker: {errors[-1]}; waiting up to {args.error_grace_s} s for results")
            elif error_seen_at is not None and elapsed - error_seen_at > args.error_grace_s:
                reason = "error marker and no DONE within grace period"
            if reason is None:
                reason = stop_reason(elapsed, dph, args.max_hours, args.max_cost, args.poll_s)
            if reason:
                break
        out(f"stopping: {reason}")
        # The log snapshot behind result_url can lag or be partial; when the poll already saw DONE, retry
        # until the collected text has it too, so results are never lost before the destroy.
        want_done = reason.startswith("DONE")
        for attempt in range(1, 7):
            try:
                text = api.logs(iid, tail=args.collect_tail)
                if want_done and not scan_markers(text)[0] and attempt < 6:
                    out(f"  collect attempt {attempt}: DONE marker not in fetched logs yet ({len(text)} bytes); retrying")
                    sleep(20)
                    continue
                try:  # keep the raw instance log: the only record of why a run failed once it is destroyed
                    os.makedirs(out_dir, exist_ok=True)
                    with open(os.path.join(out_dir, "instance.log"), "w") as fh:
                        fh.write(text)
                except OSError as e:
                    out(f"  could not save instance.log: {e}")
                rep = collect_text(text, out_dir, out=out)
                rc = 0 if rep["done"] and rep["done_status"] == "ok" and not rep["blocks_bad"] else 1
                break
            except VastError as e:
                out(f"collect attempt {attempt} failed: {e}")
                if attempt < 6:
                    sleep(20)
        if args.keep and info:
            out(f"--keep: instance {iid} left running (${dph:.3f}/h). Raw slots.bin files are under "
                f"/root/sb/repo/{SUBDIR}/runs on the instance; e.g. "
                f"scp -P {info.get('ssh_port', 'PORT')} -r root@{info.get('ssh_host', 'HOST')}:/root/sb/repo/"
                f"{SUBDIR}/runs ./runs_{iid}   then: python3 cloud/vast.py destroy {iid}")
        return rc
    except (KeyboardInterrupt, _Terminate) as e:
        out(f"interrupted ({type(e).__name__}); cleaning up")
        return 130
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)
        if iid is not None and not args.keep:
            destroy_verbose(api, iid, out)
        if iid is not None:
            spent = cost_so_far(state.get("dph_total") or 0.0, now() - t0)
            out(f"estimated spend: ${spent:.3f} over {(now() - t0) / 3600:.2f} h "
                f"(dph_total x elapsed; check the vast.ai billing page for the exact figure)")


def destroy_verbose(api, iid, out=print):
    try:
        api.destroy_instance(iid)
        out(f"instance {iid} destroyed")
        return True
    except Exception as e:  # noqa: BLE001 - must never mask the original error; report loudly instead
        out(f"!!! FAILED TO DESTROY INSTANCE {iid}: {redact(e)}\n"
            f"!!! it keeps billing: run `python3 cloud/vast.py destroy {iid} --yes` or use https://cloud.vast.ai/instances/")
        return False


# ---------------------------------------------------------------- CLI

def cmd_offers(args):
    api = Api(key=get_key(required=False))
    offers = find_offers(api, args)
    if not offers:
        print("no matching offers")
        return 1
    print(format_offers(offers))
    return 0


def cmd_launch(args):
    key = None if args.dry_run else get_key()
    api = Api(key=key)
    iid, _ = launch(api, args)
    if iid is not None:
        print(iid)
    return 0


def cmd_status(args):
    api = Api(key=get_key())
    info = api.show_instance(args.id)
    if not info:
        print(f"instance {args.id} not found")
        return 1
    keys = ["id", "label", "actual_status", "intended_status", "cur_state", "status_msg", "gpu_name", "dph_total",
            "start_date", "ssh_host", "ssh_port", "public_ipaddr", "image_uuid", "geolocation"]
    for k in keys:
        if k in info:
            print(f"{k:16} {info[k]}")
    sd = info.get("start_date")
    if isinstance(sd, (int, float)):
        el = time.time() - sd
        print(f"{'elapsed_h':16} {el / 3600:.2f}  (~${cost_so_far(info.get('dph_total'), el):.3f})")
    return 0


def cmd_logs(args):
    api = Api(key=get_key())
    sys.stdout.write(api.logs(args.id, tail=args.tail))
    return 0


def cmd_collect(args):
    api = Api(key=get_key())
    text = api.logs(args.id, tail=args.tail)
    rep = collect_text(text, args.out or os.path.join("results", str(args.id)))
    return 0 if rep["done"] and not rep["blocks_bad"] else 1


def cmd_destroy(args, input_fn=input):
    api = Api(key=get_key())
    if not args.yes:
        ans = input_fn(f"destroy instance {args.id}? this deletes its disk (raw slots.bin). type 'yes': ")
        if ans.strip().lower() != "yes":
            print("not destroyed")
            return 1
    return 0 if destroy_verbose(api, args.id) else 1


def cmd_run(args):
    api = Api(key=get_key())
    return run(api, args)


def add_offer_filters(p, gpu_required=False):
    p.add_argument("--gpu", default=None if gpu_required else "RTX 3060", help='exact vast.ai gpu_name, e.g. "RTX 3060", '
                   '"A100 SXM4", "A100 PCIE", "H100 SXM", "H100 PCIE"')
    p.add_argument("--max-dph", type=float, default=1.0, help="max total $/hour (incl. disk)")
    p.add_argument("--min-reliability", type=float, default=0.98)
    p.add_argument("--cuda", type=float, default=12.4, help="min cuda_max_good of the host driver")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--verified", action="store_true", help="only verified hosts")


def add_launch_args(p):
    p.add_argument("--offer", type=int, help="use this offer id instead of searching")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="config under slotbench/configs/ (name only)")
    p.add_argument("--branch", default=DEFAULT_BRANCH)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--script", default="onstart.sh",
                   help="script under slotbench/cloud/ the instance runs, e.g. onstart_cuphy.sh (NVIDIA cuPHY)")
    p.add_argument("--disk", type=float, default=40.0, help="GB")
    p.add_argument("--label", default=None)
    p.add_argument("--tune-us", type=float, default=200.0, help="slot_driver --tune-us target on the instance")
    p.add_argument("--runtype", default="ssh_proxy", choices=["ssh_proxy", "ssh_direc ssh_proxy", "args"],
                   help="ssh keeps SSH access (for scp with --keep); args runs the command as PID 1")
    p.add_argument("--dry-run", action="store_true", help="print the create payload, do nothing")


def build_parser():
    ap = argparse.ArgumentParser(prog="vast.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("offers", help="cheapest matching on-demand single-GPU offers (no key needed)")
    add_offer_filters(p)
    p.set_defaults(fn=cmd_offers, disk=40.0)

    p = sub.add_parser("launch", help="create an instance that runs cloud/onstart.sh")
    add_offer_filters(p)
    add_launch_args(p)
    p.set_defaults(fn=cmd_launch)

    p = sub.add_parser("status", help="show an instance")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("logs", help="print the container log")
    p.add_argument("id", type=int)
    p.add_argument("--tail", type=int, default=None)
    p.set_defaults(fn=cmd_logs)

    p = sub.add_parser("collect", help="extract result blocks from the log into --out")
    p.add_argument("id", type=int)
    p.add_argument("--out", default=None, help="default results/<id>")
    p.add_argument("--tail", type=int, default=MAX_LOG_TAIL)
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("destroy", help="destroy an instance")
    p.add_argument("id", type=int)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_destroy)

    p = sub.add_parser("run", help="launch, wait, collect, destroy (with hard time and cost caps)")
    add_offer_filters(p)
    add_launch_args(p)
    p.add_argument("--max-hours", type=float, default=3.0)
    p.add_argument("--max-cost", type=float, default=5.0, help="USD, computed locally as dph_total x elapsed")
    p.add_argument("--poll-s", type=float, default=60.0)
    p.add_argument("--poll-tail", type=int, default=400, help="log lines fetched per poll")
    p.add_argument("--collect-tail", type=int, default=MAX_LOG_TAIL)
    p.add_argument("--max-load-min", type=float, default=30.0, help="give up if not running after this")
    p.add_argument("--error-grace-s", type=float, default=600.0)
    p.add_argument("--out", default=None, help="default results/<id>")
    p.add_argument("--keep", action="store_true", help="do not destroy at the end (scp raw data, then destroy)")
    p.set_defaults(fn=cmd_run)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd in ("launch", "run") and not args.offer and not args.gpu:
        sys.exit("error: give --gpu NAME or --offer ID")
    try:
        return args.fn(args)
    except VastError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
