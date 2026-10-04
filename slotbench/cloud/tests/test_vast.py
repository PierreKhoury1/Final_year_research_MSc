"""Tests for cloud/vast.py (no network: urllib is replaced by a fake opener, the API by a fake class)."""
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import urllib.error

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CLOUD = os.path.dirname(HERE)
sys.path.insert(0, CLOUD)
import vast  # noqa: E402

KEY = "sk-test-SECRET-0123456789abcdef"


# ------------------------------------------------------------------ fakes

class FakeResp:
    def __init__(self, body):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    """Replays a list of responses: dict/bytes -> 200 body; int -> HTTPError; Exception -> raised."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, req, timeout=None):
        assert timeout is not None, "every call must have a timeout"
        self.requests.append(req)
        r = self.responses.pop(0)
        if isinstance(r, tuple):  # (status, body)
            code, body = r
            raise urllib.error.HTTPError(req.full_url, code, "err", {}, io.BytesIO(body.encode()))
        if isinstance(r, BaseException):
            raise r
        return FakeResp(r)


def api_with(responses, key=KEY):
    sleeps = []
    op = FakeOpener(responses)
    api = vast.Api(key=key, opener=op, sleep=sleeps.append)
    return api, op, sleeps


def offer(i, dph, rel=0.99, cuda=12.8, gpu="RTX 3060", n=1, verification="verified", **kw):
    o = {"id": i, "gpu_name": gpu, "num_gpus": n, "dph_total": dph, "reliability2": rel, "cuda_max_good": cuda,
         "rentable": True, "rented": False, "verification": verification, "compute_cap": 860,
         "driver_version": "570.1", "cpu_cores_effective": 8, "cpu_ram": 32768, "inet_down": 500, "inet_up": 100,
         "geolocation": "Somewhere"}
    o.update(kw)
    return o


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(vast, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    vast._SECRETS.clear()
    yield


def parse(*argv):
    return vast.build_parser().parse_args(list(argv))


# ------------------------------------------------------------------ offers

def test_offer_query_contents():
    q = vast.build_offer_query("RTX 3060", max_dph=0.5, min_rel=0.98, cuda=12.4, limit=5, verified=True, disk=40)
    assert q["gpu_name"] == {"eq": "RTX 3060"}
    assert q["num_gpus"] == {"eq": 1}
    assert q["dph_total"] == {"lte": 0.5}
    assert q["reliability2"] == {"gte": 0.98}
    assert q["cuda_max_good"] == {"gte": 12.4}
    assert q["verified"] == {"eq": True}
    assert q["type"] == "on-demand" and q["order"] == [["dph_total", "asc"]]
    assert q["limit"] >= 5 and q["allocated_storage"] == 40.0
    assert "verified" not in vast.build_offer_query("RTX 3060", verified=False)


def test_filter_offers_filters_and_sorts():
    offers = [
        offer(1, 0.30), offer(2, 0.10, rel=0.95), offer(3, 0.20), offer(4, 0.05, gpu="RTX 3090"),
        offer(5, 0.15, n=2), offer(6, 0.20, rel=0.999), offer(7, 2.0), offer(8, 0.12, cuda=12.0),
        offer(9, 0.11, rented=True), offer(10, 0.13, verification="unverified"),
    ]
    got = vast.filter_offers(offers, "RTX 3060", max_dph=1.0, min_rel=0.98, cuda=12.4, limit=10)
    assert [o["id"] for o in got] == [10, 6, 3, 1]  # price asc, ties -> higher reliability first
    got = vast.filter_offers(offers, "RTX 3060", max_dph=1.0, min_rel=0.98, cuda=12.4, limit=2, verified=True)
    assert [o["id"] for o in got] == [6, 3]


def test_find_offers_uses_post_without_key_and_filters_locally():
    api, op, _ = api_with([{"offers": [offer(1, 0.3), offer(2, 0.1, rel=0.5), offer(3, 0.2)]}], key=None)
    args = parse("offers", "--gpu", "RTX 3060", "--limit", "5")
    got = vast.find_offers(api, args)
    assert [o["id"] for o in got] == [3, 1]
    req = op.requests[0]
    assert req.get_method() == "POST" and req.full_url.endswith("/api/v0/bundles/")
    assert "Authorization" not in req.headers
    assert json.loads(req.data)["gpu_name"] == {"eq": "RTX 3060"}
    table = vast.format_offers(got)
    assert "RTX 3060" in table and "0.200" in table and "86" in table


# ------------------------------------------------------------------ payload / launch

def test_payload_and_dry_run_contain_no_key(monkeypatch, capsys):
    monkeypatch.setenv("VAST_API_KEY", KEY)
    vast.get_key()
    api, op, _ = api_with([{"offers": [offer(42, 0.2)]}])
    args = parse("launch", "--gpu", "RTX 3060", "--dry-run", "--branch", "feature/x", "--config", "cloud.toml")
    outs = []
    iid, state = vast.launch(api, args, out=outs.append)
    assert iid is None
    text = "\n".join(outs)
    assert KEY not in text and KEY not in json.dumps(state)
    p = state["payload"]
    assert p["env"]["SB_BRANCH"] == "feature/x" and p["env"]["SB_CONFIG"] == "cloud.toml"
    assert p["env"]["SB_SM"] == "86"
    assert p["image"] == vast.DEFAULT_IMAGE and p["disk"] == 40.0 and p["runtype"] == "ssh_proxy"
    assert "refs/heads/feature/x/slotbench/cloud/onstart.sh" in p["onstart"]
    assert len(p["onstart"]) < 1024
    assert "/asks/42/" in text
    assert len(op.requests) == 1  # only the search; nothing created
    assert not os.path.exists(vast.STATE_DIR)


def test_onstart_command_is_valid_shell_and_rejects_unsafe_values():
    for rt in ("ssh_proxy", "args"):
        cmd = vast.onstart_command(vast.DEFAULT_REPO, vast.DEFAULT_BRANCH, "cloud.toml", rt)
        subprocess.run(["bash", "-n", "-c", cmd], check=True)
    assert vast.onstart_command(vast.DEFAULT_REPO, "b", "c.toml", "args").endswith("sleep infinity")
    with pytest.raises(vast.VastError):
        vast.onstart_command(vast.DEFAULT_REPO, "main; rm -rf /", "cloud.toml")
    with pytest.raises(vast.VastError):
        vast.onstart_command("https://evil.example/x", "main", "cloud.toml")
    p = vast.build_create_payload("img", 10, {}, "echo hi", "lbl", runtype="args")
    assert p["runtype"] == "args" and p["args"] == ["-c", "echo hi"] and p["onstart"] == "bash"


def test_launch_creates_sends_bearer_and_saves_state_without_key():
    api, op, _ = api_with([{"offers": [offer(42, 0.2)]}, {"success": True, "new_contract": 777}])
    args = parse("launch", "--gpu", "RTX 3060")
    iid, state = vast.launch(api, args, out=lambda s: None)
    assert iid == 777
    create = op.requests[1]
    assert create.get_method() == "PUT" and create.full_url.endswith("/api/v0/asks/42/")
    assert create.headers["Authorization"] == "Bearer " + KEY
    assert KEY not in create.full_url and KEY.encode() not in create.data
    saved = open(os.path.join(vast.STATE_DIR, "777.json")).read()
    assert KEY not in saved and json.loads(saved)["instance_id"] == 777
    assert open(os.path.join(vast.STATE_DIR, ".gitignore")).read().strip() == "*"


def test_create_is_not_retried_and_ambiguous_failure_finds_label():
    # create -> 500 (maybe applied); list shows our label -> use it, never create twice
    api, op, sleeps = api_with([{"offers": [offer(42, 0.2)]}, (500, "boom"), {"instances": []}])
    args = parse("launch", "--gpu", "RTX 3060", "--label", "slotbench-unique")
    op.responses[2] = {"instances": [{"id": 9, "label": "other"}, {"id": 555, "label": "slotbench-unique"}]}
    iid, _ = vast.launch(api, args, out=lambda s: None)
    assert iid == 555
    assert [r.get_method() for r in op.requests] == ["POST", "PUT", "GET"]
    assert sleeps == []


def test_create_4xx_raises_clearly():
    api, op, _ = api_with([{"offers": [offer(42, 0.2)]}, (400, '{"error": "no_such_ask"}')])
    with pytest.raises(vast.VastError) as ei:
        vast.launch(api, parse("launch", "--gpu", "RTX 3060"), out=lambda s: None)
    assert "HTTP 400" in str(ei.value) and "no_such_ask" in str(ei.value)
    assert len(op.requests) == 2


# ------------------------------------------------------------------ HTTP behaviour and redaction

def test_retry_with_backoff_on_429_and_5xx():
    api, op, sleeps = api_with([(429, "slow down"), (503, "busy"), {"instances": {"id": 1}}])
    assert api.show_instance(1) == {"id": 1}
    assert len(op.requests) == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_retries_exhausted_raises():
    api, op, sleeps = api_with([(502, "x")] * 5)
    with pytest.raises(vast.VastError) as ei:
        api.destroy_instance(3)
    assert "HTTP 502" in str(ei.value) and len(op.requests) == 5


def test_key_redacted_in_errors():
    api, _, _ = api_with([(401, f'{{"msg": "bad key {KEY}", "h": "Bearer {KEY}"}}')])
    with pytest.raises(vast.VastError) as ei:
        api.show_instance(1)
    assert KEY not in str(ei.value) and "***" in str(ei.value)
    api, _, _ = api_with([urllib.error.URLError(f"proxy said api_key={KEY}")] * 5)
    with pytest.raises(vast.VastError) as ei:
        api.show_instance(1)
    assert KEY not in str(ei.value)
    assert vast.redact(f"x?api_key={KEY}&y=1") == "x?api_key=***&y=1"


def test_missing_key_exits_with_variable_name(capsys):
    with pytest.raises(SystemExit) as ei:
        vast.main(["status", "5"])
    assert "VAST_API_KEY" in str(ei.value.code)


def test_logs_fetches_result_url_without_auth():
    api, op, _ = api_with([{"result_url": "https://s3.example/logs/abc"}, (404, "not yet"), b"line1\nline2\n"])
    text = api.logs(12, tail=100)
    assert text == "line1\nline2\n"
    req0 = op.requests[0]
    assert req0.get_method() == "PUT" and req0.full_url.endswith("/instances/request_logs/12/")
    assert json.loads(req0.data) == {"tail": "100"}
    assert all("Authorization" not in r.headers for r in op.requests[1:])


# ------------------------------------------------------------------ markers / collect

def make_tar(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def block(name, data, sha=True, prefix=""):
    b64 = base64.b64encode(data).decode()
    lines = [b64[i:i + 76] for i in range(0, len(b64), 76)]
    h = " " + hashlib.sha256(data).hexdigest() if sha else ""
    out = [f"{prefix}=====SLOTBENCH-BEGIN {name}{h}====="] + [prefix + ln for ln in lines] + \
          [f"{prefix}=====SLOTBENCH-END {name}====="]
    return "\n".join(out) + "\n"


def test_extract_roundtrip_corruption_and_untar(tmp_path):
    good = make_tar({"cloud/M1_sgemm_d50_r0/summary.json": b'{"p50": 1}' * 50,
                     "cloud/M1_sgemm_d50_r0/status": b"ok\n"})
    other = make_tar({"cloud/M0_idle_d0_r0/summary.json": b"{}"})
    bad = make_tar({"cloud/M4_llm_d100_r0/summary.json": b"{}" * 400})
    bad_txt = block("cloud/M4_llm_d100_r0", bad)
    lines = bad_txt.splitlines()
    lines[2] = lines[2][:10] + ("A" if lines[2][10] != "A" else "B") + lines[2][11:]  # flip one base64 char
    log = ("noise\n" + block("cloud/M1_sgemm_d50_r0", good) + "\n".join(lines) + "\n"
           + block("cloud/M0_idle_d0_r0", other, prefix="2026-09-30T10:00:00Z ")
           + "=====SLOTBENCH-BEGIN cloud/trunc_r0 " + "0" * 64 + "=====\nAAAA\n"
           + "=====SLOTBENCH-DONE=====\n")
    blocks = {b["name"]: b for b in vast.extract_blocks(log)}
    assert blocks["cloud/M1_sgemm_d50_r0"]["data"] == good
    assert blocks["cloud/M0_idle_d0_r0"]["data"] == other  # timestamp prefixes tolerated
    assert blocks["cloud/M4_llm_d100_r0"]["data"] is None
    assert "sha256" in blocks["cloud/M4_llm_d100_r0"]["error"]
    assert "truncated" in blocks["cloud/trunc_r0"]["error"]

    out = tmp_path / "res"
    rep = vast.collect_text(log, str(out), out=lambda s: None)
    assert rep["done"] and rep["done_status"] == "ok"
    assert sorted(rep["blocks_ok"]) == ["cloud/M0_idle_d0_r0", "cloud/M1_sgemm_d50_r0"]
    assert {b["name"] for b in rep["blocks_bad"]} == {"cloud/M4_llm_d100_r0", "cloud/trunc_r0"}
    assert (out / "cloud/M1_sgemm_d50_r0/summary.json").read_bytes() == b'{"p50": 1}' * 50
    assert (out / "instance.log").read_text() == log
    assert json.loads((out / "collect.json").read_text())["done"] is True


def test_block_without_sha_and_resent_copy_wins(tmp_path):
    data = make_tar({"a/b/status": b"ok\n"})
    torn = "=====SLOTBENCH-BEGIN a/b " + hashlib.sha256(data).hexdigest() + "=====\nQUJD"  # killed mid-line
    log = torn + block("a/b", data, sha=False)  # next BEGIN lands on the torn line
    rep = vast.collect_text(log, str(tmp_path), out=lambda s: None)
    assert rep["blocks_ok"] == ["a/b"] and rep["blocks_bad"] == []
    assert not rep["done"]


def test_error_and_done_markers():
    done, st, errs = vast.scan_markers("x\n=====SLOTBENCH-ERROR 88 make -j4 (exit 2)=====\n"
                                       "=====SLOTBENCH-DONE status=error=====\n")
    assert done and st == "error" and errs == ["=====SLOTBENCH-ERROR 88 make -j4 (exit 2)====="]
    assert vast.scan_markers("=====SLOTBENCH-DONE status=matrix_rc_3=====")[1] == "matrix_rc_3"
    assert vast.scan_markers("still running")[0] is False


def test_untar_rejects_unsafe_paths(tmp_path):
    for name in ("../evil", "/abs/evil"):
        with pytest.raises(vast.VastError):
            vast.safe_untar(make_tar({name: b"x"}), str(tmp_path / "o"))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(vast.VastError):
        vast.safe_untar(buf.getvalue(), str(tmp_path / "o"))
    assert not (tmp_path / "evil").exists()


def test_onstart_emit_block_format_roundtrip(tmp_path):
    """Run onstart.sh's emit_block function in bash and parse its output with extract_blocks."""
    src = open(os.path.join(CLOUD, "onstart.sh")).read()
    start = src.index("emit_block() {")
    end = src.index("\n}\n", start) + 3
    (tmp_path / "d" / "c").mkdir(parents=True)
    (tmp_path / "d" / "c" / "summary.json").write_text('{"x": 1}\n' * 300)
    script = "log() { echo \"$*\"; }\n" + src[start:end] + f'\nemit_block cfg/c "{tmp_path}/d" c/summary.json\n'
    out = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True).stdout
    assert all(len(ln) <= 76 for ln in out.splitlines() if "=====" not in ln)
    rep = vast.collect_text(out, str(tmp_path / "o"), out=lambda s: None)
    assert rep["blocks_ok"] == ["cfg/c"] and rep["blocks_bad"] == []
    assert (tmp_path / "o" / "c" / "summary.json").read_text() == '{"x": 1}\n' * 300
    b = vast.extract_blocks(out)[0]
    assert b["sha256"] is not None  # the BEGIN line carries the sha256


def test_onstart_syntax():
    subprocess.run(["bash", "-n", os.path.join(CLOUD, "onstart.sh")], check=True)


# ------------------------------------------------------------------ run loop: caps and destroy-in-finally

def test_stop_reason_caps():
    assert vast.stop_reason(0, 0.5, 3, 5.0, 60) is None
    assert "time cap" in vast.stop_reason(3 * 3600 - 30, 0.01, 3, 5.0, 60)
    # $2/h: cost at elapsed+poll >= $1 -> stop; one poll before the cap is crossed
    assert "cost cap" in vast.stop_reason(1800 - 60, 2.0, 10, 1.0, 60)
    assert vast.stop_reason(1800 - 120, 2.0, 10, 1.0, 60) is None
    assert vast.cost_so_far(0.36, 3600 * 2) == pytest.approx(0.72)
    assert vast.cost_so_far(None, 100) == 0.0


class FakeApi:
    def __init__(self, logs=None, fail_logs=None, status="running", dph=0.5):
        self.created, self.destroyed = [], []
        self.log_texts = list(logs or [])
        self.fail_logs = fail_logs
        self.status = status
        self.dph = dph

    def search_offers(self, q):
        return [offer(42, self.dph)]

    def create_instance(self, offer_id, payload):
        self.created.append((offer_id, payload))
        return {"success": True, "new_contract": 1234}

    def show_instance(self, iid):
        return {"id": iid, "actual_status": self.status, "dph_total": self.dph, "ssh_host": "h", "ssh_port": 1}

    def list_instances(self):
        return []

    def logs(self, iid, tail=None, wait_s=60):
        if self.fail_logs:
            raise self.fail_logs
        return self.log_texts.pop(0) if len(self.log_texts) > 1 else self.log_texts[0]

    def destroy_instance(self, iid):
        self.destroyed.append(iid)
        return {"success": True}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def run_args(tmp_path, *extra):
    return parse("run", "--gpu", "RTX 3060", "--poll-s", "60", "--out", str(tmp_path / "res"), *extra)


def run_with(api, args, clock=None):
    clock = clock or Clock()
    msgs = []
    import time as _t
    orig = _t.time
    _t.time = clock.now  # launch() stamps created_unix with time.time()
    try:
        rc = vast.run(api, args, out=msgs.append, now=clock.now, sleep=clock.sleep)
    finally:
        _t.time = orig
    return rc, "\n".join(msgs)


def test_run_success_collects_then_destroys(tmp_path):
    data = make_tar({"cloud/M1_x/summary.json": b"{}"})
    final = block("cloud/M1_x", data) + "=====SLOTBENCH-DONE=====\n"
    api = FakeApi(logs=["building...\n", final, final])
    rc, msgs = run_with(api, run_args(tmp_path))
    assert rc == 0
    assert api.destroyed == [1234]
    assert (tmp_path / "res" / "cloud/M1_x/summary.json").exists()
    assert "estimated spend" in msgs and "destroyed" in msgs


def test_run_null_actual_status_uses_cur_state_and_collects(tmp_path, monkeypatch):
    data = make_tar({"cloud/M1_x/summary.json": b"{}"})
    final = block("cloud/M1_x", data) + "=====SLOTBENCH-DONE=====\n"
    api = FakeApi(logs=[final, final], status=None)
    show_instance = api.show_instance
    monkeypatch.setattr(api, "show_instance", lambda iid: dict(show_instance(iid), cur_state="running"))
    log_calls = []
    logs = api.logs

    def fetch_logs(iid, tail=None):
        log_calls.append(tail)
        return logs(iid, tail=tail)

    monkeypatch.setattr(api, "logs", fetch_logs)
    args = run_args(tmp_path)
    clock = Clock()
    rc, msgs = run_with(api, args, clock)
    assert rc == 0 and api.destroyed == [1234]
    assert log_calls == [args.poll_tail, args.collect_tail]
    assert clock.t == 1000.0 + args.poll_s
    assert "] running:" in msgs and "stopping: DONE (ok)" in msgs
    assert (tmp_path / "res" / "cloud/M1_x/summary.json").exists()


def test_run_actual_status_takes_precedence_over_cur_state(tmp_path, monkeypatch):
    api = FakeApi(logs=["container exited\n"], status="exited")
    show_instance = api.show_instance
    monkeypatch.setattr(api, "show_instance", lambda iid: dict(show_instance(iid), cur_state="running"))
    rc, msgs = run_with(api, run_args(tmp_path))
    assert rc == 1 and api.destroyed == [1234]
    assert "] exited:" in msgs and "stopping: instance is exited" in msgs


def test_run_cost_cap_stops_collects_and_destroys(tmp_path):
    api = FakeApi(logs=["still running\n"], dph=6.0)  # $6/h, cap $1 -> stops after ~9 min
    clock = Clock()
    rc, msgs = run_with(api, run_args(tmp_path, "--max-cost", "1.0", "--max-dph", "10"), clock)
    assert "cost cap" in msgs and api.destroyed == [1234]
    assert rc == 1  # no DONE marker
    assert vast.cost_so_far(6.0, clock.t - 1000.0) <= 1.0 + 1e-9


def test_run_destroys_on_unexpected_exception(tmp_path):
    api = FakeApi(fail_logs=RuntimeError("kaboom"))
    with pytest.raises(RuntimeError):
        run_with(api, run_args(tmp_path))
    assert api.destroyed == [1234]


def test_run_destroys_on_ctrl_c(tmp_path):
    api = FakeApi(fail_logs=KeyboardInterrupt())
    rc, msgs = run_with(api, run_args(tmp_path))
    assert rc == 130 and api.destroyed == [1234] and "interrupted" in msgs


def test_run_keep_does_not_destroy(tmp_path):
    api = FakeApi(logs=["=====SLOTBENCH-DONE=====\n"])
    rc, msgs = run_with(api, run_args(tmp_path, "--keep"))
    assert api.destroyed == [] and "scp -P 1" in msgs


def test_run_error_marker_then_grace(tmp_path):
    api = FakeApi(logs=["=====SLOTBENCH-ERROR 0 fetch-onstart=====\n"])
    rc, msgs = run_with(api, run_args(tmp_path, "--error-grace-s", "120"))
    assert rc == 1 and "grace" in msgs and api.destroyed == [1234]


def test_run_silent_instance_is_abandoned_early(tmp_path):
    api = FakeApi(logs=[""])  # running, but the container log stays empty
    clock = Clock()
    rc, msgs = run_with(api, run_args(tmp_path, "--max-silent-min", "10", "--max-hours", "2"), clock)
    assert rc == 1 and api.destroyed == [1234]
    assert "no container output 10.0 min after running" in msgs
    assert clock.t - 1000.0 <= 12 * 60 + 120  # stopped near 10 min, not at the 2 h cap


def test_run_proxy_noise_is_not_output(tmp_path):
    noise = "Warning: Permanently added 'ssh2.vast.ai' (ED25519) to the list of known hosts.\nSun Oct  4 22:27:28 UTC 2026\n"
    api = FakeApi(logs=[noise])
    rc, msgs = run_with(api, run_args(tmp_path, "--max-silent-min", "10", "--max-hours", "2"))
    assert "no container output" in msgs and api.destroyed == [1234]


def test_run_output_disables_silence_stop(tmp_path):
    api = FakeApi(logs=["[cuphy-lockstep 21:29:00] apt (limit 1200s)\n"] * 20 + ["=====SLOTBENCH-DONE=====\n"])
    rc, msgs = run_with(api, run_args(tmp_path, "--max-silent-min", "1"))
    assert "no container output" not in msgs and "DONE" in msgs


def test_run_instance_exited(tmp_path):
    api = FakeApi(logs=["x\n"], status="exited")
    rc, msgs = run_with(api, run_args(tmp_path))
    assert "exited" in msgs and api.destroyed == [1234]


def test_destroy_failure_is_loud_not_masking(tmp_path):
    api = FakeApi(logs=["=====SLOTBENCH-DONE=====\n"])

    def boom(iid):
        raise vast.VastError(f"HTTP 500 with {KEY}")
    vast.register_secret(KEY)
    api.destroy_instance = boom
    rc, msgs = run_with(api, run_args(tmp_path))
    assert "FAILED TO DESTROY INSTANCE 1234" in msgs and KEY not in msgs


def test_destroy_asks_for_confirmation(monkeypatch, capsys):
    monkeypatch.setenv("VAST_API_KEY", KEY)
    calls = []
    monkeypatch.setattr(vast.Api, "destroy_instance", lambda self, iid: calls.append(iid) or {"success": True})
    args = parse("destroy", "77")
    assert vast.cmd_destroy(args, input_fn=lambda prompt: "no") == 1 and calls == []
    assert vast.cmd_destroy(args, input_fn=lambda prompt: "yes") == 0 and calls == [77]
    assert vast.cmd_destroy(parse("destroy", "78", "--yes")) == 0 and calls == [77, 78]
