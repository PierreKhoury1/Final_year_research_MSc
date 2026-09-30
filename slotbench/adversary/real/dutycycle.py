#!/usr/bin/env python3
"""Duty-cycle any external program with SIGCONT/SIGSTOP (DESIGN.md section 5).

    python3 dutycycle.py --duty D --period-ms P [--seconds S] -- CMD ARGS...
    python3 dutycycle.py --selftest

CMD runs in its own process group. In each period (absolute schedule, no drift) the group gets
SIGCONT for D% of the period and SIGSTOP for the rest. SIGINT/SIGTERM are forwarded to the group
(after a SIGCONT, so a stopped child can act on them); after --seconds the group gets SIGTERM.
Children that do not exit within --kill-after seconds get SIGKILL. The exit status is the
child's (128 + signal number if it died from a signal).

Caveat for GPU programs: SIGSTOP only stops the CPU side; kernels already queued on the GPU keep
running, so the effective GPU duty cycle is smeared by the program's queue depth.
Stdlib only.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

_pending_signal = None


def _on_signal(signum, _frame):
    global _pending_signal
    _pending_signal = signum


def _kill_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def _reap(pid, block):
    """waitpid via wait4 so the child's rusage is available. Returns (status, rusage) or None."""
    try:
        wpid, status, ru = os.wait4(pid, 0 if block else os.WNOHANG)
    except ChildProcessError:
        return (0, None)
    if wpid == 0:
        return None
    return (status, ru)


def _exit_code(status):
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1


def _sleep_until(t, pid):
    """Sleep until monotonic time t in small chunks; return the reaped result if the child exits."""
    while True:
        r = _reap(pid, False)
        if r is not None:
            return r
        if _pending_signal is not None:
            return None
        dt = t - time.monotonic()
        if dt <= 0:
            return None
        time.sleep(min(dt, 0.02))


def run(cmd, duty, period_ms, seconds=0.0, kill_after=10.0, quiet=False):
    """Run cmd duty-cycled. Returns (exit_code, stats dict)."""
    global _pending_signal
    _pending_signal = None
    old = {s: signal.signal(s, _on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    period = period_ms / 1000.0
    on = period * duty / 100.0
    t_begin = time.monotonic()
    proc = subprocess.Popen(cmd, start_new_session=True)  # own session => own process group
    pid = pgid = proc.pid
    stats = {"periods": 0, "on_s": 0.0}
    result = None
    reason = "child_exit"
    stopped = False
    try:
        k = 0
        while result is None:
            ps = t_begin + k * period
            deadline = t_begin + seconds if seconds > 0 else None
            # ON phase
            if duty > 0:
                if stopped:
                    _kill_group(pgid, signal.SIGCONT)
                    stopped = False
                t_on = time.monotonic()
                end_on = ps + on if duty < 100 else ps + period
                if deadline is not None:
                    end_on = min(end_on, deadline)
                result = _sleep_until(end_on, pid)
                stats["on_s"] += max(0.0, time.monotonic() - t_on)
                if result is not None:
                    break
            # OFF phase
            if duty < 100 and _pending_signal is None and (deadline is None or time.monotonic() < deadline):
                _kill_group(pgid, signal.SIGSTOP)
                stopped = True
                end_off = ps + period if deadline is None else min(ps + period, deadline)
                result = _sleep_until(end_off, pid)
                if result is not None:
                    break
            stats["periods"] += 1
            if _pending_signal is not None:
                reason = signal.Signals(_pending_signal).name.lower()
                break
            if deadline is not None and time.monotonic() >= deadline:
                reason = "seconds"
                break
            # next period: the one containing now (never re-run a missed period)
            k = max(k + 1, int((time.monotonic() - t_begin) / period))
        if result is None:
            # terminate: always SIGCONT first so a stopped child can handle the signal
            sig = _pending_signal if _pending_signal is not None else signal.SIGTERM
            _kill_group(pgid, signal.SIGCONT)
            stopped = False
            _kill_group(pgid, sig)
            t_kill = time.monotonic() + kill_after
            while result is None and time.monotonic() < t_kill:
                result = _reap(pid, False)
                if result is None:
                    time.sleep(0.02)
            if result is None:
                _kill_group(pgid, signal.SIGKILL)
                result = _reap(pid, True)
                reason += "+sigkill"
    finally:
        if result is None:  # unexpected exception: never leave the group stopped or running
            _kill_group(pgid, signal.SIGCONT)
            _kill_group(pgid, signal.SIGKILL)
            result = _reap(pid, True)
        else:
            # the leader is gone; release any stopped stragglers in the group
            _kill_group(pgid, signal.SIGCONT)
        proc.returncode = 0  # reaped by us; stop Popen from waiting on it again
        for s, h in old.items():
            signal.signal(s, h)
    status, ru = result
    wall = time.monotonic() - t_begin
    stats.update({
        "wall_s": wall,
        "on_fraction": stats["on_s"] / wall if wall > 0 else 0.0,
        "child_cpu_s": (ru.ru_utime + ru.ru_stime) if ru is not None else float("nan"),
        "reason": reason,
    })
    code = _exit_code(status)
    if not quiet:
        print("dutycycle: exit=%d reason=%s wall_s=%.3f on_fraction=%.3f child_cpu_s=%.3f periods=%d"
              % (code, reason, wall, stats["on_fraction"], stats["child_cpu_s"], stats["periods"]),
              file=sys.stderr)
    return code, stats


def selftest():
    """Duty-cycle a busy python loop and check its CPU share; check signal/exit handling with sleep."""
    fails = 0
    busy = [sys.executable, "-c", "while True: pass"]
    for duty in (30, 70):
        code, st = run(busy, duty, 100, seconds=3.0, quiet=True)
        frac = st["child_cpu_s"] / st["wall_s"]
        ok = abs(frac - duty / 100.0) <= 0.10 * duty / 100.0 and code == 128 + signal.SIGTERM
        fails += not ok
        print("selftest busy duty=%d: cpu_fraction=%.3f (target %.2f +-10%%) on_fraction=%.3f exit=%d %s"
              % (duty, frac, duty / 100.0, st["on_fraction"], code, "ok" if ok else "FAIL"))
    # --seconds must end a child that is currently stopped (SIGCONT before SIGTERM)
    t = time.monotonic()
    code, st = run(["sleep", "30"], 10, 1000, seconds=0.5, kill_after=5, quiet=True)
    dt = time.monotonic() - t
    ok = code == 128 + signal.SIGTERM and dt < 2.0 and "sigkill" not in st["reason"]
    fails += not ok
    print("selftest stopped child terminated: exit=%d wall=%.2fs %s" % (code, dt, "ok" if ok else "FAIL"))
    # exit status propagation
    code, st = run(["sh", "-c", "exit 3"], 50, 100, quiet=True)
    ok = code == 3 and st["reason"] == "child_exit"
    fails += not ok
    print("selftest exit status: exit=%d %s" % (code, "ok" if ok else "FAIL"))
    # D=100 never stops; D=0 keeps the child stopped
    code, st = run(busy, 100, 100, seconds=1.0, quiet=True)
    ok = st["child_cpu_s"] / st["wall_s"] > 0.85
    fails += not ok
    print("selftest duty=100: cpu_fraction=%.3f %s" % (st["child_cpu_s"] / st["wall_s"], "ok" if ok else "FAIL"))
    code, st = run(busy, 0, 100, seconds=1.0, quiet=True)
    ok = st["child_cpu_s"] < 0.1 and code == 128 + signal.SIGTERM
    fails += not ok
    print("selftest duty=0: child_cpu_s=%.3f exit=%d %s" % (st["child_cpu_s"], code, "ok" if ok else "FAIL"))
    # forwarded SIGTERM: run a nested dutycycle and signal it
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--duty", "50", "--period-ms", "100",
                          "--", "sleep", "30"], stderr=subprocess.DEVNULL)
    time.sleep(0.4)
    p.send_signal(signal.SIGTERM)
    try:
        rc = p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
        rc = None
    ok = rc == 128 + signal.SIGTERM
    fails += not ok
    print("selftest forwarded SIGTERM: exit=%s %s" % (rc, "ok" if ok else "FAIL"))
    print("selftest %s (%d failures)" % ("PASSED" if fails == 0 else "FAILED", fails))
    return 0 if fails == 0 else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--" in argv:
        i = argv.index("--")
        opts, cmd = argv[:i], argv[i + 1:]
    else:
        opts, cmd = argv, []
    ap = argparse.ArgumentParser(description="Duty-cycle CMD with SIGCONT/SIGSTOP on its process group.",
                                 usage="%(prog)s --duty D --period-ms P [--seconds S] -- CMD ARGS... | --selftest")
    ap.add_argument("--duty", type=float, default=100.0, help="percent of each period the command runs (0-100)")
    ap.add_argument("--period-ms", type=float, default=100.0, help="duty-cycle period in ms (default 100)")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after S seconds (0 = until the child exits)")
    ap.add_argument("--kill-after", type=float, default=10.0, help="SIGKILL if the child ignores SIGTERM (s)")
    ap.add_argument("--selftest", action="store_true", help="run the built-in self-test and exit")
    a = ap.parse_args(opts)
    if a.selftest:
        return selftest()
    if not cmd:
        ap.error("missing -- CMD")
    if not 0 <= a.duty <= 100 or a.period_ms <= 0 or a.seconds < 0:
        ap.error("need 0 <= --duty <= 100, --period-ms > 0, --seconds >= 0")
    code, _ = run(cmd, a.duty, a.period_ms, a.seconds, a.kill_after)
    return code


if __name__ == "__main__":
    sys.exit(main())
