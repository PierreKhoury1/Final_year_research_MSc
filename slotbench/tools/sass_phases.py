#!/usr/bin/env python3
"""The SASS between consecutive checkpoints of k_ktrace: every checkpoint is a `CS2R SR_GLOBALTIMERLO` followed by a
`CS2R SR_CLOCKLO`; the instructions between checkpoint k's clock read and checkpoint k+1's timer read are what
phase k measured. Usage: sass_phases.py BINARY_OR_SASS [--json OUT]  (cuobjdump + nvdisasm on PATH for a binary)."""
import json, os, re, subprocess, sys

NAMES = ["entry", "cal", "loaded", "bar1_arrive", "bar1_release", "computed", "bar2_arrive", "bar2_release", "stored", "ticket", "exit"]


def listing(path):
    if path.endswith(".sass"):
        return open(path).read()
    cuobjdump = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "cuobjdump")
    if not os.path.exists(cuobjdump):
        cuobjdump = "cuobjdump"
    return subprocess.run([cuobjdump, "-sass", path], capture_output=True, text=True).stdout


def phases(sass, kernel="k_ktrace"):
    cur, ins = None, []
    for line in sass.splitlines():
        m = re.match(r"\s*Function\s*:\s*(\S+)", line)
        if m:
            if cur and kernel in cur:
                break
            cur = m.group(1); ins = []; continue
        m = re.match(r"\s*/\*([0-9a-f]+)\*/\s+((?:@!?P\d\s+)?[A-Z0-9_.]+[^;]*);", line)
        if m and cur:
            ins.append((m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()))
    if not (cur and kernel in cur):
        return None
    # checkpoint k = index of its GLOBALTIMER read; the clock read is the next instruction
    ck = [i for i, (_, t) in enumerate(ins) if "SR_GLOBALTIMERLO" in t and t.startswith("CS2R")]
    out = {}
    for k in range(len(ck) - 1):
        if k >= len(NAMES) - 1:
            break
        body = ins[ck[k] + 2:ck[k + 1]]
        hist = {}
        for _, t in body:
            op = re.sub(r"^@!?P\d\s+", "", t).split(" ")[0]
            hist[op] = hist.get(op, 0) + 1
        out[f"{NAMES[k]} -> {NAMES[k + 1]}"] = dict(n=len(body), opcodes=hist, listing=[f"{a} {t}" for a, t in body])
    return dict(kernel=cur, checkpoints=len(ck), phases=out)


def main():
    res = phases(listing(sys.argv[1]))
    if res is None:
        sys.exit("k_ktrace not found")
    print(f"{res['kernel']}: {res['checkpoints']} checkpoints")
    for name, p in res["phases"].items():
        print(f"  {name:28s} {p['n']:3d} instr  {dict(sorted(p['opcodes'].items(), key=lambda kv: -kv[1]))}")
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
