#!/usr/bin/env python3
"""Verify the per-instruction brackets in SASS: for every k_instr<K,N> kernel in a cuobjdump -sass listing, find the
two clock reads (CS2R / S2R SR_CLOCK*) and count the opcodes between them. The expected opcode for each kind must
appear exactly N times. Usage: sass_check.py BINARY [--json OUT] (needs cuobjdump on PATH or CUDA_HOME)."""
import json, os, re, subprocess, sys

KINDS = {0: ("LDG_CA", "LDG"), 1: ("LDG_CG", "LDG"), 2: ("LDG_CS", "LDG"), 3: ("LDG_NC", "LDG"), 4: ("LDS", "LDS"),
         5: ("FADD", "FADD"), 6: ("FFMA", "FFMA"), 7: ("IMAD", "IMAD"), 8: ("SHFL", "SHFL"), 9: ("ATOM_RET", "ATOM"),
         10: ("RED", "RED"), 11: ("STG_FENCE", "STG"), 12: ("BAR", "BAR")}


def parse(sass):
    funcs, cur = {}, None
    for line in sass.splitlines():
        m = re.match(r"\s*Function\s*:\s*(\S+)", line)
        if m:
            cur = m.group(1); funcs[cur] = []; continue
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d\s+)?([A-Z0-9_.]+)", line)
        if m and cur:
            funcs[cur].append(m.group(1))
    return funcs


def check(funcs):
    out = {}
    for name, ops in funcs.items():
        m = re.search(r"k_instrILi(\d+)ELi(\d+)E", name)
        if not m:
            continue
        K, N = int(m.group(1)), int(m.group(2))
        clk = [i for i, op in enumerate(ops) if op.startswith("CS2R") or (op.startswith("S2R") and False)]
        # S2R SR_CLOCKLO/HI pairs on older parts: treat any S2R as a clock read candidate
        if len(clk) < 2:
            clk = [i for i, op in enumerate(ops) if op.startswith("S2R") or op.startswith("CS2R")]
        kind, expect = KINDS.get(K, (str(K), "?"))
        entry = dict(kind=kind, N=N, clock_reads=len(clk), verified=False, between=None)
        # the measured bracket is the LAST pair of clock reads before the record store... there are two brackets
        # (warm and measured) per sample loop; both have the same body. Take the first pair with the expected opcode.
        for a, b in zip(clk, clk[1:]):
            body = ops[a + 1:b]
            n_target = sum(1 for op in body if op.split(".")[0] == expect)
            if n_target:
                entry["between"] = {}
                for op in body:
                    entry["between"][op.split(".")[0]] = entry["between"].get(op.split(".")[0], 0) + 1
                entry["target_count"] = n_target
                entry["verified"] = (n_target == N)
                entry["bracket_opcodes"] = sorted(set(op for op in body))[:40]
                break
        out[name] = entry
    return out


def main():
    binary = sys.argv[1]
    cuobjdump = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "cuobjdump")
    if not os.path.exists(cuobjdump):
        cuobjdump = "cuobjdump"
    sass = subprocess.run([cuobjdump, "-sass", binary], capture_output=True, text=True).stdout
    res = check(parse(sass))
    ok = sum(1 for v in res.values() if v["verified"]); n = len(res)
    print(f"sass_check: {ok}/{n} bracket kernels verified (exactly N target opcodes between the clock reads)")
    for name, v in sorted(res.items(), key=lambda kv: (kv[1]["kind"], kv[1]["N"])):
        if not v["verified"]:
            print(f"  NOT verified: {v['kind']} N={v['N']} target_count={v.get('target_count')} between={v.get('between')}")
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    if "--sass" in sys.argv:
        open(sys.argv[sys.argv.index("--sass") + 1], "w").write(sass)


if __name__ == "__main__":
    main()
