#!/usr/bin/env python3
"""Verify the per-instruction brackets in SASS: for every k_instr<K,N> kernel in a cuobjdump -sass listing, find the
two clock reads (CS2R / S2R SR_CLOCK*) and count the opcodes between them. The expected opcode for each kind must
appear exactly N times. Usage: sass_check.py BINARY [--json OUT] (needs cuobjdump on PATH or CUDA_HOME)."""
import json, os, re, subprocess, sys

# kind -> (name, target opcode, exact match?) ; exact: the chain's IMAD has no suffix, the compiler's moves are IMAD.MOV etc.
KINDS = {0: ("LDG_CA", "LDG", False), 1: ("LDG_CG", "LDG", False), 2: ("LDG_CS", "LDG", False), 3: ("LDG_NC", "LDG", False),
         4: ("LDS", "LDS", False), 5: ("FADD", "FADD", False), 6: ("FFMA", "FFMA", False), 7: ("IMAD", "IMAD", True),
         8: ("SHFL", "SHFL", False), 9: ("ATOM_RET", "ATOMG", False), 10: ("RED", "RED", False), 11: ("STG_FENCE", "STG", False),
         12: ("BAR", "BAR", False)}


def is_target(op, expect, exact):
    return op == expect if exact else (op == expect or op.startswith(expect + "."))


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
    """Each k_instr<K,N> has two brackets (warm, measured) with identical bodies. The clock reads are CS2R on
    sm_70+ (S2R SR_CLOCKLO/HI pairs on older parts). For every consecutive pair of clock reads, count the target
    opcode between them; the bracket is the pair with the largest count. Verified when that count == N. Everything
    else found inside (the compiler may schedule independent instructions across a clock read) is listed."""
    out = {}
    for name, ops in funcs.items():
        m = re.search(r"k_instrILi(\d+)ELi(\d+)E", name)
        if not m:
            continue
        K, N = int(m.group(1)), int(m.group(2))
        clk = [i for i, op in enumerate(ops) if op.startswith("CS2R")]
        if len(clk) < 2:
            clk = [i for i, op in enumerate(ops) if op.startswith("S2R")]
        kind, expect, exact = KINDS.get(K, (str(K), "?", False))
        entry = dict(kind=kind, N=N, clock_reads=len(clk), verified=False, target_count=0, between=None, other_inside=None)
        best = None
        for a, b in zip(clk, clk[1:]):
            body = ops[a + 1:b]
            n_target = sum(1 for op in body if is_target(op, expect, exact))
            if best is None or n_target > best[0]:
                best = (n_target, body)
        if best:
            n_target, body = best
            hist = {}
            for op in body:
                hist[op] = hist.get(op, 0) + 1
            entry.update(target_count=n_target, verified=(n_target == N), between=hist,
                         other_inside=sorted(op for op in hist if not is_target(op, expect, exact)))
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
