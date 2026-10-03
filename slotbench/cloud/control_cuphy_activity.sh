#!/bin/bash
# Separate follow-up: ordinary CPU launch, CPU launch with GPU keep-alive, GPU launch.
# Reuses the validated repeated-trial collection, CPU controls, and correctness gates.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/repeat_cuphy_lockstep.sh"

repeat_plan() {
    python3 - "$OUT" "$REPEAT_SEED" "$SLOTS" "$WARMUP" "$PERIOD" "$DEADLINE" "$TV_SHA" "$ACAR_COMMIT" "$OBSERVER_CPU" "${REPEATS:-${SB_CUPHY_REPEATS:-6}}" <<'PY'
import itertools, json, os, pathlib, random, sys
out, seed, slots, warmup, period, deadline, vector, revision, observer, repeats = sys.argv[1:]
out = pathlib.Path(out)
repeats = int(repeats)
assert repeats % 6 == 0, "REPEATS must be a multiple of 6 (one of each variant order per block of six)"
orders = list(itertools.permutations(("cpu", "gpu", "cpu_keepalive"))) * (repeats // 6)
random.Random(int(seed)).shuffle(orders)
schedule = []
for triplet, order in enumerate(orders, 1):
    for position, variant in enumerate(order, 1):
        mode = "gpu" if variant == "gpu" else "cpu"
        name = f"t{triplet:02d}_p{position}_{variant}_alone"
        schedule.append(dict(case_index=len(schedule)+1, block_index=triplet,
            pair_index=triplet, triplet_index=triplet, position=position,
            condition="alone", variant=variant, mode=mode, name=name,
            json=name+".json", raw=name+".bin", run=name+".run.json",
            pusch_log=name+".pusch.log", scheduler=name+".scheduler.json",
            telemetry_before=name+".telemetry.before.xml",
            telemetry_after=name+".telemetry.after.xml",
            telemetry_continuous=name+".telemetry.continuous.csv"))
clock = json.loads((out / "clock_control.json").read_text()) if (out / "clock_control.json").exists() else {"clocks_locked": None}
cpu = json.loads((out / "cpu_selection.json").read_text())
manifest = dict(schema_version=1, experiment_kind="idle_keepalive_control", seed=int(seed), repeats=repeats,
    randomization=f"Each of the six permutations of CPU, GPU, CPU+keepalive occurs {repeats // 6} time(s); triplet order shuffled by seed.",
    slots=int(slots), warmup=int(warmup), period_us=float(period), deadline_us=float(deadline),
    test_vector_sha256=vector, aerial_commit=revision, clocks_locked=clock.get("clocks_locked"),
    clock_control=clock, cpu_control=cpu, gates=["gate_cpu_alone", "gate_gpu_alone", "gate_cpu_keepalive_alone"],
    schedule=schedule, telemetry_interval_ms=100, observer_cpu=int(observer),
    phy_cpu_requested=int(os.environ.get("SB_CUPHY_CPU", "0")), adversary_cpu_requested=None,
    scope="Isolated fixed-vector replay; keep-alive changes activity, residency, occupancy and polling traffic, not only clocks.",
    logger_note="Retain the primary campaign's common nvlog fallback configuration for comparability.")
with (out / "experiment.json").open("x") as stream:
    json.dump(manifest, stream, indent=2)
with (out / "schedule.tsv").open("x") as stream:
    for row in schedule:
        stream.write("\t".join(str(row[k]) for k in ("case_index", "triplet_index", "position", "variant", "mode", "name")) + "\n")
PY
}

activity_validate_case() {
    python3 - "$OUT/$1.json" "$2" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
expected = sys.argv[2] == "cpu_keepalive"
keeper = data.get("keepalive", {})
if data.get("keepalive_active") is not expected or keeper.get("requested") is not expected:
    raise SystemExit("GPU activity control differs from the assigned variant")
if expected and (keeper.get("blocks") != 1 or keeper.get("threads_per_block") != 1
                 or keeper.get("covers_replay") is not True or keeper.get("stop_reason") != "host_stop"
                 or keeper.get("stream_priority") != data["stream_priority"]):
    raise SystemExit("Keep-alive configuration, lifetime, or priority check failed")
PY
}

repeat_schedule() {
    local case_index triplet position variant mode name measured_slots=$SLOTS
    # Test the added control before its measured cases; preserve ordinary gates too.
    export SB_CUPHY_LOCKSTEP_CPU_KEEPALIVE=1
    SLOTS=64
    repeat_case gate_cpu_keepalive_alone cpu none none
    activity_validate_case gate_cpu_alone cpu
    activity_validate_case gate_gpu_alone gpu
    activity_validate_case gate_cpu_keepalive_alone cpu_keepalive
    SLOTS=$measured_slots
    while IFS=$'\t' read -r case_index triplet position variant mode name; do
        repeat_no_mps
        export SB_CUPHY_LOCKSTEP_CPU_KEEPALIVE=0
        [[ $variant != cpu_keepalive ]] || export SB_CUPHY_LOCKSTEP_CPU_KEEPALIVE=1
        log "Activity control $case_index/$(( ${REPEATS:-${SB_CUPHY_REPEATS:-6}} * 3 )): triplet=$triplet position=$position variant=$variant"
        repeat_case "$name" "$mode" none none
        activity_validate_case "$name" "$variant"
    done < "$OUT/schedule.tsv"
    export SB_CUPHY_LOCKSTEP_CPU_KEEPALIVE=0
    repeat_no_mps
}

activity_main() {
    export SB_CUPHY_REPEATS="${SB_CUPHY_REPEATS:-6}" SB_CUPHY_REPEAT_SEED="${SB_CUPHY_REPEAT_SEED:-20261003}"
    [[ $SB_CUPHY_REPEATS =~ ^(6|12|18)$ ]]
    export SB_CUPHY_LOCKSTEP_CPU_KEEPALIVE=0
    repeat_main "$@"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then activity_main "$@"; fi
