#!/bin/bash
# Read-only preflight; writes reports but changes no clocks, affinities, or policies.
# Usage: cuphy_host_probe.sh OUTPUT_DIRECTORY [NVIDIA_SMI_GPU_INDEX [PUSCH_PROCESS_PID]]
# Source recommended_cpus.env explicitly only after reviewing host_probe.json.
set -euo pipefail
if (( $# < 1 || $# > 3 )) || [[ ! ${2:-0} =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 OUTPUT_DIRECTORY [GPU_INDEX [PUSCH_PROCESS_PID]]" >&2
    exit 2
fi
if [[ -n ${3:-} && ! ${3} =~ ^[1-9][0-9]*$ ]]; then
    echo "PUSCH_PROCESS_PID must be a positive integer" >&2
    exit 2
fi
mkdir -p -- "$1"
python3 - "$1" "${2:-0}" "${3:-}" <<'PY'
import csv
import datetime
import io
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys

out = Path(sys.argv[1]).resolve()
gpu = sys.argv[2]
target_pid = int(sys.argv[3]) if sys.argv[3] else None
warnings = []
commands = {}

def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None

def integer(path):
    try:
        return int(read(path))
    except (TypeError, ValueError):
        return None

def cpulist(value):
    result = set()
    if not value:
        return result
    for item in value.split(','):
        match = re.fullmatch(r'(\d+)(?:-(\d+))?', item.strip())
        if not match:
            raise ValueError('invalid Linux CPU list: ' + value)
        lo = int(match[1])
        hi = int(match[2] or match[1])
        if hi < lo or hi > 1048575:
            raise ValueError('invalid CPU range: ' + item)
        result.update(range(lo, hi + 1))
    return result

def capture(name, argv):
    try:
        completed = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=10, check=False)
        stdout, stderr, status = completed.stdout, completed.stderr, completed.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        stdout, stderr, status = '', str(error), 127
    (out / name).write_text(stdout + ('\nSTDERR:\n' + stderr if stderr else ''))
    commands[name] = dict(argv=argv, returncode=status)
    return stdout if status == 0 else ''

identity_text = capture('gpu_identity.csv', ['nvidia-smi', '-i', gpu,
    '--query-gpu=index,uuid,name,pci.bus_id,driver_version', '--format=csv'])
capture('gpu_supported_clocks.csv', ['nvidia-smi', '-i', gpu,
    '--query-supported-clocks=memory,graphics', '--format=csv'])
capture('gpu_clock_power_state.txt', ['nvidia-smi', '-i', gpu, '-q',
    '-d', 'CLOCK,POWER,PERFORMANCE,TEMPERATURE'])
capture('gpu_topology.txt', ['nvidia-smi', 'topo', '-m'])
numa_text = capture('gpu_nearest_numa.txt', ['nvidia-smi', 'topo', '-C', '-i', gpu])
capture('gpu_query_fields.txt', ['nvidia-smi', '--help-query-gpu'])
capture('lscpu.json', ['lscpu', '--json'])
identity_rows = [{k.strip(): v.strip() for k, v in row.items()}
                 for row in csv.DictReader(io.StringIO(identity_text))] if identity_text else []
identity = identity_rows[0] if len(identity_rows) == 1 else {}
if not identity:
    warnings.append('GPU identity unavailable; GPU settings and locality are not verified.')

allowed = sorted(os.sched_getaffinity(0))
if not allowed:
    raise SystemExit('no allowed CPUs')
gpu_local, locality_source, gpu_numa = set(), 'unavailable', None
pci = identity.get('pci.bus_id', '').lower()
match = re.fullmatch(r'([0-9a-f]+):([0-9a-f]{2}):([0-9a-f]{2}\.[0-7])', pci)
if match:
    pci_path = Path('/sys/bus/pci/devices') / f'{int(match[1], 16):04x}:{match[2]}:{match[3]}'
    gpu_numa = integer(pci_path / 'numa_node')
    local_text = read(pci_path / 'local_cpulist')
    if local_text:
        gpu_local = cpulist(local_text)
        locality_source = str(pci_path / 'local_cpulist')
if gpu_numa is None or gpu_numa < 0:
    match = re.search(r'(?:^|[:\s])(-?\d+)\s*$', numa_text)
    gpu_numa = int(match[1]) if match else None
if not gpu_local and gpu_numa is not None and gpu_numa >= 0:
    numa_path = Path(f'/sys/devices/system/node/node{gpu_numa}/cpulist')
    gpu_local = cpulist(read(numa_path))
    if gpu_local:
        locality_source = str(numa_path)
local_allowed = set(allowed) & gpu_local
if not local_allowed:
    warnings.append('No verified GPU-local allowed CPU; recommendations use the allowed affinity mask.')

topology = []
keys = {}
siblings = {}
for cpu in allowed:
    base = Path(f'/sys/devices/system/cpu/cpu{cpu}')
    topo = base / 'topology'
    package, die, core = [integer(topo / name) for name in
                          ('physical_package_id', 'die_id', 'core_id')]
    peers = cpulist(read(topo / 'thread_siblings_list'))
    known = bool(peers) or (package is not None and core is not None)
    if peers:
        keys[cpu] = ('siblings', tuple(sorted(peers)))
    elif known:
        keys[cpu] = ('core', package, die, core)
    else:
        keys[cpu] = ('unknown', cpu)
    siblings[cpu] = peers or {cpu}
    topology.append(dict(cpu=cpu, package=package, die=die, core=core,
        thread_siblings=sorted(peers), physical_core_known=known,
        gpu_local=cpu in gpu_local,
        governor=read(base / 'cpufreq/scaling_governor'),
        current_khz=integer(base / 'cpufreq/scaling_cur_freq')))

# Prefer GPU locality, avoid CPU 0 when an equivalent allowed core exists, then
# choose deterministic low CPU IDs. This does not claim host/IRQ isolation.
ranked = sorted(allowed, key=lambda cpu: (bool(local_allowed) and cpu not in local_allowed,
                                          cpu == 0, cpu))
selected = []
used_keys = set()
for cpu in ranked:
    if keys[cpu] not in used_keys:
        selected.append(cpu)
        used_keys.add(keys[cpu])
    if len(selected) == 3:
        break
physical_distinct = len(selected) == 3 and all(
    item['physical_core_known'] for item in topology if item['cpu'] in selected)
while len(selected) < 3:
    selected.append(selected[-1])
phy, adversary, telemetry = selected
background = sorted(set(allowed) - siblings[phy])
if not background:
    background = allowed
if not physical_distinct:
    warnings.append('Three verified distinct physical cores are unavailable; selected CPUs may share resources.')

policies = {getattr(os, name): name for name in
            ('SCHED_OTHER', 'SCHED_FIFO', 'SCHED_RR', 'SCHED_BATCH', 'SCHED_IDLE') if hasattr(os, name)}
def thread_state(tid):
    state = dict(tid=tid)
    try:
        policy = os.sched_getscheduler(tid)
        state.update(policy=policies.get(policy, str(policy)), policy_number=policy,
                     priority=os.sched_getparam(tid).sched_priority,
                     affinity=sorted(os.sched_getaffinity(tid)))
        state['comm'] = read(f'/proc/{tid}/comm')
    except OSError as error:
        state['error'] = str(error)
    return state

status = read('/proc/self/status') or ''
status_fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
cap_eff = int(status_fields.get('CapEff', '0').strip(), 16)
caps = {name: bool(cap_eff & (1 << bit)) for name, bit in
        [('CAP_SYS_ADMIN', 21), ('CAP_SYS_NICE', 23), ('CAP_IPC_LOCK', 14)]}
limits = {name: list(resource.getrlimit(getattr(resource, name))) for name in
          ('RLIMIT_RTPRIO', 'RLIMIT_RTTIME', 'RLIMIT_MEMLOCK') if hasattr(resource, name)}
threads = []
if target_pid:
    task_directory = Path(f'/proc/{target_pid}/task')
    if not task_directory.exists():
        raise SystemExit(f'PUSCH process {target_pid} does not exist')
    threads = [thread_state(int(path.name)) for path in task_directory.iterdir() if path.name.isdigit()]

recommended = dict(SB_CUPHY_CPU=phy, SB_ADVERSARY_CPU=adversary, SB_TELEMETRY_CPU=telemetry,
                   SB_BACKGROUND_CPUS=','.join(map(str, background)))
mask = os.environ.get('CUDA_VISIBLE_DEVICES')
if mask and mask not in ('0', gpu):
    warnings.append('CUDA_VISIBLE_DEVICES may remap CUDA ordinals; verify selected CUDA GPU matches probed UUID.')
report = dict(schema_version=1, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    hostname=os.uname().nodename, kernel=os.uname().release, uid=os.getuid(),
    gpu_index_requested=int(gpu), gpu_identity=identity, gpu_identity_verified=bool(identity),
    cuda_visible_devices=mask, gpu_numa_node=gpu_numa, gpu_locality_source=locality_source,
    gpu_local_cpus=sorted(gpu_local), allowed_cpus=allowed, gpu_local_allowed_cpus=sorted(local_allowed),
    cpu_topology=topology, recommended=recommended, distinct_physical_cores=physical_distinct,
    probe_thread=thread_state(os.getpid()), capability_effective_hex=f'{cap_eff:x}', capabilities=caps,
    resource_limits=limits, realtime_permission_tested=False, gpu_clock_changes_tested=False,
    kernel_scheduler={name: read('/proc/sys/kernel/' + name) for name in
                      ('sched_rt_period_us', 'sched_rt_runtime_us', 'sched_rr_timeslice_ms')},
    cgroup_membership=read('/proc/self/cgroup'),
    cgroup_cpu={name: read('/sys/fs/cgroup/' + name) for name in
                ('cpu.max', 'cpu.stat', 'cpuset.cpus.effective', 'cpuset.mems.effective')},
    target_pid=target_pid, target_threads=threads, commands=commands, warnings=warnings)
(out / 'host_probe.json').write_text(json.dumps(report, indent=2) + '\n')
(out / 'recommended_cpus.env').write_text(
    '# Recommendations only; inspect host_probe.json before using. No host settings were changed.\n' +
    ''.join(f'export {name}={value}\n' for name, value in recommended.items()))
print(json.dumps(dict(output=str(out), recommended=recommended,
                      distinct_physical_cores=physical_distinct, warnings=warnings), indent=2))
PY
