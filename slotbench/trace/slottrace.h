// slottrace: per-slot host records and the GPU residency probe, shared by the drivers, the mock
// launcher and the analyzer (trace/analyze.py parses the same layouts with struct).
//
// Every host timestamp is CLOCK_MONOTONIC_RAW in ns, the axis the kernel tracer also uses
// (trace_clock mono_raw), so scheduler/interrupt events and slot events need no conversion.
// GPU timestamps (%globaltimer) are mapped to that axis with the run's pre/post clock fits.
#pragma once
#include <cstdint>

namespace sb {

// One per measured boundary, written to --host-raw (little endian, 48 bytes).
struct SlotHostRec {
    uint64_t slot;
    int64_t t_target;   // boundary k on the host axis
    int64_t t_wake;     // sleep returned (CPU mode; 0 in GPU mode)
    int64_t t_launch;   // spin ended, launch call entered (CPU mode; 0 in GPU mode)
    int64_t t_return;   // launch call returned (CPU mode; 0 in GPU mode)
    int32_t cpu;        // CPU the launch was issued from (-1 unknown)
    uint32_t flags;     // copy of the slot's LsRec flags (1 skipped, 2 launch error, 8 timeout)
};
static_assert(sizeof(SlotHostRec) == 48, "SlotHostRec");

// GPU residency probe output (--probe-out): a header followed by gap records. A gap is an interval
// in which a resident probe thread did not observe %globaltimer advancing by less than gap_ns, i.e.
// the probe's context was not running (time-sliced out by another context) or the probe was starved.
struct ProbeHeader {
    char magic[8];        // "SBPROBE1"
    uint64_t gap_ns;      // detection threshold
    uint64_t g_first;     // first probe reading
    uint64_t g_last;      // last probe reading
    uint64_t n_gaps;      // gaps detected (may exceed capacity)
    uint64_t capacity;    // gap records stored = min(n_gaps, capacity)
    uint64_t iterations;  // probe loop iterations
    uint64_t reserved;
};
static_assert(sizeof(ProbeHeader) == 64, "ProbeHeader");

struct ProbeGap {
    uint64_t g_from;  // last reading before the gap
    uint64_t g_to;    // first reading after the gap
};
static_assert(sizeof(ProbeGap) == 16, "ProbeGap");

}  // namespace sb
