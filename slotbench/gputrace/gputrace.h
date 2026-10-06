// gputrace: on-disk record formats. Host-only header (no CUDA), shared by the tracer binary, the host test
// and analysis/gputrace.py (which parses the same layouts with struct).
//
// A run is one strategy. It writes PREFIX.gpu.bin (GpuRec), PREFIX.host.bin (HostEvent), PREFIX.json (meta) and
// the clock-sync samples PREFIX.{pre,post}.{classic,up,down}.bin in the formats of tools/pcieclock.cu, so
// analysis/pcieclock.py's fits apply unchanged. Host timestamps are CLOCK_MONOTONIC_RAW ns; GPU timestamps are
// %globaltimer ns (and clock64() SM cycles) and are mapped to the host axis by the analysis with a hard bound.
#pragma once
#include <cstdint>

namespace sb {
namespace gt {

// One per traced block (thread 0 writes it) or per sample of a resident sampler. 64 bytes.
struct GpuRec {
    uint64_t g_begin;     // %globaltimer at block start (or sample time)
    uint64_t g_end;       // %globaltimer at block end (== g_begin for point samples)
    uint64_t clk_begin;   // clock64() at start: SM cycle counter, per SM, not comparable across SMs
    uint64_t clk_end;
    uint64_t max_gap_ns;  // largest %globaltimer jump between consecutive spin iterations (0 if not spinning)
    uint32_t smid;        // %smid
    uint32_t kernel_id;   // host-assigned launch sequence number (ties the record to its HostEvents)
    uint32_t block;       // linear block index
    uint32_t tag;         // strategy-specific: stream index, priority class, role, sample kind
    uint32_t n_iters;     // spin iterations (0 if none)
    uint32_t flags;       // strategy-specific
};
static_assert(sizeof(GpuRec) == 64, "GpuRec");

// Host-side events. 32 bytes.
struct HostEvent {
    int64_t t;            // CLOCK_MONOTONIC_RAW ns
    uint32_t type;        // HostEventType
    uint32_t kernel_id;   // launch this event belongs to (0 if none)
    uint64_t a;           // type-specific
    uint64_t b;           // type-specific
};
static_assert(sizeof(HostEvent) == 32, "HostEvent");

enum HostEventType : uint32_t {
    EV_LAUNCH_ENTER = 1,   // just before the launch call (a = blocks, b = threads)
    EV_LAUNCH_RETURN = 2,  // launch call returned
    EV_SYNC_ENTER = 3,     // just before cudaStreamSynchronize / cudaEventSynchronize
    EV_SYNC_RETURN = 4,    // it returned
    EV_EVENT_SEEN = 5,     // cudaEventQuery first returned cudaSuccess (a = polls)
    EV_FLAG_SEEN = 6,      // mapped-memory completion flag first observed (a = flag value = GPU g_end, b = polls)
    EV_MARK = 7,           // strategy marker (a = code, b = value)
    EV_COPY_ENTER = 8,     // before cudaMemcpyAsync (a = bytes, b = 0 H2D / 1 D2H)
    EV_COPY_RETURN = 9,    // memcpy call returned
    EV_COPY_DONE = 10,     // copy's event observed complete (a = bytes, b = dir)
    EV_IDLE_END = 11,      // host finished an idle wait before a launch (a = idle ns requested)
    EV_GRAPH_ENTER = 12,   // before cudaGraphLaunch
    EV_GRAPH_RETURN = 13,
};

}  // namespace gt
}  // namespace sb
