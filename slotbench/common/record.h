// On-disk format of slots.bin (DESIGN.md section 4). Little endian, fixed 64-byte header and records.
#pragma once
#include <cstdint>
#include <cstring>

namespace sb {

struct SlotFileHeader {
    char magic[8];         // "SLOTBEN1"
    uint32_t version;      // 1
    uint32_t record_size;  // sizeof(SlotRecord)
    uint64_t n_records;    // written at clean close; 0 after a crash (readers use the file size)
    double period_ns;
    double deadline_ns;
    int64_t t_start_ns;    // host ns of slot boundary 0
    uint8_t reserved[16];
};

struct SlotRecord {        // all times host CLOCK_MONOTONIC_RAW ns unless noted
    uint64_t slot;         // boundary index k
    int64_t t_sched;       // scheduled boundary
    int64_t t_wake;        // return from clock_nanosleep
    int64_t t0;            // just before the launch call
    int64_t t_launched;    // launch call returned
    int64_t t1;            // completion observed
    uint64_t g0;           // GPU %globaltimer ns at the start stamp (0 if missing)
    uint64_t g1;           // GPU %globaltimer ns at the end stamp (0 if missing)
};

static_assert(sizeof(SlotFileHeader) == 64, "header must be 64 bytes");
static_assert(sizeof(SlotRecord) == 64, "record must be 64 bytes");

inline SlotFileHeader make_header(double period_ns, double deadline_ns, int64_t t_start_ns) {
    SlotFileHeader h;
    std::memset(&h, 0, sizeof h);
    std::memcpy(h.magic, "SLOTBEN1", 8);
    h.version = 1;
    h.record_size = sizeof(SlotRecord);
    h.period_ns = period_ns;
    h.deadline_ns = deadline_ns;
    h.t_start_ns = t_start_ns;
    return h;
}

}  // namespace sb
