// Slot timetable shared between the 5G slot driver and co-located GPU tenants (time-aware gating).
// The driver writes it once, before its first boundary: slot k's GPU work starts at about
// t0 + k * period_ns (host CLOCK_MONOTONIC_RAW) and occupies the GPU for at most busy_ns. A gated
// tenant only issues a chunk of GPU work when the chunk is expected to finish before the next slot.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <unistd.h>

namespace sb {

struct SlotTimetable {
    char magic[8];       // "SBTT0001"
    int64_t t0;          // first boundary (host CLOCK_MONOTONIC_RAW ns)
    int64_t period_ns;   // boundary spacing
    int64_t busy_ns;     // GPU time reserved after each boundary (slot work + margin)
    int64_t n_slots;     // boundaries in the run (gating ends after the last one)
    int64_t written_ns;  // when the driver wrote it (host CLOCK_MONOTONIC_RAW ns)
};
static_assert(sizeof(SlotTimetable) == 48, "SlotTimetable");

// Atomic publish: write a temporary file and rename it over the path.
inline bool write_timetable(const std::string &path, const SlotTimetable &tt) {
    std::string tmp = path + ".tmp." + std::to_string((long)getpid());
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(&tt, sizeof tt, 1, f) == 1;
    ok = (fclose(f) == 0) && ok;
    return ok && rename(tmp.c_str(), path.c_str()) == 0;
}

inline bool read_timetable(const std::string &path, SlotTimetable &tt) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = fread(&tt, sizeof tt, 1, f) == 1;
    fclose(f);
    return ok && memcmp(tt.magic, "SBTT0001", 8) == 0 && tt.period_ns > 0 && tt.busy_ns >= 0 && tt.n_slots > 0;
}

}  // namespace sb
