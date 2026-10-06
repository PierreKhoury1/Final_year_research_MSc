// Host-only check that the gputrace record layouts match what analysis/gputrace.py parses (numpy dtypes).
// Build: g++ -std=c++17 -Icommon -Igputrace -o bin/test_gputrace_format gputrace/tests/test_format.cpp
#include <cstddef>
#include <cstdio>
#include "gputrace.h"

using namespace sb::gt;

int main() {
    static_assert(sizeof(GpuRec) == 64 && sizeof(HostEvent) == 32, "sizes");
    static_assert(offsetof(GpuRec, g_begin) == 0 && offsetof(GpuRec, g_end) == 8 && offsetof(GpuRec, clk_begin) == 16 &&
                  offsetof(GpuRec, clk_end) == 24 && offsetof(GpuRec, max_gap_ns) == 32 && offsetof(GpuRec, smid) == 40 &&
                  offsetof(GpuRec, kernel_id) == 44 && offsetof(GpuRec, block) == 48 && offsetof(GpuRec, tag) == 52 &&
                  offsetof(GpuRec, n_iters) == 56 && offsetof(GpuRec, flags) == 60, "GpuRec layout");
    static_assert(offsetof(HostEvent, t) == 0 && offsetof(HostEvent, type) == 8 && offsetof(HostEvent, kernel_id) == 12 &&
                  offsetof(HostEvent, a) == 16 && offsetof(HostEvent, b) == 24, "HostEvent layout");
    printf("gputrace formats ok\n");
    return 0;
}
