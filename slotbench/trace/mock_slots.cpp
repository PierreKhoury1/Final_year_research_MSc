// mock_slots: the CPU-mode slot launch loop of lockstep_driver without a GPU, for developing and
// validating slottrace on machines without CUDA. It runs the same host timing (sleep to T-spin,
// spin to T, "launch call", wait for the previous slot, skip boundaries that pass meanwhile) and
// writes the same files as the driver: LsRec raw records, SlotHostRec host records and a JSON summary
// whose clock fits are the identity, because the "GPU" times are host CLOCK_MONOTONIC_RAW ns:
//   launch call = busy --launch-us; slot start = max(return + --gpu-delay-us, previous end);
//   slot end = start + --exec-us.
// --busy-poll 1 replaces the sleep with a busy wait for the whole period (as busy-polling RAN
// workers do), which with SCHED_FIFO exercises RT throttling.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sched.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "host_time.h"
#include "json_writer.h"
#include "slottrace.h"

using namespace sb;

namespace {
struct LsRecMock {  // same layout as LsRec in driver/lockstep_common.h (64 bytes)
    uint64_t slot; int64_t t_target; uint64_t g_target, g_launch, g_launch_done, g0, g1, flags;
};
static_assert(sizeof(LsRecMock) == 64, "LsRec layout");

[[noreturn]] void usage(const char *m) {
    fprintf(stderr, "mock_slots: %s\nusage: mock_slots --out F.json --raw F.bin --host-raw F.bin [--slots N] [--warmup N]\n"
                    "  [--period-us F] [--deadline-us F] [--spin-us F] [--core N] [--fifo P] [--launch-us F]\n"
                    "  [--gpu-delay-us F] [--exec-us F] [--busy-poll 0|1] [--label S]\n", m);
    exit(2);
}
}  // namespace

int main(int argc, char **argv) {
    std::string out, raw, host_raw, label;
    long slots = 20000, warmup = 200;
    double period_us = 500, deadline_us = 300, spin_us = 200, launch_us = 8, gpu_delay_us = 5, exec_us = 140;
    int core = -1, fifo = 0, busy_poll = 0;
    for (int i = 1; i < argc; i++) {
        std::string f = argv[i];
        if (i + 1 >= argc) usage(("missing value for " + f).c_str());
        const char *v = argv[++i];
        if (f == "--out") out = v; else if (f == "--raw") raw = v; else if (f == "--host-raw") host_raw = v;
        else if (f == "--label") label = v;
        else if (f == "--slots") slots = atol(v); else if (f == "--warmup") warmup = atol(v);
        else if (f == "--period-us") period_us = atof(v); else if (f == "--deadline-us") deadline_us = atof(v);
        else if (f == "--spin-us") spin_us = atof(v); else if (f == "--core") core = atoi(v);
        else if (f == "--fifo") fifo = atoi(v); else if (f == "--launch-us") launch_us = atof(v);
        else if (f == "--gpu-delay-us") gpu_delay_us = atof(v); else if (f == "--exec-us") exec_us = atof(v);
        else if (f == "--busy-poll") busy_poll = atoi(v);
        else usage(("unknown flag " + f).c_str());
    }
    if (out.empty() || raw.empty() || host_raw.empty() || slots < 1 || warmup < 0) usage("bad arguments");
    const long total = slots + warmup;
    const int64_t period = llround(period_us * 1e3), spin = llround(spin_us * 1e3), launch = llround(launch_us * 1e3),
                  gdelay = llround(gpu_delay_us * 1e3), exec = llround(exec_us * 1e3);
    std::vector<LsRecMock> rec(total);
    std::vector<SlotHostRec> hrec(total);
    memset(rec.data(), 0, rec.size() * sizeof rec[0]);
    memset(hrec.data(), 0, hrec.size() * sizeof hrec[0]);
    bool mlock_ok = lock_memory(), pin_ok = pin_thread(core), fifo_ok = set_fifo(fifo);
    prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
    const long tid = syscall(SYS_gettid);
    const int cpu0 = sched_getcpu();
    const int64_t t_start = now_ns() + 200000000LL;
    for (long k = 0; k < total; k++) {
        rec[k].slot = hrec[k].slot = (uint64_t)k;
        rec[k].t_target = hrec[k].t_target = t_start + k * period;
        rec[k].g_target = (uint64_t)rec[k].t_target;
        hrec[k].cpu = -1;
    }
    const std::string start_utc = iso_utc_now();
    int64_t prev_end = 0;
    long launched = -1, skipped = 0;
    for (long k = 0; k < total;) {
        const int64_t T = rec[k].t_target;
        if (launched >= 0) {  // wait for the previous slot; boundaries that pass meanwhile are skipped
            bool all_skipped = false;
            while (now_ns() < prev_end) {
                if (now_ns() >= rec[k].t_target) {
                    rec[k].flags = 1; skipped++;
                    if (++k >= total) { all_skipped = true; break; }
                }
                cpu_relax();
            }
            launched = -1;
            if (all_skipped || k >= total) break;
            if (rec[k].flags & 1) continue;
        }
        const int64_t Tk = rec[k].t_target;
        (void)T;
        if (busy_poll) spin_until(Tk - spin); else sleep_until_raw(Tk - spin);
        const int64_t tw = now_ns();
        spin_until(Tk);
        const int64_t t0 = now_ns();
        spin_until(t0 + launch);  // the "launch call"
        const int64_t t1 = now_ns();
        const int64_t g0 = std::max(t1 + gdelay, prev_end);
        prev_end = g0 + exec;
        rec[k].g_launch = (uint64_t)t0; rec[k].g_launch_done = (uint64_t)t1;
        rec[k].g0 = (uint64_t)g0; rec[k].g1 = (uint64_t)prev_end;
        hrec[k].t_wake = tw; hrec[k].t_launch = t0; hrec[k].t_return = t1; hrec[k].cpu = sched_getcpu();
        launched = k;
        k++;
    }
    while (now_ns() < prev_end) cpu_relax();
    const std::string end_utc = iso_utc_now();
    long misses = 0, recorded = 0;
    skipped = 0;  // measured boundaries only (warm-up excluded)
    for (long k = warmup; k < total; k++) {
        hrec[k].flags = (uint32_t)rec[k].flags;
        if (rec[k].flags & 1) { misses++; skipped++; continue; }
        recorded++;
        if ((double)((int64_t)rec[k].g1 - rec[k].t_target) > deadline_us * 1e3) misses++;
    }
    auto write = [&](const std::string &path, const void *p, size_t sz, size_t n) {
        FILE *f = fopen(path.c_str(), "wb");
        if (!f || fwrite(p, sz, n, f) != n || fclose(f) != 0) { perror(path.c_str()); exit(2); }
    };
    write(raw, rec.data() + warmup, sizeof rec[0], (size_t)slots);
    write(host_raw, hrec.data() + warmup, sizeof hrec[0], (size_t)slots);
    auto ident = [&](int64_t t) {
        Json f; f.add("ok", true).add("a", 1.0).add("b_ns", 0.0).add("g_ref", (unsigned long long)t).add("t_ref", (long long)t);
        return f.str();
    };
    Json j;
    j.add("ok", true).add("mock", true).add("mode", "cpu").add("label", label).add("gpu", "none (mock)")
        .add("slots", (long long)slots).add("warmup", (long long)warmup).add("period_us", period_us)
        .add("deadline_us", deadline_us).add("spin_us", spin_us).add("launch_us", launch_us).add("exec_us", exec_us)
        .add("busy_poll", busy_poll).add("core", core).add("pin_ok", pin_ok).add("fifo", fifo).add("fifo_ok", fifo_ok)
        .add("mlock_ok", mlock_ok).add("launcher_tid", (long long)tid).add("launcher_cpu", cpu0)
        .add("run_start_utc", start_utc).add("run_end_utc", end_utc)
        .add("recorded", (long long)recorded).add("skipped", (long long)skipped).add("misses", (long long)misses)
        .add("total_slots", (long long)slots).add("two_point_ok", true)
        .add_raw("clock_fit_pre", ident(t_start)).add_raw("clock_fit_post", ident(t_start + total * period))
        .add("host_raw", host_raw).add("probe", 0);
    FILE *f = fopen(out.c_str(), "w");
    if (!f) { perror("out"); return 2; }
    fputs(j.str().c_str(), f); fputs("\n", f); fclose(f);
    printf("mock %s: %ld slots, %ld recorded, %ld skipped, %ld misses (deadline %.0f us), tid %ld cpu %d fifo %d%s\n",
           label.c_str(), slots, recorded, skipped, misses, deadline_us, tid, cpu0, fifo, fifo_ok ? "" : " (FIFO DENIED)");
    return 0;
}
