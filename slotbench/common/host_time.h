// Host timing and real-time helpers. Every host timestamp in slotbench is CLOCK_MONOTONIC_RAW ns:
// it is never slewed by NTP/PTP and PTP_SYS_OFFSET_PRECISE reports it, so all three clocks
// (CPU, GPU %globaltimer, NIC PHC) can be put on one axis.
#pragma once
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

namespace sb {

inline int64_t ts_ns(const timespec &ts) { return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec; }

inline int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts_ns(ts);
}

inline int64_t realtime_ns() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts_ns(ts);
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
}

inline void spin_until(int64_t t_raw) {
    while (now_ns() < t_raw) cpu_relax();
}

// clock_nanosleep cannot sleep on CLOCK_MONOTONIC_RAW, so convert the raw target to a
// CLOCK_MONOTONIC deadline using the current offset between the two clocks. The error is the
// NTP slew over the sleep (<= 500 ppm of at most one period), which the spin margin absorbs.
inline void sleep_until_raw(int64_t t_raw) {
    for (;;) {
        timespec mono;
        clock_gettime(CLOCK_MONOTONIC, &mono);
        int64_t raw = now_ns();
        int64_t delta = t_raw - raw;
        if (delta <= 0) return;
        int64_t target = ts_ns(mono) + delta;
        timespec ts{(time_t)(target / 1000000000LL), (long)(target % 1000000000LL)};
        int rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
        if (rc == 0) return;
        if (rc != EINTR) return;  // any other error: fall back to the caller's spin
    }
}

inline bool pin_thread(int core) {
    if (core < 0) return true;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
}

// SCHED_FIFO for the calling thread. prio <= 0 leaves the thread on SCHED_OTHER and returns true.
inline bool set_fifo(int prio) {
    if (prio <= 0) return true;
    sched_param sp{};
    sp.sched_priority = prio;
    return pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0;
}

inline bool lock_memory() { return mlockall(MCL_CURRENT | MCL_FUTURE) == 0; }

inline std::string read_text(const std::string &path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

inline std::string hostname() {
    char buf[256] = {0};
    gethostname(buf, sizeof buf - 1);
    return buf;
}

// ISO 8601 UTC with milliseconds, e.g. 2026-09-30T21:05:03.123Z
inline std::string iso_utc_now() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    tm t;
    gmtime_r(&ts.tv_sec, &t);
    char buf[64];
    snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday,
             t.tm_hour, t.tm_min, t.tm_sec, ts.tv_nsec / 1000000L);
    return buf;
}

}  // namespace sb
