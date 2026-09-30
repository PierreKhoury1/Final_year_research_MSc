// phc_offset: NIC PTP hardware clock (PHC) vs host CLOCK_MONOTONIC_RAW, one CSV row per interval
// (DESIGN.md section 6). Methods, best first:
//   precise           PTP_SYS_OFFSET_PRECISE: hardware cross-timestamp (e.g. PCIe PTM) giving PHC,
//                     CLOCK_REALTIME and CLOCK_MONOTONIC_RAW for one instant; width 0.
//   extended_monoraw  PTP_SYS_OFFSET_EXTENDED with clockid = CLOCK_MONOTONIC_RAW (newer kernels):
//                     narrowest of --samples [sys, phc, sys] sandwiches taken directly in monoraw.
//   extended          same with CLOCK_REALTIME sandwiches, converted to monoraw with back-to-back
//                     realtime/monoraw reads taken just before and just after the ioctl.
// Output: host_monoraw_ns,host_realtime_ns,phc_ns,method,width_ns where width_ns is the full width of
// the host-time interval the PHC reading is known to lie in (sandwich + conversion uncertainty).
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/ptp_clock.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

// Old kernel headers: provide the ABI (unchanged since 4.x/5.x) ourselves.
#ifndef PTP_MAX_SAMPLES
#define PTP_MAX_SAMPLES 25
#endif
#ifndef PTP_SYS_OFFSET_PRECISE
struct ptp_sys_offset_precise {
    struct ptp_clock_time device;
    struct ptp_clock_time sys_realtime;
    struct ptp_clock_time sys_monoraw;
    unsigned int rsv[4];
};
#define PTP_SYS_OFFSET_PRECISE _IOWR(PTP_CLK_MAGIC, 8, struct ptp_sys_offset_precise)
#endif
#ifndef PTP_SYS_OFFSET_EXTENDED
struct ptp_sys_offset_extended {
    unsigned int n_samples;
    unsigned int rsv[3];
    struct ptp_clock_time ts[PTP_MAX_SAMPLES][3];
};
#define PTP_SYS_OFFSET_EXTENDED _IOWR(PTP_CLK_MAGIC, 9, struct ptp_sys_offset_extended)
#endif

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int s) { (void)s; g_stop = 1; }

static int64_t pct_ns(const struct ptp_clock_time *t) { return (int64_t)t->sec * 1000000000LL + (int64_t)t->nsec; }
static int64_t ts_ns(const struct timespec *t) { return (int64_t)t->tv_sec * 1000000000LL + t->tv_nsec; }
static int64_t clk_ns(clockid_t c) {
    struct timespec t;
    clock_gettime(c, &t);
    return ts_ns(&t);
}

// monoraw - realtime from realtime/monoraw/realtime reads; *unc = half the realtime bracket.
static int64_t raw_minus_real(int64_t *unc) {
    int64_t r0 = clk_ns(CLOCK_REALTIME), m = clk_ns(CLOCK_MONOTONIC_RAW), r1 = clk_ns(CLOCK_REALTIME);
    *unc = (r1 - r0 + 1) / 2;
    return m - (r0 + (r1 - r0) / 2);
}

static void usage(FILE *f) {
    fprintf(f, "usage: phc_offset [--dev /dev/ptp0] [--interval-ms 1000] [--count 0] [--out phc.csv] [--samples 25]\n");
}

static int parse_long(const char *s, long *v) {
    char *e = NULL;
    errno = 0;
    *v = strtol(s, &e, 10);
    return errno == 0 && e != s && *e == 0;
}

struct Sample {
    int64_t monoraw, realtime, phc, width;
    const char *method;
};

// Returns 0 on success, -1 (errno set) if the ioctl is unsupported/failed.
static int try_precise(int fd, struct Sample *s) {
    struct ptp_sys_offset_precise p;
    memset(&p, 0, sizeof p);
    if (ioctl(fd, PTP_SYS_OFFSET_PRECISE, &p) != 0) return -1;
    s->phc = pct_ns(&p.device);
    s->realtime = pct_ns(&p.sys_realtime);
    s->monoraw = pct_ns(&p.sys_monoraw);
    s->width = 0;
    s->method = "precise";
    return 0;
}

// use_raw: request CLOCK_MONOTONIC_RAW sandwiches via the clockid field that newer kernels put where
// rsv[0] was (older kernels reject a non-zero rsv with EINVAL). Written by offset so either header works.
static int try_extended(int fd, int n, int use_raw, struct Sample *s) {
    struct ptp_sys_offset_extended e;
    memset(&e, 0, sizeof e);
    e.n_samples = (unsigned)n;
    if (use_raw) {
        int cid = CLOCK_MONOTONIC_RAW;
        memcpy((char *)&e + sizeof e.n_samples, &cid, sizeof cid);
    }
    int64_t u0, u1;
    int64_t d0 = raw_minus_real(&u0);
    int rc = ioctl(fd, PTP_SYS_OFFSET_EXTENDED, &e);
    int64_t d1 = raw_minus_real(&u1);
    if (rc != 0) return -1;
    int best = -1;
    int64_t bw = INT64_MAX;
    for (int i = 0; i < n; i++) {
        int64_t w = pct_ns(&e.ts[i][2]) - pct_ns(&e.ts[i][0]);
        if (w >= 0 && w < bw) { bw = w; best = i; }
    }
    if (best < 0) { errno = EIO; return -1; }
    int64_t sys_mid = pct_ns(&e.ts[best][0]) + bw / 2;
    // Conversion uncertainty: read brackets plus any realtime step/slew between the two reads.
    int64_t d = d0 + (d1 - d0) / 2;
    int64_t conv = (u0 > u1 ? u0 : u1) + llabs(d1 - d0) / 2 + 1;
    s->phc = pct_ns(&e.ts[best][1]);
    if (use_raw) {
        s->monoraw = sys_mid;
        s->realtime = sys_mid - d;
        s->width = bw;  // realtime column is derived, the monoraw sandwich is direct
        s->method = "extended_monoraw";
    } else {
        s->realtime = sys_mid;
        s->monoraw = sys_mid + d;
        s->width = bw + 2 * conv;
        s->method = "extended";
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *dev = "/dev/ptp0", *out = "phc.csv";
    long interval_ms = 1000, count = 0, samples = 25;
    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        if (!strcmp(k, "-h") || !strcmp(k, "--help")) { usage(stdout); return 0; }
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", k); usage(stderr); return 2; }
        const char *v = argv[++i];
        int ok = 1;
        if (!strcmp(k, "--dev")) dev = v;
        else if (!strcmp(k, "--out")) out = v;
        else if (!strcmp(k, "--interval-ms")) ok = parse_long(v, &interval_ms) && interval_ms > 0;
        else if (!strcmp(k, "--count")) ok = parse_long(v, &count) && count >= 0;
        else if (!strcmp(k, "--samples")) ok = parse_long(v, &samples) && samples >= 1 && samples <= PTP_MAX_SAMPLES;
        else { fprintf(stderr, "unknown flag %s\n", k); usage(stderr); return 2; }
        if (!ok) { fprintf(stderr, "bad value for %s: %s\n", k, v); usage(stderr); return 2; }
    }

    int fd = open(dev, O_RDWR);
    if (fd < 0 && errno == EACCES) fd = open(dev, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", dev, strerror(errno)); return 1; }
    FILE *f = fopen(out, "w");
    if (!f) { fprintf(stderr, "open %s: %s\n", out, strerror(errno)); return 1; }

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    // Pick the method once: precise, then extended in monoraw, then extended in realtime.
    struct Sample s;
    int mode = 0;  // 0 precise, 1 extended_monoraw, 2 extended
    if (try_precise(fd, &s) != 0) {
        int e = errno;
        if (e != EOPNOTSUPP && e != ENOTTY && e != EINVAL && e != ENOSYS)
            fprintf(stderr, "PTP_SYS_OFFSET_PRECISE failed: %s; trying EXTENDED\n", strerror(e));
        mode = 1;
        if (try_extended(fd, (int)samples, 1, &s) != 0) {
            mode = 2;
            if (try_extended(fd, (int)samples, 0, &s) != 0) {
                fprintf(stderr, "PTP_SYS_OFFSET_EXTENDED failed on %s: %s\n", dev, strerror(errno));
                return 1;
            }
        }
    }
    fprintf(stderr, "phc_offset: %s using %s\n", dev, s.method);
    fprintf(f, "host_monoraw_ns,host_realtime_ns,phc_ns,method,width_ns\n");

    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);
    long done = 0, fails = 0;
    while (!g_stop && (count == 0 || done < count)) {
        int rc = mode == 0 ? try_precise(fd, &s) : try_extended(fd, (int)samples, mode == 1, &s);
        if (rc == 0) {
            fprintf(f, "%" PRId64 ",%" PRId64 ",%" PRId64 ",%s,%" PRId64 "\n", s.monoraw, s.realtime, s.phc, s.method,
                    s.width);
            fflush(f);
            done++;
            fails = 0;
        } else {
            fprintf(stderr, "sample failed: %s\n", strerror(errno));
            if (++fails >= 10) { fprintf(stderr, "10 consecutive failures, giving up\n"); fclose(f); return 1; }
        }
        if (count != 0 && done >= count) break;
        next.tv_nsec += (interval_ms % 1000) * 1000000L;
        next.tv_sec += interval_ms / 1000 + next.tv_nsec / 1000000000L;
        next.tv_nsec %= 1000000000L;
        while (!g_stop && clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL) == EINTR) {
        }
    }
    fclose(f);
    close(fd);
    return 0;
}
