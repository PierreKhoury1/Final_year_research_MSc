// slot_driver: the timed slot loop (DESIGN.md sections 3, 4, 6).
// Every period the driver launches one uplink slot (CUDA graph or individual stream launches), waits
// for completion (event query or mapped end-stamp spin) and records host/GPU timestamps into an SPSC
// ring that a collector thread writes to slots.bin. Pingpong clock calibration runs before and after.
//
// Overrun rule (precise form of "skip to the first boundary after t1"): after slot k completes at t1,
// the next slot is k+1 if t1 <= t_sched(k+1); otherwise it is the smallest k' with t_sched(k') > t1.
// The boundaries in between are counted as skipped and never launched.
//
// Exit codes: 0 completed, 1 selftest/tune failure, 2 usage/setup error, 3 fatal GPU error or wait
// timeout inside the run, 4 stopped by SIGINT/SIGTERM, 5 wall-clock guard hit. Every run that got as
// far as the loop still writes slots.bin and meta.json.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

#include "clock_fit.h"
#include "cuda_check.h"
#include "host_time.h"
#include "json_writer.h"
#include "record.h"
#include "slot_pipeline.h"
#include "spsc_ring.h"

namespace {

// ---------------------------------------------------------------- constants

constexpr size_t kRingCapacity = 65536;
constexpr int64_t kWaitTimeoutNs = 2000000000LL;       // one slot may not take longer than this (fatal)
constexpr int64_t kCalibFirstTimeoutNs = 2000000000LL;  // pingpong kernel start (may queue behind work)
constexpr int64_t kCalibTimeoutNs = 100000000LL;        // later pingpong answers
constexpr int64_t kCalibDrainTimeoutNs = 10000000000LL; // kernel exit after STOP; longer = GPU hung
constexpr int kCalibBatch = 1000;                       // pingpong samples per resident kernel
constexpr unsigned kStop = 0xFFFFFFFFu;
constexpr int kTuneWarm = 20, kTuneRuns = 300;

volatile sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

// ---------------------------------------------------------------- options

struct Opts {
    std::string out;
    long long slots = 1000000, warmup = 2000, calib_samples = 20000;
    double period_us = 500, deadline_us = 500, spin_us = 50;
    std::string prio = "high", mode = "graph", wait = "event", label;
    int core = -1, collector_core = -1, fifo = 0, gpu = 0;
    bool selftest = false, print_config = false, tune = false;
    double tune_us = 0;
    sb::PhyConfig phy;

    int64_t period_ns() const { return (int64_t)std::llround(period_us * 1000.0); }
    int64_t deadline_ns() const { return (int64_t)std::llround(deadline_us * 1000.0); }
    int64_t spin_ns() const { return (int64_t)std::llround(spin_us * 1000.0); }

    std::string json() const {
        sb::Json j;
        j.add("out", out).add("slots", slots).add("warmup", warmup).add("period_us", period_us)
            .add("deadline_us", deadline_us).add("prio", prio).add("mode", mode).add("wait", wait)
            .add("core", core).add("collector_core", collector_core).add("fifo", fifo).add("spin_us", spin_us)
            .add("gpu", gpu).add("calib_samples", calib_samples).add("label", label)
            .add("period_ns", (long long)period_ns()).add("deadline_ns", (long long)deadline_ns())
            .add("spin_ns", (long long)spin_ns()).add_raw("phy", phy.json());
        return j.str();
    }
};

void usage(FILE *f) {
    fprintf(f,
            "usage: slot_driver --out DIR [flags]        (all flags are --name value; see DESIGN.md section 3)\n"
            "  --slots N (1000000)  --warmup N (2000)  --period-us F (500)  --deadline-us F (500)\n"
            "  --prio high|default|low (high)  --mode graph|streams (graph)  --wait event|flag (event)\n"
            "  --core N (-1)  --collector-core N (-1)  --fifo P (0)  --spin-us F (50)  --gpu N (0)\n"
            "  --calib-samples N (20000)  --label S\n"
            "  sizes: --fft 4096 --symbols 14 --antennas 4 --layers 4 --subcarriers 3276 --qam 256\n"
            "         --ldpc-cb 50 --ldpc-iters 20 --ldpc-rows 46 --ldpc-z 384\n"
            "  --selftest            GPU decoder check + one pipeline run, exit 0/1 (no --out needed)\n"
            "  --tune-us F           largest ldpc-cb (then ldpc-iters) with idle median <= F us, print flags\n"
            "  --print-config        print resolved config JSON and exit (no GPU needed)\n"
            "exit codes: 0 ok, 1 selftest/tune failed, 2 usage/setup, 3 fatal in run, 4 signal, 5 wall guard\n");
}

[[noreturn]] void bad_usage(const std::string &msg) {
    fprintf(stderr, "slot_driver: %s\n", msg.c_str());
    usage(stderr);
    exit(2);
}

long long parse_ll(const std::string &flag, const char *s, long long lo, long long hi) {
    errno = 0;
    char *end = nullptr;
    long long v = strtoll(s, &end, 10);
    if (errno || end == s || *end != '\0' || v < lo || v > hi)
        bad_usage(flag + ": expected an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) + "], got '" +
                  s + "'");
    return v;
}

double parse_d(const std::string &flag, const char *s, double lo, double hi) {
    errno = 0;
    char *end = nullptr;
    double v = strtod(s, &end);
    if (errno || end == s || *end != '\0' || !std::isfinite(v) || v < lo || v > hi)
        bad_usage(flag + ": expected a number in [" + std::to_string(lo) + ", " + std::to_string(hi) + "], got '" + s +
                  "'");
    return v;
}

Opts parse_args(int argc, char **argv) {
    Opts o;
    for (int i = 1; i < argc; i++) {
        std::string f = argv[i];
        if (f == "--help" || f == "-h") { usage(stdout); exit(0); }
        if (f == "--selftest") { o.selftest = true; continue; }
        if (f == "--print-config") { o.print_config = true; continue; }
        if (f.rfind("--", 0) != 0) bad_usage("unexpected argument '" + f + "'");
        if (i + 1 >= argc) bad_usage(f + " needs a value");
        const char *v = argv[++i];
        const long long IMAX = INT_MAX;
        if (f == "--out") o.out = v;
        else if (f == "--slots") o.slots = parse_ll(f, v, 1, (long long)1e9);
        else if (f == "--warmup") o.warmup = parse_ll(f, v, 0, (long long)1e12);
        else if (f == "--period-us") o.period_us = parse_d(f, v, 10, 1e7);
        else if (f == "--deadline-us") o.deadline_us = parse_d(f, v, 0.001, 1e9);
        else if (f == "--prio") {
            o.prio = v;
            if (o.prio != "high" && o.prio != "default" && o.prio != "low") bad_usage("--prio: high, default or low");
        } else if (f == "--mode") {
            o.mode = v;
            if (o.mode != "graph" && o.mode != "streams") bad_usage("--mode: graph or streams");
        } else if (f == "--wait") {
            o.wait = v;
            if (o.wait != "event" && o.wait != "flag") bad_usage("--wait: event or flag");
        } else if (f == "--core") o.core = (int)parse_ll(f, v, -1, 4095);
        else if (f == "--collector-core") o.collector_core = (int)parse_ll(f, v, -1, 4095);
        else if (f == "--fifo") o.fifo = (int)parse_ll(f, v, 0, 99);
        else if (f == "--spin-us") o.spin_us = parse_d(f, v, 0, 1e7);
        else if (f == "--gpu") o.gpu = (int)parse_ll(f, v, 0, 1023);
        else if (f == "--calib-samples") o.calib_samples = parse_ll(f, v, 0, 100000000);
        else if (f == "--label") o.label = v;
        else if (f == "--fft") o.phy.fft = (int)parse_ll(f, v, 1, IMAX);
        else if (f == "--symbols") o.phy.symbols = (int)parse_ll(f, v, 1, IMAX);
        else if (f == "--antennas") o.phy.antennas = (int)parse_ll(f, v, 1, IMAX);
        else if (f == "--layers") o.phy.layers = (int)parse_ll(f, v, 1, IMAX);
        else if (f == "--subcarriers") o.phy.subcarriers = (int)parse_ll(f, v, 1, IMAX);
        else if (f == "--qam") o.phy.qam = (int)parse_ll(f, v, 1, IMAX);
        else if (f == "--ldpc-cb") o.phy.ldpc_cb = (int)parse_ll(f, v, 1, IMAX);
        else if (f == "--ldpc-iters") o.phy.ldpc_iters = (int)parse_ll(f, v, 1, IMAX);
        else if (f == "--ldpc-rows") o.phy.ldpc_rows = (int)parse_ll(f, v, 1, IMAX);
        else if (f == "--ldpc-z") o.phy.ldpc_z = (int)parse_ll(f, v, 1, IMAX);
        else if (f == "--tune-us") { o.tune = true; o.tune_us = parse_d(f, v, 0.001, 1e9); }
        else bad_usage("unknown flag " + f);
    }
    int modes = (int)o.selftest + (int)o.tune;
    if (modes > 1) bad_usage("--selftest and --tune-us are exclusive");
    if (!o.print_config && modes == 0 && o.out.empty()) bad_usage("--out is required");
    if (o.spin_ns() >= o.period_ns()) bad_usage("--spin-us must be smaller than --period-us");
    return o;
}

// ---------------------------------------------------------------- small helpers

// mkdir -p; returns an error message or "".
std::string mkdir_p(const std::string &path) {
    if (path.empty()) return "empty path";
    std::string cur;
    size_t pos = 0;
    while (pos <= path.size()) {
        size_t next = path.find('/', pos);
        if (next == std::string::npos) next = path.size();
        cur = path.substr(0, next);
        pos = next + 1;
        if (cur.empty() || cur == "." || cur == "..") continue;
        if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return cur + ": " + strerror(errno);
    }
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return path + ": not a directory";
    return "";
}

// Nearest-rank percentile of an already sorted vector (p in [0, 100]).
int64_t pct_sorted(const std::vector<int64_t> &v, double p) {
    if (v.empty()) return 0;
    size_t r = (size_t)std::ceil(p / 100.0 * (double)v.size());
    if (r < 1) r = 1;
    if (r > v.size()) r = v.size();
    return v[r - 1];
}

std::string uuid_str(const cudaUUID_t &u) {
    const unsigned char *b = (const unsigned char *)u.bytes;
    char s[64];
    snprintf(s, sizeof s, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1], b[2],
             b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return s;
}

// ---------------------------------------------------------------- device

struct Device {
    int id = 0;
    cudaDeviceProp prop{};
    int clock_khz = 0, mem_clock_khz = 0, driver_ver = 0, runtime_ver = 0;
    unsigned flags = 0;
    bool spin_flag_ok = false;
    int prio_least = 0, prio_greatest = 0;
    char pci[32] = {0};

    std::string json() const {
        sb::Json j;
        j.add("id", id).add("name", prop.name).add("uuid", uuid_str(prop.uuid)).add("pci_bus_id", pci)
            .add("sm_count", prop.multiProcessorCount)
            .add("cc", std::to_string(prop.major) + "." + std::to_string(prop.minor))
            .add("clock_khz", clock_khz).add("mem_clock_khz", mem_clock_khz)
            .add("total_mem_bytes", (unsigned long long)prop.totalGlobalMem)
            .add("device_flags", flags).add("schedule_spin", spin_flag_ok);
        return j.str();
    }
};

// cudaSetDevice + cudaSetDeviceFlags first, as DESIGN.md requires; the flags are read back and checked.
void init_device(int gpu, Device &d) {
    d.id = gpu;
    cudaError_t e = cudaSetDevice(gpu);
    if (e != cudaSuccess) {
        fprintf(stderr, "slot_driver: cudaSetDevice(%d): %s\n", gpu, cudaGetErrorString(e));
        exit(2);
    }
    CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost));
    CK(cudaGetDeviceFlags(&d.flags));
    d.spin_flag_ok = (d.flags & cudaDeviceScheduleMask) == cudaDeviceScheduleSpin;
    if (!d.spin_flag_ok) fprintf(stderr, "slot_driver: warning: device flags 0x%x lack ScheduleSpin\n", d.flags);
    CK(cudaGetDeviceProperties(&d.prop, gpu));
    CK(cudaDeviceGetAttribute(&d.clock_khz, cudaDevAttrClockRate, gpu));
    CK(cudaDeviceGetAttribute(&d.mem_clock_khz, cudaDevAttrMemoryClockRate, gpu));
    CK(cudaDriverGetVersion(&d.driver_ver));
    CK(cudaRuntimeGetVersion(&d.runtime_ver));
    CK(cudaDeviceGetPCIBusId(d.pci, sizeof d.pci, gpu));
    CK(cudaDeviceGetStreamPriorityRange(&d.prio_least, &d.prio_greatest));
}

int prio_value(const Device &d, const std::string &p) {
    if (p == "high") return d.prio_greatest;  // numerically lowest
    if (p == "low") return d.prio_least;
    return 0;
}

// ---------------------------------------------------------------- pingpong clock calibration

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Resident 1-thread kernel: for each request base+i+1 written by the host into ctl[0], read the GPU
// clock, publish it and acknowledge in ctl[1]. kStop in ctl[0] ends it early.
__global__ void pingpong_kernel(volatile unsigned *ctl, volatile unsigned long long *gt, int n, unsigned base) {
    for (int i = 0; i < n; i++) {
        unsigned want = base + (unsigned)i + 1u, c;
        while ((c = ctl[0]) != want)
            if (c == kStop) return;
        gt[i] = gtimer();
        __threadfence_system();
        ctl[1] = want;
        __threadfence_system();
    }
}

struct Calibrator {
    enum Status { OK = 0, INCOMPLETE = 1, HUNG = 2 };
    volatile unsigned *ctl_h = nullptr;
    unsigned *ctl_d = nullptr;
    volatile unsigned long long *gt_h = nullptr;
    unsigned long long *gt_d = nullptr;
    unsigned seq = 0;
    cudaStream_t s = nullptr;

    void init(int prio) {
        void *p;
        CK(cudaHostAlloc(&p, 64, cudaHostAllocMapped));
        memset(p, 0, 64);
        ctl_h = (volatile unsigned *)p;
        CK(cudaHostGetDevicePointer((void **)&ctl_d, p, 0));
        CK(cudaHostAlloc(&p, kCalibBatch * sizeof(unsigned long long), cudaHostAllocMapped));
        memset(p, 0, kCalibBatch * sizeof(unsigned long long));
        gt_h = (volatile unsigned long long *)p;
        CK(cudaHostGetDevicePointer((void **)&gt_d, p, 0));
        CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, prio));
    }

    // Wait for the stream with a timeout; false = still busy (GPU hung) or error.
    bool drain(std::string &note) {
        int64_t lim = sb::now_ns() + kCalibDrainTimeoutNs;
        for (;;) {
            cudaError_t e = cudaStreamQuery(s);
            if (e == cudaSuccess) return true;
            if (e != cudaErrorNotReady) { note += std::string("stream error: ") + cudaGetErrorString(e) + "; "; return false; }
            if (sb::now_ns() > lim) { note += "pingpong kernel did not exit after STOP; "; return false; }
            sb::cpu_relax();
        }
    }

    // Collect up to n brackets. Timeouts end a batch early; three failed batches in a row end the run.
    Status run(long long n, std::vector<sb::Bracket> &out, std::string &note) {
        out.clear();
        out.reserve((size_t)n);
        int bad_batches = 0;
        Status st = OK;
        std::vector<int64_t> t0s(kCalibBatch), t1s(kCalibBatch);
        while ((long long)out.size() < n && !g_stop) {
            int m = (int)std::min<long long>(kCalibBatch, n - (long long)out.size());
            unsigned base = seq;
            if (base > 0xF0000000u) base = seq = 0;  // never approach kStop
            ctl_h[0] = base;
            ctl_h[1] = base;
            for (int i = 0; i < m; i++) gt_h[i] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            pingpong_kernel<<<1, 1, 0, s>>>((volatile unsigned *)ctl_d, gt_d, m, base);
            cudaError_t le = cudaGetLastError();
            if (le != cudaSuccess) { note += std::string("launch: ") + cudaGetErrorString(le) + "; "; return HUNG; }
            int got = 0;
            for (int i = 0; i < m; i++) {
                unsigned w = base + (unsigned)i + 1u;
                int64_t lim_ns = i == 0 ? kCalibFirstTimeoutNs : kCalibTimeoutNs;
                int64_t t0 = sb::now_ns();
                std::atomic_signal_fence(std::memory_order_seq_cst);
                ctl_h[0] = w;
                bool ok = true;
                unsigned spins = 0;
                while (ctl_h[1] != w) {
                    if ((++spins & 1023u) == 0 && sb::now_ns() - t0 > lim_ns) { ok = false; break; }
                }
                int64_t t1 = sb::now_ns();
                if (!ok) break;
                t0s[i] = t0;
                t1s[i] = t1;
                got++;
            }
            ctl_h[0] = kStop;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (!drain(note)) return HUNG;
            std::atomic_thread_fence(std::memory_order_acquire);
            for (int i = 0; i < got; i++) out.push_back({t0s[i], t1s[i], (uint64_t)gt_h[i]});
            seq = base + (unsigned)m + 1u;
            if (got < m) {
                st = INCOMPLETE;
                note += "timeout after " + std::to_string(got) + "/" + std::to_string(m) + " samples; ";
                if (++bad_batches >= 3) { note += "giving up; "; break; }
            } else {
                bad_batches = 0;
            }
        }
        return st;
    }
};

bool write_calib_csv(const std::string &path, const std::vector<sb::Bracket> &br) {
    FILE *f = fopen(path.c_str(), "w");
    if (!f) return false;
    fprintf(f, "t0_ns,t1_ns,gpu_ns\n");
    for (const auto &b : br) fprintf(f, "%" PRId64 ",%" PRId64 ",%" PRIu64 "\n", b.t0, b.t1, b.g);
    bool ok = fflush(f) == 0;
    ok = (fsync(fileno(f)) == 0) && ok;
    return (fclose(f) == 0) && ok;
}

// ---------------------------------------------------------------- collector

struct Collector {
    sb::SpscRing<sb::SlotRecord> *ring = nullptr;
    FILE *f = nullptr;
    int core = -1;
    cpu_set_t orig_mask;
    std::atomic<bool> header_ready{false}, stop{false};
    sb::SlotFileHeader header{};
    uint64_t written = 0;
    bool header_written = false, write_ok = true, pin_ok = true;
    std::string err;
    std::thread th;

    void start() { th = std::thread([this] { body(); }); }

    void body() {
        pthread_setname_np(pthread_self(), "sb-collector");
        // Never inherit the driver's core or SCHED_FIFO.
        sched_param sp{};
        pthread_setschedparam(pthread_self(), SCHED_OTHER, &sp);
        if (core >= 0) pin_ok = sb::pin_thread(core);
        else pthread_setaffinity_np(pthread_self(), sizeof orig_mask, &orig_mask);

        while (!header_ready.load(std::memory_order_acquire)) {
            if (stop.load(std::memory_order_acquire)) break;
            usleep(1000);
        }
        if (header_ready.load(std::memory_order_acquire)) header_written = put(&header, sizeof header, 1);
        std::vector<sb::SlotRecord> buf(4096);
        for (;;) {
            bool stopping = stop.load(std::memory_order_acquire);
            size_t n = ring->pop_many(buf.data(), buf.size());
            if (n) {
                if (put(buf.data(), sizeof(sb::SlotRecord), n)) written += n;
                continue;
            }
            if (stopping) break;  // stop was seen before an empty pop: nothing more can arrive
            usleep(1000);
        }
    }

    bool put(const void *p, size_t sz, size_t n) {
        if (!write_ok) return false;
        if (fwrite(p, sz, n, f) != n) { write_ok = false; err = std::string("fwrite: ") + strerror(errno); return false; }
        return true;
    }

    // Join, patch n_records into the header, flush and fsync. Returns false on any I/O error.
    bool finish() {
        stop.store(true, std::memory_order_release);
        if (th.joinable()) th.join();
        header.n_records = written;
        // A run stopped before recording still gets a valid (empty) file with the default header.
        if (write_ok && !header_written && !put(&header, sizeof header, 1)) header_written = false;
        if (write_ok) {
            if (fflush(f) != 0 || fseek(f, 0, SEEK_SET) != 0 || fwrite(&header, sizeof header, 1, f) != 1 ||
                fflush(f) != 0 || fsync(fileno(f)) != 0) {
                write_ok = false;
                err = std::string("finalise: ") + strerror(errno);
            }
        }
        if (fclose(f) != 0 && write_ok) { write_ok = false; err = std::string("fclose: ") + strerror(errno); }
        f = nullptr;
        return write_ok;
    }
};

// ---------------------------------------------------------------- slot loop

struct Run {
    const Opts *o = nullptr;
    sb::SlotPipeline *pipe = nullptr;
    cudaStream_t s = nullptr;
    cudaEvent_t ev = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool graph = true, wait_event = true;
    int64_t P = 0, spin = 0, deadline = 0, guard_end = 0;
    uint64_t seq = 0;  // expected stamp sequence of the last launch
    sb::SpscRing<sb::SlotRecord> *ring = nullptr;
    bool fatal = false;
    std::string exit_reason = "completed";
};

struct LoopStats {
    uint64_t done = 0, skipped = 0, mismatches = 0, misses = 0, max_lat = 0;
    int64_t t_start = 0, next_boundary = 0;
};

// Runs slots on the grid t_start + k*P until n_target have completed (or stop/fatal/guard).
// record: push SlotRecords to the ring; lat[i] receives t1 - t0 of the i-th completed slot.
void run_slots(Run &r, int64_t t_start, uint64_t n_target, bool record, std::vector<int64_t> &lat, LoopStats &st) {
    const volatile sb::SlotStamps *S = r.pipe->stamps();
    st.t_start = t_start;
    uint64_t k = 0;
    while (st.done < n_target) {
        if (g_stop) { r.exit_reason = "signal"; break; }
        const int64_t t_sched = t_start + (int64_t)k * r.P;
        if (sb::now_ns() > r.guard_end) { r.exit_reason = "wall_clock_guard"; break; }
        sb::sleep_until_raw(t_sched - r.spin);
        const int64_t t_wake = sb::now_ns();
        sb::spin_until(t_sched);

        const uint64_t expect = ++r.seq;
        const int64_t t0 = sb::now_ns();
        cudaError_t e = cudaSuccess;
        if (r.graph) e = cudaGraphLaunch(r.exec, r.s);
        else r.pipe->enqueue(r.s);
        if (e == cudaSuccess && r.wait_event) e = cudaEventRecord(r.ev, r.s);
        const int64_t t_launched = sb::now_ns();
        if (e != cudaSuccess) {
            r.fatal = true;
            r.exit_reason = std::string("fatal: launch: ") + cudaGetErrorString(e);
            break;
        }
        if (r.wait_event) {
            for (;;) {
                e = cudaEventQuery(r.ev);
                if (e == cudaSuccess) break;
                if (e != cudaErrorNotReady) { r.exit_reason = std::string("fatal: event query: ") + cudaGetErrorString(e); break; }
                if (sb::now_ns() - t_launched > kWaitTimeoutNs) { r.exit_reason = "fatal: event wait timeout"; e = cudaErrorTimeout; break; }
            }
        } else {
            // end_seq counts up; wait until it reaches this launch (a jump past it is a mismatch, not a hang)
            while ((int64_t)(S->end_seq - expect) < 0) {
                if (sb::now_ns() - t_launched > kWaitTimeoutNs) { r.exit_reason = "fatal: end-stamp wait timeout"; e = cudaErrorTimeout; break; }
            }
        }
        const int64_t t1 = sb::now_ns();
        if (e != cudaSuccess) { r.fatal = true; break; }

        // seq is written after t and fenced on the GPU, so read seq first
        const uint64_t ss = S->start_seq, es = S->end_seq;
        std::atomic_thread_fence(std::memory_order_acquire);
        uint64_t g0 = S->start_t, g1 = S->end_t;
        if (ss != expect || es != expect) { g0 = g1 = 0; st.mismatches++; }

        const int64_t latency = t1 - t0;
        lat[st.done] = latency;
        if (latency > r.deadline) st.misses++;
        if (record) {
            sb::SlotRecord rec{k, t_sched, t_wake, t0, t_launched, t1, g0, g1};
            r.ring->push(rec);  // never blocks; a full ring counts an overflow
        }
        st.done++;

        const int64_t t_next = t_sched + r.P;
        uint64_t k_next = k + 1;
        if (t1 > t_next) k_next = (uint64_t)((t1 - t_start) / r.P) + 1;  // first boundary strictly after t1
        st.skipped += k_next - k - 1;
        k = k_next;
    }
    st.next_boundary = t_start + (int64_t)k * r.P;
}

// ---------------------------------------------------------------- modes without a timed run

int do_selftest(const Opts &o) {
    Device d;
    init_device(o.gpu, d);
    printf("GPU %s (%s), CC %d.%d, %d SMs\n", d.prop.name, uuid_str(d.prop.uuid).c_str(), d.prop.major, d.prop.minor,
           d.prop.multiProcessorCount);
    sb::SlotPipeline pipe(o.phy);
    printf("pipeline: %s\n", pipe.describe_json().c_str());
    std::string rep;
    bool ok = pipe.selftest(rep);
    printf("%s\n", rep.c_str());
    printf("selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Median idle graph time for one config on the high-priority stream (event wait, like the run).
double tune_measure(const sb::PhyConfig &c, cudaStream_t s, cudaEvent_t ev) {
    sb::SlotPipeline *p = new sb::SlotPipeline(c);
    cudaGraphExec_t ex = p->capture(s);
    std::vector<int64_t> t;
    t.reserve(kTuneRuns);
    for (int i = 0; i < kTuneWarm + kTuneRuns; i++) {
        int64_t t0 = sb::now_ns();
        CK(cudaGraphLaunch(ex, s));
        CK(cudaEventRecord(ev, s));
        cudaError_t e;
        while ((e = cudaEventQuery(ev)) == cudaErrorNotReady) {
            if (sb::now_ns() - t0 > kWaitTimeoutNs) { fprintf(stderr, "slot_driver: tune: wait timeout\n"); exit(3); }
        }
        CK(e);
        int64_t t1 = sb::now_ns();
        if (i >= kTuneWarm) t.push_back(t1 - t0);
    }
    CK(cudaStreamSynchronize(s));
    delete p;
    std::sort(t.begin(), t.end());
    return (double)t[t.size() / 2] / 1000.0;
}

int do_tune(const Opts &o) {
    Device d;
    init_device(o.gpu, d);
    cudaStream_t s;
    cudaEvent_t ev;
    CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, d.prio_greatest));
    CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    // candidates: ldpc_cb from the given value down to 1, then ldpc_iters down with ldpc_cb = 1
    std::vector<std::pair<int, int>> cand;
    for (int cb = o.phy.ldpc_cb; cb >= 1; cb--) cand.push_back({cb, o.phy.ldpc_iters});
    for (int it = o.phy.ldpc_iters - 1; it >= 1; it--) cand.push_back({1, it});
    std::string tried = "[";
    int best = -1;
    double best_us = NAN;
    for (size_t i = 0; i < cand.size() && !g_stop; i++) {
        sb::PhyConfig c = o.phy;
        c.ldpc_cb = cand[i].first;
        c.ldpc_iters = cand[i].second;
        double med = tune_measure(c, s, ev);
        fprintf(stderr, "tune: ldpc_cb %d ldpc_iters %d -> median %.1f us\n", c.ldpc_cb, c.ldpc_iters, med);
        sb::Json j;
        j.add("ldpc_cb", c.ldpc_cb).add("ldpc_iters", c.ldpc_iters).add("median_us", med);
        tried += (i ? "," : "") + j.str();
        if (med <= o.tune_us) { best = (int)i; best_us = med; break; }
    }
    tried += "]";
    sb::Json j;
    j.add("tune_us", o.tune_us).add("gpu", d.prop.name).add("uuid", uuid_str(d.prop.uuid)).add("ok", best >= 0);
    if (best >= 0) {
        j.add("ldpc_cb", cand[best].first).add("ldpc_iters", cand[best].second).add("median_us", best_us)
            .add("flags", "--ldpc-cb " + std::to_string(cand[best].first) + " --ldpc-iters " +
                              std::to_string(cand[best].second));
    }
    j.add("runs", kTuneRuns).add_raw("candidates", tried);
    if (best >= 0) printf("--ldpc-cb %d --ldpc-iters %d\n", cand[best].first, cand[best].second);
    else printf("no configuration reaches a median of %.1f us\n", o.tune_us);
    printf("%s\n", j.str().c_str());
    CK(cudaEventDestroy(ev));
    CK(cudaStreamDestroy(s));
    return best >= 0 ? 0 : 1;
}

}  // namespace

// ---------------------------------------------------------------- main

int main(int argc, char **argv) {
    Opts o = parse_args(argc, argv);
    if (o.print_config) {
        printf("%s\n", o.json().c_str());
        return 0;
    }

    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    if (o.selftest) return do_selftest(o);
    if (o.tune) return do_tune(o);

    const std::string wall_start = sb::iso_utc_now();
    const int64_t prog_start = sb::now_ns();
    cpu_set_t orig_mask;
    CPU_ZERO(&orig_mask);
    sched_getaffinity(0, sizeof orig_mask, &orig_mask);

    // Output directory and slots.bin first, so an unwritable --out fails before any GPU work.
    std::string err = mkdir_p(o.out);
    if (!err.empty()) { fprintf(stderr, "slot_driver: cannot create --out: %s\n", err.c_str()); return 2; }
    const std::string slots_path = o.out + "/slots.bin";
    FILE *slots_f = fopen(slots_path.c_str(), "wb");
    if (!slots_f) { fprintf(stderr, "slot_driver: cannot write %s: %s\n", slots_path.c_str(), strerror(errno)); return 2; }
    static char file_buf[8 << 20];
    setvbuf(slots_f, file_buf, _IOFBF, sizeof file_buf);

    Device d;
    init_device(o.gpu, d);
    const int prio = prio_value(d, o.prio);

    Run r;
    r.o = &o;
    r.graph = o.mode == "graph";
    r.wait_event = o.wait == "event";
    r.P = o.period_ns();
    r.spin = o.spin_ns();
    r.deadline = o.deadline_ns();
    CK(cudaStreamCreateWithPriority(&r.s, cudaStreamNonBlocking, prio));
    CK(cudaEventCreateWithFlags(&r.ev, cudaEventDisableTiming));
    int prio_read = 0;
    CK(cudaStreamGetPriority(r.s, &prio_read));

    r.pipe = new sb::SlotPipeline(o.phy);
    r.exec = r.pipe->capture(r.s);  // also in streams mode, for the node count; the graph is not launched there
    const std::string pipe_json = r.pipe->describe_json();

    // Priming launch: establishes the stamp sequence base without assuming the pipeline's initial counter.
    if (r.graph) CK(cudaGraphLaunch(r.exec, r.s));
    else r.pipe->enqueue(r.s);
    CK(cudaStreamSynchronize(r.s));
    const volatile sb::SlotStamps *S = r.pipe->stamps();
    r.seq = S->end_seq;
    const bool prime_ok = S->start_seq == S->end_seq && S->end_seq != 0;
    if (!prime_ok)
        fprintf(stderr, "slot_driver: warning: priming launch stamps start_seq %llu end_seq %llu\n",
                (unsigned long long)S->start_seq, (unsigned long long)S->end_seq);

    Calibrator cal;
    cal.init(d.prio_greatest);

    // All host memory the loop touches is allocated (and faulted in) before mlockall.
    sb::SpscRing<sb::SlotRecord> ring(kRingCapacity);
    r.ring = &ring;
    std::vector<int64_t> lat((size_t)o.slots, 0), wlat((size_t)std::max(o.warmup, 1LL), 0);
    const bool mlock_ok = sb::lock_memory();

    const bool pin_ok = sb::pin_thread(o.core);
    const bool fifo_ok = sb::set_fifo(o.fifo);
    pthread_setname_np(pthread_self(), "sb-driver");

    std::vector<sb::Bracket> br_pre, br_post;
    std::string note_pre, note_post;
    int cal_pre = cal.run(o.calib_samples, br_pre, note_pre);
    sb::ClockFit fit_pre = sb::fit_clock(br_pre), fit_post;
    int cal_post = -1;

    Collector col;
    col.ring = &ring;
    col.f = slots_f;
    col.core = o.collector_core;
    col.orig_mask = orig_mask;
    col.header = sb::make_header((double)r.P, (double)r.deadline, 0);  // t_start filled in before recording
    col.start();

    // Wall-clock guard: 10x the nominal duration plus 10 minutes.
    const double nominal_ns = (double)(o.slots + o.warmup) * (double)r.P;
    r.guard_end = sb::now_ns() + (int64_t)std::min(10.0 * nominal_ns + 600e9, 9e18 / 2);

    LoopStats ws, rs;
    if (cal_pre == Calibrator::HUNG) {
        r.fatal = true;
        r.exit_reason = "fatal: pre-calibration: " + note_pre;
    } else {
        run_slots(r, sb::now_ns() + 1000000, (uint64_t)o.warmup, false, wlat, ws);
        if (!r.fatal && r.exit_reason == "completed") {
            // The recorded grid continues the warm-up grid seamlessly.
            const int64_t t_start = o.warmup > 0 ? ws.next_boundary : sb::now_ns() + 1000000;
            col.header = sb::make_header((double)r.P, (double)r.deadline, t_start);
            col.header_ready.store(true, std::memory_order_release);
            run_slots(r, t_start, (uint64_t)o.slots, true, lat, rs);
        }
    }
    const int64_t loop_end = sb::now_ns();

    // slots.bin is finalised before anything else that could hang.
    const bool file_ok = col.finish();
    if (!file_ok) fprintf(stderr, "slot_driver: slots.bin: %s\n", col.err.c_str());

    if (!r.fatal) {
        cudaError_t e = cudaStreamSynchronize(r.s);
        if (e != cudaSuccess) { r.fatal = true; r.exit_reason = std::string("fatal: sync: ") + cudaGetErrorString(e); }
    }
    if (!r.fatal) {
        g_stop = 0;  // a stop request ends the loop, but the short post-calibration still runs (timeouts bound it)
        cal_post = cal.run(o.calib_samples, br_post, note_post);
        fit_post = sb::fit_clock(br_post);
        if (cal_post == Calibrator::HUNG) { r.fatal = true; r.exit_reason = "fatal: post-calibration: " + note_post; }
    }
    const bool csv_ok = write_calib_csv(o.out + "/calib_pre.csv", br_pre) &&
                        write_calib_csv(o.out + "/calib_post.csv", br_post);

    // ---- summary statistics over the recorded slots
    std::vector<int64_t> sorted(lat.begin(), lat.begin() + (ptrdiff_t)rs.done);
    std::sort(sorted.begin(), sorted.end());
    std::vector<int64_t> wsorted(wlat.begin(), wlat.begin() + (ptrdiff_t)ws.done);
    std::sort(wsorted.begin(), wsorted.end());
    const double warm_median_ns = wsorted.empty() ? NAN : (double)wsorted[wsorted.size() / 2];

    // ---- meta.json
    utsname un{};
    uname(&un);
    std::vector<std::string> args(argv, argv + argc);
    sb::Json counts;
    counts.add("requested", (long long)o.slots).add("recorded", (unsigned long long)rs.done)
        .add("written", (unsigned long long)col.written).add("skipped_boundaries", (unsigned long long)rs.skipped)
        .add("stamp_mismatches", (unsigned long long)rs.mismatches).add("ring_overflows", (unsigned long long)ring.overflows())
        .add("misses_latency", (unsigned long long)rs.misses)
        .add("warmup_requested", (long long)o.warmup).add("warmup_done", (unsigned long long)ws.done)
        .add("warmup_skipped_boundaries", (unsigned long long)ws.skipped)
        .add("warmup_stamp_mismatches", (unsigned long long)ws.mismatches)
        .add("warmup_misses_latency", (unsigned long long)ws.misses)
        .add("launches_total", (unsigned long long)r.seq);
    sb::Json lat_j;
    lat_j.add("p50_ns", (long long)pct_sorted(sorted, 50)).add("p99_ns", (long long)pct_sorted(sorted, 99))
        .add("p999_ns", (long long)pct_sorted(sorted, 99.9)).add("max_ns", (long long)(sorted.empty() ? 0 : sorted.back()));
    sb::Json prio_j;
    prio_j.add("least", d.prio_least).add("greatest", d.prio_greatest).add("requested", o.prio).add("value", prio)
        .add("stream_reported", prio_read).add("calibration_stream", d.prio_greatest);
    sb::Json calib_j;
    calib_j.add("method", "pingpong").add("samples_requested", (long long)o.calib_samples)
        .add("pre_status", cal_pre).add("pre_note", note_pre).add("post_status", cal_post).add("post_note", note_post)
        .add("status_codes", "0 ok, 1 incomplete (timeouts), 2 GPU hung, -1 not run");
    sb::Json m;
    m.add("tool", "slot_driver").add("label", o.label).add_raw("config", o.json())
        .add_raw("gpu", d.json()).add("driver_version", d.driver_ver).add("runtime_version", d.runtime_ver)
        .add("host", sb::hostname()).add("kernel_release", un.release).add("kernel_version", un.version)
        .add("machine", un.machine).add("proc_cmdline", sb::read_text("/proc/cmdline"))
        .add("isolated_cpus", sb::read_text("/sys/devices/system/cpu/isolated")).add("argv", args)
        .add("pid", (int)getpid()).add_raw("stream_priority", prio_j.str())
        .add("graph_node_count", (unsigned long long)r.pipe->graph_node_count())
        .add_raw("pipeline", pipe_json.empty() ? "null" : pipe_json).add_raw("counts", counts.str())
        .add_raw("latency", lat_j.str())
        .add("fifo_ok", fifo_ok).add("mlock_ok", mlock_ok).add("pin_ok", pin_ok)
        .add("collector_pin_ok", col.pin_ok).add("slots_file_ok", file_ok).add("slots_file_error", col.err)
        .add("calib_csv_ok", csv_ok).add("prime_stamps_ok", prime_ok)
        .add_raw("calibration", calib_j.str()).add_raw("clock_fit_pre", fit_pre.json())
        .add_raw("clock_fit_post", fit_post.json()).add("warmup_median_ns", warm_median_ns)
        .add("t_start_ns", (long long)rs.t_start).add("loop_end_ns", (long long)loop_end)
        .add("program_start_ns", (long long)prog_start)
        .add("wait_timeout_ns", (long long)kWaitTimeoutNs).add("wall_guard_ns", (long long)(r.guard_end - prog_start))
        .add("overrun_rule", "next = k+1 if t1 <= t_sched(k+1) else first boundary strictly after t1")
        .add("start_wall", wall_start).add("end_wall", sb::iso_utc_now()).add("exit_reason", r.exit_reason);
    const std::string meta_path = o.out + "/meta.json";
    bool meta_ok = false;
    if (FILE *f = fopen(meta_path.c_str(), "w")) {
        meta_ok = fputs(m.str().c_str(), f) >= 0 && fputc('\n', f) != EOF;
        meta_ok = fflush(f) == 0 && fsync(fileno(f)) == 0 && meta_ok;
        meta_ok = fclose(f) == 0 && meta_ok;
    }
    if (!meta_ok) fprintf(stderr, "slot_driver: failed to write %s\n", meta_path.c_str());

    // ---- human summary
    auto us = [](int64_t ns) { return (double)ns / 1000.0; };
    printf("slot_driver: %s  recorded %llu/%lld  exit %s\n", d.prop.name, (unsigned long long)rs.done, o.slots,
           r.exit_reason.c_str());
    printf("  latency t1-t0 us: p50 %.1f  p99 %.1f  p99.9 %.1f  max %.1f  (warm-up median %.1f)\n",
           us(pct_sorted(sorted, 50)), us(pct_sorted(sorted, 99)), us(pct_sorted(sorted, 99.9)),
           us(sorted.empty() ? 0 : sorted.back()), warm_median_ns / 1000.0);
    printf("  misses (> %.1f us) %llu  skipped boundaries %llu  ring overflows %llu  stamp mismatches %llu\n",
           o.deadline_us, (unsigned long long)rs.misses, (unsigned long long)rs.skipped,
           (unsigned long long)ring.overflows(), (unsigned long long)rs.mismatches);
    printf("  clock fit pre: ok %d rate %.3f ppm eps %.0f ns; post: ok %d rate %.3f ppm eps %.0f ns\n", fit_pre.ok,
           fit_pre.rate_ppm(), fit_pre.eps_ns, fit_post.ok, fit_post.rate_ppm(), fit_post.eps_ns);

    int rc = 0;
    if (r.fatal) {
        fflush(stdout);
        _exit(3);  // the GPU may be wedged: skip teardown, which could block
    }
    if (r.exit_reason == "signal") rc = 4;
    else if (r.exit_reason == "wall_clock_guard") rc = 5;
    if (!file_ok || !meta_ok || !csv_ok) rc = rc ? rc : 2;
    delete r.pipe;
    CK(cudaEventDestroy(r.ev));
    CK(cudaStreamDestroy(r.s));
    return rc;
}
