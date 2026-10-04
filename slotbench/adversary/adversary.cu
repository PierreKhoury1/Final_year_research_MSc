// adversary: co-located GPU workload with a duty cycle (DESIGN.md section 5).
// W1 sgemm (long FP32 GEMMs), W2 llm (fp16 GEMV chain = LLM decode step, memory bound),
// W3 vision (YOLOv8n-like conv-as-GEMM pyramid + SiLU, many small kernels), idle (context only).
// Within each period P the loop issues whole units (GEMM / token / frame), synchronising the
// stream after each, while (now - period_start) < D% * P, then sleeps to the period end.
// All buffers are allocated before the timed loop; one warm-up unit (not counted) loads modules.
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "host_time.h"
#include "json_writer.h"
#include "timetable.h"

// ---------------------------------------------------------------- config

struct Config {
    std::string workload = "sgemm";
    double duty = 100;
    double period_ms = 100;
    double seconds = 0;
    std::string prio = "default";
    int gpu = 0;
    std::string out = "adversary.json";
    std::string timeline;
    int size = 4096;
    double llm_params = 1e9;
    int llm_layers = 16;
    double vision_scale = 1.0;
    bool print_config = false;
    // time-aware gating: only issue a unit when it is expected to finish before the next 5G slot
    std::string gate;          // timetable path written by the slot driver (--timetable)
    double gate_guard_us = 30; // safety margin before the next boundary
    std::string sync = "blocking";  // blocking|spin (spin learns completion sooner; uses a CPU core)
    std::string gate_mode = "on";   // on: wait for slot gaps; observe: never wait, only count (ungated control)
};

static void usage(FILE *f) {
    fprintf(f,
            "usage: adversary [--name value]...\n"
            "  --workload sgemm|llm|vision|idle  (sgemm)   W1 / W2 / W3 proxy, idle = context only\n"
            "  --duty D            (100)   percent of each period spent issuing work (0 = idle)\n"
            "  --period-ms F       (100)   duty-cycle period\n"
            "  --seconds F         (0)     run time, 0 = until SIGINT/SIGTERM\n"
            "  --prio high|default|low (default) stream priority\n"
            "  --gpu N             (0)     CUDA device\n"
            "  --out FILE          (adversary.json) summary JSON written at exit\n"
            "  --timeline FILE     (\"\")    per-second CSV t_s,units,units_per_s\n"
            "  --size N            (4096)  sgemm M=N=K\n"
            "  --llm-params F      (1e9)   llm proxy parameter count (fp16 weights = 2 bytes each)\n"
            "  --llm-layers N      (16)    llm proxy layers\n"
            "  --vision-scale F    (1.0)   vision proxy spatial scale (640x640 input at 1.0)\n"
            "  --gate FILE         (\"\")    slot timetable written by the slot driver; issue a unit only when it is\n"
            "                              expected to finish before the next slot boundary\n"
            "  --gate-mode on|observe (on) observe = never wait, only count units/overruns against the timetable\n"
            "  --gate-guard-us F   (30)    margin kept before each boundary\n"
            "  --sync blocking|spin (blocking) how the host waits for a unit to complete\n"
            "  --print-config              print resolved config JSON and exit (no GPU needed)\n");
}

static bool parse_double(const char *s, double &v) {
    char *e = nullptr;
    errno = 0;
    v = strtod(s, &e);
    return errno == 0 && e != s && *e == '\0' && std::isfinite(v);
}
static bool parse_int(const char *s, int &v) {
    char *e = nullptr;
    errno = 0;
    long x = strtol(s, &e, 10);
    if (errno != 0 || e == s || *e != '\0' || x < INT32_MIN || x > INT32_MAX) return false;
    v = (int)x;
    return true;
}

static Config parse_args(int argc, char **argv) {
    Config c;
    auto bad = [](const std::string &msg) {
        fprintf(stderr, "adversary: %s\n", msg.c_str());
        usage(stderr);
        exit(2);
    };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--print-config") { c.print_config = true; continue; }
        if (a == "--help" || a == "-h") { usage(stdout); exit(0); }
        if (i + 1 >= argc) bad("missing value for " + a);
        const char *v = argv[++i];
        bool ok = true;
        if (a == "--workload") c.workload = v;
        else if (a == "--duty") ok = parse_double(v, c.duty);
        else if (a == "--period-ms") ok = parse_double(v, c.period_ms);
        else if (a == "--seconds") ok = parse_double(v, c.seconds);
        else if (a == "--prio") c.prio = v;
        else if (a == "--gpu") ok = parse_int(v, c.gpu);
        else if (a == "--out") c.out = v;
        else if (a == "--timeline") c.timeline = v;
        else if (a == "--size") ok = parse_int(v, c.size);
        else if (a == "--llm-params") ok = parse_double(v, c.llm_params);
        else if (a == "--llm-layers") ok = parse_int(v, c.llm_layers);
        else if (a == "--vision-scale") ok = parse_double(v, c.vision_scale);
        else if (a == "--gate") c.gate = v;
        else if (a == "--gate-guard-us") ok = parse_double(v, c.gate_guard_us);
        else if (a == "--sync") c.sync = v;
        else if (a == "--gate-mode") c.gate_mode = v;
        else bad("unknown flag " + a);
        if (!ok) bad("bad value for " + a + ": " + v);
    }
    if (c.workload != "sgemm" && c.workload != "llm" && c.workload != "vision" && c.workload != "idle")
        bad("--workload must be sgemm|llm|vision|idle");
    if (c.prio != "high" && c.prio != "default" && c.prio != "low") bad("--prio must be high|default|low");
    if (c.duty < 0 || c.duty > 100) bad("--duty must be in [0, 100]");
    if (c.period_ms <= 0) bad("--period-ms must be > 0");
    if (c.seconds < 0) bad("--seconds must be >= 0");
    if (c.size < 1 || c.size > 32768) bad("--size must be in [1, 32768]");
    if (c.llm_params < 1e6) bad("--llm-params must be >= 1e6");
    if (c.llm_layers < 1 || c.llm_layers > 1024) bad("--llm-layers must be in [1, 1024]");
    if (c.vision_scale <= 0 || c.vision_scale > 8) bad("--vision-scale must be in (0, 8]");
    if (c.out.empty()) bad("--out must not be empty");
    if (c.sync != "blocking" && c.sync != "spin") bad("--sync must be blocking|spin");
    if (c.gate_mode != "on" && c.gate_mode != "observe") bad("--gate-mode must be on|observe");
    if (c.gate_guard_us < 0 || c.gate_guard_us > 1e6) bad("--gate-guard-us out of range");
    return c;
}

// ---------------------------------------------------------------- workload shapes (host only)

// LLM proxy: GPT-style layer = qkv (3h x h), o (h x h), up (4h x h), down (h x 4h) = 12 h^2 weights.
// h is rounded to a multiple of 64 so every matrix stays aligned.
struct LlmShape {
    int layers = 0;
    int hidden = 0;
    double params = 0;           // actual = 12 h^2 L
    size_t weight_bytes = 0;     // 2 * params
};
static LlmShape llm_shape(double params, int layers) {
    LlmShape s;
    s.layers = layers;
    double h = std::sqrt(params / (12.0 * layers));
    s.hidden = std::max(64, (int)std::lround(h / 64.0) * 64);
    s.params = 12.0 * s.hidden * (double)s.hidden * layers;
    s.weight_bytes = (size_t)s.params * 2;
    return s;
}

// Vision proxy: conv-as-GEMM out(Cout x HW) = W(Cout x K) * cols(K x HW), K = k*k*Cin.
// YOLOv8n backbone at 640x640 (Conv s2 stages, C2f blocks with their 1x1/3x3 convs, SPPF 1x1s)
// plus one 1x1 detection conv per output scale (80, 40, 20); the neck is omitted to keep about
// 30 GEMMs + 30 SiLU = 60 kernels per frame. --vision-scale scales the spatial side only.
struct ConvLayer { int cin, cout, k, side; };
static std::vector<ConvLayer> vision_layers(double scale) {
    // side = output spatial side at 640 input
    const ConvLayer base[] = {
        {3, 16, 3, 320},                                                     // Conv s2
        {16, 32, 3, 160},                                                    // Conv s2
        {32, 32, 1, 160}, {16, 16, 3, 160}, {16, 16, 3, 160}, {48, 32, 1, 160},   // C2f(32, n=1)
        {32, 64, 3, 80},                                                     // Conv s2
        {64, 64, 1, 80}, {32, 32, 3, 80}, {32, 32, 3, 80}, {32, 32, 3, 80}, {32, 32, 3, 80},
        {128, 64, 1, 80},                                                    // C2f(64, n=2)
        {64, 128, 3, 40},                                                    // Conv s2
        {128, 128, 1, 40}, {64, 64, 3, 40}, {64, 64, 3, 40}, {64, 64, 3, 40}, {64, 64, 3, 40},
        {256, 128, 1, 40},                                                   // C2f(128, n=2)
        {128, 256, 3, 20},                                                   // Conv s2
        {256, 256, 1, 20}, {128, 128, 3, 20}, {128, 128, 3, 20}, {384, 256, 1, 20},  // C2f(256, n=1)
        {256, 128, 1, 20}, {512, 256, 1, 20},                                // SPPF 1x1 convs
        {64, 144, 1, 80}, {128, 144, 1, 40}, {256, 144, 1, 20},              // detect (reg+cls) per scale
    };
    std::vector<ConvLayer> v;
    for (const ConvLayer &l : base) {
        ConvLayer s = l;
        s.side = std::max(1, (int)std::lround(l.side * scale));
        v.push_back(s);
    }
    return v;
}

// ---------------------------------------------------------------- kernels

// Deterministic pseudo-random fill in [-amp, amp] (integer hash, no RNG state).
__device__ __forceinline__ float hash_unit(uint64_t i, uint32_t seed) {
    uint32_t x = (uint32_t)i * 2654435761u ^ (uint32_t)(i >> 32) ^ seed * 0x9e3779b9u;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return (float)(x & 0xffffff) / 8388608.0f - 1.0f;
}
__global__ void fill_f32(float *p, size_t n, float amp, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = amp * hash_unit(i, seed);
}
__global__ void fill_f16(__half *p, size_t n, float amp, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = __float2half(amp * hash_unit(i, seed));
}

// Elementwise kernels are grid-stride (blocks_for caps the grid at 65535).
// SiLU in place, clamped so values cannot drift to inf/NaN across frames.
__global__ void silu_f32(float *p, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = p[i];
        v = v / (1.0f + __expf(-v));
        p[i] = fminf(fmaxf(v, -8.0f), 8.0f);
    }
}
// LLM "attention" proxy: a[i] = q[i] * sigmoid(k[i]) + v[i] from the fused qkv output.
__global__ void attn_mix(const __half *qkv, __half *a, int h) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < h; i += gridDim.x * blockDim.x) {
        float q = __half2float(qkv[i]), k = __half2float(qkv[h + i]), v = __half2float(qkv[2 * h + i]);
        float r = q / (1.0f + __expf(-k)) + v;
        a[i] = __float2half(fminf(fmaxf(r, -4.0f), 4.0f));
    }
}
// Residual: x = 0.5 * (x + y), clamped.
__global__ void residual(__half *x, const __half *y, int h) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < h; i += gridDim.x * blockDim.x) {
        float r = 0.5f * (__half2float(x[i]) + __half2float(y[i]));
        x[i] = __float2half(fminf(fmaxf(r, -4.0f), 4.0f));
    }
}
// MLP activation: SiLU in place on fp16.
__global__ void silu_f16(__half *p, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = __half2float(p[i]);
        v = v / (1.0f + __expf(-v));
        p[i] = __float2half(fminf(fmaxf(v, -4.0f), 4.0f));
    }
}

static inline int blocks_for(size_t n, int t = 256) { return (int)std::min<size_t>((n + t - 1) / t, 65535); }

// ---------------------------------------------------------------- state and fatal handling

static volatile sig_atomic_t g_stop_signal = 0;
static void on_signal(int s) { g_stop_signal = s; }

struct Summary {
    Config cfg;
    std::string gpu_name = "", gpu_uuid = "";
    int prio_least = 0, prio_greatest = 0, prio_value = 0;
    int runtime_version = 0, driver_version = 0;
    std::string start_time, end_time, exit_reason = "not_started";
    bool ok = false;
    double seconds_total = 0, seconds_active = 0;
    unsigned long long units = 0, periods = 0, overrun_periods = 0, warmup_units = 0;
    double unit_ms_min = 0, unit_ms_max = 0, unit_s_sum = 0;
    double flop_per_unit = 0, bytes_per_unit = 0;   // FLOP (sgemm, vision) / weight bytes read (llm)
    int kernels_per_unit = 0;
    LlmShape llm;
    bool llm_scaled_down = false;
    std::vector<ConvLayer> vision;
    size_t device_bytes = 0;
    size_t free_mem_before = 0, total_mem = 0;
};
static Summary g_sum;

// ---------------------------------------------------------------- time-aware gate
// The slot driver publishes a timetable: slot k's GPU work starts near S_k = t0 + k*P and is done by
// S_k + busy. The gate lets a unit start only inside [S_k + busy, S_{k+1} - guard - est], where est is
// 1.1 x the longest of the last 32 measured unit durations. Before the timetable appears and after
// its last slot the tenant runs ungated. In observe mode it never waits but counts the same things,
// which is the ungated control measured over the identical window.
struct Gate {
    bool enabled = false, observe = false, loaded = false;
    std::string path;
    sb::SlotTimetable tt{};
    int64_t start_ns = 0, next_poll = 0, guard_ns = 0;
    int64_t dur[32] = {0};
    int ndur = 0;
    // statistics (units "in window" started in [t0, t0 + n*P))
    unsigned long long units_in_window = 0, waits = 0, overrun_units = 0, busy_starts = 0;
    double wait_s = 0, unit_s_in_window = 0;
    int64_t unit_slot_end = 0;   // boundary the current unit must finish before (0 = none)
    bool unit_in_window = false;

    int64_t est() const {
        int64_t m = 0;
        for (int i = 0; i < std::min(ndur, 32); i++) m = std::max(m, dur[i]);
        return (int64_t)(1.1 * (double)m);
    }
    int64_t window_end() const { return tt.t0 + tt.n_slots * tt.period_ns; }
    void poll_file(int64_t now) {
        if (loaded || now < next_poll) return;
        next_poll = now + 1000000;   // at most once per ms
        sb::SlotTimetable t{};
        if (sb::read_timetable(path, t) && t.written_ns >= start_ns) {
            tt = t;
            loaded = true;
            fprintf(stderr, "adversary: gate timetable loaded t0_in=%.3f ms period=%lld ns busy=%lld ns slots=%lld\n",
                    (t.t0 - now) * 1e-6, (long long)t.period_ns, (long long)t.busy_ns, (long long)t.n_slots);
        }
    }
    // floor((now - t0) / P) for any sign
    int64_t slot_of(int64_t now) const {
        int64_t d = now - tt.t0;
        return d >= 0 ? d / tt.period_ns : -((-d + tt.period_ns - 1) / tt.period_ns);
    }
    template <class Sleep>
    void wait_until(int64_t t, Sleep sleep_to) {
        int64_t now = sb::now_ns();
        if (t <= now) return;
        waits++;
        wait_s += (t - now) * 1e-9;
        // Gaps are sub-millisecond and sleep wake-ups overshoot by 100+ us on VMs, so short waits spin
        // (the gated tenant therefore uses one CPU core; pin it away from the slot launcher).
        if (t - now > 2000000) sleep_to(t - 1000000);
        while (sb::now_ns() < t && !g_stop_signal) sb::cpu_relax();
    }
    // Called before each unit. Returns once the unit may start.
    template <class Sleep>
    void before(Sleep sleep_to) {
        unit_slot_end = 0;
        unit_in_window = false;
        if (!enabled) return;
        for (;;) {
            int64_t now = sb::now_ns();
            poll_file(now);
            if (!loaded || g_stop_signal) return;
            const int64_t P = tt.period_ns, k = slot_of(now);
            if (k >= tt.n_slots) return;                         // after the last slot: ungated
            const int64_t Sk = tt.t0 + k * P, Sn = k < 0 ? tt.t0 : Sk + P;   // before t0 the next slot is slot 0
            const bool next_is_slot = k + 1 < tt.n_slots;
            if (k >= 0 && now < Sk + tt.busy_ns) {               // slot k is (assumed) on the GPU
                if (observe) { busy_starts++; }
                else { wait_until(Sk + tt.busy_ns, sleep_to); continue; }
            } else if (next_is_slot && now + est() + guard_ns > Sn) {   // would not finish before slot k+1
                if (!observe) { wait_until(Sn + tt.busy_ns, sleep_to); continue; }
            }
            unit_in_window = k >= 0;
            unit_slot_end = next_is_slot ? Sn : 0;
            if (k >= 0 && now < Sk + tt.busy_ns) unit_slot_end = Sk;   // observe: already overlapping slot k
            return;
        }
    }
    // Called after each unit with its start (after the gate) and completion times.
    void after(int64_t t_start, int64_t t_done) {
        if (!enabled) return;
        dur[ndur++ % 32] = t_done - t_start;
        if (!loaded) return;
        if (unit_in_window) { units_in_window++; unit_s_in_window += (t_done - t_start) * 1e-9; }
        if (unit_slot_end && (t_done > unit_slot_end || unit_slot_end <= t_start)) overrun_units++;
    }
};
static Gate g_gate;

static std::string summary_json(const Summary &s) {
    const Config &c = s.cfg;
    sb::Json j;
    j.add("ok", s.ok).add("exit_reason", s.exit_reason);
    j.add("workload", c.workload).add("duty", c.duty).add("period_ms", c.period_ms).add("prio", c.prio);
    j.add("prio_value", s.prio_value);
    sb::Json pr;
    pr.add("least", s.prio_least).add("greatest", s.prio_greatest);
    j.add("prio_range", pr);
    j.add("seconds_requested", c.seconds);
    j.add("seconds_total", s.seconds_total).add("seconds_active", s.seconds_active);
    j.add("active_fraction", s.seconds_total > 0 ? s.seconds_active / s.seconds_total : NAN);
    j.add("units", s.units).add("warmup_units", s.warmup_units);
    j.add("periods", s.periods).add("overrun_periods", s.overrun_periods);
    j.add("units_per_s", s.seconds_total > 0 ? s.units / s.seconds_total : NAN);
    j.add("units_per_s_active", s.seconds_active > 0 ? s.units / s.seconds_active : NAN);
    j.add("unit_ms_mean", s.units ? 1e3 * s.unit_s_sum / s.units : NAN);
    j.add("unit_ms_min", s.units ? s.unit_ms_min : NAN).add("unit_ms_max", s.units ? s.unit_ms_max : NAN);
    double act = s.seconds_active;
    bool w1 = c.workload == "sgemm", w2 = c.workload == "llm", w3 = c.workload == "vision";
    j.add("tflops", w1 && act > 0 ? s.flop_per_unit * s.units / act / 1e12 : NAN);
    j.add("gb_per_s", w2 && act > 0 ? s.bytes_per_unit * s.units / act / 1e9 : NAN);
    j.add("fps", w3 && act > 0 ? s.units / act : NAN);
    j.add("flop_per_unit", s.flop_per_unit).add("bytes_per_unit", s.bytes_per_unit);
    j.add("kernels_per_unit", s.kernels_per_unit);
    // requested vs actual sizes
    sb::Json sz;
    sz.add("size_requested", c.size).add("size_actual", w1 ? c.size : 0);
    sz.add("llm_params_requested", c.llm_params).add("llm_params_actual", w2 ? s.llm.params : 0.0);
    sz.add("llm_layers", c.llm_layers).add("llm_hidden", w2 ? s.llm.hidden : 0);
    sz.add("llm_weight_bytes", (unsigned long long)(w2 ? s.llm.weight_bytes : 0));
    sz.add("llm_scaled_down", s.llm_scaled_down);
    sz.add("vision_scale", c.vision_scale).add("vision_convs", w3 ? (int)s.vision.size() : 0);
    sz.add("device_bytes_allocated", (unsigned long long)s.device_bytes);
    j.add("sizes", sz);
    j.add("gpu", c.gpu).add("gpu_name", s.gpu_name).add("gpu_uuid", s.gpu_uuid);
    j.add("free_mem_before", (unsigned long long)s.free_mem_before).add("total_mem", (unsigned long long)s.total_mem);
    j.add("cuda_runtime_version", s.runtime_version).add("cuda_driver_version", s.driver_version);
    j.add("sync", c.sync == "spin" ? "cudaDeviceScheduleSpin + cudaStreamSynchronize after each unit"
                                   : "cudaDeviceScheduleBlockingSync + cudaStreamSynchronize after each unit");
    j.add("sgemm_math", "CUBLAS_DEFAULT_MATH (FP32, no TF32)");
    j.add("host", sb::hostname()).add("pid", (long)getpid());
    const char *mps = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
    if (mps) j.add("mps_active_thread_percentage", mps); else j.add_null("mps_active_thread_percentage");
    const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
    if (cvd) j.add("cuda_visible_devices", cvd); else j.add_null("cuda_visible_devices");
    j.add("start_time", s.start_time).add("end_time", s.end_time);
    j.add("timeline", c.timeline);
    sb::Json gj;
    gj.add("path", c.gate).add("mode", c.gate.empty() ? "off" : c.gate_mode).add("guard_us", c.gate_guard_us);
    gj.add("sync", c.sync).add("timetable_loaded", g_gate.loaded);
    gj.add("t0_ns", (long long)g_gate.tt.t0).add("period_ns", (long long)g_gate.tt.period_ns);
    gj.add("busy_ns", (long long)g_gate.tt.busy_ns).add("n_slots", (long long)g_gate.tt.n_slots);
    double win_s = g_gate.loaded ? g_gate.tt.n_slots * (double)g_gate.tt.period_ns * 1e-9 : NAN;
    gj.add("window_s", win_s).add("units_in_window", g_gate.units_in_window);
    gj.add("units_per_s_in_window", g_gate.loaded ? g_gate.units_in_window / win_s : NAN);
    gj.add("gpu_busy_fraction_in_window", g_gate.loaded ? g_gate.unit_s_in_window / win_s : NAN);
    gj.add("waits", g_gate.waits).add("wait_s", g_gate.wait_s);
    gj.add("overrun_units", g_gate.overrun_units).add("starts_during_slot_busy", g_gate.busy_starts);
    gj.add("est_unit_us_final", g_gate.est() * 1e-3);
    j.add("gate", gj);
    return j.str();
}

static void mkdir_parents(const std::string &file) {
    size_t slash = file.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return;
    std::string dir = file.substr(0, slash);
    for (size_t p = 1; p <= dir.size(); p++) {
        if (p == dir.size() || dir[p] == '/') {
            std::string d = dir.substr(0, p);
            if (mkdir(d.c_str(), 0775) != 0 && errno != EEXIST) return;
        }
    }
}

// Atomic write: temp file in the same directory, fsync, rename.
static bool write_summary(const Summary &s) {
    mkdir_parents(s.cfg.out);
    std::string tmp = s.cfg.out + ".tmp." + std::to_string((long)getpid());
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) { fprintf(stderr, "adversary: cannot write %s: %s\n", tmp.c_str(), strerror(errno)); return false; }
    std::string js = summary_json(s) + "\n";
    bool ok = fwrite(js.data(), 1, js.size(), f) == js.size();
    ok = fflush(f) == 0 && ok;
    ok = fsync(fileno(f)) == 0 && ok;
    ok = fclose(f) == 0 && ok;
    if (ok && rename(tmp.c_str(), s.cfg.out.c_str()) != 0) ok = false;
    if (!ok) { fprintf(stderr, "adversary: failed to write %s: %s\n", s.cfg.out.c_str(), strerror(errno)); unlink(tmp.c_str()); }
    return ok;
}

// Any error after argument parsing still leaves a summary (ok=false) behind, then exits 1.
[[noreturn]] static void fatal(const std::string &msg) {
    fprintf(stderr, "adversary: fatal: %s\n", msg.c_str());
    g_sum.ok = false;
    g_sum.exit_reason = "error: " + msg;
    g_sum.end_time = sb::iso_utc_now();
    write_summary(g_sum);
    exit(1);
}
#define ACK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    fatal(std::string(#x) + ": " + cudaGetErrorString(e_) + " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")"); } while (0)
#define BCK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) \
    fatal(std::string(#x) + ": cuBLAS status " + std::to_string((int)s_)); } while (0)

// ---------------------------------------------------------------- workloads

struct Gpu {
    cudaStream_t s = nullptr;
    cublasHandle_t h = nullptr;
    std::vector<void *> allocs;
    template <class T> T *alloc(size_t n) {
        void *p = nullptr;
        ACK(cudaMalloc(&p, n * sizeof(T)));
        allocs.push_back(p);
        g_sum.device_bytes += n * sizeof(T);
        return (T *)p;
    }
};

static const float kOne = 1.0f, kZero = 0.0f;

struct Sgemm {
    int n = 0;
    float *a = nullptr, *b = nullptr, *c = nullptr;
    void init(Gpu &g, int n_) {
        n = n_;
        size_t e = (size_t)n * n;
        a = g.alloc<float>(e); b = g.alloc<float>(e); c = g.alloc<float>(e);
        float amp = 1.0f / std::sqrt((float)n);
        fill_f32<<<blocks_for(e), 256, 0, g.s>>>(a, e, 1.0f, 1);
        fill_f32<<<blocks_for(e), 256, 0, g.s>>>(b, e, amp, 2);
        fill_f32<<<blocks_for(e), 256, 0, g.s>>>(c, e, 0.0f, 3);
        g_sum.flop_per_unit = 2.0 * n * (double)n * n;
        g_sum.kernels_per_unit = 1;
    }
    void unit(Gpu &g) {
        BCK(cublasSgemm(g.h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &kOne, a, n, b, n, &kZero, c, n));
    }
};

struct Llm {
    LlmShape sh;
    __half *w = nullptr;                 // all weights, contiguous, per layer: qkv | o | up | down
    __half *x = nullptr, *qkv = nullptr, *att = nullptr, *tmp = nullptr, *up = nullptr;
    void init(Gpu &g, const LlmShape &s) {
        sh = s;
        size_t hh = (size_t)sh.hidden * sh.hidden;
        size_t nw = 12 * hh * sh.layers;
        w = g.alloc<__half>(nw);
        x = g.alloc<__half>(sh.hidden);
        qkv = g.alloc<__half>(3 * (size_t)sh.hidden);
        att = g.alloc<__half>(sh.hidden);
        tmp = g.alloc<__half>(sh.hidden);
        up = g.alloc<__half>(4 * (size_t)sh.hidden);
        fill_f16<<<blocks_for(nw), 256, 0, g.s>>>(w, nw, 1.0f / std::sqrt((float)sh.hidden), 11);
        fill_f16<<<blocks_for(sh.hidden), 256, 0, g.s>>>(x, sh.hidden, 1.0f, 12);
        g_sum.bytes_per_unit = (double)nw * 2;
        g_sum.flop_per_unit = 2.0 * (double)nw;
        g_sum.kernels_per_unit = 8 * sh.layers;
    }
    // y(m) = W(m x k) x(k), W column-major with lda = m. fp16 in/out, fp32 compute.
    void gemv(Gpu &g, const __half *W, const __half *in, __half *out, int m, int k) {
        BCK(cublasGemmEx(g.h, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, k, &kOne, W, CUDA_R_16F, m, in, CUDA_R_16F, k,
                         &kZero, out, CUDA_R_16F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    void unit(Gpu &g) {   // one token through all layers
        const int h = sh.hidden;
        size_t hh = (size_t)h * h;
        for (int l = 0; l < sh.layers; l++) {
            const __half *wl = w + 12 * hh * l;
            gemv(g, wl, x, qkv, 3 * h, h);
            attn_mix<<<blocks_for(h), 256, 0, g.s>>>(qkv, att, h);
            gemv(g, wl + 3 * hh, att, tmp, h, h);
            residual<<<blocks_for(h), 256, 0, g.s>>>(x, tmp, h);
            gemv(g, wl + 4 * hh, x, up, 4 * h, h);
            silu_f16<<<blocks_for(4 * (size_t)h), 256, 0, g.s>>>(up, 4 * h);
            gemv(g, wl + 8 * hh, up, tmp, h, 4 * h);
            residual<<<blocks_for(h), 256, 0, g.s>>>(x, tmp, h);
        }
    }
};

struct Vision {
    std::vector<ConvLayer> layers;
    std::vector<size_t> woff;
    float *w = nullptr, *act[2] = {nullptr, nullptr};
    // Shapes only: GEMM i reads K x HW from act[i%2] and writes Cout x HW to act[(i+1)%2]
    // (no im2col kernel; the data flow is a proxy, the kernel shapes are what matter).
    void init(Gpu &g, const std::vector<ConvLayer> &ls) {
        layers = ls;
        size_t nw = 0, na = 0;
        double flop = 0;
        for (const ConvLayer &l : layers) {
            size_t K = (size_t)l.k * l.k * l.cin, hw = (size_t)l.side * l.side;
            woff.push_back(nw);
            nw += (size_t)l.cout * K;
            na = std::max(na, std::max(K * hw, (size_t)l.cout * hw));
            flop += 2.0 * l.cout * (double)K * hw;
        }
        w = g.alloc<float>(nw);
        act[0] = g.alloc<float>(na);
        act[1] = g.alloc<float>(na);
        fill_f32<<<blocks_for(nw), 256, 0, g.s>>>(w, nw, 0.3f, 21);
        fill_f32<<<blocks_for(na), 256, 0, g.s>>>(act[0], na, 1.0f, 22);
        fill_f32<<<blocks_for(na), 256, 0, g.s>>>(act[1], na, 1.0f, 23);
        g_sum.flop_per_unit = flop;
        g_sum.kernels_per_unit = 2 * (int)layers.size();
    }
    void unit(Gpu &g) {   // one frame
        for (size_t i = 0; i < layers.size(); i++) {
            const ConvLayer &l = layers[i];
            int K = l.k * l.k * l.cin, hw = l.side * l.side;
            float *in = act[i % 2], *out = act[(i + 1) % 2];
            // out(Cout x HW) = W(Cout x K) * in(K x HW), column-major
            BCK(cublasSgemm(g.h, CUBLAS_OP_N, CUBLAS_OP_N, l.cout, hw, K, &kOne, w + woff[i], l.cout, in, K, &kZero,
                            out, l.cout));
            silu_f32<<<blocks_for((size_t)l.cout * hw), 256, 0, g.s>>>(out, l.cout * hw);
        }
    }
};

// ---------------------------------------------------------------- main

static std::string uuid_str(const cudaUUID_t &u) {
    const unsigned char *b = (const unsigned char *)u.bytes;
    char s[64];
    snprintf(s, sizeof s, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1], b[2],
             b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return s;
}

static std::string config_json(const Config &c) {
    sb::Json j;
    j.add("workload", c.workload).add("duty", c.duty).add("period_ms", c.period_ms).add("seconds", c.seconds);
    j.add("prio", c.prio).add("gpu", c.gpu).add("out", c.out).add("timeline", c.timeline).add("size", c.size);
    j.add("llm_params", c.llm_params).add("llm_layers", c.llm_layers).add("vision_scale", c.vision_scale);
    LlmShape ls = llm_shape(c.llm_params, c.llm_layers);
    sb::Json lj;
    lj.add("hidden", ls.hidden).add("params_actual", ls.params).add("weight_bytes", (unsigned long long)ls.weight_bytes);
    j.add("llm_derived", lj);
    std::vector<ConvLayer> vl = vision_layers(c.vision_scale);
    double flop = 0;
    for (const ConvLayer &l : vl) flop += 2.0 * l.cout * (double)l.k * l.k * l.cin * l.side * l.side;
    sb::Json vj;
    vj.add("convs", (int)vl.size()).add("kernels_per_frame", 2 * (int)vl.size()).add("gflop_per_frame", flop / 1e9);
    j.add("vision_derived", vj);
    j.add("sgemm_flop_per_unit", 2.0 * c.size * (double)c.size * c.size);
    return j.str();
}

// Per-second timeline rows: t_s (seconds since the measured start), units in the interval, rate.
struct Timeline {
    FILE *f = nullptr;
    int64_t t_begin = 0, t_last = 0, next_tick = 0;
    unsigned long long units_last = 0;
    void open(const std::string &path, int64_t t0) {
        if (path.empty()) return;
        mkdir_parents(path);
        f = fopen(path.c_str(), "w");
        if (!f) fatal("cannot open timeline " + path + ": " + strerror(errno));
        setvbuf(f, nullptr, _IOFBF, 1 << 16);
        fprintf(f, "t_s,units,units_per_s\n");
        fflush(f);
        t_begin = t_last = t0;
        next_tick = t0 + 1000000000LL;
    }
    void row(int64_t now, unsigned long long units) {
        double dt = (now - t_last) * 1e-9;
        unsigned long long du = units - units_last;
        fprintf(f, "%.3f,%llu,%.6g\n", (now - t_begin) * 1e-9, du, dt > 0 ? du / dt : 0.0);
        fflush(f);
        t_last = now;
        units_last = units;
    }
    void poll(int64_t now, unsigned long long units) {
        if (!f || now < next_tick) return;
        row(now, units);
        while (next_tick <= now) next_tick += 1000000000LL;
    }
    void close(int64_t now, unsigned long long units) {
        if (!f) return;
        if (now > t_last) row(now, units);
        fclose(f);
        f = nullptr;
    }
};

// Duty-cycle loop (DESIGN.md section 5). Period p starts at t_begin + p*P (integer ns, no drift).
// While (now - period_start) < D% * P: run one unit (issue + stream sync) and count it once
// completed; then sleep to the period end. D=100 never sleeps; work=false (idle, D=0) only sleeps.
// A unit that runs past later boundaries resumes with the period containing now (overrun_periods).
// Fills g_sum.units/periods/overrun_periods/seconds_active/seconds_total and the unit time stats.
template <class RunUnit>
static void duty_loop(const Config &cfg, bool work, RunUnit run_unit) {
    const int64_t P = std::max<int64_t>(1, (int64_t)std::llround(cfg.period_ms * 1e6));
    const int64_t active_ns = (int64_t)std::llround(P * cfg.duty / 100.0);
    const bool full = work && cfg.duty >= 100;   // D=100 never sleeps; idle / D=0 always sleep
    const int64_t t_begin = sb::now_ns();
    const int64_t t_end = cfg.seconds > 0 ? t_begin + (int64_t)std::llround(cfg.seconds * 1e9) : INT64_MAX;
    const int64_t kMaxSleep = 50000000;   // sleep in <= 50 ms chunks so signals/timeline stay responsive
    Timeline tl;
    tl.open(cfg.timeline, t_begin);
    g_sum.exit_reason = "running";

    auto stop_now = [&](int64_t now) { return g_stop_signal != 0 || now >= t_end; };
    // Sleep until target (host monoraw ns), waking early for stop conditions and timeline ticks.
    auto sleep_to = [&](int64_t target) {
        for (;;) {
            int64_t now = sb::now_ns();
            tl.poll(now, g_sum.units);
            if (now >= target || stop_now(now)) return;
            int64_t t = std::min(std::min(target, t_end), now + kMaxSleep);
            if (tl.f) t = std::min(t, tl.next_tick);
            sb::sleep_until_raw(t);
        }
    };

    int64_t p = 0;   // period index; period p starts at t_begin + p * P (integer ns, no drift)
    while (!stop_now(sb::now_ns())) {
        const int64_t ps = t_begin + p * P;
        g_sum.periods++;
        if (work) {
            for (;;) {
                g_gate.before(sleep_to);   // time-aware gate (no-op unless --gate)
                int64_t now = sb::now_ns();
                if (stop_now(now) || (!full && now - ps >= active_ns) || (full && now - ps >= P)) break;
                run_unit();   // issues one unit and synchronises the stream
                int64_t done = sb::now_ns();
                g_gate.after(now, done);
                double dt_s = (done - now) * 1e-9;
                g_sum.units++;
                g_sum.unit_s_sum += dt_s;
                g_sum.seconds_active += dt_s;
                double ms = dt_s * 1e3;
                if (g_sum.units == 1 || ms < g_sum.unit_ms_min) g_sum.unit_ms_min = ms;
                if (ms > g_sum.unit_ms_max) g_sum.unit_ms_max = ms;
                tl.poll(done, g_sum.units);
            }
        }
        if (!full) sleep_to(ps + P);
        // Next period: normally p+1; if the last unit ran past later boundaries, continue with the
        // period that contains now (those boundaries are counted as overrun periods).
        int64_t now = sb::now_ns();
        int64_t pn = (now - t_begin) / P;
        if (pn > p + 1) g_sum.overrun_periods += (unsigned long long)(pn - p - 1);
        p = std::max(p + 1, pn);
    }
    int64_t t_stop = sb::now_ns();
    tl.close(t_stop, g_sum.units);
    g_sum.seconds_total = (t_stop - t_begin) * 1e-9;
}

#ifndef ADVERSARY_NO_MAIN
int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);
    if (cfg.print_config) { printf("%s\n", config_json(cfg).c_str()); return 0; }
    g_sum.cfg = cfg;
    g_sum.start_time = sb::iso_utc_now();

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;   // no SA_RESTART: sleeps return early
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    fprintf(stderr, "adversary: start %s\n", config_json(cfg).c_str());

    // Blocking sync: the adversary must not spin a CPU core that the slot driver may share.
    ACK(cudaSetDevice(cfg.gpu));
    ACK(cudaSetDeviceFlags(cfg.sync == "spin" ? cudaDeviceScheduleSpin : cudaDeviceScheduleBlockingSync));
    ACK(cudaFree(nullptr));
    cudaDeviceProp prop;
    ACK(cudaGetDeviceProperties(&prop, cfg.gpu));
    g_sum.gpu_name = prop.name;
    g_sum.gpu_uuid = uuid_str(prop.uuid);
    ACK(cudaRuntimeGetVersion(&g_sum.runtime_version));
    ACK(cudaDriverGetVersion(&g_sum.driver_version));
    ACK(cudaMemGetInfo(&g_sum.free_mem_before, &g_sum.total_mem));

    ACK(cudaDeviceGetStreamPriorityRange(&g_sum.prio_least, &g_sum.prio_greatest));
    g_sum.prio_value = cfg.prio == "high" ? g_sum.prio_greatest : cfg.prio == "low" ? g_sum.prio_least : 0;
    Gpu g;
    ACK(cudaStreamCreateWithPriority(&g.s, cudaStreamNonBlocking, g_sum.prio_value));

    Sgemm w1;
    Llm w2;
    Vision w3;
    const bool idle = cfg.workload == "idle";
    if (!idle) {
        BCK(cublasCreate(&g.h));
        BCK(cublasSetStream(g.h, g.s));
        const size_t ws_bytes = 32u << 20;   // preallocated cuBLAS workspace: no allocation in the loop
        void *ws = g.alloc<char>(ws_bytes);
        BCK(cublasSetWorkspace(g.h, ws, ws_bytes));
    }
    if (idle) {
        g.alloc<char>(1 << 20);
    } else if (cfg.workload == "sgemm") {
        w1.init(g, cfg.size);
    } else if (cfg.workload == "llm") {
        LlmShape ls = llm_shape(cfg.llm_params, cfg.llm_layers);
        // Keep weights within 80% of free memory minus 256 MiB headroom for the context and the driver.
        double budget = 0.8 * (double)g_sum.free_mem_before - 256.0 * (1 << 20);
        if ((double)ls.weight_bytes > budget) {
            if (budget < 2.0 * 12 * 64 * 64 * cfg.llm_layers) fatal("not enough free GPU memory for the llm proxy");
            ls = llm_shape(budget / 2.0, cfg.llm_layers);
            while ((double)ls.weight_bytes > budget && ls.hidden > 64) {
                ls.hidden -= 64;
                ls.params = 12.0 * ls.hidden * (double)ls.hidden * ls.layers;
                ls.weight_bytes = (size_t)ls.params * 2;
            }
            g_sum.llm_scaled_down = true;
            fprintf(stderr, "adversary: llm params scaled down to %.4g (free memory %.2f GiB)\n", ls.params,
                    g_sum.free_mem_before / 1073741824.0);
        }
        g_sum.llm = ls;
        w2.init(g, ls);
    } else {
        g_sum.vision = vision_layers(cfg.vision_scale);
        w3.init(g, g_sum.vision);
    }
    ACK(cudaGetLastError());
    ACK(cudaStreamSynchronize(g.s));

    auto issue_unit = [&]() {
        if (cfg.workload == "sgemm") w1.unit(g);
        else if (cfg.workload == "llm") w2.unit(g);
        else w3.unit(g);
    };
    // Warm-up: one unit (lazy module loading, cuBLAS heuristics) before the clock starts.
    if (!idle && cfg.duty > 0) {
        issue_unit();
        ACK(cudaGetLastError());
        ACK(cudaStreamSynchronize(g.s));
        g_sum.warmup_units = 1;
    }
    if (!cfg.gate.empty()) {
        g_gate.enabled = true;
        g_gate.observe = cfg.gate_mode == "observe";
        g_gate.path = cfg.gate;
        g_gate.guard_ns = (int64_t)std::llround(cfg.gate_guard_us * 1e3);
        g_gate.start_ns = sb::now_ns();
        // Seed the duration estimate with 8 measured units (not counted).
        for (int i = 0; i < 8 && !idle && cfg.duty > 0; i++) {
            int64_t a = sb::now_ns();
            issue_unit();
            ACK(cudaStreamSynchronize(g.s));
            g_gate.dur[g_gate.ndur++ % 32] = sb::now_ns() - a;
        }
        fprintf(stderr, "adversary: gate %s path=%s est_unit_us=%.1f guard_us=%.1f\n", cfg.gate_mode.c_str(),
                cfg.gate.c_str(), g_gate.est() * 1e-3, cfg.gate_guard_us);
    }

    duty_loop(cfg, !idle && cfg.duty > 0, [&]() {
        issue_unit();
        cudaError_t e = cudaStreamSynchronize(g.s);
        if (e != cudaSuccess) fatal(std::string("unit sync: ") + cudaGetErrorString(e));
        e = cudaGetLastError();
        if (e != cudaSuccess) fatal(std::string("unit launch: ") + cudaGetErrorString(e));
    });
    g_sum.end_time = sb::iso_utc_now();
    g_sum.exit_reason = g_stop_signal == SIGINT ? "sigint" : g_stop_signal == SIGTERM ? "sigterm" : "seconds";
    g_sum.ok = true;
    bool wrote = write_summary(g_sum);

    fprintf(stderr, "adversary: done workload=%s duty=%g units=%llu total_s=%.3f active_s=%.3f units_per_s=%.4g "
                    "units_per_s_active=%.4g overrun_periods=%llu exit=%s\n",
            cfg.workload.c_str(), cfg.duty, g_sum.units, g_sum.seconds_total, g_sum.seconds_active,
            g_sum.seconds_total > 0 ? g_sum.units / g_sum.seconds_total : 0.0,
            g_sum.seconds_active > 0 ? g_sum.units / g_sum.seconds_active : 0.0, g_sum.overrun_periods,
            g_sum.exit_reason.c_str());

    if (g.h) cublasDestroy(g.h);
    for (void *ptr : g.allocs) cudaFree(ptr);
    cudaStreamDestroy(g.s);
    return wrote ? 0 : 1;
}
#endif  // ADVERSARY_NO_MAIN
