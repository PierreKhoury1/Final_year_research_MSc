// Fixed-vector replay of the actual configured cuPHY PUSCH graph. No signal-processing substitutes.
#include "cuphy_lockstep.h"
#include "cuphy_lockstep_stamps.h"
#include "clock_fit.h"
#include "host_time.h"
#include "json_writer.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace sb;
using U64 = unsigned long long;

struct Record {
    U64 slot;
    long long t_target;
    U64 g_target, g_launch, g_launch_done, g0, g1, flags;
};
static_assert(sizeof(Record) == 64, "lockstep raw record ABI");

struct ExecutiveState {
    int k = 0, pending = -1, finished = 0, first_error = 0;
    U64 previous_end = 0, deadline = 0, generations = 0, sequence = 0;
};

struct Options {
    std::string mode, out, raw, label;
    int slots = 1000, warmup = 100;
    double period_us = 500, deadline_us = 500;
};

std::string env_string(const char* name, const char* fallback = "") {
    const char* value = std::getenv(name);
    return value ? value : fallback;
}
double env_number(const char* name, double fallback, double lower, double upper, bool integer = false) {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    char* end = nullptr;
    errno = 0;
    double number = std::strtod(value, &end);
    if (end == value || *end || errno == ERANGE || !std::isfinite(number) || number < lower || number > upper
        || (integer && std::floor(number) != number)) throw std::runtime_error(std::string("invalid ") + name);
    return number;
}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
#define SB_CUDA(call) check((call), #call)

cudaError_t wait_stream(cudaStream_t stream, int64_t limit) {
    cudaError_t e;
    while ((e = cudaStreamQuery(stream)) == cudaErrorNotReady) {
        if (now_ns() >= limit) return e;
        cpu_relax();
    }
    return e;
}

bool write_json(const std::string& path, const Json& result) {
    FILE* file = std::fopen(path.c_str(), "w");
    if (!file) return false;
    bool ok = std::fputs((result.str() + "\n").c_str(), file) >= 0;
    return std::fclose(file) == 0 && ok;
}

__device__ __forceinline__ U64 timer_ns() {
    U64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

constexpr unsigned stop_sequence = 0xFFFFFFFEu;
__global__ void pingpong(volatile unsigned* control, U64* times, int count, unsigned base) {
    for (int i = 0; i < count; ++i) {
        const unsigned want = base + i + 1;
        const U64 limit = timer_ns() + 5000000000ull;
        while (control[0] != want) {
            if (control[0] == stop_sequence || timer_ns() > limit) return;
        }
        times[i] = timer_ns();
        __threadfence_system();
        control[1] = want;
        __threadfence_system();
    }
}

__global__ void measure_tick(U64* value) {
    U64 prior = timer_ns(), smallest = ~0ull;
    for (int i = 0; i < 100000; ++i) {
        U64 current = timer_ns();
        if (current != prior) {
            if (current - prior < smallest) smallest = current - prior;
            prior = current;
        }
    }
    *value = smallest;
}

// Each tail generation starts only after the previous full-slot graph and all its children retire.
// Hence exactly one cuPHY executable is reused safely, without cloning stale setup parameters.
__global__ void executive(cudaGraphExec_t slot, const U64* targets, int count, Record* records,
                          ExecutiveState* state, const volatile SbCuPhyStamps* stamps,
                          volatile int* finished_host) {
    ExecutiveState& st = *state;
    ++st.generations;
    if (st.pending >= 0) {
        if (stamps->sequence != st.sequence + 1 || !stamps->start || stamps->end < stamps->start) {
            records[st.pending].flags |= 8ull;
            st.finished = 4;
            __threadfence_system();
            *finished_host = st.finished;
            return;
        }
        st.sequence = stamps->sequence;
        records[st.pending].g0 = stamps->start;
        records[st.pending].g1 = stamps->end;
        st.previous_end = stamps->end;
        st.pending = -1;
    }
    // Both modes skip exactly those supplied boundaries covered by the previous PHY execution.
    while (st.k < count && targets[st.k] <= st.previous_end) records[st.k++].flags = 1ull;
    if (st.k >= count) {
        st.finished = 1;
    } else if (timer_ns() > st.deadline) {
        while (st.k < count) records[st.k++].flags = 8ull;
        st.finished = 3;
    } else {
        while (timer_ns() < targets[st.k]) {}
        Record& r = records[st.k];
        r.flags = 4ull;
        r.g_launch = timer_ns();
        cudaError_t e = cudaGraphLaunch(slot, cudaStreamGraphFireAndForget);
        r.g_launch_done = timer_ns();
        if (e != cudaSuccess) {
            r.flags |= 2ull;
            if (!st.first_error) st.first_error = static_cast<int>(e);
        } else {
            st.pending = st.k;
        }
        ++st.k;
        e = cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
        if (e != cudaSuccess) {
            st.finished = 2;
            if (!st.first_error) st.first_error = static_cast<int>(e);
        }
    }
    if (st.finished) {
        __threadfence_system();
        *finished_host = st.finished;
    }
}

struct Calibration {
    volatile unsigned* host_control = nullptr;
    unsigned* device_control = nullptr;
    U64 *host_times = nullptr, *device_times = nullptr;
    unsigned sequence = 0;

    Calibration() {
        SB_CUDA(cudaHostAlloc((void**)&host_control, 64, cudaHostAllocMapped));
        SB_CUDA(cudaHostGetDevicePointer((void**)&device_control, (void*)host_control, 0));
        SB_CUDA(cudaHostAlloc((void**)&host_times, 400 * sizeof(U64), cudaHostAllocMapped));
        SB_CUDA(cudaHostGetDevicePointer((void**)&device_times, host_times, 0));
    }

    ClockFit fit(cudaStream_t stream, int batches) {
        std::vector<Bracket> brackets;
        for (int batch = 0; batch < batches; ++batch) {
            unsigned base = sequence;
            host_control[0] = base;
            host_control[1] = base;
            __sync_synchronize();
            pingpong<<<1, 1, 0, stream>>>(device_control, device_times, 400, base);
            SB_CUDA(cudaGetLastError());
            std::vector<std::pair<int64_t, int64_t>> host;
            for (int i = 0; i < 400; ++i) {
                unsigned want = base + i + 1;
                int64_t before = now_ns();
                host_control[0] = want;
                __sync_synchronize();
                int64_t limit = before + 2000000000LL;
                while (host_control[1] != want && now_ns() < limit) cpu_relax();
                int64_t after = now_ns();
                if (host_control[1] != want) break;
                host.emplace_back(before, after);
            }
            host_control[0] = stop_sequence;
            __sync_synchronize();
            SB_CUDA(wait_stream(stream, now_ns() + 5000000000LL));
            for (size_t i = 0; i < host.size(); ++i)
                brackets.push_back({host[i].first, host[i].second, host_times[i]});
            sequence = base + 401;
            if (host.size() != 400) throw std::runtime_error("incomplete clock calibration batch");
            if (batch + 1 < batches) sleep_until_raw(now_ns() + 2000000000LL / batches);
        }
        ClockFit result = fit_clock(brackets);
        if (!result.ok || !std::isfinite(result.a) || result.a <= 0)
            throw std::runtime_error("invalid clock calibration");
        return result;
    }
};

double delta_us(U64 a, U64 b) { return static_cast<double>(static_cast<long long>(a - b)) / 1000.; }
double percentile(std::vector<double> data, double p) {
    if (data.empty()) return NAN;
    std::sort(data.begin(), data.end());
    return data[static_cast<size_t>(std::ceil(p * (data.size() - 1) / 100.))];
}
std::string statistics(const std::vector<double>& data) {
    Json j;
    j.add("n", static_cast<long long>(data.size()));
    if (!data.empty()) j.add("min", *std::min_element(data.begin(), data.end()))
        .add("p50", percentile(data, 50)).add("p90", percentile(data, 90)).add("p99", percentile(data, 99))
        .add("p99_9", percentile(data, 99.9)).add("max", *std::max_element(data.begin(), data.end()));
    return j.str();
}
}

int sb_cuphy_lockstep_run(cudaGraphExec_t graph, cudaStream_t stream,
                         int (*validate)(void*), void* validation_context) {
    using namespace sb;
    Options o;
    o.mode = env_string("SB_CUPHY_LOCKSTEP_MODE");
    o.out = env_string("SB_CUPHY_LOCKSTEP_OUT", "cuphy_lockstep.json");
    o.raw = env_string("SB_CUPHY_LOCKSTEP_RAW", "cuphy_lockstep.bin");
    o.label = env_string("SB_CUPHY_LOCKSTEP_LABEL");
    bool before_ok = false, after_ok = false;
    try {
        if (o.mode != "cpu" && o.mode != "gpu") throw std::runtime_error("mode must be cpu or gpu");
        o.slots = static_cast<int>(env_number("SB_CUPHY_LOCKSTEP_SLOTS", 1000, 1, 1000000, true));
        o.warmup = static_cast<int>(env_number("SB_CUPHY_LOCKSTEP_WARMUP", 100, 0, 100000, true));
        o.period_us = env_number("SB_CUPHY_LOCKSTEP_PERIOD_US", 500, 1, 1000000);
        o.deadline_us = env_number("SB_CUPHY_LOCKSTEP_DEADLINE_US", 500, 1, 1000000);
        if (!graph || !validate) throw std::runtime_error("missing graph or validation callback");
        SB_CUDA(wait_stream(stream, now_ns() + 5000000000LL));
        before_ok = validate(validation_context) == 0;
        if (!before_ok) throw std::runtime_error("NVIDIA baseline decoded-data/CRC validation failed");
        SB_CUDA(wait_stream(stream, now_ns() + 5000000000LL));
        SbCuPhyStamps *stamps_h = nullptr, *stamps_d = nullptr;
        SB_CUDA(sb_cuphy_lockstep_get_stamps(&stamps_h, &stamps_d));
        SB_CUDA(cudaGraphUpload(graph, stream));

        int device = 0, priority = 0, driver = 0, runtime = 0;
        cudaDeviceProp properties{};
        SB_CUDA(cudaGetDevice(&device));
        SB_CUDA(cudaGetDeviceProperties(&properties, device));
        SB_CUDA(cudaStreamGetPriority(stream, &priority));
        SB_CUDA(cudaDriverGetVersion(&driver));
        SB_CUDA(cudaRuntimeGetVersion(&runtime));
        U64 *tick_d = nullptr, tick = 0;
        SB_CUDA(cudaMalloc(&tick_d, sizeof(U64)));
        measure_tick<<<1, 1, 0, stream>>>(tick_d);
        SB_CUDA(cudaGetLastError());
        SB_CUDA(wait_stream(stream, now_ns() + 5000000000LL));
        SB_CUDA(cudaMemcpy(&tick, tick_d, sizeof(U64), cudaMemcpyDeviceToHost));

        const int count = o.slots + o.warmup;
        std::vector<Record> records(count);
        std::vector<U64> targets(count);
        U64* targets_d = nullptr;
        Record* records_d = nullptr;
        ExecutiveState* state_d = nullptr;
        volatile int* finished_h = nullptr;
        int* finished_d = nullptr;
        cudaGraph_t executive_graph = nullptr;
        cudaGraphExec_t executive_exec = nullptr;
        if (o.mode == "gpu") {
            SB_CUDA(cudaMalloc(&targets_d, count * sizeof(U64)));
            SB_CUDA(cudaMalloc(&records_d, count * sizeof(Record)));
            SB_CUDA(cudaMalloc(&state_d, sizeof(ExecutiveState)));
            SB_CUDA(cudaHostAlloc((void**)&finished_h, sizeof(int), cudaHostAllocMapped));
            *finished_h = 0;
            SB_CUDA(cudaHostGetDevicePointer((void**)&finished_d, (void*)finished_h, 0));
            SB_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
            executive<<<1, 1, 0, stream>>>(graph, targets_d, count, records_d, state_d, stamps_d, finished_d);
            SB_CUDA(cudaGetLastError());
            SB_CUDA(cudaStreamEndCapture(stream, &executive_graph));
            SB_CUDA(cudaGraphInstantiateWithFlags(&executive_exec, executive_graph,
                cudaGraphInstantiateFlagDeviceLaunch | cudaGraphInstantiateFlagUseNodePriority));
            SB_CUDA(cudaGraphUpload(executive_exec, stream));
        }
        SB_CUDA(wait_stream(stream, now_ns() + 5000000000LL));
        Calibration calibration;
        ClockFit pre = calibration.fit(stream, 50);
        const int64_t first_target = now_ns() + 200000000LL;
        const int64_t period_ns = static_cast<int64_t>(std::llround(o.period_us * 1000));
        for (int k = 0; k < count; ++k) {
            records[k].slot = k;
            records[k].t_target = first_target + k * period_ns;
            records[k].g_target = targets[k] = pre.gpu_of(records[k].t_target);
        }
        ExecutiveState state;
        state.sequence = stamps_h->sequence;
        state.deadline = targets.back() + 5000000000ull;
        const std::string start_utc = iso_utc_now();
        const int64_t start_wall = realtime_ns();
        if (now_ns() >= first_target) throw std::runtime_error("preparation overran first target");
        if (o.mode == "cpu") {
            U64 previous_end = 0;
            for (int k = 0; k < count; ++k) {
                auto& r = records[k];
                if (r.g_target <= previous_end) { r.flags = 1; continue; }
                sleep_until_raw(r.t_target - 200000);
                spin_until(r.t_target);
                r.g_launch = pre.gpu_of(now_ns());
                cudaError_t e = cudaGraphLaunch(graph, stream);
                r.g_launch_done = pre.gpu_of(now_ns());
                if (e != cudaSuccess) {
                    r.flags = 2;
                    if (!state.first_error) state.first_error = static_cast<int>(e);
                    continue;
                }
                SB_CUDA(wait_stream(stream, now_ns() + 5000000000LL));
                const volatile SbCuPhyStamps* visible = stamps_h;
                if (visible->sequence != state.sequence + 1 || !visible->start || visible->end < visible->start)
                    throw std::runtime_error("missing or stale PHY completion stamps");
                state.sequence = visible->sequence;
                r.g0 = visible->start;
                r.g1 = previous_end = visible->end;
            }
            state.finished = 1;
        } else {
            SB_CUDA(cudaMemcpyAsync(targets_d, targets.data(), count * sizeof(U64), cudaMemcpyHostToDevice, stream));
            SB_CUDA(cudaMemcpyAsync(records_d, records.data(), count * sizeof(Record), cudaMemcpyHostToDevice, stream));
            SB_CUDA(cudaMemcpyAsync(state_d, &state, sizeof(state), cudaMemcpyHostToDevice, stream));
            SB_CUDA(wait_stream(stream, now_ns() + 5000000000LL));
            if (now_ns() >= first_target) throw std::runtime_error("input upload overran first target");
            SB_CUDA(cudaGraphLaunch(executive_exec, stream));
            const int64_t limit = first_target + count * period_ns + 10000000000LL;
            SB_CUDA(wait_stream(stream, limit));
            if (*finished_h != 1) throw std::runtime_error("GPU executive did not finish successfully");
            SB_CUDA(cudaMemcpy(&state, state_d, sizeof(state), cudaMemcpyDeviceToHost));
            SB_CUDA(cudaMemcpy(records.data(), records_d, count * sizeof(Record), cudaMemcpyDeviceToHost));
        }
        const std::string end_utc = iso_utc_now();
        const int64_t end_wall = realtime_ns();
        ClockFit post = calibration.fit(stream, 13);
        after_ok = validate(validation_context) == 0;
        SB_CUDA(wait_stream(stream, now_ns() + 5000000000LL));

        const int64_t anchor_a = pre.t_ref + static_cast<int64_t>(std::llround(pre.b));
        const int64_t anchor_b = post.t_ref + static_cast<int64_t>(std::llround(post.b));
        if (anchor_b <= anchor_a || post.g_ref <= pre.g_ref) throw std::runtime_error("invalid post-run clock anchors");
        const double rate = static_cast<double>(post.g_ref - pre.g_ref) / (anchor_b - anchor_a);
        std::vector<double> start_error, precision, prediction, launch_error, launch_call, queue, execution, latency;
        long long recorded = 0, skipped = 0, errors = 0, timeouts = 0, misses = 0;
        for (int k = o.warmup; k < count; ++k) {
            auto& r = records[k];
            if (r.flags & 1) { ++skipped; ++misses; continue; }
            if (r.flags & 2) { ++errors; ++misses; continue; }
            if ((r.flags & 8) || !r.g0 || !r.g1) { r.flags |= 8; ++timeouts; ++misses; continue; }
            ++recorded;
            U64 corrected = pre.g_ref + static_cast<long long>(std::llround((r.t_target - anchor_a) * rate));
            start_error.push_back(delta_us(r.g0, corrected));
            precision.push_back(delta_us(r.g0, r.g_target));
            prediction.push_back(delta_us(r.g_target, corrected));
            launch_error.push_back(delta_us(r.g_launch, r.g_target));
            launch_call.push_back(delta_us(r.g_launch_done, r.g_launch));
            queue.push_back(delta_us(r.g0, r.g_launch));
            execution.push_back(delta_us(r.g1, r.g0));
            latency.push_back(delta_us(r.g1, corrected));
            if (latency.back() > o.deadline_us) ++misses;
        }
        const bool ok = before_ok && after_ok && !errors && !timeouts && !state.first_error && recorded > 0
            && recorded + skipped + errors + timeouts == o.slots && state.finished == 1;
        FILE* raw = std::fopen(o.raw.c_str(), "wb");
        if (!raw) throw std::runtime_error("cannot open raw output");
        bool raw_ok = std::fwrite(records.data() + o.warmup, sizeof(Record), o.slots, raw) == static_cast<size_t>(o.slots);
        if (std::fclose(raw)) raw_ok = false;
        if (!raw_ok) throw std::runtime_error("raw output write failed");
        Json result;
        result.add("ok", ok).add("mode", o.mode).add("label", o.label).add("load", "external")
            .add("error", ok ? "" : "correctness, launch, or record validation failed")
            .add("correctness_before", before_ok).add("correctness_after", after_ok)
            .add("implementation", "NVIDIA Aerial cuPHY PUSCH fixed-vector full-slot replay")
            .add("slot_variant", "nvidia_cuphy_full_slot").add("gpu", properties.name).add("sm", properties.multiProcessorCount)
            .add("driver", driver).add("runtime", runtime).add("stream_priority", priority).add("globaltimer_tick_ns", tick)
            .add("slots", o.slots).add("warmup", o.warmup).add("period_us", o.period_us).add("deadline_us", o.deadline_us)
            .add("recorded", recorded).add("skipped", skipped).add("launch_errors", errors).add("timeouts", timeouts)
            .add("total_slots", recorded + skipped + errors + timeouts).add("misses", misses).add("miss_rate", double(misses) / o.slots)
            .add("run_start_utc", start_utc).add("run_end_utc", end_utc)
            .add("run_wall_start_ns", static_cast<long long>(start_wall)).add("run_wall_end_ns", static_cast<long long>(end_wall))
            .add_raw("start_error_us", statistics(start_error)).add_raw("launch_precision_us", statistics(precision))
            .add_raw("target_pred_error_us", statistics(prediction)).add_raw("launch_error_us", statistics(launch_error))
            .add_raw("launch_call_us", statistics(launch_call)).add_raw("launch_to_start_us", statistics(queue))
            .add_raw("exec_us", statistics(execution)).add_raw("latency_from_target_us", statistics(latency))
            .add_raw("clock_fit_pre", pre.json()).add_raw("clock_fit_post", post.json())
            .add("two_point_ok", true).add("two_point_rate_ppm", (rate - 1.) * 1e6)
            .add("executive_finished", state.finished).add("executive_chunks", state.generations)
            .add("executive_first_error", state.first_error);
        if (!write_json(o.out, result)) throw std::runtime_error("JSON output write failed");
        std::printf("CUPHY_LOCKSTEP mode=%s correctness=%s recorded=%lld skipped=%lld p99_start_us=%.3f misses=%lld/%d\n",
                    o.mode.c_str(), before_ok && after_ok ? "pass" : "FAIL", recorded, skipped, percentile(start_error, 99), misses, o.slots);
        std::fflush(stdout);
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        Json failure;
        failure.add("ok", false).add("mode", o.mode).add("label", o.label).add("error", error.what())
            .add("correctness_before", before_ok).add("correctness_after", after_ok)
            .add("slot_variant", "nvidia_cuphy_full_slot");
        write_json(o.out, failure);
        std::fprintf(stderr, "CUPHY_LOCKSTEP failed: %s\n", error.what());
        std::fflush(stderr);
        // A failed stream guard may leave a resident graph: never enter blocking CUDA destructors.
        _exit(1);
    }
}
