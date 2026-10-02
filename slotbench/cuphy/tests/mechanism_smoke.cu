// Synthetic helper test: this program does not execute NVIDIA cuPHY or decode PUSCH.
// From slotbench/ on Linux:
/* nvcc -std=c++17 -O2 -arch=sm_80 -rdc=true -Icommon -Icuphy \
     cuphy/tests/mechanism_smoke.cu cuphy/cuphy_lockstep.cu \
     cuphy/cuphy_lockstep_stamps.cu -lcudadevrt -o /tmp/cuphy-mechanism-smoke */
// SB_CUPHY_LOCKSTEP_MODE=cpu timeout 30 /tmp/cuphy-mechanism-smoke
// SB_CUPHY_LOCKSTEP_MODE=gpu timeout 30 /tmp/cuphy-mechanism-smoke

#include "cuphy_lockstep.h"
#include "cuphy_lockstep_stamps.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
constexpr int width = 1024;
struct Payload {
    int left[width], right[width], sum[width], independent_leaf[width];
    unsigned long long executions[4];
};

__global__ void root_left(Payload* p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) p->left[i] = 3 * i + 7;
    if (i == 0) ++p->executions[0];
}

__global__ void root_right(Payload* p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) p->right[i] = 5 * i + 11;
    if (i == 0) ++p->executions[1];
}

__global__ void leaf_join(Payload* p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) p->sum[i] = p->left[i] + p->right[i];
    if (i == 0) ++p->executions[2];
}

__global__ void leaf_left(Payload* p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) p->independent_leaf[i] = p->left[i] ^ 0x5a5a;
    if (i == 0) ++p->executions[3];
}

void check(cudaError_t e, const char* operation) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(e));
}
#define SMOKE_CUDA(call) check((call), #call)

void wait_stream(cudaStream_t stream) {
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    cudaError_t e;
    while ((e = cudaStreamQuery(stream)) == cudaErrorNotReady) {
        if (std::chrono::steady_clock::now() >= limit)
            throw std::runtime_error("synthetic smoke stream timeout");
    }
    check(e, "synthetic smoke stream completion");
}

void default_env(const char* name, const std::string& value) {
    if (!std::getenv(name) && setenv(name, value.c_str(), 0) != 0)
        throw std::runtime_error(std::string("cannot set ") + name);
}

struct Validation {
    Payload* device;
    cudaStream_t stream;
    SbCuPhyStamps* stamps;
    unsigned calls = 0;
};

int validate(void* opaque) {
    auto& state = *static_cast<Validation*>(opaque);
    try {
        Payload actual{};
        SMOKE_CUDA(cudaMemcpyAsync(&actual, state.device, sizeof(actual), cudaMemcpyDeviceToHost, state.stream));
        wait_stream(state.stream);
        const unsigned long long sequence = state.stamps->sequence;
        if (!sequence || !state.stamps->start || state.stamps->end < state.stamps->start)
            throw std::runtime_error("invalid synthetic DAG completion stamps");
        for (int i = 0; i < width; ++i) {
            if (actual.left[i] != 3 * i + 7 || actual.right[i] != 5 * i + 11 ||
                actual.sum[i] != 8 * i + 18 || actual.independent_leaf[i] != ((3 * i + 7) ^ 0x5a5a))
                throw std::runtime_error("synthetic DAG output mismatch at index " + std::to_string(i));
        }
        for (int kernel = 0; kernel < 4; ++kernel) {
            if (actual.executions[kernel] != sequence)
                throw std::runtime_error("kernel execution count disagrees with completed graph count");
        }
        if (state.calls == 0) {
            if (sequence != 1) throw std::runtime_error("expected exactly one baseline launch");
            // Poison all four arrays after baseline validation. Retain execution counters.
            // The final callback must observe work from replay, not stale baseline output.
            SMOKE_CUDA(cudaMemsetAsync(state.device, 0xa5, 4 * width * sizeof(int), state.stream));
            wait_stream(state.stream);
        } else if (state.calls != 1 || sequence < 2) {
            throw std::runtime_error("expected a final callback after at least one replay");
        }
        ++state.calls;
        std::printf("MECHANISM_SMOKE validation=%u graphs=%llu kernels_each=%llu payload=pass\n",
                    state.calls, sequence, actual.executions[0]);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MECHANISM_SMOKE validation failed: %s\n", error.what());
        return 1;
    }
}

// The production helper emits cuPHY identity fields. Relabel its successful/ordinary
// failure report for this synthetic caller without altering any result or timing field.
void relabel_report(const char* path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open smoke helper report");
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    input.close();
    const auto replace = [&](const std::string& from, const std::string& to) {
        const auto position = text.find(from);
        if (position == std::string::npos) throw std::runtime_error("missing report identity field");
        text.replace(position, from.size(), to);
    };
    replace("\"NVIDIA Aerial cuPHY PUSCH fixed-vector full-slot replay\"",
            "\"Synthetic branched-DAG executive mechanism smoke; no cuPHY\"");
    replace("\"nvidia_cuphy_full_slot\"", "\"synthetic_mechanism_smoke\"");
    std::ofstream output(path, std::ios::trunc);
    output << text;
    output.close();
    if (!output) throw std::runtime_error("cannot write smoke report identity");
}

cudaGraphNode_t add_kernel(cudaGraph_t graph, void* kernel, Payload* device,
                           const cudaGraphNode_t* dependencies = nullptr, size_t count = 0) {
    void* arguments[] = {&device};
    cudaKernelNodeParams parameters{};
    parameters.func = kernel;
    parameters.gridDim = dim3(width / 128);
    parameters.blockDim = dim3(128);
    parameters.kernelParams = arguments;
    cudaGraphNode_t node = nullptr;
    SMOKE_CUDA(cudaGraphAddKernelNode(&node, graph, dependencies, count, &parameters));
    return node;
}
} // namespace

int main() {
    const char* mode = std::getenv("SB_CUPHY_LOCKSTEP_MODE");
    if (!mode || (std::string(mode) != "cpu" && std::string(mode) != "gpu")) {
        std::fprintf(stderr, "Set SB_CUPHY_LOCKSTEP_MODE=cpu or gpu. Synthetic helper smoke only; no cuPHY.\n");
        return 2;
    }
    try {
        default_env("SB_CUPHY_LOCKSTEP_SLOTS", "64");
        default_env("SB_CUPHY_LOCKSTEP_WARMUP", "8");
        default_env("SB_CUPHY_LOCKSTEP_OUT", std::string("mechanism_smoke_") + mode + ".json");
        default_env("SB_CUPHY_LOCKSTEP_RAW", std::string("mechanism_smoke_") + mode + ".bin");
        if (setenv("SB_CUPHY_LOCKSTEP_LABEL", "SYNTHETIC helper mechanism smoke; no NVIDIA cuPHY", 1) != 0)
            throw std::runtime_error("cannot set synthetic report label");
        std::puts("MECHANISM_SMOKE synthetic DAG: two roots, two leaves, four kernels; no NVIDIA cuPHY");
        std::fflush(stdout);

        cudaStream_t stream = nullptr;
        SMOKE_CUDA(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1));
        int priority = 0;
        SMOKE_CUDA(cudaStreamGetPriority(stream, &priority));
        if (priority != -1) throw std::runtime_error("test requires stream priority -1");
        Payload* device = nullptr;
        SMOKE_CUDA(cudaMalloc(&device, sizeof(Payload)));
        SMOKE_CUDA(cudaMemsetAsync(device, 0, sizeof(Payload), stream));
        cudaGraph_t graph = nullptr;
        SMOKE_CUDA(cudaGraphCreate(&graph, 0));
        cudaGraphNode_t roots[] = {
            add_kernel(graph, reinterpret_cast<void*>(root_left), device),
            add_kernel(graph, reinterpret_cast<void*>(root_right), device)
        };
        add_kernel(graph, reinterpret_cast<void*>(leaf_join), device, roots, 2);
        add_kernel(graph, reinterpret_cast<void*>(leaf_left), device, roots, 1);
        SMOKE_CUDA(sb_cuphy_lockstep_attach_stamps(graph, priority));
        cudaGraphExec_t executable = nullptr;
        SMOKE_CUDA(cudaGraphInstantiateWithFlags(&executable, graph,
            cudaGraphInstantiateFlagDeviceLaunch | cudaGraphInstantiateFlagUseNodePriority));
        SMOKE_CUDA(cudaGraphUpload(executable, stream));
        SMOKE_CUDA(cudaGraphLaunch(executable, stream));
        wait_stream(stream);
        SbCuPhyStamps *stamps_host = nullptr, *stamps_device = nullptr;
        SMOKE_CUDA(sb_cuphy_lockstep_get_stamps(&stamps_host, &stamps_device));
        Validation validation{device, stream, stamps_host};
        const int result = sb_cuphy_lockstep_run(executable, stream, validate, &validation);
        relabel_report(std::getenv("SB_CUPHY_LOCKSTEP_OUT"));
        if (result || validation.calls != 2) throw std::runtime_error("helper or callback-count check failed");
        SMOKE_CUDA(cudaGraphExecDestroy(executable));
        SMOKE_CUDA(cudaGraphDestroy(graph));
        SMOKE_CUDA(cudaFree(device));
        SMOKE_CUDA(cudaStreamDestroy(stream));
        std::printf("MECHANISM_SMOKE mode=%s PASS (synthetic DAG, not a cuPHY validation)\n", mode);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MECHANISM_SMOKE failed: %s\n", error.what());
        std::fflush(stderr);
        // Timeout paths can retain GPU work; avoid any CUDA destruction or exit cleanup.
        std::_Exit(1);
    }
}
