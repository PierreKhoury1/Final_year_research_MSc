#include "cuphy_lockstep_stamps.h"

#include <algorithm>
#include <vector>

namespace {
thread_local SbCuPhyStamps* stamps_host = nullptr;
thread_local SbCuPhyStamps* stamps_device = nullptr;

__device__ unsigned long long timer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

__global__ void stamp_start(volatile SbCuPhyStamps* s) {
    s->start = timer_ns();
    s->end = 0;
}

__global__ void stamp_end(volatile SbCuPhyStamps* s) {
    s->end = timer_ns();
    const auto next = s->sequence + 1;
    __threadfence_system();
    s->sequence = next;
}

cudaError_t set_priority(cudaGraph_t graph, int priority) {
    size_t count = 0;
    cudaError_t e = cudaGraphGetNodes(graph, nullptr, &count);
    if (e != cudaSuccess) return e;
    std::vector<cudaGraphNode_t> nodes(count);
    e = cudaGraphGetNodes(graph, nodes.data(), &count);
    if (e != cudaSuccess) return e;
    for (auto node : nodes) {
        cudaGraphNodeType type;
        e = cudaGraphNodeGetType(node, &type);
        if (e != cudaSuccess) return e;
        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeAttrValue attribute{};
            attribute.priority = priority;
            e = cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributePriority, &attribute);
        } else if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child = nullptr;
            e = cudaGraphChildGraphNodeGetGraph(node, &child);
            if (e == cudaSuccess) e = set_priority(child, priority);
        }
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}
}

cudaError_t sb_cuphy_lockstep_attach_stamps(cudaGraph_t graph, int priority) {
    if (stamps_host) return cudaErrorInvalidValue;  // multiple pipelines are deliberately unsupported
    size_t count = 0;
    cudaError_t e = cudaGraphGetNodes(graph, nullptr, &count);
    if (e != cudaSuccess || count == 0) return e != cudaSuccess ? e : cudaErrorInvalidValue;
    std::vector<cudaGraphNode_t> nodes(count), roots, leaves;
    e = cudaGraphGetNodes(graph, nodes.data(), &count);
    if (e != cudaSuccess) return e;
    for (auto node : nodes) {
        size_t dependencies = 0, dependents = 0;
        e = cudaGraphNodeGetDependencies(node, nullptr, &dependencies);
        if (e != cudaSuccess) return e;
        e = cudaGraphNodeGetDependentNodes(node, nullptr, &dependents);
        if (e != cudaSuccess) return e;
        if (!dependencies) roots.push_back(node);
        if (!dependents) leaves.push_back(node);
    }
    if (roots.empty() || leaves.empty()) return cudaErrorInvalidValue;
    e = cudaHostAlloc(reinterpret_cast<void**>(&stamps_host), sizeof(SbCuPhyStamps), cudaHostAllocMapped);
    if (e != cudaSuccess) return e;
    *stamps_host = {};
    e = cudaHostGetDevicePointer(reinterpret_cast<void**>(&stamps_device), stamps_host, 0);
    if (e != cudaSuccess) return e;
    void* args[] = {&stamps_device};
    cudaKernelNodeParams params{};
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = args;
    params.func = reinterpret_cast<void*>(stamp_start);
    cudaGraphNode_t first = nullptr, last = nullptr;
    e = cudaGraphAddKernelNode(&first, graph, nullptr, 0, &params);
    if (e != cudaSuccess) return e;
    std::vector<cudaGraphNode_t> starts(roots.size(), first);
    e = cudaGraphAddDependencies(graph, starts.data(), roots.data(), roots.size());
    if (e != cudaSuccess) return e;
    params.func = reinterpret_cast<void*>(stamp_end);
    e = cudaGraphAddKernelNode(&last, graph, leaves.data(), leaves.size(), &params);
    if (e != cudaSuccess) return e;
    return set_priority(graph, priority);
}

cudaError_t sb_cuphy_lockstep_get_stamps(SbCuPhyStamps** host, SbCuPhyStamps** device) {
    if (!host || !device || !stamps_host || !stamps_device) return cudaErrorInvalidValue;
    *host = stamps_host;
    *device = stamps_device;
    return cudaSuccess;
}
