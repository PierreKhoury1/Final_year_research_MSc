// One 5G NR uplink slot of GPU signal processing (DESIGN.md section 2) as a reusable object.
// The driver owns streams and timing; the pipeline owns buffers, library handles and the graph.
#pragma once
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

namespace sb {

struct PhyConfig {
    int fft = 4096;          // OFDM FFT size
    int symbols = 14;        // OFDM symbols per slot
    int antennas = 4;        // receive antennas (pipeline supports exactly 4)
    int layers = 4;          // spatial layers (pipeline supports exactly 4)
    int subcarriers = 3276;  // used subcarriers (<= fft)
    int qam = 256;           // 4, 16, 64 or 256
    int ldpc_cb = 0;         // codewords per slot; 0 = auto (fill the slot's coded bits)
    int ldpc_iters = 10;     // fixed decoder iterations
    int ldpc_rows = 8;       // BG1 rows used (4..46): 8 -> code rate 0.79, typical with 256-QAM
    int ldpc_z = 384;        // lifting size
    int ldpc_bg = 1;         // base graph
    float sigma2 = 0.01f;    // noise variance used in the MMSE Gram matrix and LLR scaling
    unsigned seed = 1;       // synthetic input data
    std::string json() const;
};

// Written by the stamp kernels into mapped pinned host memory. The GPU writes t before seq and
// fences, so a host that sees seq == k also sees that launch's t.
struct alignas(64) SlotStamps {
    unsigned long long start_seq;
    unsigned long long start_t;  // %globaltimer ns at S0
    unsigned long long end_seq;
    unsigned long long end_t;    // %globaltimer ns at S12
};

class SlotPipeline {
public:
    // Allocates every buffer, creates cuFFT plan and cuBLAS handle (with preallocated workspace),
    // fills the synthetic received signal. Must be called after cudaSetDevice/cudaSetDeviceFlags.
    // Exits the process with a message on any CUDA/library error.
    explicit SlotPipeline(const PhyConfig &cfg);
    ~SlotPipeline();
    SlotPipeline(const SlotPipeline &) = delete;
    SlotPipeline &operator=(const SlotPipeline &) = delete;

    // Streams mode: enqueue S0..S12 as individual launches on s (binds cuBLAS/cuFFT to s first).
    // Every enqueue (and every graph launch) increments the stamp sequence number by exactly 1.
    void enqueue(cudaStream_t s);

    // Graph mode: capture enqueue(s) into a graph once and instantiate it. Returns the executable
    // graph, owned by the pipeline. Calling again returns the same graph.
    cudaGraphExec_t capture(cudaStream_t s);
    size_t graph_node_count() const { return graph_nodes_; }

    // Host pointer to the stamps (mapped pinned memory).
    const volatile SlotStamps *stamps() const { return stamps_host_; }

    // GPU decoder correctness: encodes random info on the host, BPSK + AWGN at a few Eb/N0 values,
    // decodes on the GPU with the same kernel the slot uses, compares with the host decoder.
    // Also runs one full slot and checks every output is finite. Appends a readable report.
    bool selftest(std::string &report);

    const PhyConfig &config() const { return cfg_; }
    // Sizes, FLOP and byte estimates per slot, dynamic shared memory used by the decoder.
    std::string describe_json() const;

private:
    struct Impl;
    Impl *impl_ = nullptr;
    PhyConfig cfg_;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    size_t graph_nodes_ = 0;
    SlotStamps *stamps_host_ = nullptr;
};

}  // namespace sb
