// SlotPipeline: buffers, cuFFT plan, cuBLAS handle, the S0..S12 enqueue sequence, graph capture,
// a synthetic but physically consistent received signal, and the GPU correctness selftest.
#include "slot_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#include <cublas_v2.h>
#include <cufft.h>

#include "cuda_check.h"
#include "json_writer.h"
#include "ldpc_code.h"
#include "phy_kernels.cuh"
#include "phy_ops.h"

namespace sb {

// Defined in ldpc_code.cpp (not in the frozen ldpc_code.h): host decoder with fp16 a posteriori
// storage, to mirror the GPU decoder when its fp32 array does not fit in shared memory.
void ldpc_decode_host_ex(const LdpcCode &code, const float *llr_in, int iters, float alpha, uint8_t *bits_out,
                         bool app_fp16);

namespace {

constexpr int kAnt = 4;
constexpr float kAlpha = 0.75f;                   // normalized min-sum scale (DESIGN.md S10)
constexpr size_t kBlasWorkspace = 32u << 20;      // cuBLAS workspace, preallocated

[[noreturn]] void config_error(const std::string &msg) {
    fprintf(stderr, "slot_pipeline: invalid config: %s\n", msg.c_str());
    exit(2);
}

template <typename T> T *dev_alloc(size_t n) {
    T *p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
    return p;
}

template <typename T> std::vector<T> to_host(const T *d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

Cf cnormal(std::mt19937_64 &rng) {  // CN(0, 1)
    std::normal_distribution<float> nd(0.0f, 0.70710678f);
    float a = nd(rng);
    return Cf{a, nd(rng)};
}

}  // namespace

std::string PhyConfig::json() const {
    Json j;
    j.add("fft", fft).add("symbols", symbols).add("antennas", antennas).add("layers", layers);
    j.add("subcarriers", subcarriers).add("qam", qam).add("ldpc_cb", ldpc_cb).add("ldpc_iters", ldpc_iters);
    char s2[32];
    snprintf(s2, sizeof s2, "%.9g", (double)sigma2);  // float precision, so 0.01 prints as 0.01
    j.add("ldpc_rows", ldpc_rows).add("ldpc_z", ldpc_z).add("ldpc_bg", ldpc_bg).add_raw("sigma2", s2);
    j.add("seed", seed);
    return j.str();
}

struct SlotPipeline::Impl {
    int device = 0;
    int n_data = 0, qm = 0;
    long long n_llr = 0;
    LdpcCode code;
    LdpcDevCode dcode;
    int *d_row_start = nullptr, *d_edge_col = nullptr, *d_edge_shift = nullptr;
    bool app_fp16 = false;
    size_t dec_smem = 0;
    cudaEvent_t prof_ev[16] = {};  // selftest stage profile: events recorded between stages when prof_on
    bool prof_on = false;
    int prof_n = 0;
    int smem_optin = 0;

    cufftHandle plan = 0;
    cublasHandle_t blas = nullptr;
    void *d_ws = nullptr;

    Cf *d_time = nullptr, *d_freq = nullptr, *d_Yp = nullptr, *d_Yd = nullptr, *d_Xp = nullptr;
    Cf *d_H = nullptr, *d_G = nullptr, *d_R = nullptr;
    cuComplex **d_Gptr = nullptr, **d_Rptr = nullptr;
    int *d_piv = nullptr, *d_info = nullptr;
    int getrs_info = 0;  // cublasCgetrsBatched reports argument errors through a host int
    float *d_llr = nullptr, *d_cw = nullptr, *d_msg = nullptr, *d_app = nullptr;
    uint32_t *d_bits = nullptr;
    int words_per_cw = 0;

    SlotStamps *d_stamps = nullptr;
    unsigned long long *d_counter = nullptr;

    // cuBLAS scalars (CUBLAS_POINTER_MODE_HOST), alive as long as the pipeline
    cuComplex c_quarter{0.25f, 0.0f}, c_one{1.0f, 0.0f}, c_zero{0.0f, 0.0f};

    cudaStream_t bound = nullptr;
    bool bound_valid = false;

    std::vector<Cf> tx_sym;  // transmitted data symbols, index (k * n_data + t) * 4 + l

    void bind(cudaStream_t s) {
        if (bound_valid && bound == s) return;
        CUBLAS_CK(cublasSetStream(blas, s));
        // cublasSetStream resets the workspace to the default pool: set ours again.
        CUBLAS_CK(cublasSetWorkspace(blas, d_ws, kBlasWorkspace));
        CUFFT_CK(cufftSetStream(plan, s));
        bound = s;
        bound_valid = true;
    }
};

SlotPipeline::SlotPipeline(const PhyConfig &cfg_in) : cfg_(cfg_in) {
    PhyConfig &cfg = cfg_;  // ldpc_cb 0 is resolved below, so config() reports what actually runs
    // ---- validation ----
    if (cfg.antennas != kAnt || cfg.layers != kAnt) config_error("antennas and layers must both be 4");
    if (cfg.fft < 16 || cfg.fft > (1 << 16)) config_error("fft must be in [16, 65536]");
    if (cfg.subcarriers < 2 || cfg.subcarriers > cfg.fft || cfg.subcarriers % 2)
        config_error("subcarriers must be even and in [2, fft]");
    if (cfg.symbols < 12 || cfg.symbols > 28) config_error("symbols must be in [12, 28] (DMRS on symbols 2 and 11)");
    if (qam_bits(cfg.qam) == 0) config_error("qam must be 4, 16, 64 or 256");
    if (cfg.ldpc_cb < 0 || cfg.ldpc_cb > 65535) config_error("ldpc-cb must be in [0, 65535] (0 = auto)");
    if (cfg.ldpc_iters < 0 || cfg.ldpc_iters > 1000) config_error("ldpc-iters must be in [0, 1000]");
    if (!(cfg.sigma2 > 0.0f) || !std::isfinite(cfg.sigma2)) config_error("sigma2 must be > 0");
    if (ldpc_set_index(cfg.ldpc_z) < 0) config_error("ldpc-z " + std::to_string(cfg.ldpc_z) + " is not a 38.212 lifting size");
    impl_ = new Impl;
    Impl &m = *impl_;
    try {
        m.code = make_ldpc_code(cfg.ldpc_bg, cfg.ldpc_z, cfg.ldpc_rows);
    } catch (const std::exception &e) {
        config_error(e.what());
    }
    m.n_data = cfg.symbols - 2;
    m.qm = qam_bits(cfg.qam);
    m.n_llr = (long long)cfg.subcarriers * m.n_data * kAnt * m.qm;
    // auto: as many codewords as the slot's coded bits fill (38.212 segmentation, no filler bits)
    if (cfg.ldpc_cb == 0) cfg.ldpc_cb = (int)((m.n_llr + m.code.n_tx_bits() - 1) / m.code.n_tx_bits());
    if (cfg.ldpc_cb > 65535) config_error("auto ldpc-cb exceeds 65535");
    const int S = cfg.subcarriers, nd = m.n_data;
    const LdpcCode &code = m.code;

    CK(cudaGetDevice(&m.device));
    if (!ldpc_decoder_setup(m.device, code.cols, code.Z, &m.app_fp16, &m.dec_smem, &m.smem_optin))
        config_error("LDPC decoder needs " + std::to_string(2 * code.cols * code.Z) +
                     " bytes of shared memory even in fp16, device allows " + std::to_string(m.smem_optin));

    // ---- buffers ----
    const size_t n_time = (size_t)cfg.symbols * kAnt * cfg.fft;
    m.d_time = dev_alloc<Cf>(n_time);
    m.d_freq = dev_alloc<Cf>(n_time);
    m.d_Yp = dev_alloc<Cf>((size_t)S * 16);
    m.d_Yd = dev_alloc<Cf>((size_t)S * kAnt * nd);
    m.d_Xp = dev_alloc<Cf>(16);
    m.d_H = dev_alloc<Cf>((size_t)S * 16);
    m.d_G = dev_alloc<Cf>((size_t)S * 16);
    m.d_R = dev_alloc<Cf>((size_t)S * kAnt * nd);
    m.d_Gptr = dev_alloc<cuComplex *>(S);
    m.d_Rptr = dev_alloc<cuComplex *>(S);
    m.d_piv = dev_alloc<int>((size_t)S * kAnt);
    m.d_info = dev_alloc<int>(S);
    m.d_llr = dev_alloc<float>((size_t)m.n_llr);
    m.d_cw = dev_alloc<float>((size_t)cfg.ldpc_cb * code.n_bits());
    m.d_msg = dev_alloc<float>((size_t)cfg.ldpc_cb * code.n_edges() * code.Z);
    m.d_app = dev_alloc<float>((size_t)cfg.ldpc_cb * code.n_bits());
    m.words_per_cw = (code.k_bits() + 31) / 32;
    m.d_bits = dev_alloc<uint32_t>((size_t)cfg.ldpc_cb * m.words_per_cw);
    m.d_counter = dev_alloc<unsigned long long>(1);
    CK(cudaMalloc(&m.d_ws, kBlasWorkspace));

    CK(cudaHostAlloc((void **)&stamps_host_, sizeof(SlotStamps), cudaHostAllocMapped));
    *stamps_host_ = SlotStamps{};
    CK(cudaHostGetDevicePointer((void **)&m.d_stamps, stamps_host_, 0));

    // edge lists
    m.d_row_start = dev_alloc<int>(code.row_start.size());
    m.d_edge_col = dev_alloc<int>(code.edge_col.size());
    m.d_edge_shift = dev_alloc<int>(code.edge_shift.size());
    CK(cudaMemcpy(m.d_row_start, code.row_start.data(), code.row_start.size() * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(m.d_edge_col, code.edge_col.data(), code.edge_col.size() * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(m.d_edge_shift, code.edge_shift.data(), code.edge_shift.size() * sizeof(int), cudaMemcpyHostToDevice));
    m.dcode.row_start = m.d_row_start;
    m.dcode.edge_col = m.d_edge_col;
    m.dcode.edge_shift = m.d_edge_shift;
    m.dcode.rows = code.rows;
    m.dcode.cols = code.cols;
    m.dcode.Z = code.Z;
    m.dcode.n_edges = code.n_edges();

    // pilot matrix (layers x pilot columns, column-major) and getrf/getrs pointer arrays
    Cf xp[16];
    for (int j = 0; j < 4; j++)
        for (int l = 0; l < 4; l++) xp[l + 4 * j] = Cf{pilot_value(l, j), 0.0f};
    CK(cudaMemcpy(m.d_Xp, xp, sizeof xp, cudaMemcpyHostToDevice));
    std::vector<cuComplex *> gp(S), rp(S);
    for (int k = 0; k < S; k++) {
        gp[k] = reinterpret_cast<cuComplex *>(m.d_G + (size_t)k * 16);
        rp[k] = reinterpret_cast<cuComplex *>(m.d_R + (size_t)k * kAnt * nd);
    }
    CK(cudaMemcpy(m.d_Gptr, gp.data(), S * sizeof(cuComplex *), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(m.d_Rptr, rp.data(), S * sizeof(cuComplex *), cudaMemcpyHostToDevice));

    // ---- libraries ----
    int n = cfg.fft;
    CUFFT_CK(cufftPlanMany(&m.plan, 1, &n, nullptr, 1, cfg.fft, nullptr, 1, cfg.fft, CUFFT_C2C, cfg.symbols * kAnt));
    CUBLAS_CK(cublasCreate(&m.blas));
    CUBLAS_CK(cublasSetPointerMode(m.blas, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CK(cublasSetWorkspace(m.blas, m.d_ws, kBlasWorkspace));

    // ---- synthetic received signal ----
    // Per subcarrier pair: H = I + 0.25 CN(0,1) (well conditioned, constant over the pair so the
    // pair's shared pilot block estimates it); pilots from Xp, data random Gray QAM; AWGN of
    // variance sigma2. Built in frequency, then inverse FFT once so S1 recovers it.
    std::mt19937_64 rng(cfg.seed);
    std::vector<Cf> freq(n_time, Cf{0.0f, 0.0f});
    m.tx_sym.assign((size_t)S * nd * kAnt, Cf{0.0f, 0.0f});
    const float nstd = std::sqrt(cfg.sigma2), inv_fft = 1.0f / (float)cfg.fft;
    Cf H[16];
    for (int k = 0; k < S; k++) {
        if ((k & 1) == 0)
            for (int a = 0; a < 4; a++)
                for (int l = 0; l < 4; l++) H[a + 4 * l] = cadd(Cf{a == l ? 1.0f : 0.0f, 0.0f}, cscale(cnormal(rng), 0.25f));
        const int bin = subcarrier_bin(k, S, cfg.fft);
        auto rx = [&](int sym, const Cf *x) {
            for (int a = 0; a < 4; a++) {
                Cf y = cscale(cnormal(rng), nstd);
                for (int l = 0; l < 4; l++) y = cadd(y, cmul(H[a + 4 * l], x[l]));
                freq[((size_t)sym * kAnt + a) * cfg.fft + bin] = cscale(y, inv_fft);
            }
        };
        for (int p = 0; p < 2; p++) {
            int j = 2 * (k & 1) + p;  // pilot column of (subcarrier k, DMRS symbol p)
            Cf x[4];
            for (int l = 0; l < 4; l++) x[l] = Cf{pilot_value(l, j), 0.0f};
            rx(pilot_symbol(j), x);
        }
        for (int t = 0; t < nd; t++) {
            Cf x[4];
            for (int l = 0; l < 4; l++) {
                uint8_t bits[8];
                for (int b = 0; b < m.qm; b++) bits[b] = (uint8_t)(rng() & 1u);
                x[l] = qam_map(bits, cfg.qam);
                m.tx_sym[((size_t)k * nd + t) * kAnt + l] = x[l];
            }
            rx(data_symbol(t), x);
        }
    }
    CK(cudaMemcpy(m.d_freq, freq.data(), n_time * sizeof(Cf), cudaMemcpyHostToDevice));
    CUFFT_CK(cufftSetStream(m.plan, 0));
    CUFFT_CK(cufftExecC2C(m.plan, reinterpret_cast<cufftComplex *>(m.d_freq), reinterpret_cast<cufftComplex *>(m.d_time),
                          CUFFT_INVERSE));
    CK(cudaMemset(m.d_freq, 0, n_time * sizeof(Cf)));
    // Init work ran on the legacy default stream, which does not order with the non-blocking slot
    // streams: finish it here so no slot can overlap a late memset or copy.
    CK(cudaDeviceSynchronize());
}

SlotPipeline::~SlotPipeline() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
    if (impl_) {
        Impl &m = *impl_;
        if (m.plan) cufftDestroy(m.plan);
        if (m.blas) cublasDestroy(m.blas);
        void *bufs[] = {m.d_time, m.d_freq, m.d_Yp, m.d_Yd, m.d_Xp, m.d_H, m.d_G, m.d_R, m.d_Gptr, m.d_Rptr,
                        m.d_piv, m.d_info, m.d_llr, m.d_cw, m.d_msg, m.d_app, m.d_bits, m.d_counter, m.d_ws,
                        m.d_row_start, m.d_edge_col, m.d_edge_shift};
        for (void *p : bufs)
            if (p) cudaFree(p);
        delete impl_;
    }
    if (stamps_host_) cudaFreeHost(stamps_host_);
}

void SlotPipeline::enqueue(cudaStream_t s) {
    Impl &m = *impl_;
    const int S = cfg_.subcarriers, nd = m.n_data;
    const LdpcCode &code = m.code;
    m.bind(s);
    auto cx = [](Cf *p) { return reinterpret_cast<cuComplex *>(p); };

    m.prof_n = 0;
    auto mark = [&]() {
        if (m.prof_on) CK(cudaEventRecord(m.prof_ev[m.prof_n++], s));
    };
    // S0
    launch_stamp_start(m.d_stamps, m.d_counter, s);
    mark();
    // S1: out of place, so the time-domain input is identical every slot
    CUFFT_CK(cufftExecC2C(m.plan, reinterpret_cast<cufftComplex *>(m.d_time), reinterpret_cast<cufftComplex *>(m.d_freq),
                          CUFFT_FORWARD));
    mark();
    // S2
    launch_pilot_gather(m.d_freq, m.d_Yp, m.d_Yd, cfg_.fft, S, nd, s);
    mark();
    // S3: H = Yp Xp^H / 4   (antennas x layers)
    CUBLAS_CK(cublasCgemmStridedBatched(m.blas, CUBLAS_OP_N, CUBLAS_OP_C, 4, 4, 4, &m.c_quarter, cx(m.d_Yp), 4, 16,
                                        cx(m.d_Xp), 4, 0, &m.c_zero, cx(m.d_H), 4, 16, S));
    mark();
    // S4: G = H^H H + sigma2 I
    CUBLAS_CK(cublasCgemmStridedBatched(m.blas, CUBLAS_OP_C, CUBLAS_OP_N, 4, 4, 4, &m.c_one, cx(m.d_H), 4, 16,
                                        cx(m.d_H), 4, 16, &m.c_zero, cx(m.d_G), 4, 16, S));
    launch_add_sigma2(m.d_G, S, cfg_.sigma2, s);
    mark();
    // S5: R = H^H Yd   (layers x data symbols)
    CUBLAS_CK(cublasCgemmStridedBatched(m.blas, CUBLAS_OP_C, CUBLAS_OP_N, 4, nd, 4, &m.c_one, cx(m.d_H), 4, 16,
                                        cx(m.d_Yd), 4, (long long)4 * nd, &m.c_zero, cx(m.d_R), 4, (long long)4 * nd, S));
    mark();
    // S6, S7: G X = R, X overwrites R
    CUBLAS_CK(cublasCgetrfBatched(m.blas, 4, m.d_Gptr, 4, m.d_piv, m.d_info, S));
    mark();
    CUBLAS_CK(cublasCgetrsBatched(m.blas, CUBLAS_OP_N, 4, nd, (const cuComplex *const *)m.d_Gptr, 4, m.d_piv,
                                  m.d_Rptr, 4, &m.getrs_info, S));
    mark();
    // S8: LLR scale 1/sigma2 (post-equalisation noise is not tracked; the data is synthetic)
    launch_demod(m.d_R, m.d_llr, S, nd, cfg_.qam, 1.0f / cfg_.sigma2, s);
    mark();
    // S9
    launch_rate_dematch(m.d_llr, m.n_llr, m.d_cw, cfg_.ldpc_cb, code.n_bits(), 2 * code.Z, s);
    mark();
    // S10
    launch_ldpc_decode(m.dcode, m.d_cw, m.d_msg, m.d_app, cfg_.ldpc_cb, cfg_.ldpc_iters, kAlpha, m.app_fp16,
                       m.dec_smem, s);
    mark();
    // S11
    launch_hard_pack(m.d_app, code.n_bits(), code.k_bits(), m.d_bits, cfg_.ldpc_cb, s);
    mark();
    // S12
    launch_stamp_end(m.d_stamps, m.d_counter, s);
}

cudaGraphExec_t SlotPipeline::capture(cudaStream_t s) {
    if (graph_exec_) return graph_exec_;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    enqueue(s);
    CK(cudaStreamEndCapture(s, &graph_));
    CK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
    size_t n = 0;
    CK(cudaGraphGetNodes(graph_, nullptr, &n));
    graph_nodes_ = n;
    return graph_exec_;
}

std::string SlotPipeline::describe_json() const {
    const Impl &m = *impl_;
    const LdpcCode &c = m.code;
    const double S = cfg_.subcarriers, nd = m.n_data, fft = cfg_.fft, batch = (double)cfg_.symbols * kAnt;
    const double nsym = S * nd * kAnt, cb = cfg_.ldpc_cb, it = cfg_.ldpc_iters;
    const double levels = (double)(1 << (m.qm / 2));
    struct St { const char *name; double flops, bytes; };
    // Estimates: complex MAC = 8 flops; FFT 5 N log2 N; bytes = main global reads + writes.
    const St st[] = {
        {"S1_fft", 5.0 * fft * std::log2(fft) * batch, 2.0 * 8 * fft * batch},
        {"S2_pilot_gather", 0, 2.0 * 8 * S * (16 + 4 * nd)},
        {"S3_chan_est", 8.0 * 64 * S, 8.0 * S * 32},
        {"S4_gram", 8.0 * 64 * S + 4 * S, 8.0 * S * 32},
        {"S5_matched_filter", 8.0 * 16 * nd * S, 8.0 * S * (16 + 8 * nd)},
        {"S6_lu", 8.0 * 64 / 3 * S, 8.0 * S * 32},
        {"S7_solve", 8.0 * 16 * nd * S, 8.0 * S * (16 + 8 * nd)},
        {"S8_demod", nsym * 2 * levels * (3 + m.qm / 2), nsym * (8 + 4.0 * m.qm)},
        {"S9_rate_dematch", 0, 8.0 * cb * c.n_bits()},
        {"S10_ldpc_decode", it * c.n_edges() * c.Z * cb * 8, it * c.n_edges() * c.Z * cb * 8 + 8.0 * cb * c.n_bits()},
        {"S11_hard_decision", 0, 4.0 * cb * c.k_bits() + cb * c.k_bits() / 8.0},
    };
    std::string stages = "[";
    double tf = 0, tb = 0;
    for (size_t i = 0; i < sizeof st / sizeof st[0]; i++) {
        Json j;
        j.add("stage", st[i].name).add("flops", st[i].flops).add("bytes", st[i].bytes);
        stages += (i ? "," : "") + j.str();
        tf += st[i].flops;
        tb += st[i].bytes;
    }
    stages += "]";
    Json j;
    j.add_raw("config", cfg_.json());
    j.add("data_symbols", m.n_data).add("bits_per_symbol", m.qm).add("llrs_per_slot", (long long)m.n_llr);
    j.add("ldpc_code", c.describe()).add("ldpc_k_bits", c.k_bits()).add("ldpc_n_bits", c.n_bits());
    j.add("ldpc_n_tx_bits", c.n_tx_bits()).add("ldpc_edges", c.n_edges()).add("ldpc_alpha", (double)kAlpha);
    j.add("ldpc_llr_wrap", (double)cfg_.ldpc_cb * c.n_tx_bits() / (double)m.n_llr);
    j.add("decoder_smem_bytes", (unsigned long long)m.dec_smem).add("decoder_smem_optin_max", m.smem_optin);
    j.add("decoder_app_storage", m.app_fp16 ? "fp16" : "fp32").add("decoder_msg_storage", "compressed check-node state, 8 B (2x fp16 min + signs + index)");
    j.add("cublas_workspace_bytes", (unsigned long long)kBlasWorkspace);
    j.add("graph_nodes", (unsigned long long)graph_nodes_);
    j.add_raw("stages", stages).add("total_flops", tf).add("total_bytes", tb);
    return j.str();
}

bool SlotPipeline::selftest(std::string &report) {
    Impl &m = *impl_;
    const LdpcCode &c = m.code;
    char line[512];
    bool ok = true;
    auto say = [&](const char *s) { report += s; report += "\n"; };

    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    // ---- 1. GPU decoder vs host reference on noisy codewords ----
    const int n_cw = 8, iters = std::max(cfg_.ldpc_iters, 10);
    const double rate = (double)c.k_bits() / c.n_tx_bits();
    const double shift_db = std::max(0.0, 8.0 * (rate - 1.0 / 3.0));  // higher rate needs more SNR
    const double ebn0[3] = {0.5 + shift_db, 1.5 + shift_db, 3.0 + shift_db};
    snprintf(line, sizeof line, "selftest decoder: %s, %d iterations, alpha %.2f, app storage %s, check-node state 8 B (fp16 mins), %d codewords/point",
             c.describe().c_str(), iters, kAlpha, m.app_fp16 ? "fp16" : "fp32", n_cw);
    say(line);
    const size_t nb = c.n_bits();
    float *d_llr = dev_alloc<float>(n_cw * nb), *d_app = dev_alloc<float>(n_cw * nb);
    float *d_msg = dev_alloc<float>((size_t)n_cw * c.n_edges() * c.Z);
    uint32_t *d_bits = dev_alloc<uint32_t>((size_t)n_cw * m.words_per_cw);
    std::mt19937_64 rng(20260930);
    long long agree = 0, compared = 0;
    for (int p = 0; p < 3; p++) {
        double sigma2 = 1.0 / (2.0 * rate * std::pow(10.0, ebn0[p] / 10.0));
        std::normal_distribution<double> nd(0.0, std::sqrt(sigma2));
        std::vector<float> llr(n_cw * nb, 0.0f);
        std::vector<uint8_t> truth(n_cw * nb);
        long ch_err = 0;
        for (int k = 0; k < n_cw; k++) {
            std::vector<uint8_t> info(c.k_bits());
            for (auto &b : info) b = (uint8_t)(rng() & 1u);
            std::vector<uint8_t> cw = ldpc_encode(c, info);
            std::copy(cw.begin(), cw.end(), truth.begin() + k * nb);
            for (size_t i = 2 * c.Z; i < nb; i++) {
                double y = (cw[i] ? -1.0 : 1.0) + nd(rng);
                llr[k * nb + i] = (float)(2.0 * y / sigma2);
                ch_err += (y < 0) != (cw[i] != 0);
            }
        }
        // Ordered on s: a plain cudaMemcpy runs on the legacy stream, which a non-blocking stream does not
        // wait for, and a pageable H2D copy can return before the DMA lands (this let the decoder read
        // the previous point's LLRs on the first RTX 3060 run).
        CK(cudaMemcpyAsync(d_llr, llr.data(), llr.size() * sizeof(float), cudaMemcpyHostToDevice, s));
        launch_ldpc_decode(m.dcode, d_llr, d_msg, d_app, n_cw, iters, kAlpha, m.app_fp16, m.dec_smem, s);
        launch_hard_pack(d_app, c.n_bits(), c.k_bits(), d_bits, n_cw, s);
        CK(cudaStreamSynchronize(s));
        std::vector<float> app = to_host(d_app, n_cw * nb);
        std::vector<uint32_t> packed = to_host(d_bits, (size_t)n_cw * m.words_per_cw);
        long gpu_err = 0, host_err = 0, gpu_cw = 0, host_cw = 0, pack_bad = 0, nonfinite = 0;
        std::vector<uint8_t> hbits(nb);
        for (int k = 0; k < n_cw; k++) {
            ldpc_decode_host_ex(c, &llr[k * nb], iters, kAlpha, hbits.data(), m.app_fp16);
            long ge = 0, he = 0;
            for (size_t i = 0; i < nb; i++) {
                float a = app[k * nb + i];
                nonfinite += !std::isfinite(a);
                uint8_t g = a < 0.0f;
                agree += g == hbits[i];
                compared++;
                if ((int)i < c.k_bits()) {
                    ge += g != truth[k * nb + i];
                    he += hbits[i] != truth[k * nb + i];
                    uint8_t pb = (packed[(size_t)k * m.words_per_cw + i / 32] >> (i % 32)) & 1u;
                    pack_bad += pb != g;
                }
            }
            gpu_err += ge;
            host_err += he;
            gpu_cw += ge > 0;
            host_cw += he > 0;
        }
        const double kbits = (double)c.k_bits() * n_cw;
        snprintf(line, sizeof line,
                 "  Eb/N0 %5.2f dB: channel BER %.4f | GPU BER %.3e cw errors %ld/%d | host BER %.3e cw errors %ld/%d"
                 " | pack mismatches %ld | non-finite %ld",
                 ebn0[p], ch_err / ((double)c.n_tx_bits() * n_cw), gpu_err / kbits, gpu_cw, n_cw, host_err / kbits,
                 host_cw, n_cw, pack_bad, nonfinite);
        say(line);
        if (pack_bad || nonfinite) ok = false;
        if (p == 2 && gpu_err) {
            say("  FAIL: errors at the high-SNR point");
            ok = false;
        }
        if (p == 2) {  // diagnostics: serial check-node order, and 0 iterations (channel decisions only)
            const int variants[2][2] = {{1, iters}, {0, 0}};
            const char *names[2] = {"serial z order", "0 iterations"};
            for (int vi = 0; vi < 2; vi++) {
                launch_ldpc_decode_debug(m.dcode, d_llr, d_msg, d_app, n_cw, variants[vi][1], kAlpha, m.app_fp16,
                                         m.dec_smem, variants[vi][0], s);
                CK(cudaStreamSynchronize(s));
                std::vector<float> a2 = to_host(d_app, n_cw * nb);
                long e2 = 0, e2all = 0;
                for (int k = 0; k < n_cw; k++)
                    for (size_t i = 0; i < nb; i++) {
                        bool bad = (a2[k * nb + i] < 0.0f) != (truth[k * nb + i] != 0);
                        e2all += bad && i >= (size_t)2 * c.Z;
                        if ((int)i < c.k_bits()) e2 += bad;
                    }
                snprintf(line, sizeof line, "  diag %-15s: GPU info BER %.3e, transmitted-bit BER %.3e", names[vi],
                         e2 / kbits, e2all / ((double)c.n_tx_bits() * n_cw));
                say(line);
            }
        }
    }
    double agreement = (double)agree / (double)compared;
    snprintf(line, sizeof line, "  GPU/host hard-decision agreement %.6f over %lld bits (need >= 0.999)", agreement, compared);
    say(line);
    if (agreement < 0.999) ok = false;
    cudaFree(d_llr);
    cudaFree(d_app);
    cudaFree(d_msg);
    cudaFree(d_bits);

    // ---- 1b. stage profile (median over repeats, events between stages) ----
    {
        static const char *names[] = {"S1 FFT", "S2 gather", "S3 chan est", "S4 Gram+sigma", "S5 matched filt",
                                      "S6 LU (getrf)", "S7 solve (getrs)", "S8 demod", "S9 de-match", "S10 LDPC",
                                      "S11 hard pack"};
        const int nst = 11, reps = 30;
        for (auto &e : m.prof_ev) CK(cudaEventCreate(&e));
        std::vector<std::vector<float>> t(nst);
        m.prof_on = true;
        for (int r = 0; r < reps + 3; r++) {
            enqueue(s);
            CK(cudaStreamSynchronize(s));
            if (r < 3) continue;  // warm-up
            for (int i = 0; i < nst && i + 1 < m.prof_n; i++) {
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, m.prof_ev[i], m.prof_ev[i + 1]));
                t[i].push_back(ms * 1000.0f);
            }
        }
        m.prof_on = false;
        for (auto &e : m.prof_ev) CK(cudaEventDestroy(e));
        say("selftest stage profile (median us over 30 slots, event-to-event):");
        double tot = 0;
        for (int i = 0; i < nst; i++) {
            if (t[i].empty()) continue;
            std::sort(t[i].begin(), t[i].end());
            double med = t[i][t[i].size() / 2];
            tot += med;
            snprintf(line, sizeof line, "  %-18s %9.1f", names[i], med);
            say(line);
        }
        snprintf(line, sizeof line, "  %-18s %9.1f", "total", tot);
        say(line);
    }

    // ---- 2. one full slot on the synthetic signal ----
    const unsigned long long seq0 = stamps_host_->end_seq;
    enqueue(s);
    CK(cudaStreamSynchronize(s));
    const unsigned long long ss = stamps_host_->start_seq, se = stamps_host_->end_seq;
    const unsigned long long ts = stamps_host_->start_t, te = stamps_host_->end_t;
    bool stamps_ok = ss == seq0 + 1 && se == ss && ts > 0 && te >= ts;
    snprintf(line, sizeof line, "selftest slot: stamps seq %llu/%llu (expected %llu), gpu time %.1f us%s", ss, se,
             seq0 + 1, (te - ts) / 1000.0, stamps_ok ? "" : "  FAIL");
    say(line);
    ok = ok && stamps_ok;

    const int S = cfg_.subcarriers, nd = m.n_data;
    std::vector<Cf> X = to_host(m.d_R, (size_t)S * kAnt * nd);
    std::vector<float> llr = to_host(m.d_llr, (size_t)m.n_llr);
    std::vector<float> cw = to_host(m.d_cw, (size_t)cfg_.ldpc_cb * c.n_bits());
    std::vector<float> app = to_host(m.d_app, (size_t)cfg_.ldpc_cb * c.n_bits());
    std::vector<uint32_t> bits = to_host(m.d_bits, (size_t)cfg_.ldpc_cb * m.words_per_cw);
    std::vector<int> info = to_host(m.d_info, (size_t)S);
    long nonfinite = 0, llr_bad = 0, dematch_bad = 0, pack_bad = 0, singular = 0, bit_err = 0;
    double evm_num = 0;
    for (int k = 0; k < S; k++) singular += info[k] != 0;
    for (size_t q = 0; q < m.tx_sym.size(); q++) {
        // X is per subcarrier layers x data symbols column-major: q = (k*nd + t)*4 + l
        size_t k = q / (kAnt * nd), t = (q / kAnt) % nd, l = q % kAnt;
        Cf x = X[k * kAnt * nd + l + 4 * t];
        nonfinite += !std::isfinite(x.x) || !std::isfinite(x.y);
        evm_num += cabs2(csub(x, m.tx_sym[q]));
        float ref[8];
        qam_llr(x, cfg_.qam, 1.0f / cfg_.sigma2, ref);
        // transmitted bits of this symbol: demod of the noiseless symbol
        float txl[8];
        qam_llr(m.tx_sym[q], cfg_.qam, 1.0f, txl);
        for (int b = 0; b < m.qm; b++) {
            float g = llr[q * m.qm + b];
            nonfinite += !std::isfinite(g);
            if (std::fabs(g - ref[b]) > 1e-3f * std::max(1.0f, std::fabs(ref[b]))) llr_bad++;
            bit_err += (g < 0) != (txl[b] < 0);
        }
    }
    const int nbits = c.n_bits(), punct = 2 * c.Z;
    for (int k = 0; k < cfg_.ldpc_cb; k++)
        for (int j = 0; j < nbits; j++) {
            float want = j < punct ? 0.0f : llr[((long long)k * (nbits - punct) + (j - punct)) % m.n_llr];
            dematch_bad += cw[(size_t)k * nbits + j] != want;
            float a = app[(size_t)k * nbits + j];
            nonfinite += !std::isfinite(a);
            if (j < c.k_bits()) pack_bad += ((bits[(size_t)k * m.words_per_cw + j / 32] >> (j % 32)) & 1u) != (a < 0.0f);
        }
    const double evm = evm_num / m.tx_sym.size();
    const double evm_limit = std::max(0.25, 10.0 * cfg_.sigma2);
    snprintf(line, sizeof line,
             "  equaliser EVM %.1f dB (limit %.1f dB), demod BER vs transmitted %.4f, LLR mismatches %ld, "
             "de-match mismatches %ld, pack mismatches %ld, singular LU %ld, non-finite %ld",
             10 * std::log10(std::max(evm, 1e-30)), 10 * std::log10(evm_limit), (double)bit_err / (m.tx_sym.size() * m.qm),
             llr_bad, dematch_bad, pack_bad, singular, nonfinite);
    say(line);
    if (evm > evm_limit || llr_bad || dematch_bad || pack_bad || singular || nonfinite) {
        say("  FAIL: slot outputs inconsistent");
        ok = false;
    }
    CK(cudaStreamDestroy(s));
    m.bound_valid = false;  // the selftest stream is gone: rebind on the next enqueue
    say(ok ? "selftest: PASS" : "selftest: FAIL");
    return ok;
}

}  // namespace sb
