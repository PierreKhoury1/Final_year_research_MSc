// Host side of the 5G NR LDPC code: lifting, edge lists, GF(2) encoder and the reference layered
// min-sum decoder (same arithmetic as the GPU kernel through phy_ops.h ms_row_update).
#include "ldpc_code.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "ldpc_tables.h"
#include "phy_ops.h"

namespace sb {

namespace {

const int kLiftA[8] = {2, 3, 5, 7, 9, 11, 13, 15};

// Bit-packed dense GF(2) matrix, row-major.
struct Gf2Mat {
    int n = 0, words = 0;
    std::vector<uint64_t> d;
    void init(int n_) {
        n = n_;
        words = (n + 63) / 64;
        d.assign((size_t)n * words, 0);
    }
    uint64_t *row(int r) { return &d[(size_t)r * words]; }
    const uint64_t *row(int r) const { return &d[(size_t)r * words]; }
    void flip(int r, int c) { row(r)[c >> 6] ^= 1ull << (c & 63); }
    bool get(int r, int c) const { return (row(r)[c >> 6] >> (c & 63)) & 1u; }
};

// Inverse of the 4Z x 4Z core parity matrix (rows 0..3 x columns kb..kb+3) by Gauss-Jordan.
std::shared_ptr<const Gf2Mat> core_inverse(const LdpcCode &code) {
    static std::mutex mu;
    static std::map<std::pair<int, int>, std::shared_ptr<const Gf2Mat>> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto key = std::make_pair(code.bg, code.Z);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    const int Z = code.Z, n = 4 * Z;
    Gf2Mat a, inv;
    a.init(n);
    inv.init(n);
    for (int r = 0; r < 4; r++)
        for (int e = code.row_start[r]; e < code.row_start[r + 1]; e++) {
            int c = code.edge_col[e] - code.kb;
            if (c < 0 || c >= 4) continue;
            for (int i = 0; i < Z; i++) a.flip(r * Z + i, c * Z + (i + code.edge_shift[e]) % Z);
        }
    for (int i = 0; i < n; i++) inv.flip(i, i);
    for (int c = 0; c < n; c++) {
        int p = c;
        while (p < n && !a.get(p, c)) p++;
        if (p == n) throw std::logic_error("ldpc: core parity matrix is singular for " + code.describe());
        if (p != c)
            for (int w = 0; w < a.words; w++) {
                std::swap(a.row(p)[w], a.row(c)[w]);
                std::swap(inv.row(p)[w], inv.row(c)[w]);
            }
        for (int r = 0; r < n; r++) {
            if (r == c || !a.get(r, c)) continue;
            uint64_t *ar = a.row(r), *ac = a.row(c), *ir = inv.row(r), *ic = inv.row(c);
            for (int w = 0; w < a.words; w++) {
                ar[w] ^= ac[w];
                ir[w] ^= ic[w];
            }
        }
    }
    auto sp = std::make_shared<const Gf2Mat>(std::move(inv));
    cache[key] = sp;
    return sp;
}

// Host storage for ms_row_update: compressed check-node state, fp32 or fp16-rounded a posteriori LLRs.
struct HostAcc {
    const LdpcCode *code;
    float *appv;
    CnWord *cnv;  // [row][z]
    int Z, z;
    bool fp16;
    int col(int e) const { return code->edge_col[e]; }
    int shift(int e) const { return code->edge_shift[e]; }
    float app(int v) const { return appv[v]; }
    void set_app(int v, float x) {
        x = llr_clamp(x);
        appv[v] = fp16 ? f16_round(x) : x;
    }
    CnWord cn(int r) const { return cnv[(size_t)r * Z + z]; }
    void set_cn(int r, CnWord w) { cnv[(size_t)r * Z + z] = w; }
};

}  // namespace

std::string LdpcCode::describe() const {
    char b[160];
    snprintf(b, sizeof b, "BG%d Z=%d set=%d rows=%d cols=%d kb=%d edges=%d K=%d N=%d Ntx=%d rate=%.4f", bg, Z,
             i_ls, rows, cols, kb, n_edges(), k_bits(), n_bits(), n_tx_bits(), (double)k_bits() / n_tx_bits());
    return b;
}

int ldpc_set_index(int Z) {
    if (Z < 2 || Z > 384) return -1;
    for (int i = 0; i < 8; i++)
        for (int z = kLiftA[i]; z <= 384; z *= 2)
            if (z == Z) return i;
    return -1;
}

LdpcCode make_ldpc_code(int bg, int Z, int rows) {
    if (bg != 1 && bg != 2) throw std::invalid_argument("ldpc: base graph must be 1 or 2");
    int i_ls = ldpc_set_index(Z);
    if (i_ls < 0) throw std::invalid_argument("ldpc: Z=" + std::to_string(Z) + " is not a 38.212 lifting size");
    const int max_rows = bg == 1 ? kBG1Rows : kBG2Rows;
    if (rows < 4 || rows > max_rows)
        throw std::invalid_argument("ldpc: rows must be in [4, " + std::to_string(max_rows) + "] for BG" +
                                    std::to_string(bg));
    LdpcCode c;
    c.bg = bg;
    c.Z = Z;
    c.i_ls = i_ls;
    c.kb = bg == 1 ? 22 : 10;
    c.rows = rows;
    c.cols = c.kb + rows;
    const LdpcBaseEntry *tab = bg == 1 ? kBG1 : kBG2;
    const int n_tab = bg == 1 ? kBG1Entries : kBG2Entries;
    c.row_start.assign(rows + 1, 0);
    for (int t = 0; t < n_tab; t++) {
        const LdpcBaseEntry &e = tab[t];
        if (e.row >= rows) break;  // table is row-major
        if (e.col >= c.cols) continue;
        c.edge_col.push_back(e.col);
        c.edge_shift.push_back(e.shift[i_ls] % Z);
        c.row_start[e.row + 1]++;
    }
    for (int r = 0; r < rows; r++) {
        c.max_row_degree = std::max(c.max_row_degree, c.row_start[r + 1]);
        c.row_start[r + 1] += c.row_start[r];
    }
    if (c.max_row_degree > kMaxRowDegree)
        throw std::logic_error("make_ldpc_code: row degree exceeds kMaxRowDegree in phy_ops.h");
    return c;
}

bool ldpc_check(const LdpcCode &code, const std::vector<uint8_t> &cw) {
    if ((int)cw.size() != code.n_bits()) return false;
    const int Z = code.Z;
    for (int r = 0; r < code.rows; r++)
        for (int i = 0; i < Z; i++) {
            unsigned s = 0;
            for (int e = code.row_start[r]; e < code.row_start[r + 1]; e++)
                s ^= cw[(size_t)code.edge_col[e] * Z + (i + code.edge_shift[e]) % Z] & 1u;
            if (s) return false;
        }
    return true;
}

std::vector<uint8_t> ldpc_encode(const LdpcCode &code, const std::vector<uint8_t> &info) {
    if ((int)info.size() != code.k_bits()) throw std::invalid_argument("ldpc_encode: info size != k_bits");
    const int Z = code.Z, kb = code.kb;
    std::vector<uint8_t> cw(code.n_bits(), 0);
    for (int i = 0; i < code.k_bits(); i++) cw[i] = info[i] & 1u;

    // Core: A p = s, s = syndrome of the information part over rows 0..3.
    Gf2Mat syn;
    syn.init(4 * Z);  // only row 0 used as a bit vector
    for (int r = 0; r < 4; r++)
        for (int e = code.row_start[r]; e < code.row_start[r + 1]; e++) {
            if (code.edge_col[e] >= kb) continue;
            for (int i = 0; i < Z; i++)
                if (cw[(size_t)code.edge_col[e] * Z + (i + code.edge_shift[e]) % Z]) syn.flip(0, r * Z + i);
        }
    auto inv = core_inverse(code);
    const uint64_t *s = syn.row(0);
    for (int j = 0; j < 4 * Z; j++) {
        const uint64_t *ir = inv->row(j);
        uint64_t acc = 0;
        for (int w = 0; w < inv->words; w++) acc ^= ir[w] & s[w];
        cw[(size_t)kb * Z + j] = (uint8_t)(__builtin_popcountll(acc) & 1);
    }

    // Extension rows: row r >= 4 has one entry on its own column kb + r and otherwise only
    // columns < kb + r, so its parity bits follow directly.
    for (int r = 4; r < code.rows; r++) {
        int own = -1;
        for (int e = code.row_start[r]; e < code.row_start[r + 1]; e++) {
            if (code.edge_col[e] == kb + r) own = e;
            else if (code.edge_col[e] > kb + r) throw std::logic_error("ldpc_encode: unexpected extension structure");
        }
        if (own < 0) throw std::logic_error("ldpc_encode: extension row without own parity column");
        for (int i = 0; i < Z; i++) {
            unsigned p = 0;
            for (int e = code.row_start[r]; e < code.row_start[r + 1]; e++)
                if (e != own) p ^= cw[(size_t)code.edge_col[e] * Z + (i + code.edge_shift[e]) % Z];
            cw[(size_t)(kb + r) * Z + (i + code.edge_shift[own]) % Z] = (uint8_t)p;
        }
    }
    if (!ldpc_check(code, cw)) throw std::logic_error("ldpc_encode: result fails parity check for " + code.describe());
    return cw;
}

// Reference decoder with a selectable a posteriori storage precision. The GPU uses fp16 storage
// when the fp32 array does not fit in shared memory; selftest calls this with the GPU's choice.
void ldpc_decode_host_ex(const LdpcCode &code, const float *llr_in, int iters, float alpha, uint8_t *bits_out,
                         bool app_fp16) {
    const int Z = code.Z, n = code.n_bits();
    std::vector<float> app(n);
    std::vector<CnWord> cn((size_t)code.rows * Z, CnWord{0u, 0u});
    for (int v = 0; v < n; v++) {
        float x = llr_clamp(llr_in[v]);
        app[v] = app_fp16 ? f16_round(x) : x;
    }
    HostAcc acc{};
    acc.code = &code;
    acc.appv = app.data();
    acc.cnv = cn.data();
    acc.Z = Z;
    acc.fp16 = app_fp16;
    for (int it = 0; it < iters; it++)
        for (int r = 0; r < code.rows; r++) {
            int e0 = code.row_start[r], deg = code.row_start[r + 1] - e0;
            for (int z = 0; z < Z; z++) {
                acc.z = z;
                ms_row_update(acc, r, e0, deg, z, Z, alpha, it == 0);
            }
        }
    for (int v = 0; v < n; v++) bits_out[v] = app[v] < 0.0f ? 1 : 0;
}

void ldpc_decode_host(const LdpcCode &code, const float *llr_in, int iters, float alpha, uint8_t *bits_out) {
    ldpc_decode_host_ex(code, llr_in, iters, alpha, bits_out, false);
}

}  // namespace sb
