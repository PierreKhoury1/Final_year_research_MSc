// Host tests for the 5G LDPC tables, lifting, encoder and reference layered min-sum decoder.
// Plain counters; prints a summary and exits nonzero on any failure.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "ldpc_code.h"
#include "ldpc_tables.h"
#include "phy_ops.h"

namespace sb {
// Defined in ldpc_code.cpp (not in the frozen header): decoder with fp16 a posteriori storage.
void ldpc_decode_host_ex(const LdpcCode &code, const float *llr_in, int iters, float alpha, uint8_t *bits_out,
                         bool app_fp16);
}  // namespace sb

using namespace sb;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, ...) do { if (cond) g_pass++; else { g_fail++; printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); } } while (0)

static std::vector<uint8_t> random_bits(std::mt19937_64 &rng, int n) {
    std::vector<uint8_t> v(n);
    for (auto &b : v) b = (uint8_t)(rng() & 1u);
    return v;
}

// BPSK (bit 0 -> +1) over AWGN at Eb/N0 (dB) for the code's transmitted rate; punctured first 2Z
// positions get LLR 0. Returns channel hard-decision errors over the transmitted bits.
static long bpsk_awgn(const LdpcCode &c, const std::vector<uint8_t> &cw, double ebn0_db, std::mt19937_64 &rng,
                      std::vector<float> &llr) {
    double rate = (double)c.k_bits() / c.n_tx_bits();
    double sigma2 = 1.0 / (2.0 * rate * std::pow(10.0, ebn0_db / 10.0));
    std::normal_distribution<double> nd(0.0, std::sqrt(sigma2));
    llr.assign(c.n_bits(), 0.0f);
    long err = 0;
    for (int i = 2 * c.Z; i < c.n_bits(); i++) {
        double y = (cw[i] ? -1.0 : 1.0) + nd(rng);
        llr[i] = (float)(2.0 * y / sigma2);
        err += ((y < 0) != (cw[i] != 0));
    }
    return err;
}

static void test_tables() {
    CHECK(kBG1Entries == 316 && kBG2Entries == 197, "entry counts %d %d", kBG1Entries, kBG2Entries);
    for (int bg = 1; bg <= 2; bg++) {
        const LdpcBaseEntry *t = bg == 1 ? kBG1 : kBG2;
        int n = bg == 1 ? kBG1Entries : kBG2Entries, rows = bg == 1 ? 46 : 42, kb = bg == 1 ? 22 : 10;
        std::vector<int> rdeg(rows, 0);
        bool sorted = true, ext_ok = true;
        for (int i = 0; i < n; i++) {
            rdeg[t[i].row]++;
            if (i && (t[i].row < t[i - 1].row || (t[i].row == t[i - 1].row && t[i].col <= t[i - 1].col))) sorted = false;
        }
        for (int r = 4; r < rows; r++) {  // extension rows: identity (V = 0 for all sets) on column kb + r
            bool found = false;
            for (int i = 0; i < n; i++)
                if (t[i].row == r && t[i].col == kb + r) {
                    found = true;
                    for (int s = 0; s < 8; s++) ext_ok = ext_ok && t[i].shift[s] == 0;
                }
            ext_ok = ext_ok && found;
        }
        CHECK(sorted, "BG%d table not row-major sorted", bg);
        CHECK(ext_ok, "BG%d extension rows are not identity on their own column", bg);
        int max_deg = 0;
        for (int d : rdeg) max_deg = std::max(max_deg, d);
        CHECK(max_deg == (bg == 1 ? 19 : 10), "BG%d max row degree %d", bg, max_deg);
        LdpcCode c = make_ldpc_code(bg, 384, rows);
        CHECK(c.n_edges() == n && c.cols == (bg == 1 ? 68 : 52), "BG%d full code edges %d cols %d", bg, c.n_edges(), c.cols);
        CHECK(c.max_row_degree == max_deg, "BG%d code max degree %d", bg, c.max_row_degree);
    }
    // row 0 of BG1 has 19 entries, rows 4.. of BG1 are sparse (degree 3 for row 4: cols 0, 1, 26)
    LdpcCode c = make_ldpc_code(1, 384, 5);
    CHECK(c.row_start[1] - c.row_start[0] == 19, "BG1 row 0 degree");
    CHECK(c.row_start[5] - c.row_start[4] == 3, "BG1 row 4 degree %d", c.row_start[5] - c.row_start[4]);
    // shift = V mod Z: BG1 (0,0) has V = 307 for set 1 (Z = 3 and Z = 384 are both set 1)
    LdpcCode c3 = make_ldpc_code(1, 3, 46);  // Z = 3 is set 1
    CHECK(c3.edge_shift[0] == 307 % 3, "shift mod Z %d", c3.edge_shift[0]);
    LdpcCode c384 = make_ldpc_code(1, 384, 46);
    CHECK(c384.edge_shift[0] == 307, "BG1(0,0) set1 shift %d", c384.edge_shift[0]);
    bool threw = false;
    try { make_ldpc_code(1, 384, 47); } catch (const std::exception &) { threw = true; }
    CHECK(threw, "rows 47 accepted");
    threw = false;
    try { make_ldpc_code(1, 17, 46); } catch (const std::exception &) { threw = true; }
    CHECK(threw, "Z=17 accepted");
}

static void test_set_index() {
    int n_valid = 0;
    for (int z = 0; z <= 400; z++) n_valid += ldpc_set_index(z) >= 0;
    CHECK(n_valid == 51, "lifting sizes %d != 51", n_valid);
    const int a[8] = {2, 3, 5, 7, 9, 11, 13, 15};
    for (int i = 0; i < 8; i++)
        for (int z = a[i]; z <= 384; z *= 2) CHECK(ldpc_set_index(z) == i, "set_index(%d) = %d != %d", z, ldpc_set_index(z), i);
    CHECK(ldpc_set_index(1) < 0 && ldpc_set_index(17) < 0 && ldpc_set_index(416) < 0, "invalid sizes accepted");
}

static void test_encoder(std::mt19937_64 &rng) {
    int z_list[3] = {384, 208, 64}, r_list[3] = {4, 10, 46};
    for (int Z : z_list)
        for (int rows : r_list) {
            LdpcCode c = make_ldpc_code(1, Z, rows);
            for (int k = 0; k < 2; k++) {
                auto cw = ldpc_encode(c, random_bits(rng, c.k_bits()));
                CHECK(ldpc_check(c, cw), "BG1 Z=%d rows=%d codeword invalid", Z, rows);
            }
        }
    for (int Z : {384, 52, 10})
        for (int rows : {4, 20, 42}) {
            LdpcCode c = make_ldpc_code(2, Z, rows);
            auto cw = ldpc_encode(c, random_bits(rng, c.k_bits()));
            CHECK(ldpc_check(c, cw), "BG2 Z=%d rows=%d codeword invalid", Z, rows);
        }
    // every lifting size, both graphs, full rows: the core must be invertible and encoding valid
    int ok = 0, tot = 0;
    for (int Z = 2; Z <= 384; Z++) {
        if (ldpc_set_index(Z) < 0) continue;
        for (int bg = 1; bg <= 2; bg++) {
            LdpcCode c = make_ldpc_code(bg, Z, bg == 1 ? 46 : 42);
            tot++;
            try {
                auto cw = ldpc_encode(c, random_bits(rng, c.k_bits()));
                ok += ldpc_check(c, cw);
            } catch (const std::exception &e) {
                printf("  %s\n", e.what());
            }
        }
    }
    CHECK(ok == tot && tot == 102, "all-Z encode %d/%d", ok, tot);
    // a flipped bit must break the check
    LdpcCode c = make_ldpc_code(1, 64, 46);
    auto cw = ldpc_encode(c, random_bits(rng, c.k_bits()));
    cw[100] ^= 1;
    CHECK(!ldpc_check(c, cw), "check misses a flipped bit");
}

struct DecStats {
    long ch_err = 0, bits = 0, dec_err = 0, cw_err = 0, n_cw = 0;
};

static DecStats run_decode(const LdpcCode &c, double ebn0, int n_cw, int iters, bool fp16, std::mt19937_64 &rng) {
    DecStats s;
    std::vector<float> llr;
    std::vector<uint8_t> out(c.n_bits());
    for (int k = 0; k < n_cw; k++) {
        auto cw = ldpc_encode(c, random_bits(rng, c.k_bits()));
        s.ch_err += bpsk_awgn(c, cw, ebn0, rng, llr);
        ldpc_decode_host_ex(c, llr.data(), iters, 0.75f, out.data(), fp16);
        long e = 0;
        for (int i = 0; i < c.k_bits(); i++) e += out[i] != cw[i];
        s.dec_err += e;
        s.cw_err += e > 0;
        s.bits += c.k_bits();
        s.n_cw++;
    }
    return s;
}

static void test_decoder(std::mt19937_64 &rng) {
    LdpcCode c = make_ldpc_code(1, 384, 46);
    printf("  code: %s\n", c.describe().c_str());
    // high SNR: error free (fp32 and fp16 storage)
    DecStats hi = run_decode(c, 3.0, 8, 20, false, rng);
    DecStats hi16 = run_decode(c, 3.0, 8, 20, true, rng);
    printf("  Eb/N0 3.0 dB: channel BER %.4f, decoded bit errors fp32 %ld fp16 %ld\n",
           (double)hi.ch_err / (c.n_tx_bits() * 8.0), hi.dec_err, hi16.dec_err);
    CHECK(hi.dec_err == 0 && hi16.dec_err == 0, "errors at 3 dB: %ld %ld", hi.dec_err, hi16.dec_err);
    // moderate SNR: corrects many channel errors
    DecStats mid = run_decode(c, 1.5, 16, 20, false, rng);
    double ber_ch = (double)mid.ch_err / (c.n_tx_bits() * 16.0), ber_dec = (double)mid.dec_err / mid.bits;
    printf("  Eb/N0 1.5 dB: channel BER %.4f -> decoded BER %.2e, codeword errors %ld/%ld\n", ber_ch, ber_dec,
           mid.cw_err, mid.n_cw);
    CHECK(ber_ch > 0.1 && ber_dec < ber_ch / 100.0, "weak correction %.4f -> %.4f", ber_ch, ber_dec);
    DecStats lo = run_decode(c, 0.5, 8, 20, false, rng);
    printf("  Eb/N0 0.5 dB: decoded BER %.2e, codeword errors %ld/%ld (informational)\n",
           (double)lo.dec_err / lo.bits, lo.cw_err, lo.n_cw);

    // high-rate and BG2 codes at high SNR (the selftest shifts its Eb/N0 points by rate)
    struct { int bg, Z, rows; double ebn0; } hc[] = {{1, 384, 4, 3.0 + 8.0 * (22.0 / 24 - 1.0 / 3)},
                                                     {1, 384, 10, 3.0 + 8.0 * (22.0 / 30 - 1.0 / 3)},
                                                     {2, 384, 42, 3.0}, {1, 64, 46, 3.0}};
    for (auto &h : hc) {
        LdpcCode ch = make_ldpc_code(h.bg, h.Z, h.rows);
        DecStats d = run_decode(ch, h.ebn0, 4, 20, false, rng);
        printf("  %s @ %.2f dB: bit errors %ld\n", ch.describe().c_str(), h.ebn0, d.dec_err);
        CHECK(d.dec_err == 0, "BG%d Z=%d rows=%d errors %ld", h.bg, h.Z, h.rows, d.dec_err);
    }

    // all-zero trap: a random codeword decodes to itself, not to zeros; deterministic across runs
    auto cw = ldpc_encode(c, random_bits(rng, c.k_bits()));
    std::vector<float> llr;
    bpsk_awgn(c, cw, 2.5, rng, llr);
    std::vector<uint8_t> a(c.n_bits()), b(c.n_bits()), f(c.n_bits());
    ldpc_decode_host(c, llr.data(), 20, 0.75f, a.data());
    ldpc_decode_host(c, llr.data(), 20, 0.75f, b.data());
    ldpc_decode_host_ex(c, llr.data(), 20, 0.75f, f.data(), true);
    long ones = 0, diff = 0, dfp16 = 0;
    for (int i = 0; i < c.n_bits(); i++) {
        ones += a[i];
        diff += a[i] != cw[i];
        dfp16 += a[i] != f[i];
    }
    CHECK(ones > c.n_bits() / 4 && diff == 0, "random codeword: ones %ld diff %ld", ones, diff);
    CHECK(a == b, "decoder not deterministic");
    CHECK(dfp16 == 0, "fp16 storage decode differs in %ld bits", dfp16);
    // zero iterations = hard decision on the (clamped) channel LLRs
    ldpc_decode_host(c, llr.data(), 0, 0.75f, a.data());
    long hd = 0;
    for (int i = 2 * c.Z; i < c.n_bits(); i++) hd += a[i] != (llr[i] < 0);
    CHECK(hd == 0, "0-iteration output != channel hard decision (%ld)", hd);
}

int main() {
    auto t0 = std::chrono::steady_clock::now();
    std::mt19937_64 rng(12345);
    test_tables();
    test_set_index();
    test_encoder(rng);
    test_decoder(rng);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("test_ldpc: %d passed, %d failed (%.1f s)\n", g_pass, g_fail, s);
    return g_fail ? 1 : 0;
}
