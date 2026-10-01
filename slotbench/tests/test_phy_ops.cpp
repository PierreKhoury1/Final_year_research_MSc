// Host tests for phy/phy_ops.h: Gray QAM mapping and max-log LLRs, min-sum helpers against a
// brute-force reference, software fp16 rounding, slot geometry helpers.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "phy_ops.h"

using namespace sb;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, ...) do { if (cond) g_pass++; else { g_fail++; printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); } } while (0)

// Every constellation point: unit energy, distinct, Gray (neighbours differ in one bit), zero-noise
// LLR signs recover the bits, magnitudes equal a 2-D brute-force max-log over all M points.
static void test_qam() {
    for (int qam : {4, 16, 64, 256}) {
        int q = qam_bits(qam);
        std::vector<Cf> pts(qam);
        double energy = 0;
        int sign_bad = 0, mag_bad = 0, gray_bad = 0, noisy_bad = 0;
        for (int s = 0; s < qam; s++) {
            uint8_t b[8];
            for (int i = 0; i < q; i++) b[i] = (s >> i) & 1;
            pts[s] = qam_map(b, qam);
            energy += cabs2(pts[s]);
        }
        energy /= qam;
        CHECK(std::fabs(energy - 1.0) < 1e-5, "QAM%d mean energy %.7f", qam, energy);
        // distinct points on the square grid with spacing 2*scale
        float sc = qam_scale(qam);
        int m = q / 2, side = 1 << m;
        std::vector<int> seen(qam, 0);
        for (int s = 0; s < qam; s++) {
            int ix = (int)std::lround((pts[s].x / sc + side - 1) / 2), iy = (int)std::lround((pts[s].y / sc + side - 1) / 2);
            if (ix >= 0 && ix < side && iy >= 0 && iy < side) seen[ix * side + iy]++;
        }
        CHECK(std::count(seen.begin(), seen.end(), 1) == qam, "QAM%d points not a full grid", qam);
        // Gray: points at distance 2*scale differ in exactly one bit
        for (int s = 0; s < qam; s++)
            for (int t = 0; t < qam; t++) {
                Cf d = csub(pts[s], pts[t]);
                if (std::fabs(std::sqrt(cabs2(d)) - 2 * sc) < 1e-4f && __builtin_popcount(s ^ t) != 1) gray_bad++;
            }
        CHECK(gray_bad == 0, "QAM%d not Gray (%d pairs)", qam, gray_bad);
        // zero-noise LLRs and brute-force max-log
        const float inv_n0 = 10.0f;
        for (int s = 0; s < qam; s++) {
            float llr[8];
            qam_llr(pts[s], qam, inv_n0, llr);
            for (int i = 0; i < q; i++) {
                int bit = (s >> i) & 1;
                if (!((bit == 0 && llr[i] > 0) || (bit == 1 && llr[i] < 0))) sign_bad++;
                double d0 = 1e30, d1 = 1e30;
                for (int t = 0; t < qam; t++) {
                    double d = cabs2(csub(pts[s], pts[t]));
                    if ((t >> i) & 1) d1 = std::min(d1, d); else d0 = std::min(d0, d);
                }
                double ref = (d1 - d0) * inv_n0;
                if (std::fabs(llr[i] - ref) > 1e-4 * std::max(1.0, std::fabs(ref))) mag_bad++;
            }
        }
        CHECK(sign_bad == 0, "QAM%d zero-noise LLR signs wrong: %d", qam, sign_bad);
        CHECK(mag_bad == 0, "QAM%d LLR magnitudes differ from brute force: %d", qam, mag_bad);
        // noisy points vs brute force (max-log is separable per axis for square QAM)
        std::mt19937 rng(7 + qam);
        std::normal_distribution<float> nd(0.0f, 0.3f);
        for (int k = 0; k < 2000; k++) {
            Cf y = cadd(pts[rng() % qam], Cf{nd(rng), nd(rng)});
            float llr[8];
            qam_llr(y, qam, 3.0f, llr);
            for (int i = 0; i < q; i++) {
                double d0 = 1e30, d1 = 1e30;
                for (int t = 0; t < qam; t++) {
                    double d = cabs2(csub(y, pts[t]));
                    if ((t >> i) & 1) d1 = std::min(d1, d); else d0 = std::min(d0, d);
                }
                double ref = std::clamp((d1 - d0) * 3.0, -(double)kLlrClamp, (double)kLlrClamp);
                if (std::fabs(llr[i] - ref) > 1e-3 * std::max(1.0, std::fabs(ref))) noisy_bad++;
            }
        }
        CHECK(noisy_bad == 0, "QAM%d noisy LLRs differ from brute force: %d", qam, noisy_bad);
    }
    // 38.211 5.1.3 spot values for 16QAM (scale 1/sqrt(10))
    uint8_t b0[4] = {0, 0, 0, 0}, b1[4] = {1, 1, 1, 1}, b2[4] = {0, 1, 1, 0};
    Cf p0 = qam_map(b0, 16), p1 = qam_map(b1, 16), p2 = qam_map(b2, 16);
    float s = 1.0f / std::sqrt(10.0f);
    CHECK(std::fabs(p0.x - s) < 1e-6f && std::fabs(p0.y - s) < 1e-6f, "16QAM 0000 -> %f %f", p0.x, p0.y);
    CHECK(std::fabs(p1.x + 3 * s) < 1e-6f && std::fabs(p1.y + 3 * s) < 1e-6f, "16QAM 1111 -> %f %f", p1.x, p1.y);
    // b0=0,b2=1 -> I = 1*(2-(-1)) = 3; b1=1,b3=0 -> Q = -1*(2-1) = -1
    CHECK(std::fabs(p2.x - 3 * s) < 1e-6f && std::fabs(p2.y + s) < 1e-6f, "16QAM 0110 -> %f %f", p2.x, p2.y);
    // huge input stays finite (clamped)
    float llr[8];
    qam_llr(Cf{1e20f, -1e20f}, 256, 1e10f, llr);
    bool fin = true;
    for (float v : llr) fin = fin && std::isfinite(v) && std::fabs(v) <= kLlrClamp;
    CHECK(fin, "LLR not clamped");
}

// Reference check-to-variable message for edge j.
static float ms_ref(const std::vector<float> &t, int j, float alpha) {
    float mn = 1e30f;
    int neg = 0;
    for (int i = 0; i < (int)t.size(); i++) {
        if (i == j) continue;
        mn = std::min(mn, std::fabs(t[i]));
        neg ^= t[i] < 0;
    }
    return neg ? -alpha * mn : alpha * mn;
}

static void test_minsum() {
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> ud(-20.0f, 20.0f);
    int bad = 0, cases = 0;
    for (int k = 0; k < 20000; k++) {
        int deg = 2 + rng() % 19;
        std::vector<float> t(deg);
        for (auto &v : t) {
            v = ud(rng);
            if (rng() % 8 == 0) v = std::round(v);   // ties and exact zeros
            if (rng() % 16 == 0) v = 0.0f;
        }
        if (rng() % 4 == 0) t[rng() % deg] = t[rng() % deg];  // duplicate minima
        MsAcc a;
        ms_init(a);
        for (int j = 0; j < deg; j++) ms_add(a, t[j], j);
        for (int j = 0; j < deg; j++) {
            float o = ms_out(a, t[j], j, 0.75f), r = ms_ref(t, j, 0.75f);
            cases++;
            if (o != r && !(o == 0.0f && r == 0.0f)) bad++;
        }
    }
    CHECK(bad == 0, "min-sum helper mismatches %d / %d", bad, cases);

    // ms_row_update on a toy row against a direct computation (Z = 5, 3 edges), two iterations so the
    // second one subtracts the old messages rebuilt from the compressed state
    struct S {
        float *ap;
        CnWord *cnv;
        const int *cols, *shifts;
        int Z, z;
        int col(int e) const { return cols[e]; }
        int shift(int e) const { return shifts[e]; }
        float app(int v) const { return ap[v]; }
        void set_app(int v, float x) { ap[v] = llr_clamp(x); }
        CnWord cn(int r) const { return cnv[r * Z + z]; }
        void set_cn(int r, CnWord w) { cnv[r * Z + z] = w; }
    };
    const int Z = 5, cols[3] = {0, 2, 3}, shifts[3] = {1, 4, 0};
    std::vector<float> app(4 * Z);
    std::vector<CnWord> cn(Z, CnWord{0u, 0u});
    for (auto &v : app) v = ud(rng);
    int rbad = 0;
    for (int it = 0; it < 2; it++) {
        std::vector<float> app0 = app;
        std::vector<CnWord> cn0 = cn;
        for (int z = 0; z < Z; z++) {
            S s{app.data(), cn.data(), cols, shifts, Z, z};
            ms_row_update(s, 0, 0, 3, z, Z, 0.75f, it == 0);
            std::vector<float> t(3);
            int v[3];
            for (int j = 0; j < 3; j++) {
                v[j] = cols[j] * Z + (z + shifts[j]) % Z;
                t[j] = app0[v[j]] - (it == 0 ? 0.0f : cn_msg(cn0[z], j, 0.75f));
            }
            for (int j = 0; j < 3; j++) {
                float n = ms_ref(t, j, 0.75f);
                if (app[v[j]] != llr_clamp(t[j] + n)) rbad++;
            }
        }
    }
    CHECK(rbad == 0, "ms_row_update mismatches %d", rbad);
}

static void test_fp16() {
    // round trip of every finite half and inf/NaN handling
    int rt_bad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        float f = f16_bits_to_f32((uint16_t)h);
        uint16_t back = f32_to_f16_bits(f);
        int exp = (h >> 10) & 0x1f;
        if (exp == 31 && (h & 0x3ff)) {
            if (!std::isnan(f) || ((back >> 10) & 0x1f) != 31 || !(back & 0x3ff)) rt_bad++;
        } else if (back != h) {
            rt_bad++;
        }
    }
    CHECK(rt_bad == 0, "fp16 round trip failures %d", rt_bad);
    // round to nearest even at every midpoint between adjacent positive finite halves
    int rne_bad = 0;
    for (uint32_t h = 0; h < 0x7bff; h++) {
        double lo = f16_bits_to_f32((uint16_t)h), hi = f16_bits_to_f32((uint16_t)(h + 1));
        float mid = (float)((lo + hi) / 2);  // exact in fp32
        uint16_t even = (h & 1) ? (uint16_t)(h + 1) : (uint16_t)h;
        if (f32_to_f16_bits(mid) != even) rne_bad++;
        if (f32_to_f16_bits(std::nextafter(mid, 1e9f)) != h + 1) rne_bad++;
        if (f32_to_f16_bits(std::nextafter(mid, -1e9f)) != h) rne_bad++;
        if (f32_to_f16_bits(-mid) != (even | 0x8000)) rne_bad++;
    }
    CHECK(rne_bad == 0, "fp16 RNE failures %d", rne_bad);
    CHECK(f32_to_f16_bits(65504.0f) == 0x7bff && f32_to_f16_bits(65519.0f) == 0x7bff &&
          f32_to_f16_bits(65520.0f) == 0x7c00 && f32_to_f16_bits(1e10f) == 0x7c00 &&
          f32_to_f16_bits(-1e10f) == 0xfc00, "fp16 overflow handling");
    CHECK(f32_to_f16_bits(2.9802322e-8f) == 0 && f32_to_f16_bits(2.9802326e-8f) == 1 &&
          f32_to_f16_bits(1e-30f) == 0 && f32_to_f16_bits(-0.0f) == 0x8000, "fp16 underflow handling");
    CHECK(f16_round(kLlrClamp) == kLlrClamp && f16_round(-kLlrClamp) == -kLlrClamp, "clamp not representable");
    CHECK(llr_clamp(1e30f) == kLlrClamp && llr_clamp(-INFINITY) == -kLlrClamp && llr_clamp(NAN) == -kLlrClamp &&
          llr_clamp(3.5f) == 3.5f, "llr_clamp");
}

static void test_geometry() {
    // pilot matrix orthogonality Xp Xp^H = 4 I
    int bad = 0;
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++) {
            float s = 0;
            for (int j = 0; j < 4; j++) s += pilot_value(a, j) * pilot_value(b, j);
            if (s != (a == b ? 4.0f : 0.0f)) bad++;
        }
    CHECK(bad == 0, "pilot matrix not orthogonal");
    // both subcarriers of a pair gather the same 4 (subcarrier, symbol) positions in the same order
    for (int k = 0; k < 100; k++)
        for (int j = 0; j < 4; j++)
            if (pilot_subcarrier(k, j) != pilot_subcarrier(k ^ 1, j) || (pilot_subcarrier(k, j) >> 1) != (k >> 1)) bad++;
    CHECK(bad == 0, "pilot pairing");
    std::vector<int> seen;
    for (int t = 0; t < 12; t++) seen.push_back(data_symbol(t));
    std::vector<int> want = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13};
    CHECK(seen == want, "data symbols");
    std::vector<int> bins;
    for (int k = 0; k < 3276; k++) bins.push_back(subcarrier_bin(k, 3276, 4096));
    std::sort(bins.begin(), bins.end());
    CHECK(std::unique(bins.begin(), bins.end()) == bins.end() && bins.front() >= 0 && bins.back() < 4096,
          "subcarrier bins not distinct/in range");
    CHECK(subcarrier_bin(1638, 3276, 4096) == 0 && subcarrier_bin(0, 3276, 4096) == 4096 - 1638, "bin centring");
}

// The compile-time-M demod used by the GPU kernel must equal the runtime-qam reference bit for bit.
void test_qam_m() {
    const int orders[4] = {4, 16, 64, 256};
    long bad = 0, n = 0;
    for (int oi = 0; oi < 4; oi++) {
        int qam = orders[oi], q = qam_bits(qam);
        for (int i = 0; i < 2000; i++) {
            Cf y{(float)((i * 7919) % 2001 - 1000) / 700.0f, (float)((i * 104729) % 2001 - 1000) / 700.0f};
            float a[8], b[8];
            qam_llr(y, qam, 37.0f, a);
            switch (q / 2) {
                case 1: qam_llr_m<1>(y, 37.0f, b); break;
                case 2: qam_llr_m<2>(y, 37.0f, b); break;
                case 3: qam_llr_m<3>(y, 37.0f, b); break;
                default: qam_llr_m<4>(y, 37.0f, b); break;
            }
            for (int k = 0; k < q; k++, n++) bad += a[k] != b[k];
        }
    }
    CHECK(bad == 0 && n > 0, "qam_llr_m differs from qam_llr");
}

// Compressed check-node state reproduces every outgoing message (up to the fp16 rounding of the mins).
void test_cn_pack() {
    long bad = 0;
    for (int trial = 0; trial < 500; trial++) {
        int deg = 2 + trial % 18;
        float t[19];
        MsAcc a;
        ms_init(a);
        uint32_t signs = 0;
        for (int j = 0; j < deg; j++) {
            t[j] = (float)(((trial * 31 + j * 17) % 401) - 200) / 8.0f;
            ms_add(a, t[j], j);
            signs |= (t[j] < 0.0f ? 1u : 0u) << j;
        }
        CnWord w = cn_pack(a, signs);
        for (int j = 0; j < deg; j++) {
            float want = ms_out(a, t[j], j, 0.75f);
            float got = cn_msg(w, j, 0.75f);
            float tol = 1e-3f * std::max(1.0f, std::fabs(want));
            bad += std::fabs(got - want) > tol || ((got < 0) != (want < 0) && want != 0.0f);
        }
    }
    CHECK(bad == 0, "cn_msg does not reproduce ms_out");
}

int main() {
    test_qam();
    test_qam_m();
    test_cn_pack();
    test_minsum();
    test_fp16();
    test_geometry();
    printf("test_phy_ops: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
