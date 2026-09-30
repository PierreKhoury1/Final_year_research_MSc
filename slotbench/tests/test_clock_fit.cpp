// Host test for common/clock_fit.h. Synthetic brackets around a known GPU->host mapping
// (rate offset in ppm, large absolute offsets, 6-hour span at ns resolution) with asymmetric,
// heavy-tailed bracket widths; checks the recovered rate, offset, error bound, the host_of/gpu_of
// round trip and degenerate inputs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "clock_fit.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(c) do { if (c) g_pass++; else { g_fail++; fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

// True mapping: host ns at which the GPU read g = T0 + a_true * (g - G0).
struct Truth {
    int64_t T0 = 314159265358979LL;       // ~3.6 days of CLOCK_MONOTONIC_RAW
    uint64_t G0 = 1790000000123456789ULL;  // %globaltimer is Unix-epoch ns scale
    double a_true = 1.0;
    double host(uint64_t g) const { return (double)T0 + a_true * (double)(int64_t)(g - G0); }
};

// One side of a bracket: fixed latency, exponential jitter, 5% heavy (Pareto) tail up to ~1 ms.
static double side(std::mt19937_64 &rng, double base, double mean_exp) {
    std::exponential_distribution<double> ex(1.0 / mean_exp);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double w = base + ex(rng);
    if (u(rng) < 0.05) w += std::min(1e6, 500.0 / std::pow(u(rng) + 1e-9, 1.5));
    return w;
}

static std::vector<sb::Bracket> make_brackets(const Truth &tr, int n, double span_ns, uint64_t seed,
                                              bool asymmetric = true) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, span_ns);
    std::vector<sb::Bracket> br;
    br.reserve(n);
    for (int i = 0; i < n; i++) {
        uint64_t g = tr.G0 + (uint64_t)u(rng);
        double T = tr.host(g);
        // asymmetric: request path ~450 ns + exp(150), answer path ~650 ns + exp(250)
        // symmetric: both paths ~550 ns + exp(200)
        double w1 = asymmetric ? side(rng, 450, 150) : side(rng, 550, 200);
        double w2 = asymmetric ? side(rng, 650, 250) : side(rng, 550, 200);
        br.push_back({(int64_t)std::floor(T - w1), (int64_t)std::ceil(T + w2), g});
    }
    return br;
}

// Largest |host_of(g) - truth| over the sample GPU readings and a dense grid across the span.
static double max_err(const sb::ClockFit &f, const Truth &tr, const std::vector<sb::Bracket> &br, double span_ns) {
    double m = 0;
    for (const auto &b : br) m = std::max(m, std::fabs(f.host_of(b.g) - tr.host(b.g)));
    for (int i = 0; i <= 1000; i++) {
        uint64_t g = tr.G0 + (uint64_t)(span_ns * i / 1000.0);
        m = std::max(m, std::fabs(f.host_of(g) - tr.host(g)));
    }
    return m;
}

static void test_known_mapping(double ppm, uint64_t seed) {
    Truth tr;
    tr.a_true = 1.0 + ppm * 1e-6;
    const double span = 6.0 * 3600e9;  // 6 hours
    auto br = make_brackets(tr, 20000, span, seed);
    sb::ClockFit f = sb::fit_clock(br);
    CHECK(f.ok);
    CHECK(f.n == 20000);
    CHECK(f.n_kept >= 200);
    double rate_err = std::fabs(f.rate_ppm() - ppm);
    // Offset: the mapping error at mid-span. The asymmetric paths shift bracket midpoints by
    // (w2 - w1)/2 = 100 ns plus jitter difference; a midpoint fit cannot see a constant asymmetry,
    // so the reference is the true instant shifted by the kept brackets' mean asymmetry.
    std::vector<int64_t> ws;
    for (const auto &b : br) ws.push_back(b.t1 - b.t0);
    std::sort(ws.begin(), ws.end());
    const int64_t thr = ws[std::max<size_t>((size_t)std::ceil(0.01 * ws.size()), 50) - 1];  // fit_clock's kept set
    double asym = 0;
    int nk = 0;
    for (const auto &b : br)
        if (b.t1 - b.t0 <= thr) {
            asym += 0.5 * ((double)b.t0 + (double)b.t1) - tr.host(b.g);
            nk++;
        }
    asym /= nk;
    uint64_t gmid = tr.G0 + (uint64_t)(span / 2);
    double off_err = std::fabs(f.host_of(gmid) - (tr.host(gmid) + asym));
    double err = max_err(f, tr, br, span);
    printf("  %+7.2f ppm seed %llu: rate err %.2e ppm, offset err %.2f ns (asym %.1f), max err %.1f ns, eps %.1f ns, "
           "kept %d, min width %.0f\n",
           ppm, (unsigned long long)seed, rate_err, off_err, asym, err, f.eps_ns, f.n_kept, f.min_width_ns);
    CHECK(rate_err < 0.01);
    CHECK(nk == f.n_kept);
    CHECK(off_err < 5.0);
    CHECK(err <= f.eps_ns);          // the bound covers the true error everywhere in the span
    CHECK(f.eps_ns < 2000.0);        // and is not vacuous
    CHECK(f.resid_rms_ns < f.kept_median_width_ns);
    CHECK(f.min_width_ns >= 1100.0 - 2 && f.median_width_ns > f.min_width_ns);
}

// Symmetric paths: the fitted offset itself must be within a few ns of the truth.
static void test_symmetric_offset() {
    Truth tr;
    tr.a_true = 1.0 + 37e-6;
    const double span = 6.0 * 3600e9;
    auto br = make_brackets(tr, 20000, span, 21, false);
    sb::ClockFit f = sb::fit_clock(br);
    uint64_t gmid = tr.G0 + (uint64_t)(span / 2);
    double off = std::fabs(f.host_of(gmid) - tr.host(gmid));
    double err = max_err(f, tr, br, span);
    printf("  symmetric +37 ppm: rate err %.2e ppm, offset err %.2f ns, max err %.2f ns, eps %.1f ns\n",
           std::fabs(f.rate_ppm() - 37.0), off, err, f.eps_ns);
    CHECK(f.ok);
    CHECK(std::fabs(f.rate_ppm() - 37.0) < 0.01);
    CHECK(off < 5.0);
    CHECK(err < 10.0 && err <= f.eps_ns);
}

static void test_round_trip() {
    Truth tr;
    tr.a_true = 1.0 + 37e-6;
    auto br = make_brackets(tr, 5000, 6.0 * 3600e9, 7);
    sb::ClockFit f = sb::fit_clock(br);
    CHECK(f.ok);
    int bad = 0;
    std::mt19937_64 rng(3);
    for (int i = 0; i < 10000; i++) {
        uint64_t g = tr.G0 + (rng() % (uint64_t)(6.0 * 3600e9));
        int64_t t = (int64_t)std::llround(f.host_of(g));
        uint64_t g2 = f.gpu_of(t);
        double h2 = f.host_of(f.gpu_of(t));
        if (std::llabs((long long)(g2 - g)) > 1 || std::fabs(h2 - (double)t) > 1.0) bad++;
    }
    CHECK(bad == 0);
    // mapping works before g_ref too (negative x)
    uint64_t early = f.g_ref - 1000000000ULL;
    CHECK(std::llabs((long long)(f.gpu_of((int64_t)std::llround(f.host_of(early))) - early)) <= 1);
}

static void test_degenerate() {
    std::vector<sb::Bracket> none;
    sb::ClockFit f0 = sb::fit_clock(none);
    CHECK(!f0.ok && f0.n == 0);

    std::vector<sb::Bracket> one{{1000, 2000, 5000}};
    CHECK(!sb::fit_clock(one).ok);

    // invalid samples (g == 0, t1 < t0) are ignored
    std::vector<sb::Bracket> invalid{{1000, 2000, 0}, {3000, 2000, 7}, {1000, 900, 9}};
    CHECK(!sb::fit_clock(invalid).ok);

    // all brackets saw the same GPU reading: no rate information, must assume a = 1 and stay finite
    std::vector<sb::Bracket> same;
    for (int i = 0; i < 100; i++) same.push_back({1000000 + i, 1001000 + i, 42});
    sb::ClockFit fs = sb::fit_clock(same);
    CHECK(fs.a == 1.0);
    CHECK(std::isfinite(fs.b) && std::isfinite(fs.host_of(42)));
    CHECK(fs.host_of(42) >= 1000000.0 && fs.host_of(42) <= 1001100.0);
    CHECK(!fs.ok || std::isfinite(fs.eps_ns));

    // two samples: exact line through the midpoints
    std::vector<sb::Bracket> two{{1000, 1010, 100}, {2000, 2010, 1100}};
    sb::ClockFit f2 = sb::fit_clock(two);
    CHECK(f2.ok && std::fabs(f2.a - 1.0) < 1e-12 && std::fabs(f2.host_of(600) - 1505.0) < 1e-6);

    // small n: min_keep (50) wins over 1%
    Truth tr;
    auto br = make_brackets(tr, 1000, 60e9, 11);
    sb::ClockFit fk = sb::fit_clock(br);
    CHECK(fk.ok && fk.n_kept >= 50 && fk.n_kept < 100);

    // json is produced and has no NaN tokens even for a failed fit
    std::string js = f0.json();
    CHECK(js.find("nan") == std::string::npos && js.find("\"ok\":false") != std::string::npos);
}

int main() {
    for (uint64_t seed : {1ULL, 2ULL, 3ULL}) test_known_mapping(37.0, seed);
    test_known_mapping(-37.0, 4);
    test_known_mapping(0.0, 5);
    test_known_mapping(250.0, 6);
    test_symmetric_offset();
    test_round_trip();
    test_degenerate();
    printf("test_clock_fit: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
