// GPU %globaltimer -> host CLOCK_MONOTONIC_RAW mapping from bracket samples (DESIGN.md section 6).
// A bracket (t0, t1, g) says the GPU read g at some host instant inside [t0, t1]. Keeping only the
// tightest brackets and fitting host_mid = a * (g - g_ref) + b gives the mapping; the bound eps says
// how far the mapping can be from the truth (half the typical kept width plus any bracket it misses).
// Regression runs on centred values so 6-hour spans (1e13 ns) keep sub-ns precision in doubles.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "json_writer.h"

namespace sb {

struct Bracket {
    int64_t t0;  // host ns before the request
    int64_t t1;  // host ns after the answer arrived
    uint64_t g;  // GPU %globaltimer ns read in between
};

struct ClockFit {
    bool ok = false;
    double a = 1.0;        // host ns per GPU ns
    double b = 0.0;        // host ns offset from t_ref at g == g_ref
    uint64_t g_ref = 0;    // GPU reference reading
    int64_t t_ref = 0;     // host reference time
    double eps_ns = NAN;   // mapping error bound
    double resid_rms_ns = NAN, resid_max_ns = NAN;  // over kept brackets, |mapped - midpoint|
    double min_width_ns = NAN, median_width_ns = NAN, kept_median_width_ns = NAN;
    int n = 0, n_kept = 0;

    // Host ns (as double, relative to nothing: absolute CLOCK_MONOTONIC_RAW) for a GPU reading.
    double host_of(uint64_t g) const {
        double x = (double)(int64_t)(g - g_ref);
        return (double)t_ref + b + a * x;
    }
    // GPU reading expected at a host time.
    uint64_t gpu_of(int64_t t) const {
        double x = ((double)(t - t_ref) - b) / a;
        return g_ref + (uint64_t)(int64_t)std::llround(x);
    }
    double rate_ppm() const { return (a - 1.0) * 1e6; }

    std::string json() const {
        Json j;
        j.add("ok", ok).add("a", a).add("rate_ppm", rate_ppm()).add("b_ns", b)
            .add("g_ref", (unsigned long long)g_ref).add("t_ref", (long long)t_ref)
            .add("eps_ns", eps_ns).add("resid_rms_ns", resid_rms_ns).add("resid_max_ns", resid_max_ns)
            .add("min_width_ns", min_width_ns).add("median_width_ns", median_width_ns)
            .add("kept_median_width_ns", kept_median_width_ns).add("n", n).add("n_kept", n_kept);
        return j.str();
    }
};

// keep_frac: fraction of tightest brackets used (at least min_keep, at most all).
inline ClockFit fit_clock(const std::vector<Bracket> &br, double keep_frac = 0.01, int min_keep = 50) {
    ClockFit f;
    f.n = (int)br.size();
    std::vector<size_t> valid;
    valid.reserve(br.size());
    for (size_t i = 0; i < br.size(); i++)
        if (br[i].t1 >= br[i].t0 && br[i].g != 0) valid.push_back(i);
    if (valid.size() < 2) return f;

    std::vector<int64_t> widths;
    widths.reserve(valid.size());
    for (size_t i : valid) widths.push_back(br[i].t1 - br[i].t0);
    std::vector<int64_t> ws = widths;
    std::sort(ws.begin(), ws.end());
    f.min_width_ns = (double)ws.front();
    f.median_width_ns = (double)ws[ws.size() / 2];

    size_t keep = (size_t)std::ceil(keep_frac * (double)valid.size());
    keep = std::max(keep, (size_t)std::max(min_keep, 2));
    keep = std::min(keep, valid.size());
    int64_t thr = ws[keep - 1];

    std::vector<size_t> kept;
    for (size_t j = 0; j < valid.size(); j++)
        if (widths[j] <= thr) kept.push_back(valid[j]);
    // ties at the threshold can keep more than `keep`; that is fine
    if (kept.size() < 2) return f;

    f.g_ref = br[kept[0]].g;
    f.t_ref = br[kept[0]].t0;
    auto X = [&](size_t i) { return (double)(int64_t)(br[i].g - f.g_ref); };
    auto Y = [&](size_t i) { return 0.5 * (double)(br[i].t0 - f.t_ref) + 0.5 * (double)(br[i].t1 - f.t_ref); };

    double mx = 0, my = 0;
    for (size_t i : kept) { mx += X(i); my += Y(i); }
    mx /= (double)kept.size();
    my /= (double)kept.size();
    double sxx = 0, sxy = 0;
    for (size_t i : kept) {
        double dx = X(i) - mx, dy = Y(i) - my;
        sxx += dx * dx;
        sxy += dx * dy;
    }
    f.a = sxx > 0 ? sxy / sxx : 1.0;  // all samples at one GPU instant: assume equal rates
    f.b = my - f.a * mx;

    double ss = 0, rmax = 0, viol = 0;
    std::vector<double> kw;
    for (size_t i : kept) {
        double pred = f.b + f.a * X(i);
        double lo = (double)(br[i].t0 - f.t_ref), hi = (double)(br[i].t1 - f.t_ref);
        double r = pred - Y(i);
        ss += r * r;
        rmax = std::max(rmax, std::fabs(r));
        viol = std::max(viol, std::max(lo - pred, pred - hi));
        kw.push_back(hi - lo);
    }
    std::sort(kw.begin(), kw.end());
    f.n_kept = (int)kept.size();
    f.resid_rms_ns = std::sqrt(ss / (double)kept.size());
    f.resid_max_ns = rmax;
    f.kept_median_width_ns = kw[kw.size() / 2];
    f.eps_ns = f.kept_median_width_ns / 2 + std::max(viol, 0.0);
    f.ok = true;
    return f;
}

}  // namespace sb
