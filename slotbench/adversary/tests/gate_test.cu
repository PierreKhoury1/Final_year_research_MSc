// CPU-only checks of the time-aware gate: synthetic "units" (busy spins) against a 500 us timetable.
// Build: nvcc -O2 -std=c++17 -Icommon -Itrace -o gate_test adversary/tests/gate_test.cu -lcublas
#define ADVERSARY_NO_MAIN
#include "../adversary.cu"
#include <cstdlib>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #c); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while (0)

struct Result { int units = 0, violations = 0; double per_slot = 0; };

// unit_ns(i) gives the duration of the i-th unit (lets a test inject one slow unit).
template <class UnitNs>
static Result run(bool observe, int64_t P, int64_t busy, int n_slots, UnitNs unit_ns) {
    const char *path = "/tmp/sb_gate_test.tt";
    unlink(path);
    g_gate = Gate{};
    g_gate.enabled = true; g_gate.observe = observe; g_gate.path = path; g_gate.guard_ns = 30000;
    g_gate.start_ns = sb::now_ns();
    for (int i = 0; i < 8; i++) g_gate.dur[g_gate.ndur++] = unit_ns(-1);
    sb::SlotTimetable tt{};
    memcpy(tt.magic, "SBTT0001", 8);
    tt.t0 = sb::now_ns() + 5000000; tt.period_ns = P; tt.busy_ns = busy; tt.n_slots = n_slots;
    tt.written_ns = sb::now_ns();
    bool wrote = sb::write_timetable(path, tt);
    CHECK(wrote, "write_timetable");
    auto sleep_to = [](int64_t t) { sb::sleep_until_raw(t); };
    Result r;
    const int64_t end = tt.t0 + tt.n_slots * P + 2000000;
    for (int i = 0; sb::now_ns() < end; i++) {
        g_gate.before(sleep_to);
        int64_t a = sb::now_ns();
        sb::spin_until(a + unit_ns(i));
        int64_t b = sb::now_ns();
        g_gate.after(a, b);
        if (g_gate.loaded && a >= tt.t0 && a < tt.t0 + tt.n_slots * P) {
            r.units++;
            int64_t k = (a - tt.t0) / P, Sk = tt.t0 + k * P;
            r.violations += a < Sk + busy || (k + 1 < tt.n_slots && b > Sk + P);
        }
    }
    unlink(path);
    r.per_slot = r.units / (double)n_slots;
    CHECK(g_gate.loaded, "timetable not loaded");
    CHECK((int)g_gate.units_in_window == r.units, "units_in_window %llu != %d", g_gate.units_in_window, r.units);
    return r;
}

int main() {
    const int64_t P = 500000, busy = 200000;
    auto unit100 = [](int) -> int64_t { return 100000; };
    // 1. gated: 2 units fit per 300 us gap (100 us units, est 110, guard 30); none may violate the reservation
    Result g = run(false, P, busy, 200, unit100);
    printf("gated:   units=%d per_slot=%.2f violations=%d fits=%d\n", g.units, g.per_slot, g.violations, g_gate.fits_at_load);
    CHECK(g.violations <= 2, "gated violations %d (only host preemption may cause any)", g.violations);
    CHECK(g.per_slot > 1.7, "gated per_slot %.2f, expected about 2", g.per_slot);
    CHECK(g_gate.fits_at_load, "should fit");
    // 2. observe: never waits, about 5 units per slot, most of them overlap the reserved time
    Result o = run(true, P, busy, 200, unit100);
    printf("observe: units=%d per_slot=%.2f violations=%d waits=%llu\n", o.units, o.per_slot, o.violations, g_gate.waits);
    CHECK(g_gate.waits == 0, "observe waited");
    CHECK(o.per_slot > 4.5, "observe per_slot %.2f", o.per_slot);
    CHECK(g_gate.overrun_units > 0 && o.violations > 0, "observe should overlap the reservation");
    // 3. reservation longer than the period (the 2026-10-04 bug): gate never opens and says so
    Result z = run(false, P, 600000, 100, unit100);
    printf("busy>P:  units=%d fits=%d\n", z.units, g_gate.fits_at_load);
    CHECK(z.units == 0 && !g_gate.fits_at_load, "busy > period must report fits=0 and run nothing");
    // 4. one slow unit (400 us) must not shut the gate for the rest of the window
    Result s = run(false, P, busy, 200, [](int i) -> int64_t { return i == 40 ? 400000 : 100000; });
    printf("latch:   units=%d per_slot=%.2f est_end=%.1f us\n", s.units, s.per_slot, g_gate.est_window_us);
    CHECK(s.per_slot > 1.6, "a single slow unit latched the gate (per_slot %.2f)", s.per_slot);
    // 5. several slow but clean units (finish before the next slot) must not close the gate for good
    Result c = run(false, P, busy, 400, [](int i) -> int64_t { return i >= 100 && i < 106 ? 250000 : 100000; });
    printf("aging:   units=%d per_slot=%.2f aged=%llu\n", c.units, c.per_slot, g_gate.aged);
    CHECK(c.per_slot > 1.5, "slow clean units closed the gate (per_slot %.2f)", c.per_slot);
    if (g_fail) { fprintf(stderr, "%d check(s) failed\n", g_fail); return 1; }
    printf("all gate checks passed\n");
    return 0;
}
