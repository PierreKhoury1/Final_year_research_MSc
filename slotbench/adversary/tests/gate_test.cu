// CPU-only check of the time-aware gate: synthetic 100 us "units" against a 500 us timetable.
#define ADVERSARY_NO_MAIN
#include "../adversary.cu"
#include <cassert>

int main() {
    const char *path = "/tmp/sb_gate_test.tt";
    unlink(path);
    const int64_t P = 500000, busy = 200000, unit = 100000;
    for (int observe = 0; observe < 2; observe++) {
        g_gate = Gate{};
        g_gate.enabled = true; g_gate.observe = observe; g_gate.path = path; g_gate.guard_ns = 30000;
        g_gate.start_ns = sb::now_ns();
        for (int i = 0; i < 8; i++) g_gate.dur[g_gate.ndur++] = unit;
        sb::SlotTimetable tt{};
        memcpy(tt.magic, "SBTT0001", 8);
        tt.t0 = sb::now_ns() + 5000000; tt.period_ns = P; tt.busy_ns = busy; tt.n_slots = 200;
        tt.written_ns = sb::now_ns();
        assert(sb::write_timetable(path, tt));
        auto sleep_to = [](int64_t t) { sb::sleep_until_raw(t); };
        int violations = 0, units = 0;
        const int64_t end = tt.t0 + tt.n_slots * P + 2000000;
        while (sb::now_ns() < end) {
            g_gate.before(sleep_to);
            int64_t a = sb::now_ns();
            sb::spin_until(a + unit);
            int64_t b = sb::now_ns();
            g_gate.after(a, b);
            if (g_gate.loaded && a >= tt.t0 && a < tt.t0 + tt.n_slots * P) {
                units++;
                int64_t k = (a - tt.t0) / P, Sk = tt.t0 + k * P;
                bool bad = a < Sk + busy || (k + 1 < tt.n_slots && b > Sk + P);
                violations += bad;
            }
        }
        printf("%s: loaded=%d units_in_window=%llu (%d) overrun_units=%llu violations=%d waits=%llu per_slot=%.2f\n",
               observe ? "observe" : "gated", g_gate.loaded, g_gate.units_in_window, units, g_gate.overrun_units,
               violations, g_gate.waits, units / 200.0);
        if (!observe) { assert(g_gate.loaded && units > 0); }
    }
    unlink(path);
}
