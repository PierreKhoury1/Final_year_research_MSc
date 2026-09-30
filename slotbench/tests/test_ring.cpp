// Host test for common/spsc_ring.h: capacity validation, single-thread order and overflow counting,
// a 10M-item producer/consumer run (ordered, nothing lost or duplicated, overflow count exact), a paced
// run with a collector-style consumer (sleeps 1 ms when empty) that must see zero overflows, and a
// stalled consumer that must produce overflows without corrupting what was accepted.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

#include "host_time.h"
#include "record.h"
#include "spsc_ring.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(c) do { if (c) g_pass++; else { g_fail++; fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

static void test_capacity() {
    for (size_t bad : {0, 1, 3, 6, 1000, 65535}) {
        bool threw = false;
        try { sb::SpscRing<int> r(bad); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw);
    }
    for (size_t good : {2, 4, 1024, 65536}) {
        bool threw = false;
        try {
            sb::SpscRing<int> r(good);
            CHECK(r.capacity() == good);
        } catch (...) { threw = true; }
        CHECK(!threw);
    }
}

static void test_single_thread() {
    sb::SpscRing<uint64_t> r(1024);
    CHECK(r.size() == 0);
    uint64_t v;
    CHECK(!r.pop(v));
    int accepted = 0;
    for (uint64_t i = 0; i < 5000; i++) accepted += r.push(i) ? 1 : 0;
    CHECK(accepted == 1024);
    CHECK(r.overflows() == 5000 - 1024);
    CHECK(r.size() == 1024);
    CHECK(r.pushed() == 1024);
    std::vector<uint64_t> out(4096);
    size_t n = r.pop_many(out.data(), 100);
    CHECK(n == 100);
    n += r.pop_many(out.data() + 100, out.size() - 100);
    CHECK(n == 1024);
    bool ordered = true;
    for (size_t i = 0; i < n; i++) ordered &= out[i] == i;
    CHECK(ordered);
    CHECK(r.size() == 0);
    // wrap-around many times with interleaved push/pop
    bool ok = true;
    uint64_t next_in = 0, next_out = 0;
    for (int round = 0; round < 10000; round++) {
        for (int j = 0; j < 7; j++) ok &= r.push(next_in++);
        for (int j = 0; j < 7; j++) ok &= r.pop(v) && v == next_out++;
    }
    CHECK(ok);
    CHECK(r.overflows() == 5000 - 1024);
}

// Producer retries rejected items so the consumer must see the complete sequence in order;
// every rejection must be counted once in overflows().
static void test_threaded_10m() {
    const uint64_t N = 10000000;
    sb::SpscRing<uint64_t> r(65536);
    std::atomic<bool> bad{false};
    std::atomic<uint64_t> received{0};
    std::thread cons([&] {
        std::vector<uint64_t> buf(4096);
        uint64_t expect = 0;
        while (expect < N) {
            size_t n = r.pop_many(buf.data(), buf.size());
            for (size_t i = 0; i < n; i++)
                if (buf[i] != expect++) bad = true;
            if (!n) sb::cpu_relax();
        }
        received = expect;
    });
    uint64_t rejected = 0;
    for (uint64_t i = 0; i < N; i++)
        while (!r.push(i)) { rejected++; sb::cpu_relax(); }
    cons.join();
    CHECK(!bad);
    CHECK(received == N);
    CHECK(r.overflows() == rejected);
    CHECK(r.pushed() == N);
    CHECK(r.size() == 0);
    printf("  10M items ordered, %llu retried pushes counted as overflows\n", (unsigned long long)rejected);
}

// Collector-style consumer (sleep 1 ms when empty) keeping up with a paced producer of SlotRecords.
static void test_paced_no_loss() {
    const uint64_t N = 1000000;
    const int64_t pace_ns = 1000;  // 1 record per us: 500x the real rate of one record per 500 us slot
    sb::SpscRing<sb::SlotRecord> r(65536);
    std::atomic<bool> stop{false}, bad{false};
    uint64_t got = 0;
    std::thread cons([&] {
        std::vector<sb::SlotRecord> buf(4096);
        uint64_t expect = 0;
        for (;;) {
            bool s = stop.load(std::memory_order_acquire);
            size_t n = r.pop_many(buf.data(), buf.size());
            for (size_t i = 0; i < n; i++) {
                const sb::SlotRecord &x = buf[i];
                if (x.slot != expect || x.t0 != (int64_t)expect * 3 || x.g1 != expect + 7) bad = true;
                expect++;
            }
            if (n) continue;
            if (s) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        got = expect;
    });
    int64_t t = sb::now_ns();
    for (uint64_t i = 0; i < N; i++) {
        t += pace_ns;
        sb::spin_until(t);
        sb::SlotRecord rec{i, 1, 2, (int64_t)i * 3, 4, 5, 6, i + 7};
        r.push(rec);
    }
    stop.store(true, std::memory_order_release);
    cons.join();
    CHECK(r.overflows() == 0);
    CHECK(got == N);
    CHECK(!bad);
    printf("  paced 1M records: received %llu, overflows %llu\n", (unsigned long long)got,
           (unsigned long long)r.overflows());
}

// Consumer stalls: overflows are counted, accepted items stay ordered, accepted + overflows == attempts.
static void test_stalled_consumer() {
    const uint64_t N = 200000;
    sb::SpscRing<uint64_t> r(4096);
    std::atomic<bool> go{false}, stop{false}, bad{false};
    uint64_t got = 0;
    std::thread cons([&] {
        while (!go.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // stall while the producer floods
        std::vector<uint64_t> buf(1024);
        int64_t last = -1;
        for (;;) {
            bool s = stop.load(std::memory_order_acquire);
            size_t n = r.pop_many(buf.data(), buf.size());
            for (size_t i = 0; i < n; i++) {
                if ((int64_t)buf[i] <= last) bad = true;
                last = (int64_t)buf[i];
            }
            got += n;
            if (n) continue;
            if (s) break;
            sb::cpu_relax();
        }
    });
    go = true;
    uint64_t accepted = 0;
    for (uint64_t i = 0; i < N; i++) accepted += r.push(i) ? 1 : 0;
    stop.store(true, std::memory_order_release);
    cons.join();
    CHECK(r.overflows() > 0);
    CHECK(accepted + r.overflows() == N);
    CHECK(got == accepted);
    CHECK(!bad);
    printf("  stalled consumer: accepted %llu, overflows %llu\n", (unsigned long long)accepted,
           (unsigned long long)r.overflows());
}

int main() {
    test_capacity();
    test_single_thread();
    test_threaded_10m();
    test_paced_no_loss();
    test_stalled_consumer();
    printf("test_ring: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
