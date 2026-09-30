// Lock-free single-producer single-consumer ring. The producer (slot driver) never blocks:
// push() on a full ring drops the item and counts an overflow, which invalidates the run.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sb {

template <typename T>
class SpscRing {
public:
    explicit SpscRing(size_t capacity_pow2) : buf_(capacity_pow2), mask_(capacity_pow2 - 1) {
        if (capacity_pow2 < 2 || (capacity_pow2 & mask_) != 0)
            throw std::invalid_argument("SpscRing capacity must be a power of two >= 2");
    }

    // Producer only.
    bool push(const T &v) {
        uint64_t h = head_.load(std::memory_order_relaxed);
        uint64_t t = tail_.load(std::memory_order_acquire);
        if (h - t >= buf_.size()) {
            overflows_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        buf_[h & mask_] = v;
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    // Consumer only. Copies up to max items into out, returns the count.
    size_t pop_many(T *out, size_t max) {
        uint64_t t = tail_.load(std::memory_order_relaxed);
        uint64_t h = head_.load(std::memory_order_acquire);
        size_t n = (size_t)(h - t);
        if (n > max) n = max;
        for (size_t i = 0; i < n; i++) out[i] = buf_[(t + i) & mask_];
        tail_.store(t + n, std::memory_order_release);
        return n;
    }

    bool pop(T &out) { return pop_many(&out, 1) == 1; }

    size_t size() const {
        return (size_t)(head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire));
    }
    size_t capacity() const { return buf_.size(); }
    uint64_t overflows() const { return overflows_.load(std::memory_order_relaxed); }
    uint64_t pushed() const { return head_.load(std::memory_order_acquire); }

private:
    std::vector<T> buf_;
    const uint64_t mask_;
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) std::atomic<uint64_t> tail_{0};
    alignas(64) std::atomic<uint64_t> overflows_{0};
};

}  // namespace sb
