#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "geo.hpp"

// Max-heap that keeps the k best results seen so far, ordered by ResultOrder
// (dist_km, then id). The root is the worst of the kept results, so a new
// candidate only has to beat the root to get in. Shared by both trees.
class BoundedMaxHeap {
public:
    BoundedMaxHeap(size_t capacity, const std::vector<Location>& pts)
        : capacity_(capacity), less_{&pts} {
        items_.reserve(capacity);
    }

    size_t size() const { return items_.size(); }
    bool full() const { return items_.size() == capacity_; }

    // The worst kept result. Only valid when size() > 0.
    const Result& top() const { return items_[0]; }

    // O(log k). Returns true if the candidate was kept.
    bool offer(const Result& candidate) {
        if (capacity_ == 0) return false;
        if (!full()) {
            items_.push_back(candidate);
            siftUp(items_.size() - 1);
            return true;
        }
        if (!less_(candidate, items_[0])) return false;
        items_[0] = candidate;
        siftDown(0);
        return true;
    }

    // Empties the heap and returns its contents best-first.
    std::vector<Result> takeSorted() {
        std::vector<Result> sorted(items_.size());
        for (size_t i = sorted.size(); i > 0; --i) {
            sorted[i - 1] = items_[0];
            items_[0] = items_.back();
            items_.pop_back();
            if (!items_.empty()) siftDown(0);
        }
        return sorted;
    }

private:
    void siftUp(size_t i) {
        while (i > 0) {
            const size_t parent = (i - 1) / 2;
            if (!less_(items_[parent], items_[i])) break;
            std::swap(items_[parent], items_[i]);
            i = parent;
        }
    }

    void siftDown(size_t i) {
        const size_t n = items_.size();
        while (true) {
            const size_t left = 2 * i + 1;
            const size_t right = left + 1;
            size_t largest = i;
            if (left < n && less_(items_[largest], items_[left])) largest = left;
            if (right < n && less_(items_[largest], items_[right])) largest = right;
            if (largest == i) break;
            std::swap(items_[i], items_[largest]);
            i = largest;
        }
    }

    size_t capacity_;
    ResultOrder less_;
    std::vector<Result> items_;
};
