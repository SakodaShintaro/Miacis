#include "segment_tree.hpp"
#include "../common.hpp"

SegmentTree::SegmentTree(uint64_t n) : next_push_index_(-1) {
    n_ = 1ull << MSB64(n);
    sum_.resize(2 * n_ - 1, 0);
    min_.resize(2 * n_ - 1, 0);
}

void SegmentTree::update(uint64_t x, float v) {
    sum_[x + n_ - 1] = v;
    min_[x + n_ - 1] = v;
    for (uint64_t i = (x + n_ - 2) / 2;; i = (i - 1) / 2) {
        sum_[i] = sum_[2 * i + 1] + sum_[2 * i + 2];
        min_[i] = std::min(min_[2 * i + 1], min_[2 * i + 2]);
        if (i == 0) {
            break;
        }
    }
}

uint64_t SegmentTree::getIndex(float value, uint64_t k) {
    if (k >= n_ - 1) {
        //最下段まで来ていたらindexを返す
        return k - (n_ - 1);
    }
    return (value <= sum_[2 * k + 1] ? getIndex(value, 2 * k + 1) : getIndex(value - sum_[2 * k + 1], 2 * k + 2));
}

uint64_t SegmentTree::getIndexToPush(uint64_t k) {
    return (++next_push_index_) %= n_;

    //最も小さいpriorityを持つもの
    if (k >= n_ - 1) {
        //最下段まで来ていたらindexを返す
        return k - (n_ - 1);
    }
    return (min_[2 * k + 1] <= min_[2 * k + 2] ? getIndexToPush(2 * k + 1) : getIndexToPush(2 * k + 2));
}

float SegmentTree::getSum() { return sum_.front(); }

float SegmentTree::operator[](uint64_t i) { return sum_[i + n_ - 1]; }

void SegmentTree::print() const {
    for (uint64_t i = 0; i < n_; i++) {
        printf("%4.1f%c", sum_[i + n_ - 1], " \n"[i == n_ - 1]);
    }
}