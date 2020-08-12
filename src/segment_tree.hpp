#ifndef MIACIS_SEGMENT_TREE_HPP
#define MIACIS_SEGMENT_TREE_HPP

#include <cstdint>
#include <vector>

//1点更新,区間和
class SegmentTree {
public:
    explicit SegmentTree(uint64_t n);

    void update(uint64_t x, float v);

    uint64_t getIndex(float value, uint64_t k = 0);

    uint64_t getIndexToPush(uint64_t k = 0);

    float getSum();

    float operator[](uint64_t i);

    void print() const;

private:
    //最下段,要素の数:2のべき乗
    uint64_t n_;

    //実際に保持している情報:2 * n_ - 1個の配列
    std::vector<float> sum_, min_;

    //次に入れる場所
    int64_t next_push_index_;
};

#endif //MIACIS_SEGMENT_TREE_HPP