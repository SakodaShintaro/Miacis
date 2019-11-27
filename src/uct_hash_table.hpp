#ifndef UCT_HASH_ENTRY_HPP
#define UCT_HASH_ENTRY_HPP

#include"neural_network.hpp"
#include"common.hpp"
#include"include_switch.hpp"

using Index = int32_t;

struct UctHashEntry {
    int32_t sum_N;
    int32_t virtual_sum_N;
    std::vector<Move> moves;
    std::vector<Index> child_indices;
    std::vector<int32_t> N;
    std::vector<int32_t> virtual_N;
    std::vector<FloatType> nn_policy;
    ValueType value;
    bool evaled;

    //識別用データ
    //ハッシュ値だけでは衝突が発生するので手数も持つ
    int64_t hash;
    uint16_t turn_number;

    uint16_t age;

#ifdef USE_CATEGORICAL
    UctHashEntry() :
        sum_N(0), virtual_sum_N(0), value({}),
        evaled(false), hash(0), turn_number(0), age(0) {}
#else
    UctHashEntry() :
        sum_N(0), virtual_sum_N(0), value(0.0),
        evaled(false), hash(0), turn_number(0), age(0) {}
#endif
};

class UctHashTable {
public:
    explicit UctHashTable(int64_t hash_size) : used_num_(0), age_(1), table_(1ull << (MSB64(hash_size) + 1)) {}

    UctHashEntry& operator[](Index i) {
        return table_[i];
    }

    const UctHashEntry& operator[](Index i) const {
        return table_[i];
    }

    //未使用のインデックスを探して返す関数(開番地法)
    Index searchEmptyIndex(const Position& pos);

    //この局面に対応するインデックスがあるか調べて返す関数
    Index findSameHashIndex(const Position& pos);

    //posの局面をindexへ保存する関数
    void saveUsedHash(Position& pos, Index index);

    //現在の局面,及びそこから到達できる局面以外を削除する関数
    void deleteOldHash(Position& root, bool leave_root);

    double getUsageRate() const {
        return (double)used_num_ / table_.size();
    }

    bool hasEnoughSize() {
        return used_num_ < table_.size();
    }

    uint64_t size() {
        return table_.size();
    }

    // 未展開のノードのインデックス
    static constexpr Index NOT_EXPANDED = -1;

private:
    Index hashToIndex(int64_t hash) {
        return (Index)(((hash & 0xffffffff) ^ ((hash >> 32) & 0xffffffff)) & (table_.size() - 1));
    }

    uint64_t used_num_;
    uint16_t age_;
    std::vector<UctHashEntry> table_;
};

#endif