#include "hash_table_for_mcts_net.hpp"

Index HashTableForMCTSNet::searchEmptyIndex(const Position& pos) {
    int64_t hash = pos.hashValue();
    uint64_t key = hashToIndex(hash);
    uint64_t i = key;
    while (true) {
        if (table_[i].age != age_) {
            //世代が違うならここを上書きして良い
            table_[i].hash = hash;
            table_[i].turn_number = static_cast<int16_t>(pos.turnNumber());
            table_[i].age = age_;
            used_num_++;
            return i;
        }

        i++;

        //たぶんmodを取るより分岐の方が速かったはず
        if (i >= table_.size()) {
            i = 0;
        }

        //一周したのなら空きがなかったということなのでsize()を返す
        if (i == key) {
            return (Index)table_.size();
        }
    }
}

Index HashTableForMCTSNet::findSameHashIndex(const Position& pos) {
    int64_t hash = pos.hashValue();
    uint64_t key = hashToIndex(hash);
    uint64_t i = key;
    while (true) {
        if (table_[i].age != age_) {
            //根本的に世代が異なるなら同じものはないのでsize()を返す
            return (Index)table_.size();
        } else if (table_[i].hash == hash && table_[i].turn_number == pos.turnNumber()) {
            //完全に一致したのでここが記録されていたエントリ
            return i;
        }

        i++;
        if (i >= table_.size()) {
            i = 0;
        }
        if (i == key) {
            return (Index)table_.size();
        }
    }
}

void HashTableForMCTSNet::deleteOldHash() {
    //置換表全体を消去
    used_num_ = 0;
    age_++;
}