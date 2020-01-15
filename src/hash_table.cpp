#include"hash_table.hpp"

Index HashTable::searchEmptyIndex(const Position& pos) {
    int64_t hash = pos.hashValue();
    uint64_t key = hashToIndex(hash);
    uint64_t i = key;
    while (true) {
        if (table_[i].age != age_) {
            table_[i].hash = hash;
            table_[i].turn_number = static_cast<int16_t>(pos.turnNumber());
            table_[i].age = age_;
            used_num_++;
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

Index HashTable::findSameHashIndex(const Position& pos) {
    int64_t hash = pos.hashValue();
    uint64_t key = hashToIndex(hash);
    uint64_t i = key;
    while (true) {
        if (table_[i].age != age_) {
            return (Index)table_.size();
        } else if (table_[i].hash == hash
            && table_[i].turn_number == pos.turnNumber()) {
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

void HashTable::saveUsedHash(Position& pos, Index index) {
    table_[index].age = age_;
    used_num_++;

    HashEntry& current_node = table_[index];
    std::vector<Index>& child_indices = current_node.child_indices;
    for (uint64_t i = 0; i < current_node.moves.size(); i++) {
        if (child_indices[i] != NOT_EXPANDED && table_[child_indices[i]].age != age_) {
            pos.doMove(current_node.moves[i]);
            saveUsedHash(pos, child_indices[i]);
            pos.undo();
        }
    }
}

void HashTable::deleteOldHash(Position& next_root, bool leave_root) {
    uint64_t next_root_index = findSameHashIndex(next_root);

    used_num_ = 0;
    age_++;

    if (next_root_index != table_.size()) { //見つかったということ
        saveUsedHash(next_root, next_root_index);

        if (!leave_root) {
            //root_indexのところは初期化
            table_[next_root_index].age = age_ - 1;
            used_num_--;
        }
    }
}

ValueType HashTable::QfromNextValue(const HashEntry& node, int32_t i) const {
    //展開されていない場合は基本的にMIN_SCOREで扱うが、探索回数が0でないときは詰み探索がそこへ詰みありと言っているということなのでMAX_SCOREを返す
#ifdef USE_CATEGORICAL
    if (node.child_indices[i] == HashTable::NOT_EXPANDED) {
        return (node.N[i] == 0 ? onehotDist(MIN_SCORE) : onehotDist(MAX_SCORE));
    }
    ValueType v = table_[node.child_indices[i]].value;
    std::reverse(v.begin(), v.end());
    return v;
#else
    if (node.child_indices[i] == HashTable::NOT_EXPANDED) {
        return (node.N[i] == 0 ? MIN_SCORE : MAX_SCORE);
    }
    return MAX_SCORE + MIN_SCORE - table_[node.child_indices[i]].value;
#endif
}

FloatType HashTable::expQfromNext(const HashEntry& node, int32_t i) const {
#ifdef USE_CATEGORICAL
    return expOfValueDist(QfromNextValue(node, i));
#else
    return QfromNextValue(node, i);
#endif
}