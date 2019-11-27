#include"uct_hash_table.hpp"

Index UctHashTable::searchEmptyIndex(const ShogiPosition& pos) {
    auto hash = pos.hashValue();
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

Index UctHashTable::findSameHashIndex(const ShogiPosition& pos) {
    auto hash = pos.hashValue();
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

void UctHashTable::saveUsedHash(ShogiPosition& pos, Index index) {
    table_[index].age = age_;
    used_num_++;

    auto& current_node = table_[index];
    auto& child_indices = current_node.child_indices;
    for (uint64_t i = 0; i < current_node.moves.size(); i++) {
        if (child_indices[i] != NOT_EXPANDED && table_[child_indices[i]].age != age_) {
            pos.doMove(current_node.moves[i]);
            saveUsedHash(pos, child_indices[i]);
            pos.undo();
        }
    }
}

void UctHashTable::deleteOldHash(ShogiPosition& root, bool leave_root) {
    uint64_t root_index = findSameHashIndex(root);

    used_num_ = 0;
    age_++;

    if (root_index != table_.size()) { //見つかったということ
        saveUsedHash(root, root_index);

        if (!leave_root) {
            //root_indexのところは初期化
            table_[root_index].age = age_ - 1;
            used_num_--;
        }
    }
}