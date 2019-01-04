#include"uct_hash_table.hpp"

UctHashTable::UctHashTable(int64_t hash_size) : age_(1) {
    setSize(hash_size);
}

void UctHashTable::setSize(int64_t megabytes) {
    int64_t bytes = megabytes * 1024 * 1024;
    size_ = 1ll << MSB64(bytes / sizeof(UctHashEntry));
    uct_hash_limit_ = size_ * 9 / 10;
    used_ = 0;
    enough_size_ = true;
    table_.resize((unsigned long)size_);
}

Index UctHashTable::searchEmptyIndex(int64_t hash, int16_t turn_number) {
    auto key = hashToIndex(hash);
    auto i = key;
    while (true) {
        if (table_[i].age != age_) {
            table_[i].hash = hash;
            table_[i].turn_number = turn_number;
            table_[i].age = age_;
            if (++used_ > uct_hash_limit_) {
                enough_size_ = false;
            }
            return i;
        }

        i++;
        if (i >= size_) {
            i = 0;
        }
        if (i == key) {
            return (Index)size_;
        }
    }
}

Index UctHashTable::findSameHashIndex(int64_t hash, int16_t turn_number) {
    auto key = hashToIndex(hash);
    auto i = key;
    while (true) {
        if (table_[i].age != age_) {
            return (Index)size_;
        } else if (table_[i].hash == hash
            && table_[i].turn_number == turn_number) {
            return i;
        }

        i++;
        if (i >= size_) {
            i = 0;
        }
        if (i == key) {
            return (Index)size_;
        }
    }
}

void UctHashTable::saveUsedHash(Position& pos, Index index) {
    table_[index].age = age_;
    used_++;

    auto& current_node = table_[index];
    auto& child_indices = current_node.child_indices;
    for (int32_t i = 0; i < current_node.child_num; i++) {
        if (child_indices[i] != NOT_EXPANDED && table_[child_indices[i]].age != age_) {
            pos.doMove(current_node.legal_moves[i]);
            saveUsedHash(pos, child_indices[i]);
            pos.undo();
        }
    }
}

void UctHashTable::deleteOldHash(Position& root, bool leave_root) {
    auto root_index = findSameHashIndex(root.hash_value(), (int16_t)root.turn_number());

    used_ = 0;
    age_++;
    //for (int i = 0; i < size_; i++) {
    //    table_[i].flag = false;
    //}

    if (leave_root && root_index != size_) { //見つかったということ
        saveUsedHash(root, root_index);
    }

    enough_size_ = true;
}