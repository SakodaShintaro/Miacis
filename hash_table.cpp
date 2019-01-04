#include "hash_table.hpp"
#include "common.hpp"
#include <iostream>

static constexpr int64_t MAX_SEARCH_NUM = 5;

HashEntry* HashTable::find(int64_t hash_value) {
	//if (table_[hash_value & key_mask_].flag_ == false
 //       || table_[hash_value & key_mask_].hash_val != hash_value) return nullptr;
	//return &table_[hash_value & key_mask_];
    const int64_t key = hash_value & key_mask_;
    for (int64_t i = 0; i < MAX_SEARCH_NUM; i++) {
        if (table_[(key + i) % size_].flag_ && table_[(key + i) % size_].hash_val == hash_value) {
            return &table_[(key + i) % size_];
        }
    }
    return nullptr;
}

#ifdef USE_NN
void HashTable::save(int64_t hash_value, Move move, Score score, Depth depth, std::vector<Move> sorted_moves) {
    HashEntry* target = &table_[hash_value & key_mask_];
    target->hash_val = hash_value;
    target->best_move = move;
    target->depth = depth;
    target->best_move.score = score;
    target->sorted_moves = sorted_moves;
    if (!target->flag_) {
        target->flag_ = true;
        hashfull_++;
    }
}
#endif
void HashTable::save(int64_t hash_value, Move move, Score score, Depth depth) {
    const int64_t key = hash_value & key_mask_;
    HashEntry& target = table_[key];

    if (target.flag_) {
        //flagがオフなものを探しつつ,一番消して良さそうなものを保持しておく
        //とりあえず探索深さの小さいものから消して良いとする
        for (int64_t i = 0; i < MAX_SEARCH_NUM; i++) {
            if (!table_[(key + i) % size_].flag_) {
                target = table_[(key + i) % size_];
                break;
            } else if (table_[(key + i) % size_].depth < target.depth) {
                target = table_[(key + i) % size_];
            }
        }
    }

	target.hash_val = hash_value;
	target.best_move = move;
    target.best_move.score = score;
	target.depth = depth;
	if (!target.flag_) {
		target.flag_ = true;
		hashfull_++;
    }
}

void HashTable::setSize(int64_t megabytes) {
    int64_t bytes = megabytes * 1024 * 1024;
    size_ = (megabytes > 0 ? (1ull << MSB64(bytes / sizeof(HashEntry))) : 0);
    age_ = 0;
    key_mask_ = size_ - 1;
    hashfull_ = 0;
	table_.resize(size_);
}