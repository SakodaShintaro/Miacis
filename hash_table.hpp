#ifndef HASH_TABLE_HPP
#define HASH_TABLE_HPP

#include"move.hpp"
#include<vector>
#include<mutex>

struct HashEntry {
    //ハッシュの値
    int64_t hash_val;

    //その局面における(仮の)最善手
    //評価値込み
    Move best_move;

    //残り探索深さ
    Depth depth;

    //登録されているか
    bool flag_;

#ifdef USE_NN
    std::vector<Move> sorted_moves;
#endif

    HashEntry() :
        hash_val(0),
        best_move(NULL_MOVE),
        depth(Depth(-1)),
        flag_(false) {}

    void print() {
        printf("hash_val = %llx\n", (unsigned long long)hash_val);
        printf("best_move = "); best_move.printWithScore();
        printf("depth = %d\n", depth);
        printf("flag_ = %s\n", flag_ ? "YES" : "NO");
    }
};

class HashTable{
public:
    HashTable() { setSize(64); }
	HashEntry* find(int64_t key);
#ifdef USE_NN
    void save(int64_t key, Move move, Score score, Depth depth, std::vector<Move> sorted_moves);
#endif
    void save(int64_t key, Move move, Score score, Depth depth);
    void setSize(int64_t megabytes);
	double hashfull() { return static_cast<double>(hashfull_) / static_cast<double>(size_) * 1000.0; }
	void clear() { 
        table_.clear();
    }
private:
    //ハッシュエントリのvector:これが本体
    std::vector<HashEntry> table_;

    //ハッシュテーブルの要素数
	//これtable.size()では?
	//上位ソフトだといくつかのエントリをまとめて一つにしてるからこれが別に必要なんだろうか
    //それにしてもそのまとめた数さえ保持しておけばいいんじゃないのか
    size_t size_;

    //ハッシュキーからテーブルのインデックスを求めるためのマスク
    size_t key_mask_;

    //使用済みエントリの数
    size_t hashfull_;

    //ハッシュテーブルに入っている情報の古さ
    uint8_t age_;
};

#endif