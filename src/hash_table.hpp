#ifndef HASH_TABLE_HPP
#define HASH_TABLE_HPP

#include "common.hpp"
#include "include_switch.hpp"
#include "neural_network.hpp"

using Index = int32_t;

struct HashEntry {
    //探索回数の合計
    int32_t sum_N;

    //まだGPU評価されてない探索回数の合計
    int32_t virtual_sum_N;

    //以下vectorとなっているものは全てmovesのサイズと同じサイズになる
    //行動
    std::vector<Move> moves;

    //各行動を取った後の置換表エントリに対応するインデックス
    std::vector<Index> child_indices;

    //各行動を探索した回数
    std::vector<int32_t> N;

    //まだGPU評価されてない各行動を探索した回数
    std::vector<int32_t> virtual_N;

    //ニューラルネットワークの出力方策
    std::vector<FloatType> nn_policy;

    //価値。漸進的に更新され、常にこのノードを根とする部分木内の価値の平均となる
    ValueType value;

    //ニューラルネットワークによる評価が行われたかを示すフラグ
    bool evaled;

    //排他制御用のmutex
    std::mutex mutex;

    //局面に対応するハッシュ値
    int64_t hash;

    //手数。ハッシュ値だけでは衝突頻度が高いので
    uint16_t turn_number;

    //置換表のageと照らし合わせて現在の情報かどうかを判断する変数
    uint16_t age;

    HashEntry() : sum_N(0), virtual_sum_N(0), value{}, evaled(false), hash(0), turn_number(0), age(0) {}
};

class HashTable {
public:
    explicit HashTable(int64_t hash_size)
        : root_index(HashTable::NOT_EXPANDED), used_num_(0), age_(1), table_(1ull << (MSB64(hash_size) + 1)) {}

    //取得用のオペレータ
    HashEntry& operator[](Index i) { return table_[i]; }
    const HashEntry& operator[](Index i) const { return table_[i]; }

    //未使用のインデックスを探して返す関数(開番地法)
    Index searchEmptyIndex(const Position& pos);

    //この局面に対応するインデックスがあるか調べて返す関数
    Index findSameHashIndex(const Position& pos);

    //posの局面をindexへ保存する関数
    void saveUsedHash(Position& pos, Index index);

    //次にルートとなる局面及びそこから到達できる局面だけ残してそれ以外を削除する関数
    void deleteOldHash(Position& next_root, bool leave_root);

    //node局面におけるi番目の指し手の行動価値を返す関数
    //Scalarのときは実数を一つ、Categoricalのときは分布を返す
    ValueType QfromNextValue(const HashEntry& node, int32_t i) const;

    //node局面におけるi番目の指し手の行動価値(期待値)を返す関数
    //Scalarのときは実数をそのまま返し、Categoricalのときはその期待値を返す
    FloatType expQfromNext(const HashEntry& node, int32_t i) const;

    //置換表の利用率を返す関数。USI対応GUIだと表示できるので
    double getUsageRate() const { return (double)used_num_ / table_.size(); }

    //置換表にmargin個以上の空きがあるかどうかを判定する関数
    bool hasEmptyEntries(int64_t margin) { return used_num_ + margin <= table_.size(); }

    //サイズを返す関数。searchEmptyIndexなどはダメだったときに置換表サイズを返すので、この関数の返り値と比較して適切なものが返ってきたか判定する
    uint64_t size() { return table_.size(); }

    //未展開のノードのインデックス
    static constexpr Index NOT_EXPANDED = -1;

    //現在思考しているノードに相当するエントリーのインデックス
    Index root_index;

private:
    //局面のハッシュ値から置換表におけるキーへ変換する。この処理が最適かは不明
    Index hashToIndex(int64_t hash) { return (Index)(((hash & 0xffffffff) ^ ((hash >> 32) & 0xffffffff)) & (table_.size() - 1)); }

    //使用されている数
    uint64_t used_num_;

    //情報の世代。この世代をインクリメントすることが置換表全体をclearすることに相当する
    uint16_t age_;

    //置換表本体
    std::vector<HashEntry> table_;
};

#endif