#ifndef HASH_FOR_MCTS_NET_TABLE_HPP
#define HASH_FOR_MCTS_NET_TABLE_HPP

#include"../../common.hpp"
#include"../../include_switch.hpp"
#include<torch/torch.h>

using Index = int32_t;

struct HashEntryForMCTSNet {
    //埋め込みベクトル
    torch::Tensor embedding_vector;

    //局面に対応するハッシュ値
    int64_t hash;

    //手数。ハッシュ値だけでは衝突頻度が高いので
    uint16_t turn_number;

    //置換表のageと照らし合わせて現在の情報かどうかを判断する変数
    uint16_t age;
};

class HashTableForMCTSNet {
public:
    explicit HashTableForMCTSNet(int64_t hash_size) : root_index(HashTableForMCTSNet::NOT_EXPANDED), used_num_(0), age_(1),
                                            table_(1ull << (MSB64(hash_size) + 1)) {}

    //取得用のオペレータ
    HashEntryForMCTSNet& operator[](Index i) {
        return table_[i];
    }
    const HashEntryForMCTSNet& operator[](Index i) const {
        return table_[i];
    }

    //未使用のインデックスを探して返す関数(開番地法)
    Index searchEmptyIndex(const Position& pos);

    //この局面に対応するインデックスがあるか調べて返す関数
    Index findSameHashIndex(const Position& pos);

    //次にルートとなる局面及びそこから到達できる局面だけ残してそれ以外を削除する関数
    void deleteOldHash();

    //サイズを返す関数。searchEmptyIndexなどはダメだったときに置換表サイズを返すので、この関数の返り値と比較して適切なものが返ってきたか判定する
    uint64_t size() {
        return table_.size();
    }

    //未展開のノードのインデックス
    static constexpr Index NOT_EXPANDED = -1;

    //現在思考しているノードに相当するエントリーのインデックス
    Index root_index;

private:
    //局面のハッシュ値から置換表におけるキーへ変換する。この処理が最適かは不明
    Index hashToIndex(int64_t hash) {
        return (Index)(((hash & 0xffffffff) ^ ((hash >> 32) & 0xffffffff)) & (table_.size() - 1));
    }

    //使用されている数
    uint64_t used_num_;

    //情報の世代。この世代をインクリメントすることが置換表全体をclearすることに相当する
    uint16_t age_;

    //置換表本体
    std::vector<HashEntryForMCTSNet> table_;
};

#endif