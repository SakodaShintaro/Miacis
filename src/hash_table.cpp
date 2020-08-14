#include "hash_table.hpp"

Index HashTable::searchEmptyIndex(const Position& pos) {
    uint64_t hash = pos.hashValue();
    uint64_t key = hashToIndex(hash);
    uint64_t i = key;
    while (true) {
        std::unique_lock<std::mutex> lock(table_[i].mutex);
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

Index HashTable::findSameHashIndex(const Position& pos) {
    uint64_t hash = pos.hashValue();
    uint64_t key = hashToIndex(hash);
    uint64_t i = key;
    while (true) {
        std::unique_lock<std::mutex> lock(table_[i].mutex);
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

void HashTable::saveUsedHash(Position& pos, Index index) {
    //エントリの世代を合わせれば情報を持ち越すことができる
    table_[index].age = age_;
    used_num_++;

    //再帰的に子ノードを探索していく
    HashEntry& curr_node = table_[index];

    const std::vector<Index>& child_indices = curr_node.child_indices;
    for (uint64_t i = 0; i < curr_node.moves.size(); i++) {
        if (child_indices[i] != NOT_EXPANDED && table_[child_indices[i]].age != age_) {
            pos.doMove(curr_node.moves[i]);
            saveUsedHash(pos, child_indices[i]);
            pos.undo();
        }
    }
}

void HashTable::deleteOldHash(Position& next_root, bool leave_root) {
    //次のルート局面に相当するノード以下の部分木だけを残すためにインデックスを取得
    uint64_t next_root_index = findSameHashIndex(next_root);

    //置換表全体を消去
    used_num_ = 0;
    age_++;

    if (next_root_index == table_.size()) {
        //そもそも存在しないならここで終了
        return;
    }

    //ルート以下をsave
    saveUsedHash(next_root, next_root_index);

    //強化学習のデータ生成中ではノイズを入れる関係で次のルート局面だけは消去したいので選べるようにしてある
    if (!leave_root) {
        //root_indexのところは初期化
        table_[next_root_index].age = age_ - 1;
        used_num_--;
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

float HashTable::expQfromNext(const HashEntry& node, int32_t i) const {
#ifdef USE_CATEGORICAL
    return expOfValueDist(QfromNextValue(node, i));
#else
    return QfromNextValue(node, i);
#endif
}