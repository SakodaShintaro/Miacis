#include "searcher_for_mate.hpp"

SearcherForMate::SearcherForMate(HashTable& hash_table, const SearchOptions& search_options)
: stop_signal(false), hash_table_(hash_table), search_options_(search_options) {}

void SearcherForMate::mateSearch(Position pos, int32_t depth_limit) {
    stop_signal = false;
    HashEntry& curr_node = hash_table_[hash_table_.root_index];
    for (int32_t depth = 1; !stop_signal && depth <= depth_limit; depth += 2) {
        for (uint64_t i = 0; i < curr_node.moves.size(); i++) {
            pos.doMove(curr_node.moves[i]);
            bool result = mateSearchForEvader(pos, depth - 1);
            pos.undo();
            if (result) {
                //この手に書き込み
                //search_limitだけ足せば必ずこの手が選ばれるようになる
                //mutexにロックをかけるべきかどうか
                curr_node.mutex.lock();
                curr_node.N[i]  += search_options_.search_limit;
                curr_node.sum_N += search_options_.search_limit;

                if (curr_node.child_indices[i] != HashTable::NOT_EXPANDED) {
                    HashEntry& child_node = hash_table_[curr_node.child_indices[i]];
                    child_node.mutex.lock();
                    //ここでvalueを上書きしても、backup途中のエージェントがいるとその後に上書きが発生するからダメそうな気がする
#ifdef USE_CATEGORICAL
                    child_node.value = onehotDist(MIN_SCORE);
#else
                    child_node.value = MIN_SCORE;
#endif
                    child_node.mutex.unlock();
                }
                curr_node.mutex.unlock();
                return;
            }
        }
    }
}

bool SearcherForMate::mateSearchForAttacker(Position& pos, int32_t depth) {
    assert(depth % 2 == 1);
    if (stop_signal) {
        return false;
    }
    //全ての手を試してみる。どれか一つでも勝ちになる行動があるなら勝ち
    for (const Move& move : pos.generateAllMoves()) {
        pos.doMove(move);
        bool result = mateSearchForEvader(pos, depth - 1);
        pos.undo();
        if (result) {
            return true;
        }
    }
    return false;
}

bool SearcherForMate::mateSearchForEvader(Position& pos, int32_t depth) {
    assert(depth % 2 == 0);
    if (stop_signal || pos.canSkipMateSearch()) {
        return false;
    }

    //負けかどうか確認
    float score;
    if (pos.isFinish(score)) {
        return (score == MIN_SCORE);
    }

    if (depth == 0) {
        return false;
    }

    //全ての手を試してみる。どれか一つでも負けを逃れる行動があるなら負けではない
    for (const Move& move : pos.generateAllMoves()) {
        pos.doMove(move);
        bool result = mateSearchForAttacker(pos, depth - 1);
        pos.undo();
        if (!result) {
            return false;
        }
    }

    return true;
}

bool SearcherForMate::search(Position& pos, int32_t depth) {
    if (stop_signal) {
        return false;
    }

    //depthの偶奇で攻め手か受け手かを判断
    bool is_attacker = depth % 2;

    //明らかに負けにならない局面を枝刈り(将棋で王手がかかっていないときを想定)
    if (!is_attacker && pos.canSkipMateSearch()) {
        return false;
    }

    //攻め手のとき勝ち、あるいは受け手のとき負けかどうか確認
    float score;
    if (pos.isFinish(score)) {
        return (score == (is_attacker ? MAX_SCORE : MIN_SCORE));
    }

    //全ての手を試してみる
    //攻め手の場合:どれか一つでも勝ちになる行動があるなら勝ち
    //受け手の場合:どれか一つでも負けを逃れる行動があるなら負けではない
    //つまりresultとis_attackerの一致性を見れば良い
    for (const Move& move : pos.generateAllMoves()) {
        pos.doMove(move);
        bool result = search(pos, depth - 1);
        pos.undo();
        if (result == is_attacker) {
            return is_attacker;
        }
    }

    //全ての手を試し終わったとき
    //攻め手の場合:勝ちは見つけられなかったのでfalse
    //受け手の場合:負けを逃れる手が見つけられなかったのでtrue
    return !is_attacker;
}