#ifndef SEARCH_HPP
#define SEARCH_HPP

#include"position.hpp"
#include"move.hpp"
#include"hash_table.hpp"
#include"usi_options.hpp"
#include"search_stack.hpp"
#include"pv_table.hpp"
#include"history.hpp"
#include<chrono>

#ifndef USE_MCTS

//1スレッド分に相当する探索クラス
class AlphaBetaSearcher {
public:
    //コンストラクタ
    AlphaBetaSearcher(int64_t hash_size, int64_t thread_num);

    //与えられた局面に対して思考をする関数
    std::pair<Move, TeacherType> think(Position& root);

    //探索で再帰的に用いる通常の関数
    template<bool isPVNode>
    Score search(Position &pos, Score alpha, Score beta, Depth depth, int distance_from_root);

    //pvを取り出す関数
    std::vector<Move> pv() {
        std::vector<Move> pv;
        for (Move m : pv_table_) {
            pv.push_back(m);
        }
        return pv;
    }

    //pvのリセットをする関数:BonanzaMethodで呼ばれるためpublicに置いているがprivateにできるかも
    void resetPVTable() {
        pv_table_.reset();
    }

    //historyのリセットをする関数:これもBonanzaMethodで呼ばれる
    //RootStrapでも呼ばれないとおかしいか?
    void clearHistory() {
        history_.clear();
    }

    static int64_t limit_msec;
    static std::atomic<bool> stop_signal;
    static bool print_usi_info;
    static bool train_mode;
private:
    //--------------------
    //    内部メソッド
    //--------------------
    //静止探索をする関数:searchのtemplateパラメータを変えたものとして実装できないか
    template<bool isPVNode>
    Score qsearch(Position &pos, Score alpha, Score beta, Depth depth, int distance_from_root);

    //GUIへ情報を送る関数
    void sendInfo(Depth depth, std::string cp_or_mate, Score score, Bound bound);

    //停止すべきか確認する関数
    inline bool shouldStop();

    //futilityMarginを計算する関数:razoringのマージンと共通化すると棋力が落ちる.そういうものか
    inline static int futilityMargin(int32_t depth);

    //------------------
    //    メンバ変数
    //------------------
    //探索局面数
    uint64_t node_number_;

    //時間
    std::chrono::steady_clock::time_point start_;

    //History 
    History history_;

    //MoveHistory:強くならなかったので外している
#ifdef USE_MOVEHISTORY
    MoveHistory move_history_;
#endif
    //思考する局面における合法手
    std::vector<Move> root_moves_;

    //seldpth
    Depth seldepth_;

    //技巧風のPV_Table
    PVTable pv_table_;

    //Search Stackとそれを参照する関数
#ifdef USE_SEARCH_STACK
    SearchStack stack_[DEPTH_MAX / PLY]; //65KBほど？
    SearchStack* searchInfoAt(int32_t distance_from_root) {
        //深さ0でも前二つが参照できるようにずらしておかなければならない
        return &stack_[distance_from_root + 2];
    }
#endif

    HashTable hash_table_;
};

#endif

#endif