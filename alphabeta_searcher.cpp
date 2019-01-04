#include"searcher.hpp"
#include"move.hpp"
#include"move_picker.hpp"
#include"usi_options.hpp"
#include"types.hpp"
#include"network.hpp"
#include<iostream>
#include<cstdio>
#include<string>
#include<algorithm>
#include<utility>
#include<functional>
#include<iomanip>

#ifndef USE_MCTS

int64_t AlphaBetaSearcher::limit_msec;
std::atomic<bool> AlphaBetaSearcher::stop_signal;
bool AlphaBetaSearcher::print_usi_info;
bool AlphaBetaSearcher::train_mode;

struct SearchLog {
    uint64_t hash_cut_num;
    uint64_t razoring_num;
    uint64_t futility_num;
    uint64_t null_move_num;
    void print() const {
        std::cout << "hash_cut_num  = " << hash_cut_num << std::endl;
        std::cout << "razoring_num  = " << razoring_num << std::endl;
        std::cout << "futility_num  = " << futility_num << std::endl;
        std::cout << "null_move_num = " << null_move_num << std::endl;
    }
};
static SearchLog search_log;

AlphaBetaSearcher::AlphaBetaSearcher(int64_t hash_size, int64_t thread_num) {
    assert(thread_num == 1);
    hash_table_.setSize(hash_size);
}

std::pair<Move, TeacherType> AlphaBetaSearcher::think(Position& root) {
    //思考開始時間をセット
    start_ = std::chrono::steady_clock::now();

    //History初期化
    history_.clear();

    resetPVTable();

    //ルート局面の合法手を設定
    root_moves_ = root.generateAllMoves();

#ifdef USE_SEARCH_STACK
    SearchStack* ss = searchInfoAt(0);
    ss->killers[0] = ss->killers[1] = NULL_MOVE;
    ss->can_null_move = true;
#endif

    //合法手が0だったら投了
    if (root_moves_.empty()) {
        return { NULL_MOVE, TeacherType() };
    }

    //合法手が1つだったらすぐ送る
    //これ別にいらないよな
    //これもUSIオプション化した方が良いか
    //if (root_moves_.size() == 1) {
    //    if (role_ == MAIN) {
    //        std::cout << "bestmove " << root_moves_[0] << std::endl;
    //    }
    //    return;
    //}

    //指定された手数まで完全ランダムに指す
    if (root.turn_number() + 1 <= usi_option.random_turn) {
        static std::random_device rd;
        auto rnd = rd() % root_moves_.size();
        root_moves_[rnd].score = MIN_SCORE;
        return { root_moves_[rnd], TeacherType() };
    }

    //探索局面数を初期化
    node_number_ = 0;

    //探索
    //反復深化
    static constexpr auto DEFAULT_ASPIRATION_WINDOW_SIZE = Score(256);
    Score aspiration_window_size = DEFAULT_ASPIRATION_WINDOW_SIZE;
    Score best_score{}, alpha, beta, previous_best_score{};

    for (Depth depth = PLY; depth <= std::min(usi_option.depth_limit, DEPTH_MAX); depth += PLY) {
        //seldepth_の初期化
        seldepth_ = depth;

        //探索窓の設定
        if (depth <= 4 * PLY) { //深さ4まではASPIRATION_WINDOWを使わずフルで探索する
            alpha = MIN_SCORE;
            beta = MAX_SCORE;
        } else {
            alpha = std::max(previous_best_score - aspiration_window_size, MIN_SCORE);
            beta = std::min(previous_best_score + aspiration_window_size, MAX_SCORE);
        }

        while (!shouldStop()) { //exactな評価値が返ってくるまでウィンドウを広げつつ探索
            //指し手のスコアを最小にしておかないと変になる
            for (auto& root_move : root_moves_) {
                root_move.score = MIN_SCORE;
            }

            best_score = search<true>(root, alpha, beta, depth, 0);

            //詰んでいたら抜ける
            if (isMatedScore(best_score) || shouldStop()) {
                break;
            }

            if (best_score <= alpha) {
                //fail-low
                if (AlphaBetaSearcher::print_usi_info) {
                    sendInfo(depth, "cp", best_score, UPPER_BOUND);
                }

                beta = (alpha + beta) / 2;
                alpha -= aspiration_window_size;
                aspiration_window_size *= 4;
            } else if (best_score >= beta) {
                //fail-high
                if (AlphaBetaSearcher::print_usi_info) {
                    sendInfo(depth, "cp", best_score, LOWER_BOUND);
                }

                alpha = (alpha + beta) / 2;
                beta += aspiration_window_size;
                aspiration_window_size *= 4;
            } else {
                aspiration_window_size = DEFAULT_ASPIRATION_WINDOW_SIZE;
                break;
            }
        }

        //停止確認してダメだったら保存せずループを抜ける
        if (shouldStop()) {
            break;
        }

        //指し手の並び替え
        std::stable_sort(root_moves_.begin(), root_moves_.end(), std::greater<>());

        //GUIへ読みの情報を送る
        if (AlphaBetaSearcher::print_usi_info) {
            if (MATE_SCORE_UPPER_BOUND < root_moves_[0].score && root_moves_[0].score < MATE_SCORE_LOWER_BOUND) {
                //詰みなし
                sendInfo(depth, "cp", root_moves_[0].score, EXACT_BOUND);
            } else {
                //詰みあり
                Score mate_num = MAX_SCORE - std::abs(root_moves_[0].score);
                mate_num *= ((int64_t)mate_num % 2 == 0 ? -1 : 1);
                sendInfo(depth, "mate", mate_num, EXACT_BOUND);
            }
        }

        //置換表への保存
#ifdef USE_NN
        hash_table_.save(root.hash_value(), root_moves_[0], root_moves_[0].score, depth, root_moves_);
#else
        hash_table_.save(root.hash_value(), root_moves_[0], root_moves_[0].score, depth);
#endif

        //詰みがあったらすぐ返す
        if (isMatedScore(root_moves_[0].score)) {
            break;
        }

        //今回のイテレーションにおけるスコアを記録
        previous_best_score = best_score;

        //PVtableをリセットしていいよな？
        resetPVTable();
    }

    //GUIへBestMoveの情報を送る
    if (AlphaBetaSearcher::print_usi_info) {
        //ログを出力
        search_log.print();
    }
    return { root_moves_[0], sigmoid((int32_t)root_moves_[0].score, CP_GAIN) };
}

template<bool isPVNode>
Score AlphaBetaSearcher::qsearch(Position &pos, Score alpha, Score beta, Depth depth, int distance_from_root) {
    // -----------------------
    //     最初のチェック
    // -----------------------

    pv_table_.closePV(distance_from_root);
    seldepth_ = std::max(seldepth_, distance_from_root * PLY);

    //assert(isPVNode || alpha + 1 == beta);
    assert(0 <= distance_from_root && distance_from_root <= DEPTH_MAX);

    //探索局面数を増やす
    node_number_++;

    // -----------------------
    //     停止確認
    // -----------------------

#ifdef USE_NN
    //if (pos.generateAllMoves().size() == 0) {
    //    //詰み
    //    Score s = MIN_SCORE + distance_from_root;
    //    if (pos.lastMove().isDrop() && kind(pos.lastMove().subject()) == PAWN) {
    //        //打ち歩詰めなので勝ち
    //        return -s;
    //    }
    //    return s;
    //}

    return pos.scoreForTurn();
#endif

    //ここでやる必要があるのかはわからない
    if (shouldStop()) return SCORE_ZERO;

    // -----------------------
    //     置換表を見る
    // -----------------------
    auto hash_entry = hash_table_.find(pos.hash_value());
    Score tt_score = hash_entry ? hash_entry->best_move.score : MIN_SCORE;

    //tt_moveの合法性判定はここでやらなくてもMovePickerで確認するはず
    Move tt_move = hash_entry ? hash_entry->best_move : NULL_MOVE;
    Depth tt_depth = hash_entry ? hash_entry->depth : Depth(0);

    //置換表による枝刈り
    if (!isPVNode
        && !pos.isKingChecked()
        && hash_entry
        && tt_depth >= depth
        && tt_score >= beta) {
        search_log.hash_cut_num++;
        return tt_score;
    }

    //指し手を生成
    MovePicker mp(pos, tt_move, depth, history_);
    Score best_score = (pos.isKingChecked() && depth == 0 ? MIN_SCORE + distance_from_root //王手がかかっていたら最低点から始める
        : pos.scoreForTurn());                            //そうでなかったら評価関数を呼び出した値から始める
    if (best_score >= beta) {
        return best_score;
    }

    //変数宣言
    Move best_move = NULL_MOVE;
    int64_t move_count = 0;
    
    for (Move current_move = mp.nextMove(); current_move != NULL_MOVE; current_move = mp.nextMove()) {
        //停止確認
        if (shouldStop()) {
            return Score(0);
        }

        //Null Window Searchはした方がいいんだろうか.今はしてない
        pos.doMove(current_move);
        Score score = -qsearch<isPVNode>(pos, -beta, -alpha, depth - PLY, distance_from_root + 1);

        if (score > best_score) {
            best_score = score;
            best_move = current_move;
            pv_table_.update(best_move, distance_from_root);

            if (best_score >= beta) {
                //fail-high
                pos.undo();
                break; //beta-cut
            }

            alpha = std::max(alpha, best_score);
        }

        pos.undo();

        ++move_count;
    }

    if (isMatedScore(best_score) && pos.lastMove().isDrop() && kind(pos.lastMove().subject()) == PAWN) {
        //打ち歩詰めなので勝ち
        return -best_score;
    }

    hash_table_.save(pos.hash_value(), best_move, best_score, depth);

    return best_score;
}

void AlphaBetaSearcher::sendInfo(Depth depth, std::string cp_or_mate, Score score, Bound bound) {
    if (bound != EXACT_BOUND) {
        //lower_boundとか表示する意味がない気がしてきた
        return;
    }

    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    std::cout << "info time " << std::setw(6) << elapsed.count();

    //GUIへ読み途中の情報を返す
    std::cout << " depth " << std::setw(2) << depth / PLY;
    std::cout << " seldepth " << std::setw(2) << seldepth_ / PLY;
    std::cout << " nodes " << std::setw(10) << node_number_;
    std::cout << " score " << std::setw(4) << cp_or_mate << " " << std::setw(6) << score;
    int64_t nps = (elapsed.count() == 0 ? 0 : (int64_t)((double)(node_number_) / elapsed.count() * 1000.0));
    std::cout << " nps " << std::setw(10) << nps;
    std::cout << " hashfull " << std::setw(4) << (int)hash_table_.hashfull();
    std::cout << " pv ";
    if (pv_table_.size() == 0) {
        pv_table_.update(root_moves_[0], 0);
    }
    for (auto move : pv_table_) {
        std::cout << move << " ";
    }
    std::cout << std::endl;
}

inline int AlphaBetaSearcher::futilityMargin(int32_t depth) {
    return 175 * depth / PLY;
    //return PLY / 2 + depth * 2;
}

inline bool AlphaBetaSearcher::shouldStop() {
    if (Searcher::stop_signal) {
        return true;
    }

    //探索深さの制限も加えるべきか?
    auto now_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - start_);
    if (elapsed.count() >= Searcher::limit_msec - usi_option.byoyomi_margin) {
        //停止信号をオンにする
        Searcher::stop_signal = true;
        return true;
    }
    return false;
}

#define OMIT_PRUNINGS

template<bool isPVNode>
Score AlphaBetaSearcher::search(Position &pos, Score alpha, Score beta, Depth depth, int distance_from_root) {
    if (depth < PLY) {
        return qsearch<isPVNode>(pos, alpha, beta, Depth(0), distance_from_root);
    }

    // nodeの種類
    bool isRootNode = (distance_from_root == 0);

    //-----------------------------
    // Step1. 初期化
    //-----------------------------

    //assert類
    assert(isPVNode || (alpha + 1 == beta));
    assert(MIN_SCORE <= alpha && alpha < beta && beta <= MAX_SCORE);

    //探索局面数を増やす
    ++node_number_;

    if (isPVNode) {
        seldepth_ = std::max(seldepth_, distance_from_root * PLY);
    }

    //-----------------------------
    // RootNode以外での処理
    //-----------------------------

    if (!isRootNode) {

        //-----------------------------
        // Step2. 探索の停止と引き分けの確認
        //-----------------------------

        //停止確認
        if (!train_mode && shouldStop()) {
            return SCORE_ZERO;
        }

        //引き分けの確認
        Score repeate_score;
        if (pos.isRepeating(repeate_score)) {
            pv_table_.closePV(distance_from_root);
            return repeate_score;
        }

        //-----------------------------
        // Step3. Mate distance pruning
        //-----------------------------

        //合ってるのか怪しいぞ
        alpha = std::max(MIN_SCORE + distance_from_root, alpha);
        beta = std::min(MAX_SCORE - distance_from_root + 1, beta);
        if (alpha >= beta) {
            return alpha;
        }
    }

#ifdef USE_SEARCH_STACK
    //-----------------------------
    // SearchStackの初期化
    //-----------------------------

    SearchStack* ss = searchInfoAt(distance_from_root);
    (ss + 1)->killers[0] = (ss + 1)->killers[1] = NULL_MOVE;
    (ss + 1)->can_null_move = true;
#endif

    //-----------------------------
    // Step4. 置換表を見る
    //-----------------------------

    auto hash_entry = hash_table_.find(pos.hash_value());
    if (train_mode) {
        hash_entry = nullptr;
    }
    Score tt_score = hash_entry ? hash_entry->best_move.score : MIN_SCORE;

    //tt_moveの合法性判定はここではなくMovePicker内で確認している
    Move tt_move = (hash_entry ? hash_entry->best_move : NULL_MOVE);
    Depth tt_depth = (hash_entry ? hash_entry->depth : Depth(0));

    //置換表の値による枝刈り
    if (!isPVNode
        && hash_entry
        && tt_depth >= depth
        && tt_score >= beta) {

        //tt_moveがちゃんとしたMoveならこれでHistory更新
        if (tt_move != NULL_MOVE) {
            history_.updateBetaCutMove(tt_move, depth);
        }
        search_log.hash_cut_num++;
        return tt_score;
    }

    //-----------------------------
    // Step5. 局面の静的評価
    //-----------------------------

#ifndef USE_NN

    //Score static_score = (hash_entry ? tt_score : pos.scoreForTurn());
    Score static_score = (hash_entry ? tt_score : qsearch<isPVNode>(pos, alpha, beta, Depth(0), distance_from_root));

    //王手がかかっているときは下の枝刈りはしない
    if (!pos.isKingChecked()) {
        //-----------------------------
        // static_scoreによる枝刈り
        //-----------------------------

        //-----------------------------
        // Step6. Razoring
        //-----------------------------
        static constexpr int razor_margin[] = {0, 590, 604};
        if (!isPVNode
            && depth < 3 * PLY
            && static_score + razor_margin[depth / PLY] <= alpha) {
            Score ralpha = alpha - (depth >= 2 * PLY) * razor_margin[depth / PLY];
            Score v = qsearch<isPVNode>(pos, ralpha, ralpha + 1, Depth(0), distance_from_root);
            if (depth < 2 * PLY || v <= ralpha) {
                search_log.razoring_num++;
                //printf("razoring\n");
                return v;
            }
        }

        //-----------------------------
        // Step7. Futility pruning
        //-----------------------------

        if (!isPVNode
            && (static_score - futilityMargin(depth) >= beta)) {
            search_log.futility_num++;
            //printf("futility pruning\n");
            return static_score;
        }

        //-----------------------------
        // Step8. Null Move Pruning
        //-----------------------------

#ifdef USE_SEARCH_STACK
        if (!isPVNode
            && static_score >= beta
            && ss->can_null_move) {
            Depth rdepth = depth / 2 - PLY;

            //2回連続null_moveになるのを避ける
            (ss + 1)->can_null_move = false;
            pos.doNullMove();
            Score null_score = -search<false>(pos, -alpha - 1, -alpha, rdepth - PLY,
                                                          distance_from_root + 1);

            pos.undoNullMove();
            (ss + 1)->can_null_move = true;
            if (null_score >= beta) {
                search_log.null_move_num++;
                //printf("nullmove pruning\n");
                return null_score;
            }
        }
#endif

#ifndef OMIT_PRUNINGS
        //-----------------------------
        // Step9. ProbCut
        //-----------------------------

        if (!isPVNode
            && depth >= 5 * PLY) {
            Score rbeta = std::min(beta + 300, MAX_SCORE);
            Depth rdepth = depth - 4;
            MovePicker mp(pos, tt_move, rdepth, history_, ss->killers);
            for (Move move = mp.nextMove(); move != NULL_MOVE; move = mp.nextMove()) {
                pos.doMove(move);
                Score score = -search<false>(pos, -rbeta, -rbeta + 1, rdepth, distance_from_root + 1);
                pos.undo();
                if (score >= rbeta) {
                    return score;
                }
            }
        }
#endif

        //-----------------------------
        // Step10. 多重反復深化
        //-----------------------------

        if (depth >= 6 * PLY
            && tt_move == NULL_MOVE
            && (isPVNode || static_score + 128 >= beta)
            && !train_mode) {
            Depth d = (depth * 3 / 4) - 2 * PLY;
#ifdef USE_SEARCH_STACK
            ss->can_null_move = false;
#endif
            search<isPVNode>(pos, alpha, beta, d, distance_from_root);
#ifdef USE_SEARCH_STACK
            ss->can_null_move = true;
#endif

            hash_entry = hash_table_.find(pos.hash_value());
            tt_move = hash_entry ? hash_entry->best_move : NULL_MOVE;
        }

    }
#endif

    //変数の準備
    Move non_cut_moves[600];
    uint32_t non_cut_moves_index = 0;
    int move_count = 0;
    Move pre = (pos.turn_number() > 0 ? pos.lastMove() : NULL_MOVE);
    Score best_score = MIN_SCORE + distance_from_root;
    Move best_move = NULL_MOVE;

    //指し手を生成
#ifdef USE_NN
    constexpr int32_t scale = 1000;
    constexpr int32_t threshold = scale;
    std::vector<Move> moves;
    if (hash_entry && hash_entry->sorted_moves.size() > 0) {
        moves = hash_entry->sorted_moves;
        bool flag = true;
        for (auto move : moves) {
            flag = (flag && pos.isLegalMove(move));
        }
        if (!flag) {
            moves = pos.generateAllMoves();
            Network::scoreByPolicy(moves, pos.policy(), scale);
            sort(moves.begin(), moves.end(), std::greater<Move>());
        }
    } else {
        moves = pos.generateAllMoves();
        Network::scoreByPolicy(moves, pos.policy(), scale);
        sort(moves.begin(), moves.end(), std::greater<Move>());
    }
#elif USE_SEARCH_STACK
    //MovePicker mp(pos, tt_move, depth, history_, ss->killers, move_history_[pre]);
    MovePicker mp(pos, tt_move, depth, history_, ss->killers, NULL_MOVE);
#else
    MovePicker mp(pos, tt_move, depth, history_, NULL_MOVE);
#endif

    //-----------------------------
    // Step11. Loop through moves
    //-----------------------------

    //Score base_line = pos.color() == BLACK ? pos.scores() : -pos.scores();
    //if (distance_from_root == 0) {
    //    printf("base_line = %4d\n", base_line);
    //}

#if USE_NN
    for (Move current_move : moves) {
#else
    for (Move current_move = mp.nextMove(); current_move != NULL_MOVE; current_move = mp.nextMove()) {
#endif
        //ルートノードでしか参照されない
        std::vector<Move>::iterator root_move_itr;

        if (isRootNode) {
            root_move_itr = std::find(root_moves_.begin(), root_moves_.end(), current_move);
            if (root_move_itr == root_moves_.end()) {
                //root_moves_に存在しなかったらおかしいので次の手へ
                continue;
            }
        }

        ++move_count;

        //-----------------------------
        // Step13. 動かす前での枝刈り
        //-----------------------------

#if USE_NN
        //nnの出力をベースに枝刈り
        //if (!isMatedScore(best_score) && moves[0].scores - current_move.scores >= threshold) {
        //    //ソートしてあるのでbreakして良い
        //    break;
        //}
#endif

        //-----------------------------
        // Step14. 1手進める
        //-----------------------------

        //合法性判定は必要かどうか
        //今のところ合法手しかこないはずだけど
#if DEBUG
        if (!pos.isLegalMove(current_move)) {
            pos.isLegalMove(current_move);
            current_move.print();
            pos.printForDebug();
            assert(false);
        }
#endif

        pos.doMove(current_move);

        if (isPVNode) {
            pv_table_.closePV(distance_from_root + 1);
        }

        Score score;
        bool shouldSearchFullDepth = true;

        //-----------------------------
        // Step15. Move countに応じてdepthを減らした探索(Late Move Reduction)
        //-----------------------------

#ifdef USE_NN
        //Depth new_depth = depth  - (int)(moves[0].scores - current_move.scores) * 3;
        //scores = -search<false, train_mode>(pos, -alpha - 1, -alpha, new_depth - PLY, distance_from_root + 1);
        //shouldSearchFullDepth = (scores > alpha);
#endif

        //-----------------------------
        // Step16. Full Depth Search
        //-----------------------------
        //if (distance_from_root == 0 && current_move.to() == SQ23 && current_move.subject() == BLACK_PAWN
        //    && depth == 3 * PLY) {
        //    printf("------------------------------\n");
        //}

        if (shouldSearchFullDepth) {
            //Null Window Searchでalphaを超えそうか確認
            //これ入れた方がいいのかなぁ
            score = -search<false>(pos, -alpha - 1, -alpha, depth - PLY, distance_from_root + 1);

            if (alpha < score && score < beta) {
                //いい感じのスコアだったので再探索
                score = -search<isPVNode>(pos, -beta, -alpha, depth - PLY, distance_from_root + 1);
            }
        }

        //-----------------------------
        // Step17. 1手戻す
        //-----------------------------
        pos.undo();

        //-----------------------------
        // Step18. 停止確認
        //-----------------------------

        //停止確認
        if (!train_mode && shouldStop()) {
            return Score(0);
        }

        //-----------------------------
        // 探索された値によるalpha更新
        //-----------------------------
        if (score > best_score) {
            if (isRootNode) {
                //ルートノードならスコアを更新しておく
                root_move_itr->score = score;
            }

            best_score = score;
            best_move = current_move;
            pv_table_.update(best_move, distance_from_root);

            if (score >= beta) {
                //fail-high
                break; //betaカット
            } else if (score > alpha) {
                alpha = score;
            }
        }
        non_cut_moves[non_cut_moves_index++] = current_move;
    }


    //-----------------------------
    // Step20. 詰みの確認
    //-----------------------------

    if (move_count == 0) {
        //詰み
        if (pos.lastMove().isDrop() && kind(pos.lastMove().subject()) == PAWN) {
            //打ち歩詰めなので反則勝ち
            return MAX_SCORE;
        }
        return MIN_SCORE + distance_from_root;
    }

    if (best_move != NULL_MOVE) {
        history_.updateBetaCutMove(best_move, depth);
#ifdef USE_SEARCH_STACK
        ss->updateKillers(best_move);
#endif
#ifdef USE_MOVEHISTORY
        move_history_.update(pos.lastMove(), best_move);
#endif 
    }
    for (uint32_t i = 0; i < non_cut_moves_index; i++) {
        history_.updateNonBetaCutMove(non_cut_moves[i], depth);
    }

    //-----------------------------
    // 置換表に保存
    //-----------------------------
#ifdef USE_NN
        hash_table_.save(pos.hash_value(), best_move, best_score, depth, moves);
#else
        hash_table_.save(pos.hash_value(), best_move, best_score, depth);
#endif

    return best_score;
}

#endif // !USE_MCTS