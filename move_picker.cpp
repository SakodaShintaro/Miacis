#include"move_picker.hpp"
#include"position.hpp"
#include"history.hpp"
#include"eval_params.hpp"
#include"network.hpp"
#include<functional>
#include<algorithm>

extern int piece_value[];

enum Stage {
    MAIN_SEARCH_START,
    MAIN_SEARCH_TT,
	MAIN_SEARCH_CAPTURE,
    MAIN_SEARCH_KILLERS,
    MAIN_SEARCH_COUNTER_MOVE,
	MAIN_SEARCH_NON_CAPTURE,
	MAIN_SEARCH_END,

    QSEARCH_START,
    QSEARCH_TT,
    QSEARCH_CAPTURE,
    QSEARCH_END,
};

#ifdef USE_SEARCH_STACK
MovePicker::MovePicker(Position& pos, const Move ttMove, const Depth depth, const History& history, const Move killers[2], Move counter_move)
	: pos_(pos), tt_move_(ttMove), depth(depth), history_(history), counter_move_(counter_move) {
	stage_ = MAIN_SEARCH_START - 1;
    moves_ = new Move[MAX_MOVE_LIST_SIZE];

    //まずこれら二つは先頭を指す
    cur_ = moves_;
    end_ = moves_;
    bad_capture_start_ = bad_capture_end_ = nullptr;

    killer_moves_[0] = killers[0];
    killer_moves_[1] = killers[1];
}
#else
MovePicker::MovePicker(Position & pos, const Move ttMove, const Depth depth, const History & history, Move counter_move) : pos_(pos), tt_move_(ttMove), depth(depth), history_(history), counter_move_(counter_move) {
    stage_ = MAIN_SEARCH_START - 1;
    moves_ = new Move[MAX_MOVE_LIST_SIZE];

    //まずこれら二つは先頭を指す
    cur_ = moves_;
    end_ = moves_;
}
#endif

MovePicker::MovePicker(Position& pos, const Move ttMove, const Depth depth, const History& history)
    : pos_(pos), tt_move_(ttMove), depth(depth), history_(history) {
    stage_ = QSEARCH_START - 1;
    moves_ = new Move[MAX_MOVE_LIST_SIZE];

    //まずこれら二つは先頭を指す
    cur_ = moves_;
    end_ = moves_;
    bad_capture_start_ = bad_capture_end_ = nullptr;
}

Move MovePicker::nextMove() {
    //NULL_MOVEが返ったら終わりということ
    while (true) {
        while (cur_ == end_) {
            //次のカテゴリの手を生成
            generateNextStage();
        }

        switch (stage_) {
        case MAIN_SEARCH_START:
        case MAIN_SEARCH_TT:
            return *(cur_++);
        case MAIN_SEARCH_CAPTURE:
            //tt_moveとの重複削除
            if (*cur_ != tt_move_) {
                if (cur_->score <= 0) {
                    bad_capture_start_ = cur_;
                    bad_capture_end_ = end_;
                    cur_ = end_;
                    generateNextStage();
                    break;
                }
                return *(cur_++);
            }
            cur_++;
            break;
        case MAIN_SEARCH_KILLERS:
            return *(cur_++);
        case MAIN_SEARCH_COUNTER_MOVE:
            return *(cur_++);
        case MAIN_SEARCH_NON_CAPTURE:
            //tt_move, killer_movesとの重複削除
            if (*cur_ != tt_move_ && *cur_ != killer_moves_[0] && *cur_ != killer_moves_[1] && *cur_ != counter_move_) {
                return *(cur_++);
            }
            cur_++;
            break;
        case MAIN_SEARCH_END:
            return *(cur_++);


            //静止探索
        case QSEARCH_START:
        case QSEARCH_TT:
            return *(cur_++);
        case QSEARCH_CAPTURE:
            //tt_moveとの重複削除
            if (*cur_ != tt_move_ && cur_->score > 0) {
                return *(cur_++);
            }
            cur_++;
            break;
        case QSEARCH_END:
            return *(cur_++);
        default:
            //ここにはこないはず
            assert(false);
        }
    }
}

void MovePicker::generateNextStage() {
	switch (++stage_) {
        //通常探索から呼ばれるときはここから
    case MAIN_SEARCH_START:
        if (pos_.isKingChecked()) {
            //王手がかかっていたら全部生成して最後にNULL_MOVE追加
            pos_.generateEvasionMoves(end_);
            scoringWithHistory();
            *(end_++) = NULL_MOVE;
        }
        break;
    case MAIN_SEARCH_TT:
        //tt_moveを設定する
        if (tt_move_ != NULL_MOVE && pos_.isLegalMove(tt_move_)) {
            *(end_++) = tt_move_;
        }
        break;
    case MAIN_SEARCH_CAPTURE:
        pos_.generateCaptureMoves(end_);
        scoreCapture();
        break;
    case MAIN_SEARCH_KILLERS:
        if (killer_moves_[0] != NULL_MOVE && pos_.isLegalMove(killer_moves_[0]) && killer_moves_[0] != tt_move_) {
            *(end_++) = killer_moves_[0];
        }
        if (killer_moves_[1] != NULL_MOVE && pos_.isLegalMove(killer_moves_[1]) && killer_moves_[1] != tt_move_) {
            *(end_++) = killer_moves_[1];
        }
        break;
    case MAIN_SEARCH_COUNTER_MOVE:
        if (pos_.isLegalMove(counter_move_) 
            && counter_move_ != tt_move_
            && counter_move_ != killer_moves_[0]
            && counter_move_ != killer_moves_[1]) {
            *(end_++) = counter_move_;
        }
        break;
	case MAIN_SEARCH_NON_CAPTURE:
        if (!bad_capture_end_) {
            bad_capture_start_ = cur_;
            bad_capture_end_ = end_;
        }
		pos_.generateNonCaptureMoves(bad_capture_end_);
        cur_ = bad_capture_start_;
        end_ = bad_capture_end_;
        scoringWithHistory();
        break;
	case MAIN_SEARCH_END:
        *(end_++) = NULL_MOVE;
        //これでMovePickerからNULL_MOVEが返るようになるので指し手生成が終わる
		break;


        //静止探索から呼ばれるときはここから
    case QSEARCH_START:
        if (pos_.isKingChecked() && depth == 0) {
            //王手がかかっていたら全部生成して最後にNULL_MOVE追加
            pos_.generateEvasionMoves(end_);
            scoringWithHistory();
            *(end_++) = NULL_MOVE;
        }
        break;
    case QSEARCH_TT:
        //tt_moveを設定する
        if (tt_move_ != NULL_MOVE && pos_.isLegalMove(tt_move_) && tt_move_.capture() != EMPTY) {
            *(end_++) = tt_move_;
        }
        break;
    case QSEARCH_CAPTURE:
        //取り返す手だけを生成
        if (pos_.lastMove() != NULL_MOVE) {
            pos_.generateRecaptureMovesTo(pos_.lastMove().to(), end_);
        }
        scoreCapture();
        break;
    case QSEARCH_END:
        *(end_++) = NULL_MOVE;
        //これでMovePickerからNULL_MOVEが返るようになるので指し手生成が終わる
        break;
    default:
        //ここにはこないはず
        assert(false);
    }
}

#define USE_ONLY_HISTORY
void MovePicker::scoreCapture() {
#ifdef USE_ONLY_HISTORY
    //Historyだけ使う場合
    for (auto itr = cur_; itr != end_; ++itr) {
        itr->score = std::abs(piece_value[itr->capture()]) + history_[*itr];
        if (itr->isPromote()) {
            itr->score += std::abs(piece_value[promote(itr->subject())] - piece_value[itr->subject()]);
        }
    }
    std::sort(cur_, end_, std::greater<>());
#else
    //SEEを計算する場合
    Score base_line = (pos_.color() == BLACK ? pos_.score() : -pos_.score());
    for (auto itr = cur_; itr != end_; ++itr) {
        pos_.doMove(*itr);
        itr->score = history_[*itr] - pos_.SEE() - base_line;
        pos_.undo();
    }
    std::sort(cur_, end_, std::greater<Move>());
    for (auto itr = cur_; itr != end_; ++itr) {
        itr->score -= history_[*itr];
    }
#endif
}

void MovePicker::scoringWithHistory() {
#ifdef USE_ONLY_HISTORY
    //Historyだけ使う場合
    for (auto itr = cur_; itr != end_; ++itr) {
        itr->score = std::abs(piece_value[itr->capture()]) + history_[*itr];
        if (itr->isPromote()) {
            itr->score += std::abs(piece_value[promote(itr->subject())] - piece_value[itr->subject()]);
        }
    }
    std::sort(cur_, end_, std::greater<>());
#else
    //SEEを計算する場合
    Score base_line = (pos_.color() == BLACK ? pos_.score() : -pos_.score());
    for (auto itr = cur_; itr != end_; ++itr) {
        pos_.doMove(*itr);
        itr->score = history_[*itr] - (pos_.color() == BLACK ? pos_.score() : -pos_.score()) - base_line;
        pos_.undo();
    }
    std::sort(cur_, end_, std::greater<Move>());
    for (auto itr = cur_; itr != end_; ++itr) {
        itr->score -= history_[*itr];
    }
#endif
}