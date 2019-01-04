#pragma once

#include"move.hpp"
#include<vector>

//enum Depth;
class Position;
class History;

class MovePicker {
private:
	Move* begin() { return moves_; }
    Move* end() { return end_; }

	Position& pos_;

	Move killer_moves_[2];
	Move counter_move_;
	Depth depth;
	Move tt_move_;

	//ProbCut用の指し手生成に用いる、直前の指し手で捕獲された駒の価値
	//int threshold;

	//指し手生成の段階
	int stage_;

	//次に返す手、生成された指し手の末尾、BadCaptureの終端
    Move *cur_, *end_;
    Move *bad_capture_start_, *bad_capture_end_;

    Move *moves_;

    const History& history_;

public:
	//通常探索から呼ばれる際のコンストラクタ
#ifdef USE_SEARCH_STACK
	MovePicker(Position& pos, Move ttMove, Depth depth, const History& history, const Move killers[2], Move counter_move);
#else
    MovePicker(Position& pos, const Move ttMove, const Depth depth, const History& history, Move counter_move);
#endif

    //静止探索から呼ばれる際のコンストラクタ
    MovePicker(Position& pos, Move ttMove, Depth depth, const History& history);

    ~MovePicker() {
        delete[] moves_;
    }
	Move nextMove();
	void generateNextStage();
    void scoreCapture();
    void scoringWithHistory();

    int stage() {
        return stage_;
    }

    void printAllMoves() {
        for (auto itr = cur_; itr != end_; itr++) {
            itr->printWithScore();
        }
    }
};