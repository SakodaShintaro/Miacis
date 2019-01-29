﻿#pragma once

#ifndef ALPHAZERO_TRAINER_HPP
#define ALPHAZERO_TRAINER_HPP

#include"base_trainer.hpp"
#include"game.hpp"
#include"replay_buffer.hpp"

class AlphaZeroTrainer : BaseTrainer {
public:
    //--------------------
    //    公開メソッド
    //--------------------
    //コンストラクタ
    explicit AlphaZeroTrainer(std::string settings_file_path);

    //1スレッドだけ学習器を作り、残りのスレッドは自己対局
    //これらは並列に行われる
    void startLearn();

private:
    //--------------------
    //    内部メソッド
    //--------------------
    //今ファイルに保存されているパラメータと対局して強さを測定する関数
    void evaluate(int64_t step);

    //replay_buffer_へ1局分詰め込む関数
    void pushOneGame(Game& game);

    std::vector<Game> play(int32_t game_num, bool eval);

    //---------------------------------------------
    //    ファイルから読み込むためconst化はして
    //    いないがほぼ定数であるもの
    //---------------------------------------------
    //TDLeaf(λ)のλ
    double LAMBDA;

    //引き分けの対局も学習するか
    bool USE_DRAW_GAME;

    //強くなったとみなす勝率の閾値
    double THRESHOLD;

    //評価する際のゲーム数
    int32_t EVALUATION_GAME_NUM;

    //評価する間隔
    int64_t EVALUATION_INTERVAL;

    //評価するときのランダム手数
    int32_t EVALUATION_RANDOM_TURN;

    //replay_buffer_のサイズ上限
    int64_t MAX_REPLAY_BUFFER_SIZE;

    //ステップ数
    int64_t MAX_STEP_NUM;

    //------------
    //    変数
    //------------
    //学習用に加工済の局面スタック
    ReplayBuffer replay_buffer_;

    //強くなって世代が進んだ回数
    uint64_t update_num_;
};

#endif