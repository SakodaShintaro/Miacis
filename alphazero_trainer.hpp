#pragma once

#ifndef ALPHAZERO_TRAINER_HPP
#define ALPHAZERO_TRAINER_HPP

#include"base_trainer.hpp"
#include"game.hpp"

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

    //学習のテスト
    void testLearn();

private:
    //--------------------
    //    内部メソッド
    //--------------------
    //replay_buffer_からデータを引き出し学習する関数
    void learn();

    //自己対局を行ってreplay_buffer_へデータを詰め込む関数
    void act();

    //今ファイルに保存されているパラメータと対局して強さを測定する関数
    void evaluate();

    //replay_buffer_へ1局分詰め込む関数
    void pushOneGame(Game& game);

    void parallelPlay(int32_t game_num);

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
    std::vector<std::pair<std::string, TeacherType>> replay_buffer_;

    //強くなって世代が進んだ回数
    uint64_t update_num_;
};

#endif