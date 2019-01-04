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
    void learn();

    //学習のテスト
    void testLearn();

private:
    //--------------------
    //    内部メソッド
    //--------------------
    //棋譜生成を行う関数
    void learnSlave();

    //今ファイルに保存されているパラメータと対局して強さを測定する関数
    void evaluate();

    //勝敗と探索結果を混合して教師とする関数:elmo絞りに対応
    void pushOneGame(Game& game);

    //指数減衰をかけながらnステップ後の探索の値を教師とする関数
    void pushOneGameReverse(Game& game);

    //---------------------------------------------
    //    ファイルから読み込むためconst化はして
    //    いないがほぼ定数であるもの
    //---------------------------------------------
    //損失の計算方法
    enum {
        ELMO_LEARN,    //elmo絞り   : 最終的な勝敗と現局面の深い探索の値を混合して教師信号とする
        TD_LEAF_LAMBDA //TDLeaf(λ) : 現局面から終局までの評価値を減衰させながら利用
    };
    int32_t LEARN_MODE;

    //TDLeaf(λ)のλ
    double LAMBDA;

    //引き分けの対局も学習するか
    bool USE_DRAW_GAME;

    //強くなったとみなす勝率の閾値
    double THRESHOLD;

    //深い探索にかける係数
    double DEEP_COEFFICIENT;

    //評価する際のゲーム数
    int32_t EVALUATION_GAME_NUM;

    //評価する間隔
    int64_t EVALUATION_INTERVAL;

    //評価するときのランダム手数
    int32_t EVALUATION_RANDOM_TURN;

    //スタックサイズの上限
    int64_t MAX_STACK_SIZE;

    //ステップ数
    int64_t MAX_STEP_NUM;

    //------------
    //    変数
    //------------
    //学習用に加工済の局面スタック
    std::vector<std::pair<std::string, TeacherType>> position_pool_;

    //強くなって世代が進んだ回数
    uint64_t update_num_;
};

#endif