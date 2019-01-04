#pragma once
#include"base_trainer.hpp"
#include"position.hpp"

#ifndef USE_MCTS

class TreestrapTrainer : BaseTrainer {
public:
    explicit TreestrapTrainer(std::string settings_file_path);
    void startLearn();
private:
    //学習しながら探索する関数
    Score miniMaxLearn(Position& pos, Depth depth);
    Score alphaBetaLearn(Position& pos, Score alpha, Score beta, Depth depth);

    //ラッパー
    double calcLoss(Score shallow_score, Score deep_score);
    double calcGrad(Score shallow_score, Score deep_score);

    //------------------------------
    //    学習中に値が変わる変数
    //------------------------------
    //損失
    double loss_;

    //勾配
    std::unique_ptr<EvalParams<LearnEvalType>> grad_;

    //学習局面数
    uint64_t learned_position_num_;

    //----------------------------------
    //    学習中に値が変わらない変数
    //----------------------------------
    //ステップ数
    int32_t step_size;
};

#endif