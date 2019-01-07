#pragma once

#ifndef TRAINER_HPP
#define TRAINER_HPP

#include"position.hpp"
#include"neural_network.hpp"
#include<iomanip>
#include<fstream>
#include<chrono>
#include<ctime>

//損失の型を設定する
//PolicyとValueの2つ
using LossType = std::array<double, 2>;

//各Trainerの基底となるクラス
class BaseTrainer {
protected:
    //--------------------
    //    その他関数類
    //--------------------
    //log_file_に経過時間を出力する関数
    void timestamp();

    //標準出力とlog_file_の両方に出力する関数
    template<class T> void print(T t);

    //optimizerとして入力されたものが正当か判定する関数
    bool isLegalOptimizer();

    //-----------------------------------------------------
    //    ファイルから読み込むためconst化はしていないが
    //    ほぼ定数であるもの
    //-----------------------------------------------------
    //学習率
    double LEARN_RATE;

    //学習率を減衰させるときの係数
    double LEARN_RATE_DECAY;

    //Momentumにおける混合比
    double MOMENTUM_DECAY;

    //バッチサイズ
    int32_t BATCH_SIZE;

    //optimizerの設定
    std::string OPTIMIZER_NAME;

    //並列化するスレッド数
    uint32_t THREAD_NUM;

    //policy_lossにかける係数
    double POLICY_LOSS_COEFF;

    //value_lossにかける係数
    double VALUE_LOSS_COEFF;

    //--------------------------------
    //    学習中に用いるメンバ変数
    //--------------------------------
    //ログファイル
    std::ofstream log_file_;

    //学習開始時間
    std::chrono::time_point<std::chrono::steady_clock> start_time_;

    //学習中のモデル
    NeuralNetwork<Node> learning_model_;
};

template<class T>
inline void BaseTrainer::print(T t) {
    std::cout << t << "\t";
    log_file_ << t << "\t";
}

#endif // !TRAINER_HPP