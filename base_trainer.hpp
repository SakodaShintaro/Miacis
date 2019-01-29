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
    std::string elapsedTime();

    //経過時間を小数点単位で表示
    double elapsedHours();

    //-----------------------------------------------------
    //    ファイルから読み込むためconst化はしていないが
    //    ほぼ定数であるもの
    //-----------------------------------------------------
    //学習率
    float LEARN_RATE;

    //学習率を減衰させるときの係数
    double LEARN_RATE_DECAY;

    //Momentumにおける混合比
    double MOMENTUM_DECAY;

    //バッチサイズ
    unsigned long BATCH_SIZE;

    //並列化するスレッド数
    uint32_t THREAD_NUM;

    //policy_lossにかける係数
    float POLICY_LOSS_COEFF;

    //value_lossにかける係数
    float VALUE_LOSS_COEFF;

    //--------------------------------
    //    学習中に用いるメンバ変数
    //--------------------------------
    //学習開始時間
    std::chrono::time_point<std::chrono::steady_clock> start_time_;

    //学習中のモデル
    NeuralNetwork<Node> learning_model_;
};

#endif // !TRAINER_HPP