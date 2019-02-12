#ifndef TRAINER_HPP
#define TRAINER_HPP

#include"neural_network.hpp"
#include<chrono>

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
    double MOMENTUM;

    //バッチサイズ
    uint64_t BATCH_SIZE;

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
#ifdef USE_LIBTORCH
    NeuralNetwork learning_model_;
#else
    NeuralNetwork<Node> learning_model_;
#endif
};

#endif // !TRAINER_HPP