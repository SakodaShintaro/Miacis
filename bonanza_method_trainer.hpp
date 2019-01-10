#pragma once

#ifndef BONANZA_METHOD_TRAINER_HPP
#define BONANZA_METHOD_TRAINER_HPP

#include"base_trainer.hpp"
#include"piece.hpp"
#include"common.hpp"
#include"game.hpp"
#include<mutex>
#include<atomic>

class BonanzaMethodTrainer : BaseTrainer {
public:
    //--------------------
    //    公開メソッド
    //--------------------
    //コンストラクタ
    explicit BonanzaMethodTrainer(std::string settings_file_path);

    //学習を開始する関数
    void train();

    void testTrain();

private:
    //-----------------------------------------------------
    //    定数や、ファイルから読み込むためconst化はして
    //    いないがほぼ定数であるもの
    //-----------------------------------------------------
    //学習する棋譜があるディレクトリへのパス
    std::string KIFU_PATH;

    //L1正則化の強さ
    double L1_GRAD_COEFFICIENT;

    //------------
    //    変数
    //------------
    //学習する棋譜
    std::vector<Game> games_;

    //学習する棋譜の数
    uint64_t game_num_;

    //次に学習する棋譜のid
    std::atomic<int32_t> game_index_;

    //今回のミニバッチ中で学習した局面の数
    std::atomic<uint64_t> learned_position_num;

    //学習した局面の総数
    std::atomic<uint64_t> learned_position_num_sum;

    //学習した棋譜の数
    //これがMINI_BATCH_SIZEに達するごとにパラメータを更新する
    std::atomic<uint64_t> learned_games_num;

    //学習した棋譜の総数
    std::atomic<uint64_t> learned_games_num_sum;

    //指し手が一致した局面の数
    std::atomic<int32_t> succeeded_position_num;

    //指し手の一致具合計算
    std::vector<int32_t> ordering_num_;

    //損失
    LossType loss_;
};

#endif // !BONANZA_METHOD_TRAINER_HPP