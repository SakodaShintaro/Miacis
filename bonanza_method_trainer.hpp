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

private:
    //-----------------------------------------------------
    //    定数や、ファイルから読み込むためconst化はして
    //    いないがほぼ定数であるもの
    //-----------------------------------------------------
    //学習する棋譜があるディレクトリへのパス
    std::string KIFU_PATH;

    //L1正則化の強さ
    float WEIGHT_DECAY;

    //------------
    //    変数
    //------------
    //学習する棋譜
    std::vector<Game> games_;

    //学習する棋譜の数
    uint64_t game_num_;
};

#endif // !BONANZA_METHOD_TRAINER_HPP