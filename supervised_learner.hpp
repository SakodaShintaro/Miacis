#ifndef SUPERVISED_LEARNER_HPP
#define SUPERVISED_LEARNER_HPP

#include"base_trainer.hpp"
#include"piece.hpp"
#include"common.hpp"
#include"game.hpp"
#include<mutex>
#include<atomic>

class SupervisedLearner : BaseTrainer {
public:
    //--------------------
    //    公開メソッド
    //--------------------
    //コンストラクタ
    explicit SupervisedLearner(std::string settings_file_path);

    //学習を開始する関数
    void train();

private:
    //-----------------------------------------------------
    //    定数や、ファイルから読み込むためconst化はして
    //    いないがほぼ定数であるもの
    //-----------------------------------------------------
    //学習する棋譜があるディレクトリへのパス
    std::string KIFU_PATH;

    //L2正則化の強さ
    float WEIGHT_DECAY;

    //学習する棋譜の数
    uint64_t GAME_NUM;

    //Early Stoppingする長さ
    int64_t PATIENCE;
};

#endif // !SUPERVISED_LEARNER_HPP