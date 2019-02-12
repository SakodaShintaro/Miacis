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
    //棋譜を用いてvalidationを行う関数
    void validation(int64_t step_num);

    //---------------------------------------------
    //    ファイルから読み込むためconst化はして
    //    いないがほぼ定数であるもの
    //---------------------------------------------
    //引き分けの対局も学習するか
    bool USE_DRAW_GAME;

    //評価する間隔
    int64_t EVALUATION_INTERVAL;

    //ステップ数
    int64_t MAX_STEP_NUM;

    //並列に対局を生成する数
    int64_t PARALLEL_NUM;

    //validationに用いる棋譜へのパス
    std::string VALIDATION_KIFU_PATH;

    //validationで用いる局面数
    int64_t VALIDATION_SIZE;

    //------------
    //    変数
    //------------
    //学習用に加工済の局面スタック
    ReplayBuffer replay_buffer_;
};

#endif