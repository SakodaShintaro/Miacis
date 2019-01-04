#pragma once
#include"base_trainer.hpp"
#include"types.hpp"
#include"move.hpp"
#include"eval_params.hpp"
#include"searcher.hpp"
#include"game.hpp"
#include<vector>

class RootstrapTrainer : BaseTrainer {
public:
    //--------------------
    //    公開メソッド
    //--------------------
    //コンストラクタ
    explicit RootstrapTrainer(std::string settings_file_path);

    //各スレッドが非同期的に棋譜生成とパラメータ更新を行う関数
    //learnSyncと選択で片方を使う
    void learnAsync();

    //棋譜生成だけ並列化しパラメータ更新は1スレッドで行う関数
    void learnSync();

    //1局だけ生成してそれを繰り返し学習してみるテスト用の関数
    void testLearn();

    //自己対局を行う関数:AlphaZeroTrainerでも使うのでpublicに置く
    static std::vector<Game> play(int32_t game_num);

    //並列化して対局を行う関数:AlphaZeroTrainerでも使うのでpublicに置く
    static std::vector<Game> parallelPlay(const EvalParams<DefaultEvalType>& curr, const EvalParams<DefaultEvalType>& target, int32_t game_num);

private:
    //--------------------
    //    内部メソッド
    //--------------------
    //棋譜生成・パラメータ更新を行う1スレッド分の関数
    //定期的にチェックを挟む
    void learnAsyncSlave(int32_t id);

    //今ファイルに保存されているパラメータと対局して強さを測定する関数
    void evaluate();

    //棋譜群から損失・勾配・千日手数・長手数による引き分け数を計算する関数
    void learnGames(const std::vector<Game>& games, LossType& loss, EvalParams<LearnEvalType>& grad, int32_t& draw_repeat_num, int32_t& draw_long_game_num);

    //棋譜を初手側から再生して損失・勾配を計算する関数:elmo絞りに対応
    void learnOneGame(const Game& game, EvalParams<LearnEvalType>& grad, LossType& loss, uint64_t& learn_position_num);

    //棋譜を最終手から再生して損失・勾配を計算する関数:Sarsaに対応?
    void learnOneGameReverse(const Game& game, EvalParams<LearnEvalType>& grad, LossType& loss, uint64_t& learn_position_num);

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
    int32_t EVALUATION_INTERVAL;

    //評価するときにランダムで指す手数
    int32_t EVALUATION_RANDOM_TURN;

    //------------
    //    変数
    //------------
    //学習した局面数
    uint64_t sum_learned_games_;

    //強くなって世代が進んだ回数
    uint64_t update_num_;

    //強くなったとみなせなかった回数
    uint64_t fail_num_;

    //強くなっていないことが続いている回数
    uint64_t consecutive_fail_num_;
};