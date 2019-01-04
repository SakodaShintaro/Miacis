#pragma once

#ifndef BONANZA_METHOD_TRAINER_HPP
#define BONANZA_METHOD_TRAINER_HPP

#include"base_trainer.hpp"
#include"piece.hpp"
#include"common.hpp"
#include"piece_state.hpp"
#include"game.hpp"
#include"eval_params.hpp"
#include"network.hpp"
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
    //--------------------
    //    内部メソッド
    //--------------------
    //各スレッドで動かす学習を行う関数
    void trainSlave(uint32_t thread_id);

    //pvをたどった先の局面の特徴を得る関数
    Features getLeafFeatures(Position& pos, const std::vector<Move>& pv);

    //更新作業をまとめた関数
    void update();

    //損失関数とその導関数
    double loss_function(int score_diff);
    double d_loss_function(int score_diff);

    //-----------------------------------------------------
    //    定数や、ファイルから読み込むためconst化はして
    //    いないがほぼ定数であるもの
    //-----------------------------------------------------
    //学習する棋譜があるディレクトリへのパス
    std::string KIFU_PATH;

    //L1正則化の強さ
    double L1_GRAD_COEFFICIENT;

    //損失関数にsigmoidを使うか線形関数を使うか
    int32_t LOSS_FUNCTION_MODE;

    //シグモイド関数のゲイン
    //どのくらいの値にすればいいのかはよくわからないけどとりあえずは http://woodyring.blog.so-net.ne.jp/2015-02-06-1 を参考に
    static constexpr double gain = 0.02740;

    //線形関数の傾き
    static constexpr double linear_coefficient = 1.0;

    //------------
    //    変数
    //------------
    //学習する棋譜
    std::vector<Game> games_;

    //学習する棋譜の数
    uint64_t game_num_;

    //次に学習する棋譜のid
    std::atomic<int32_t> game_index_;

    //勾配
    std::unique_ptr<EvalParams<LearnEvalType>> grad_;

    //勾配の平均(Root Mean Square):AdaGrad,RMSProp,AdaDeltaの時に使う.AdaGradの時は二乗の和だけど、まぁ別に用意するのも変な気がするので
    std::unique_ptr<EvalParams<LearnEvalType>> RMSgrad_;

    //パラメータの移動量の平均(Root Mean Square):AdaDeltaの時に使う
    std::unique_ptr<EvalParams<LearnEvalType>> RMSdelta_;

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

#ifdef USE_NN
    //指し手の一致具合計算
    std::vector<int32_t> ordering_num_;
#endif

    //損失
#ifdef USE_NN
    std::array<double, 2> loss_;
#else
    double loss_;
#endif
};

#endif // !BONANZA_METHOD_TRAINER_HPP