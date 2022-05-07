#ifndef MIACIS_GAME_HPP
#define MIACIS_GAME_HPP

#include "../model/model_common.hpp"
#ifdef _MSC_VER
#include <filesystem>
#elif __GNUC__
#include <experimental/filesystem>
#endif
#include "move.hpp"

struct OneTurnElement {
    //このターンに選択された指し手
    Shogi::Move move;

    //探索した結果として得られた方策分布, 評価値
    //それぞれデータとして小さく済む保存方法にするとPolicyは教師データの型(int32_t, float)のペア, Valueは評価値(float)だけとなる
    PolicyTeacherType policy_teacher;
    float score;

    //探索以前のニューラルネットワークの出力。PERへ最初にデータを入れる際、priorityを計算するのに必要
    PolicyType nn_output_policy;
    ValueType nn_output_value;
};

struct Game {
    //各ターンの情報
    std::vector<OneTurnElement> elements;

    //先手側から見た対局結果
    float result;

    //kifu形式でdir_path以下に保存する関数。基本的に将棋で使うことを想定している
    void writeKifuFile(const std::string& dir_path) const;
};

std::vector<Game> loadGames(const std::string& path, float rate_threshold);
void cleanGames();

#endif //MIACIS_GAME_HPP