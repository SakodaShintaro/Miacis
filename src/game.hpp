#ifndef GAME_HPP
#define GAME_HPP

#include"include_switch.hpp"
#include"neural_network.hpp"
#include<string>
#include<vector>
#include<experimental/filesystem>

struct OneTurnElement {
	Move move;
	FloatType score;
    PolicyType nn_output_policy;
    PolicyTeacherType policy_teacher;
    ValueType nn_output_value;
};

struct Game {
    std::vector<OneTurnElement> elements;
	
    //対局結果
    static constexpr double RESULT_BLACK_WIN = MAX_SCORE;
    static constexpr double RESULT_WHITE_WIN = MIN_SCORE;
    static constexpr double RESULT_DRAW_REPEAT = -2.0;
    static constexpr double RESULT_DRAW_OVER_LIMIT = -3.0;
	float result;

    void writeKifuFile(const std::string& dir_path) const;
};

#endif // !GAME_HPP