#ifndef GAME_HPP
#define GAME_HPP

#include"move.hpp"
#include"neural_network.hpp"
#include<string>
#include<vector>
#include<experimental/filesystem>

struct OneTurnElement {
	Move move;
	TeacherType teacher;
	ValueType nn_output_value;
	std::vector<float> nn_output_policy;
};

struct Game {
    std::vector<OneTurnElement> elements;
	
    //対局結果
    static constexpr double RESULT_BLACK_WIN = MAX_SCORE;
    static constexpr double RESULT_WHITE_WIN = MIN_SCORE;
    static constexpr double RESULT_DRAW_REPEAT = -2.0;
    static constexpr double RESULT_DRAW_OVER_LIMIT = -3.0;
	double result;

    void writeKifuFile(const std::string& dir_path) const;
};

Game loadGameFromCSA(const std::experimental::filesystem::path& p);
std::vector<Game> loadGames(const std::string& path, int64_t num);
void cleanGames();

#endif // !GAME_HPP