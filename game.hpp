#ifndef GAME_HPP
#define GAME_HPP

#include"move.hpp"
#include"neural_network.hpp"
#include<string>
#include<vector>
#include<experimental/filesystem>

struct Game {
	//指し手のvector
	std::vector<Move> moves;

    //教師データのvector
    std::vector<TeacherType> teachers;
	
    //対局結果
    static constexpr double RESULT_BLACK_WIN = MAX_SCORE;
    static constexpr double RESULT_WHITE_WIN = MIN_SCORE;
    static constexpr double RESULT_DRAW_REPEAT = -2.0;
    static constexpr double RESULT_DRAW_OVER_LIMIT = -3.0;
	double result;

    void writeKifuFile(std::string dir_path) const;
};

Game loadGameFromCSA(std::experimental::filesystem::path p);
std::vector<Game> loadGames(std::string path, int64_t num);
void cleanGames(std::string path);

#endif // !GAME_HPP