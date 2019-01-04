#pragma once

#ifndef LOAD_GAME_HPP
#define LOAD_GAME_HPP

#include"move.hpp"
#include"eval_params.hpp"
#include<string>
#include<vector>
#include<experimental/filesystem>

struct Game {
	//指し手のvector
	std::vector<Move> moves;

    //教師データのvector
    std::vector<TeacherType> teachers;
	
    //対局結果
    static constexpr double RESULT_BLACK_WIN = 1.0;
    static constexpr double RESULT_WHITE_WIN = 0.0;
    static constexpr double RESULT_DRAW_REPEAT = -1.0;
    static constexpr double RESULT_DRAW_OVER_LIMIT = -2.0;
	double result;

    void writeCSAFile(std::string dir_path) const;
    void writeKifuFile(std::string dir_path) const;
};

Game loadGameFromCSA(std::experimental::filesystem::path p);
std::vector<Game> loadGames(std::string path, uint64_t num);
void cleanGames(std::string path);

#endif // !LOAD_GAME_HPP