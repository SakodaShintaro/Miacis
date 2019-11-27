#ifndef MIACIS_LOAD_GAME_HPP
#define MIACIS_LOAD_GAME_HPP

#include "../game.hpp"

Game loadGameFromCSA(const std::experimental::filesystem::path& p);
std::vector<Game> loadGames(const std::string& path, int64_t num);
void cleanGames();

#endif //MIACIS_LOAD_GAME_HPP