#ifndef MIACIS_LOAD_GAME_HPP
#define MIACIS_LOAD_GAME_HPP

#include "../game.hpp"

std::vector<Game> loadGames(const std::string& path, int64_t num);
void cleanGames();

#endif //MIACIS_LOAD_GAME_HPP