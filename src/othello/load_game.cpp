#include "load_game.hpp"

namespace sys = std::experimental::filesystem;

std::vector<Game> loadGames(const std::string& path, int64_t num) {
    const sys::path dir(path);
    std::vector<Game> games;
    //必要であればここに読み込み関数を実装する
    return games;
}

void cleanGames() {
    std::cout << "棋譜のあるフォルダへのパス : ";
    std::string path;
    std::cin >> path;

    //必要に応じて実装する

    std::cout << "finish cleanGames" << std::endl;
}

void Game::writeKifuFile(const std::string& dir_path) const {
    static std::atomic<int64_t> id;
    std::string file_name = dir_path + std::to_string(id++) + ".kifu";
    std::ofstream ofs(file_name);
    if (!ofs) {
        std::cerr << "cannot open " << dir_path << std::endl;
        exit(1);
    }
    ofs << std::fixed;

    Position pos;

    for (uint64_t i = 0; i < elements.size(); i++) {
        Move m = elements[i].move;
        ofs << i + 1 << " ";
        File to_file = SquareToFile[m.to()];
        Rank to_rank = SquareToRank[m.to()];
        ofs << fileToString[to_file] << rankToString[to_rank];
        ofs << "**対局 評価値 " << (m.color() == BLACK ? elements[i].score : -elements[i].score) * 1000
            << std::endl;

        pos.doMove(m);
    }

    ofs << elements.size() + 1 << " ";
    if (result == MAX_SCORE || result == MIN_SCORE) {
        ofs << "投了" << std::endl;
    } else {
        //千日手の場合もこうなり得るが、判別する手段がないのでとりあえず全て持将棋としておく
        ofs << "持将棋" << std::endl;
    }
}