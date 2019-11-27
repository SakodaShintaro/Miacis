#include "../game.hpp"

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
    if (result == RESULT_BLACK_WIN || result == RESULT_WHITE_WIN) {
        ofs << "投了" << std::endl;
    } else if (result == RESULT_DRAW_REPEAT) {
        ofs << "千日手" << std::endl;
    } else {
        ofs << "持将棋" << std::endl;
    }
}