#include "../game.hpp"

namespace sys = std::experimental::filesystem;

std::vector<Game> loadGames(const std::string& path) {
    const sys::path dir(path);
    std::vector<Game> games;

    constexpr double RATE_THRESHOLD = 1000;

    //Generic Game Format(GGF : https://skatgame.net/mburo/ggsa/ggf)を読み込むようにする

    //key[value]となっているvalueの部分を探す
    //valueと、valueの右カッコの次のインデックスを返す(連続的に読み込みたい場合に次回読み込み始める場所)
    auto extractByGGF = [](const std::string& source, const std::string& key, int64_t start = 0) {
        auto index = source.find(key + "[", start);
        if (index == std::string::npos) {
            //見つからなかったということ
            return std::make_pair(std::string("NULL_KEY"), std::string::npos);
        }

        //見つかった場合、key[value]となっているvalueの部分を返す
        //index以降の最初の[と]を探す
        auto left = source.find('[', index);
        if (left == std::string::npos) {
            return std::make_pair(std::string("NULL_LEFT"), std::string::npos);
        }
        auto right = source.find(']', index);
        if (right == std::string::npos) {
            return std::make_pair(std::string("NULL_RIGHT"), std::string::npos);
        }

        return std::make_pair(source.substr(left + 1, right - left - 1), right + 1);
    };

    for (sys::directory_iterator p(dir); p != sys::directory_iterator(); p++) {
        std::ifstream ifs(p->path());
        std::string buf;
        while (getline(ifs, buf)) {
            //std::cout << buf << std::endl;
            Position pos;
            Game game;

            //レート確認
            double black_rate = 0, white_rate = 0;
            std::string black_rate_str = extractByGGF(buf, "RB").first;
            std::string white_rate_str = extractByGGF(buf, "RW").first;
            black_rate = stod(black_rate_str);
            white_rate = stod(white_rate_str);

            if (black_rate < RATE_THRESHOLD || white_rate < RATE_THRESHOLD) {
                continue;
            }

            //盤面確認
            //初期局面からではない対局を混じっているようなので、一応弾く
            //別に弾く必要もないかもしれないが
            auto board_value = extractByGGF(buf, "BO");
            std::string board_str = board_value.first;
            const std::string initial_board = "8 -------- -------- -------- ---O*--- ---*O--- -------- -------- -------- *";
            if (board_str != initial_board) {
                continue;
            }

            int64_t index = board_value.second;
            std::string key = "B";
            while (true) {
                auto move_value = extractByGGF(buf, key, index);
                if (move_value.second == std::string::npos) {
                    break;
                }
                std::string move_str = move_value.first.substr(0, move_value.first.find('/'));
                Move move = stringToMove(move_str);
                if (!pos.isLegalMove(move)) {
                    std::cout << "illegal move!" << std::endl;
                    pos.print();
                    std::cout << "move_str =" << move_value.first << std::endl;
                    std::cout << "move -> " << move.toPrettyStr() << std::endl;
                    std::cout << buf << std::endl;
                    std::exit(1);
                }

                OneTurnElement element;
                element.move = move;
                game.elements.push_back(element);

                pos.doMove(move);
                key = (key == "B" ? "W" : "B");
                index = move_value.second;
            }

            auto result_value = extractByGGF(buf, "RE");
            if (result_value.second == std::string::npos) {
                continue;
            }
            std::string result_str = result_value.first;
            if (result_str.find(':') != std::string::npos) {
                //追加フラグがある場合、:rなら投了、:tならtime up、:sならmutual score
                //どれも怪しい気がするのでとりあえず除去
                continue;
            }
            int64_t score = std::stoll(result_str);

            game.result = (score == 0 ? (MAX_SCORE + MIN_SCORE) / 2 : score > 0 ? MAX_SCORE : MIN_SCORE);
            games.push_back(game);
        }
    }

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
        ofs << "**対局 評価値 " << (m.color() == BLACK ? elements[i].score : -elements[i].score) * 5000
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