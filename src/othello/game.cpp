#include "../game.hpp"

//key[value]となっているvalueの部分を探す
//valueと、valueの右カッコの次のインデックスを返す(連続的に読み込みたい場合に次回読み込み始める場所)
std::pair<std::string, uint64_t> extractValue(const std::string& source, const std::string& key, int64_t start = 0) {
    uint64_t index = source.find(key + "[", start);
    if (index == std::string::npos) {
        //見つからなかったということ
        return std::make_pair(std::string("NULL_KEY"), std::string::npos);
    }

    //見つかった場合、key[value]となっているvalueの部分を返す
    //index以降の最初の[と]を探す
    uint64_t left = source.find('[', index);
    if (left == std::string::npos) {
        return std::make_pair(std::string("NULL_LEFT"), std::string::npos);
    }
    uint64_t right = source.find(']', index);
    if (right == std::string::npos) {
        return std::make_pair(std::string("NULL_RIGHT"), std::string::npos);
    }

    return std::make_pair(source.substr(left + 1, right - left - 1), right + 1);
}

std::pair<Game, bool> parseGGF(const std::string& ggf_str, float rate_threshold) {
    //合法性確認
    //(1)レート
    //読み込んで比較
    std::string black_rate_str = extractValue(ggf_str, "RB").first;
    std::string white_rate_str = extractValue(ggf_str, "RW").first;
    float black_rate = stod(black_rate_str);
    float white_rate = stod(white_rate_str);
    if (black_rate < rate_threshold || white_rate < rate_threshold) {
        return std::make_pair(Game(), false);
    }

    //(2)初期局面
    //初期局面からではない対局を混じっているようなので、一応弾く

    //初期局面のstring
    const std::string INITIAL_BOARD_STR = "8 -------- -------- -------- ---O*--- ---*O--- -------- -------- -------- *";

    //読み込んで比較
    std::pair<std::string, uint64_t> board_value = extractValue(ggf_str, "BO");
    std::string board_str = board_value.first;
    if (board_str != INITIAL_BOARD_STR) {
        return std::make_pair(Game(), false);
    }

    //(3)最終結果
    std::pair<std::string, uint64_t> result_value = extractValue(ggf_str, "RE");
    if (result_value.second == std::string::npos) {
        return std::make_pair(Game(), false);
    }
    std::string result_str = result_value.first;
    if (result_str.find(':') != std::string::npos) {
        //追加フラグがある場合、:rなら投了、:tならtime up、:sならmutual score
        //どれも怪しい気がするのでとりあえず除去
        return std::make_pair(Game(), false);
    }

    Position pos;
    Game game;

    //盤面の後に指し手が続くという前提で、そこから探していく
    int64_t index = board_value.second;

    std::string key = "B";
    while (true) {
        std::pair<std::string, uint64_t> move_value = extractValue(ggf_str, key, index);
        if (move_value.second == std::string::npos) {
            break;
        }

        //指し手の後に秒数や評価値が記述されている場合もあるので最初の'/'までで切る
        std::string move_str = move_value.first.substr(0, move_value.first.find('/'));

        Move move = stringToMove(move_str);
        if (!pos.isLegalMove(move)) {
            pos.print();
            std::cout << "move_str = [" << move_value.first << "]" << std::endl;
            std::cout << "move     = [" << move.toPrettyStr() << "]" << std::endl;
            std::cout << ggf_str << std::endl;
            return std::make_pair(Game(), false);
        }

        OneTurnElement element;
        element.move = move;
        game.elements.push_back(element);

        pos.doMove(move);
        key = (key == "B" ? "W" : "B");
        index = move_value.second;
    }

    int64_t score = std::stoll(result_value.first);

    game.result = (score == 0 ? (MAX_SCORE + MIN_SCORE) / 2 : score > 0 ? MAX_SCORE : MIN_SCORE);
    return std::make_pair(game, true);
}

std::vector<Game> loadGames(const std::string& path, float rate_threshold) {
    namespace sys = std::experimental::filesystem;
    const sys::path dir(path);
    std::vector<Game> games;

    //Generic Game Format(GGF : https://skatgame.net/mburo/ggsa/ggf)を読み込むようにする
    for (sys::directory_iterator p(dir); p != sys::directory_iterator(); p++) {
        std::ifstream ifs(p->path());
        std::string buf;
        while (getline(ifs, buf)) {
            //1行に2対局以上が記述されている場合があるので分解する
            uint64_t start_index = 0;
            while (true) {
                uint64_t left = buf.find("(;", start_index);
                if (left == std::string::npos) {
                    break;
                }
                uint64_t right = buf.find(";)", start_index) + 2;
                //[left, right)のところが1ゲーム分に相当
                std::string game_str = buf.substr(left, right - left);
                std::pair<Game, bool> parse_result = parseGGF(game_str, rate_threshold);
                if (parse_result.second) {
                    games.push_back(parse_result.first);
                }
                start_index = right;
            }
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
        ofs << "**対局 評価値 " << (m.color() == BLACK ? elements[i].score : -elements[i].score) * 5000 << std::endl;

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