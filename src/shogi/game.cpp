#include "../game.hpp"

#ifdef _MSC_VER
namespace sys = std::filesystem;
#elif __GNUC__
namespace sys = std::experimental::filesystem;
#endif

//対応関係をunordered_mapで引けるようにしておく
static std::unordered_map<std::string, Piece> CSAstringToPiece = {
    { "FU", PAWN },           { "KY", LANCE },          { "KE", KNIGHT },         { "GI", SILVER },       { "KI", GOLD },
    { "KA", BISHOP },         { "HI", ROOK },           { "OU", KING },           { "TO", PAWN_PROMOTE }, { "NY", LANCE_PROMOTE },
    { "NK", KNIGHT_PROMOTE }, { "NG", SILVER_PROMOTE }, { "UM", BISHOP_PROMOTE }, { "RY", ROOK_PROMOTE },
};

std::tuple<Game, bool> loadCSAOneGame(std::ifstream& ifs, bool rate_threshold) {
    Position pos;
    Game game;
    std::string buf;
    float black_rate = 0, white_rate = 0;
    while (getline(ifs, buf)) {
        //レート読み込み
        if (buf.find("'black_rate") < buf.size()) {
            black_rate = std::stod(buf.substr(buf.rfind(':') + 1));
        } else if (buf.find("'white_rate") < buf.size()) {
            white_rate = std::stod(buf.substr(buf.rfind(':') + 1));
        } else if (buf[0] != '%' && (buf[0] != '+' && buf[0] != '-') || buf.size() == 1) {
            //最終結果あるいは指し手ではないものはスキップ
            continue;
        } else if (buf[0] == '+' || buf[0] == '-') {
            //指し手の情報を取得
            Square from = FRToSquare[buf[1] - '0'][buf[2] - '0'];
            Square to = FRToSquare[buf[3] - '0'][buf[4] - '0'];
            Piece subject = CSAstringToPiece[buf.substr(5, 2)];
            //手番を設定
            subject = (pos.color() == BLACK ? toBlack(subject) : toWhite(subject));
            bool isDrop = (from == WALL00);

            //CSAのフォーマットから、動くものが成済なのにfromにある駒が不成の場合、成る手
            bool isPromote = ((subject & PROMOTE) && !(pos.on(from) & PROMOTE));
            if (isPromote) {
                subject = (Piece)(subject & ~PROMOTE);
            }

            //Moveを生成し、Positionの情報を使って完全なものとする
            Move move(to, from, isDrop, isPromote, subject);
            move = pos.transformValidMove(move);

            if (!pos.isLegalMove(move)) {
                std::cerr << "There is a illegal move " << move.toPrettyStr() << std::endl;
                exit(1);
            }
            OneTurnElement element;
            element.move = move;
            game.elements.push_back(element);
            pos.doMove(move);
        } else if (buf[0] == '%') { //最終的な結果
            if (buf == "%TORYO") {
                game.result = (pos.color() == BLACK ? MIN_SCORE : MAX_SCORE);
            } else if (buf == "%SENNICHITE") {
                game.result = (MAX_SCORE + MIN_SCORE) / 2;
            } else if (buf == "%KACHI") {
                game.result = (pos.color() == BLACK ? MAX_SCORE : MIN_SCORE);
            } else if (buf == "%CHUDAN" || buf == "%+ILLEGAL_ACTION" || buf == "%-ILLEGAL_ACTION") {
                //ダメな対局であったというフラグを返す
                return std::make_tuple(game, false);
            } else {
                std::exit(1);
            }
            break;
        } else {
            std::cout << buf << std::endl;
            std::exit(1);
        }
    }

    float max_rate = std::max(black_rate, white_rate);

    //60手以上でレートが閾値以上のものを採用
    bool ok = (game.elements.size() >= 60 && max_rate >= rate_threshold);
    return std::make_tuple(game, ok);
}

std::vector<Game> loadGames(const std::string& path, float rate_threshold) {
    //CSA形式のファイルがpath以下に複数あるとする
    const sys::path dir(path);
    std::vector<Game> games;
    for (sys::directory_iterator p(dir); p != sys::directory_iterator(); p++) {
        std::ifstream ifs(p->path());
        std::string buf;
        while (true) {
            auto [game, ok] = loadCSAOneGame(ifs, rate_threshold);
            if (ok) {
                games.push_back(game);
            }
            getline(ifs, buf);
            if (buf != "/") {
                break;
            }
        }
    }
    return games;
}

void cleanGames() {
    std::cout << "棋譜のあるフォルダへのパス : ";
    std::string path;
    std::cin >> path;

    float rate_threshold{};
    std::cout << "削除するレートの閾値 : ";
    std::cin >> rate_threshold;
    constexpr int32_t move_threshold = 50;

    const sys::path dir(path);
    for (sys::directory_iterator p(dir); p != sys::directory_iterator(); p++) {
        std::ifstream ifs(p->path());
        std::string buf;
        int32_t move_count = 0;
        bool illegal_summary = false;
        float black_rate = 0, white_rate = 0;
        while (getline(ifs, buf)) {
            //レート読み込み
            if (buf.find("'black_rate") < buf.size()) {
                black_rate = std::stod(buf.substr(buf.rfind(':') + 1));
                continue;
            } else if (buf.find("'white_rate") < buf.size()) {
                white_rate = std::stod(buf.substr(buf.rfind(':') + 1));
                continue;
            }

            //summaryが投了以外のものは削除
            if (buf.find("summary") < buf.size()) {
                if (buf.find("toryo") > buf.size()) {
                    illegal_summary = true;
                    break;
                }
                continue;
            }

            //指し手じゃないものはスキップ
            if (buf[0] == '\'' || (buf[0] != '+' && buf[0] != '-') || buf.size() == 1) {
                continue;
            }

            //手数を増やす
            move_count++;
        }

        //条件を見て削除
        if (illegal_summary || move_count < move_threshold || black_rate < rate_threshold || white_rate < rate_threshold) {
            ifs.close();
            std::string s = p->path().string();
            std::cout << s << " ";
            if (remove(s.c_str()) == 0) {
                std::cout << "削除成功" << std::endl;
            } else {
                std::cout << "失敗" << std::endl;
            }
        }
    }
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
        Piece subject = (Piece)(m.subject() & (PROMOTE | PIECE_KIND_MASK));
        ofs << PieceToStr[subject];
        if (m.isDrop()) {
            ofs << "打" << std::endl;
        } else {
            if (m.isPromote()) {
                ofs << "成";
            }

            int32_t from_num = SquareToNum[m.from()];
            ofs << "(" << from_num / 9 + 1 << from_num % 9 + 1 << ")" << std::endl;
        }
        ofs << "**対局 評価値 " << (pieceToColor(m.subject()) == BLACK ? elements[i].score : -elements[i].score) * 1000
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