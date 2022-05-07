#include "../game.hpp"
#include "../shogi/position.hpp"
using namespace Shogi;

#ifdef _MSC_VER
namespace sys = std::filesystem;
#elif __GNUC__
namespace sys = std::experimental::filesystem;
#endif

//対応関係をunordered_mapで引けるようにしておく
static const std::unordered_map<std::string, Piece> CSAstringToPiece = {
    { "FU", PAWN },           { "KY", LANCE },          { "KE", KNIGHT },         { "GI", SILVER },       { "KI", GOLD },
    { "KA", BISHOP },         { "HI", ROOK },           { "OU", KING },           { "TO", PAWN_PROMOTE }, { "NY", LANCE_PROMOTE },
    { "NK", KNIGHT_PROMOTE }, { "NG", SILVER_PROMOTE }, { "UM", BISHOP_PROMOTE }, { "RY", ROOK_PROMOTE },
};

std::tuple<Game, bool> loadCSAOneGame(std::ifstream& ifs, float rate_threshold) {
    Position pos;
    Game game;
    std::string buf;
    std::string sfen_board, sfen_hand;
    float black_rate = 0, white_rate = 0;
    bool ok = false;
    while (getline(ifs, buf)) {
        //レート読み込み
        if (buf.find("'black_rate") < buf.size()) {
            black_rate = std::stod(buf.substr(buf.rfind(':') + 1));
        } else if (buf.find("'white_rate") < buf.size()) {
            white_rate = std::stod(buf.substr(buf.rfind(':') + 1));
        } else if (buf[0] == 'P') {
            //盤面の情報を表す
            if (buf[1] == 'I') {
                sfen_board = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL";
            } else if ('1' <= buf[1] && buf[1] <= '9') {
                //各段の情報が並ぶ
                //3文字ずつ情報が並んでいるので解釈する
                std::string curr_rank_str;
                int64_t curr_empty_num = 0;
                for (int64_t j = 0; j < BOARD_WIDTH; j++) {
                    std::string curr_piece_str = buf.substr(2 + j * 3, 3);
                    if (curr_piece_str == " * ") {
                        curr_empty_num++;
                    } else {
                        Piece p = CSAstringToPiece.at(curr_piece_str.substr(1));
                        p = coloredPiece((curr_piece_str[0] == '+' ? BLACK : WHITE), p);

                        if (curr_empty_num != 0) {
                            curr_rank_str += std::to_string(curr_empty_num);
                        }
                        curr_rank_str += PieceToSfenStr2[p];
                        curr_empty_num = 0;
                    }
                }
                if (curr_empty_num != 0) {
                    curr_rank_str += std::to_string(curr_empty_num);
                }

                sfen_board += curr_rank_str + (buf[1] != '9' ? "/" : "");
            } else if (buf[1] == '+' || buf[1] == '-') {
                //先手の持ち駒
                //dlshogiの保存形式を前提とする
                //(1)1行には1種類だけ
                //(2)同じ駒種は持っている枚数だけ並ぶ
                //P+00FU00FU00FU
                //P+00KE
                //P+00KA
                //という感じ
                Piece p = CSAstringToPiece.at(buf.substr(4, 2));
                p = coloredPiece((buf[1] == '+' ? BLACK : WHITE), p);
                int64_t num = (buf.size() - 2) / 4;
                sfen_hand += (num != 1 ? std::to_string(num) : "") + PieceToSfenStr2[p];
            } else {
                std::cout << "想定外 buf = " << buf << std::endl;
                std::exit(1);
            }
        } else if (buf == "+" || buf == "-") {
            //手番を示す
            //盤面を構築
            if (sfen_hand.empty()) {
                sfen_hand = "-";
            }
            std::string sfen = sfen_board + (buf == "+" ? " b " : " w ") + sfen_hand + " 1";
            pos.fromStr(sfen);
        } else if (buf[0] == '+' || buf[0] == '-') {
            //指し手の情報を取得
            Square from = FRToSquare[buf[1] - '0'][buf[2] - '0'];
            Square to = FRToSquare[buf[3] - '0'][buf[4] - '0'];
            Piece subject = CSAstringToPiece.at(buf.substr(5, 2));
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
                pos.print();
                std::cerr << "There is a illegal move " << move.toPrettyStr() << std::endl;
                exit(1);
            }
            OneTurnElement element;
            element.move = move;
            game.elements.push_back(element);
            pos.doMove(move);
        } else if (buf[0] == '%') { //最終的な結果
            if (buf.substr(0, 6) == "%TORYO") {
                game.result = (pos.color() == BLACK ? MIN_SCORE : MAX_SCORE);
                ok = true;
            } else if (buf.substr(0, 6) == "%KACHI") {
                game.result = (pos.color() == BLACK ? MAX_SCORE : MIN_SCORE);
                ok = true;
            } else if (buf.substr(0, 11) == "%SENNICHITE" || buf.substr(0, 7) == "%CHUDAN" ||
                       buf.substr(0, 16) == "%+ILLEGAL_ACTION" || buf.substr(0, 16) == "%-ILLEGAL_ACTION" ||
                       buf.substr(0, 8) == "%TIME_UP") {
                //ダメな対局であったというフラグを返す
                return std::make_tuple(game, false);
            } else {
                std::cout << buf << std::endl;
                std::exit(1);
            }
            break;
        } else {
            continue;
        }
    }

    float max_rate = std::max(black_rate, white_rate);

    //60手以上でレートが閾値以上のものを採用
    ok = ok && (game.elements.size() >= 60 && max_rate >= rate_threshold);
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
                if (games.size() >= 100000) {
                    break;
                }
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