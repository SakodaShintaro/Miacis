#include"game.hpp"
#include"position.hpp"
#include<iostream>
#include<algorithm>
#include<string>
#include<unordered_map>
#include<experimental/filesystem>
#include<atomic>

namespace sys = std::experimental::filesystem;

static std::unordered_map<std::string, Piece> CSAstringToPiece = {
	{ "FU", PAWN },
	{ "KY", LANCE },
	{ "KE", KNIGHT },
	{ "GI", SILVER },
	{ "KI", GOLD },
	{ "KA", BISHOP },
	{ "HI", ROOK },
	{ "OU", KING },
	{ "TO", PAWN_PROMOTE },
	{ "NY", LANCE_PROMOTE },
	{ "NK", KNIGHT_PROMOTE },
	{ "NG", SILVER_PROMOTE },
	{ "UM", BISHOP_PROMOTE },
	{ "RY", ROOK_PROMOTE },
};

Game loadGameFromCSA(sys::path p) {
    Position pos;
	Game game;
	std::ifstream ifs(p);
	std::string buf;
	while (getline(ifs, buf)) {
		if (buf[0] == '\'') continue;
		if (buf[0] != '+' && buf[0] != '-') continue;
		if (buf.size() == 1) continue;

		//上の分岐によりMoveだけ残る
		Square from = FRToSquare[buf[1] - '0'][buf[2] - '0'];
		Square to   = FRToSquare[buf[3] - '0'][buf[4] - '0'];
        //移動駒の種類
		Piece subject = CSAstringToPiece[buf.substr(5, 2)];
        //手番を設定
        subject = (pos.color() == BLACK ? toBlack(subject) : toWhite(subject));
        bool isDrop = (from == WALL00);

        //CSAのフォーマットから、動くものが成済なのにfromにある駒が不成の場合、成る手
        bool isPromote = ((subject & PROMOTE) && !(pos.on(from) & PROMOTE));
        if (isPromote) {
            subject = (Piece)(subject & ~PROMOTE);
        }
        Move move(to, from, isDrop, isPromote, subject);
        move = pos.transformValidMove(move);

        if (!pos.isLegalMove(move)) {
            pos.printForDebug();
            move.print();
            pos.isLegalMove(move);
            std::cout << p << std::endl;
            assert(false);
        }
		game.moves.push_back(move);
        pos.doMove(move);

	}
    game.result = ~pos.color();
	return game;
}

std::vector<Game> loadGames(std::string path, uint64_t num) {
    const sys::path dir(path);
	std::vector<Game> games;
    for (sys::directory_iterator p(dir); p != sys::directory_iterator(); p++) {
        games.push_back(loadGameFromCSA(p->path()));
        if (--num == 0) {
            break;
        }
    }
	return games;
}

void cleanGames(std::string path) {
    std::cout << "start cleanGames" << std::endl;
    const sys::path dir(path);
	for (sys::directory_iterator p(dir); p != sys::directory_iterator(); p++) {
		std::ifstream ifs(p->path());
		std::string buf;
		int move_counter = 0;
        double black_rate = 0, white_rate = 0;
		while (getline(ifs, buf)) {
			//名前読み込み
			//if (buf[0] == 'N') {
			//	if (buf[1] == '+') {
			//		std::cout << "先手名前:" << buf.substr(2) << std::endl;
			//		continue;
			//	} else if (buf[1] == '-') {
			//		std::cout << "後手名前:" << buf.substr(2) << std::endl;
			//		continue;
			//	} else {
			//		assert(false);
			//	}
			//}

			//レート読み込み
			//2800より小さかったら削除
			if (buf.find("'black_rate") < buf.size()) {
				//std::cout << "先手レート:" << buf.substr(buf.rfind(':') + 1) << std::endl;
				black_rate = std::stod(buf.substr(buf.rfind(':') + 1));
                continue;
			} else if (buf.find("'white_rate") < buf.size()) {
				//std::cout << "後手レート:" << buf.substr(buf.rfind(':') + 1) << std::endl;
				white_rate = std::stod(buf.substr(buf.rfind(':') + 1));
				continue;
			}

			//summaryが投了以外のものは削除
			if (buf.find("summary") < buf.size()) {
				if (buf.find("toryo") > buf.size()) {
					ifs.close();
					std::cout << p->path().string().c_str() << " ";
					if (remove(p->path().string().c_str()) == 0) {
						std::cout << "削除成功" << std::endl;
					} else {
						std::cout << "失敗" << std::endl;
					}
					goto END;
				}
				continue;
			}

			//std::cout << buf << std::endl;
			if (buf[0] == '\'') continue;
			if (buf[0] != '+' && buf[0] != '-') continue;
			if (buf.size() == 1) continue;
			move_counter++;
		}

        //printf("black_rate = %f, white_rate = %f\n", black_rate, white_rate);
        if (black_rate < 2800) {
            ifs.close();
            std::cout << p->path().string().c_str() << " ";
            if (remove(p->path().string().c_str()) == 0) {
                std::cout << "削除成功" << std::endl;
            } else {
                std::cout << "失敗" << std::endl;
            }
            continue;
        }
        if (white_rate < 2800) {
            ifs.close();
            std::cout << p->path().string().c_str() << " ";
            if (remove(p->path().string().c_str()) == 0) {
                std::cout << "削除成功" << std::endl;
            } else {
                std::cout << "失敗" << std::endl;
            }
            continue;
        }

		//手数が50より小さいものはおかしい気がするので削除
		if (move_counter < 50) {
			ifs.close();
			std::cout << p->path().string().c_str() << " ";
			if (remove(p->path().string().c_str()) == 0) {
				std::cout << "削除成功" << std::endl;
			} else {
				std::cout << "失敗" << std::endl;
			}
            continue;
		}

	END:;
	}
    std::cout << "finish cleanGames" << std::endl;
}

void Game::writeCSAFile(std::string dir_path) const {
    static int64_t id = 0;
    static const ArrayMap<std::string, PieceNum> pieceToCSAstring({
        { PAWN, "FU" },
        { LANCE, "KY" },
        { KNIGHT, "KE" },
        { SILVER, "GI" },
        { GOLD, "KI" },
        { BISHOP, "KA" },
        { ROOK, "HI" },
        { KING, "OU" },
        { PAWN_PROMOTE, "TO" },
        { LANCE_PROMOTE, "NY"},
        { KNIGHT_PROMOTE, "NK" },
        { SILVER_PROMOTE, "NG" },
        { BISHOP_PROMOTE, "UM" },
        { ROOK_PROMOTE, "RY" }
    });

    std::string file_name = dir_path + std::to_string(id++) + ".csa";
    std::ofstream ofs(file_name);
    if (!ofs) {
        std::cout << "cannnot open " << dir_path << std::endl;
        assert(false);
    }
    ofs << "V2.2" << std::endl;
    ofs << "P1-KY-KE-GI-KI-OU-KI-GI-KE-KY" << std::endl;
    ofs << "P2 * -HI *  *  *  *  * -KA * " << std::endl;
    ofs << "P3-FU-FU-FU-FU-FU-FU-FU-FU-FU" << std::endl;
    ofs << "P4 *  *  *  *  *  *  *  *  * " << std::endl;
    ofs << "P5 *  *  *  *  *  *  *  *  * " << std::endl;
    ofs << "P6 *  *  *  *  *  *  *  *  * " << std::endl;
    ofs << "P7+FU+FU+FU+FU+FU+FU+FU+FU+FU" << std::endl;
    ofs << "P8 * +KA *  *  *  *  * +HI * " << std::endl;
    ofs << "P9+KY+KE+GI+KI+OU+KI+GI+KE+KY" << std::endl;
    ofs << "+" << std::endl;
    for (Move m : moves) {
        ofs << (pieceToColor(m.subject()) == BLACK ? "+" : "-");
        int32_t from_num = SquareToNum[m.from()];
        int32_t to_num = SquareToNum[m.to()];
        if (m.isDrop()) {
            ofs << "00";
        } else {
            ofs << from_num / 9 + 1 << from_num % 9 + 1;
        }
        ofs << to_num / 9 + 1 << to_num % 9 + 1;
        Piece subject = m.subject();
        if (m.isPromote()) {
            subject = promote(subject);
        }
        if (pieceToColor(subject) == BLACK) {
            subject = (Piece)(subject ^ BLACK_FLAG);
        } else {
            subject = (Piece)(subject ^ WHITE_FLAG);
        }
        ofs << pieceToCSAstring[subject] << std::endl;
        ofs << "\'" << m.score << std::endl;
    }
    
    if (result == RESULT_BLACK_WIN || result == RESULT_WHITE_WIN) {
        ofs << "%TORYO" << std::endl;
    } else if (result == RESULT_DRAW_REPEAT) {
        ofs << "%SENNICHITE" << std::endl;
    } else {
        ofs << "%HIKIWAKE" << std::endl;
    }
}

void Game::writeKifuFile(std::string dir_path) const {
    static std::atomic<int64_t> id;
    std::string file_name = dir_path + std::to_string(id++) + ".kif";
    std::ofstream ofs(file_name);
    if (!ofs) {
        std::cout << "cannot open " << dir_path << std::endl;
        assert(false);
    }
    ofs << std::fixed;

    Position pos;

    for (int32_t i = 0; i < moves.size(); i++) {
        Move m = moves[i];
        ofs << i + 1 << " ";
        File to_file = SquareToFile[m.to()];
        Rank to_rank = SquareToRank[m.to()];
        ofs << fileToString[to_file] << rankToString[to_rank];
        auto subject = (Piece)(m.subject() & (PROMOTE | PIECE_KIND_MASK));
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
        ofs << "**対局 評価値 " << (pieceToColor(m.subject()) == BLACK ? m.score : -m.score) << std::endl;

        pos.doMove(m);
    }
    
    ofs << moves.size() + 1 << " ";
    if (result == RESULT_BLACK_WIN || result == RESULT_WHITE_WIN) {
        ofs << "投了" << std::endl;
    } else if (result == RESULT_DRAW_REPEAT) {
        ofs << "千日手" << std::endl;
    } else {
        ofs << "持将棋" << std::endl;
    }
}
