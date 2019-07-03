﻿#include"game.hpp"
#include"position.hpp"
#include<iostream>
#include<algorithm>
#include<string>
#include<unordered_map>
#include<experimental/filesystem>
#include<atomic>

namespace sys = std::experimental::filesystem;

Game loadGameFromCSA(const sys::path& p) {
    //対応関係をunordered_mapで引けるようにしておく
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

    Position pos;
	Game game;
	std::ifstream ifs(p);
	std::string buf;
	while (getline(ifs, buf)) {
        //指し手じゃないものはスキップ
        if (buf[0] == '\'' || (buf[0] != '+' && buf[0] != '-') || buf.size() == 1) {
            continue;
        }

        //指し手の情報を取得
		Square from = FRToSquare[buf[1] - '0'][buf[2] - '0'];
		Square to   = FRToSquare[buf[3] - '0'][buf[4] - '0'];
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
            std::cerr << "There is a illegal move in " << p << std::endl;
            move.printWithNewLine();
            exit(1);
        }
        OneTurnElement element;
        element.move = move;
		game.elements.push_back(element);
        pos.doMove(move);
	}
    game.result = (pos.color() == BLACK ? Game::RESULT_WHITE_WIN : Game::RESULT_BLACK_WIN);
	return game;
}

std::vector<Game> loadGames(const std::string& path, int64_t num) {
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

void cleanGames() {
    std::cout << "棋譜のあるフォルダへのパス : ";
    std::string path;
    std::cin >> path;

    constexpr double rate_threshold = 2800;
    constexpr int32_t move_threshold = 50;

    const sys::path dir(path);
	for (sys::directory_iterator p(dir); p != sys::directory_iterator(); p++) {
		std::ifstream ifs(p->path());
		std::string buf;
		int32_t move_count = 0;
        bool should_remove = false;
		while (getline(ifs, buf)) {
			//レート読み込み
			if (buf.find("'black_rate") < buf.size() || buf.find("'white_rate") < buf.size()) {
				double rate = std::stod(buf.substr(buf.rfind(':') + 1));
				if (rate < rate_threshold) {
				    should_remove = true;
				    break;
				}
				continue;
			}

			//summaryが投了以外のものは削除
			if (buf.find("summary") < buf.size()) {
				if (buf.find("toryo") > buf.size()) {
                    should_remove = true;
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

		//上で削除すべきと判断されたもの及び手数が短すぎるものを削除
        if (should_remove || move_count < move_threshold) {
            ifs.close();
            std::cout << p->path().c_str() << " ";
            if (remove(p->path().c_str()) == 0) {
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
    std::string file_name = dir_path + std::to_string(id++) + ".kif";
    std::ofstream ofs(file_name);
    if (!ofs) {
        std::cerr << "cannot open " << dir_path << std::endl;
        exit(1);
    }
    ofs << std::fixed;

    Position pos;

    for (int32_t i = 0; i < elements.size(); i++) {
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
        ofs << "**対局 評価値 " << (pieceToColor(m.subject()) == BLACK ? elements[i].score : -elements[i].score) * 1000 << std::endl;

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