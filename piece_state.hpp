#pragma once

#include"piece.hpp"

enum PieceState {
	black_hand_pawn = 0,
	white_hand_pawn = black_hand_pawn + 18,
	black_hand_lance = white_hand_pawn + 18,
	white_hand_lance = black_hand_lance + 4,
	black_hand_knight = white_hand_lance + 4,
	white_hand_knight = black_hand_knight + 4,
	black_hand_silver = white_hand_knight + 4,
	white_hand_silver = black_hand_silver + 4,
	black_hand_gold = white_hand_silver + 4,
	white_hand_gold = black_hand_gold + 4,
	black_hand_bishop = white_hand_gold + 4,
	white_hand_bishop = black_hand_bishop + 2,
	black_hand_rook = white_hand_bishop + 2,
	white_hand_rook = black_hand_rook + 2,
	hand_end = white_hand_rook + 2,

	black_pawn = hand_end,
	white_pawn = black_pawn + 81,
	black_lance = white_pawn + 81,
	white_lance = black_lance + 81,
	black_knight = white_lance + 81,
	white_knight = black_knight + 81,
	black_silver = white_knight + 81,
	white_silver = black_silver + 81,
	black_gold = white_silver + 81,
	white_gold = black_gold + 81,
	black_bishop = white_gold + 81,
	white_bishop = black_bishop + 81,
	black_rook = white_bishop + 81,
	white_rook = black_rook + 81,
	black_horse = white_rook + 81,
	white_horse = black_horse + 81,
	black_dragon = white_horse + 81,
	white_dragon = black_dragon + 81,
	square_end = white_dragon + 81,
	PieceStateNum = square_end,
};

extern PieceState PieceToStateIndex[PieceNum];
extern PieceState PieceStateIndex[];
extern PieceState invPieceStateIndex[PieceStateNum];

void initPieceToStateIndex();
void initInvPieceState();
inline PieceState invPieceState(PieceState ps) {
    return invPieceStateIndex[ps];
}

PieceState mirrorPieceState(PieceState ps);

std::ostream& operator<<(std::ostream& os, PieceState ps);
PieceState pieceState(Piece p, int square_or_num, Color c = BLACK);