#ifndef MIACIS_SHOGI_HUFFMANCODED_POS_HPP
#define MIACIS_SHOGI_HUFFMANCODED_POS_HPP

// #include "bitboard.hpp"
// #include "hand.hpp"
#include "piece.hpp"
#include "square.hpp"
#include <algorithm>
#include <iterator>
#include <memory>
#include <stack>

enum GameResult : int8_t { Draw, BlackWin, WhiteWin, GameResultNum };

class BitStream {
public:
    // 読み込む先頭データのポインタをセットする。
    BitStream(uint8_t* d) : data_(d), curr_() {}
    // 読み込む先頭データのポインタをセットする。
    void set(uint8_t* d) {
        data_ = d;
        curr_ = 0;
    }
    // １ bit 読み込む。どこまで読み込んだかを表す bit の位置を 1 個進める。
    uint8_t getBit() {
        const uint8_t result = (*data_ & (1 << curr_++)) ? 1 : 0;
        if (curr_ == 8) {
            ++data_;
            curr_ = 0;
        }
        return result;
    }
    // numOfBits bit読み込む。どこまで読み込んだかを表す bit の位置を numOfBits 個進める。
    uint8_t getBits(const int numOfBits) {
        assert(numOfBits <= 8);
        uint8_t result = 0;
        for (int i = 0; i < numOfBits; ++i) result |= getBit() << i;
        return result;
    }
    // 1 bit 書き込む。
    void putBit(const uint8_t bit) {
        assert(bit <= 1);
        *data_ |= bit << curr_++;
        if (curr_ == 8) {
            ++data_;
            curr_ = 0;
        }
    }
    // val の値を numOfBits bit 書き込む。8 bit まで。
    void putBits(uint8_t val, const int numOfBits) {
        assert(numOfBits <= 8);
        for (int i = 0; i < numOfBits; ++i) {
            const uint8_t bit = val & 1;
            val >>= 1;
            putBit(bit);
        }
    }
    uint8_t* data() const { return data_; }
    int curr() const { return curr_; }

private:
    uint8_t* data_;
    int curr_; // 1byte 中の bit の位置
};

union HuffmanCode {
    struct {
        uint8_t code;      // 符号化時の bit 列
        uint8_t numOfBits; // 使用 bit 数
    };
    uint16_t key; // std::unordered_map の key として使う。
};

struct HuffmanCodeToPieceHash : public std::unordered_map<uint16_t, Piece> {
    Piece value(const uint16_t key) const {
        const auto it = find(key);
        if (it == std::end(*this)) {
            return PieceNum;
        } else {
            return it->second;
        }
    }
};

// Huffman 符号化された局面のデータ構造。256 bit で局面を表す。
struct HuffmanCodedPos {
    static const HuffmanCode boardCodeTable[PieceNum];
    static const HuffmanCode handCodeTable[HAND_PIECE_KIND_NUM][ColorNum];
    static HuffmanCodeToPieceHash boardCodeToPieceHash;
    static HuffmanCodeToPieceHash handCodeToPieceHash;
    static void init() {
        // 空マス
        boardCodeToPieceHash[boardCodeTable[EMPTY].key] = EMPTY;
        for (Piece pc : PieceList) {
            // 玉は位置で符号化するので、駒の種類では符号化しない。
            if (kind(pc) != KING) {
                boardCodeToPieceHash[boardCodeTable[pc].key] = pc;
            }
        }
        for (int32_t hp = PAWN; hp < KING; ++hp) {
            for (int32_t c = BLACK; c < ColorNum; ++c) {
                handCodeToPieceHash[handCodeTable[hp - 1][c].key] = coloredPiece(Color(c), Piece(hp));
            }
        }
    }
    void clear() { std::fill(std::begin(data), std::end(data), 0); }
    Color color() const { return (Color)(data[0] & 1); }
    bool operator==(const HuffmanCodedPos& other) const {
        const auto* data4 = (const uint64_t*)data;
        const auto* other4 = (const uint64_t*)other.data;
        if (data4[0] != other4[0]) return false;
        if (data4[1] != other4[1]) return false;
        if (data4[2] != other4[2]) return false;
        if (data4[3] != other4[3]) return false;
        return true;
    }
    bool isOK() const {
        HuffmanCodedPos tmp = *this; // ローカルにコピー
        BitStream bs(tmp.data);

        // 手番
        static_cast<Color>(bs.getBit());

        // 玉の位置
        const Square sq0 = (Square)bs.getBits(7);
        if (sq0 >= SquareNum) return false;
        const Square sq1 = (Square)bs.getBits(7);
        if (sq1 >= SquareNum) return false;

        // 盤上の駒
        for (Square sq : SquareList) {
            if (sq == sq0 || sq == sq1) // piece(sq) は BKing, WKing, Empty のどれか。
                continue;
            HuffmanCode hc = { 0, 0 };
            while (hc.numOfBits <= 8) {
                hc.code |= bs.getBit() << hc.numOfBits++;
                if (HuffmanCodedPos::boardCodeToPieceHash.value(hc.key) != PieceNum) {
                    break;
                }
            }
            if (HuffmanCodedPos::boardCodeToPieceHash.value(hc.key) == PieceNum) return false;
        }
        while (bs.data() != std::end(tmp.data)) {
            HuffmanCode hc = { 0, 0 };
            while (hc.numOfBits <= 8) {
                hc.code |= bs.getBit() << hc.numOfBits++;
                const Piece pc = HuffmanCodedPos::handCodeToPieceHash.value(hc.key);
                if (pc != PieceNum) {
                    break;
                }
            }
            if (HuffmanCodedPos::handCodeToPieceHash.value(hc.key) == PieceNum) return false;
        }

        return true;
    }

    uint8_t data[32];
};
static_assert(sizeof(HuffmanCodedPos) == 32, "");

struct HuffmanCodedPosAndEval {
    HuffmanCodedPos hcp;
    int16_t eval;
    uint16_t bestMove16;   // 使うかは分からないが教師データ生成時についでに取得しておく。
    GameResult gameResult; // 自己対局で勝ったかどうか。
};
static_assert(sizeof(HuffmanCodedPosAndEval) == 38, "");

#endif