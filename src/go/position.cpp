#include "position.hpp"
#include <algorithm>
#include <bitset>
#include <iostream>

namespace Go {

uint64_t Position::HashSeed[PieceNum][SQUARE_NUM];

Position::Position() { init(); }

void Position::init() {
    //盤上の初期化
    for (int32_t i = 0; i < SQUARE_NUM; i++) {
        board_[i] = EMPTY;
    }

    //手番
    color_ = BLACK;

    //手数
    turn_number_ = 0;

    //ハッシュ値の初期化
    initHashValue();

    //履歴系を初期化
    board_history_.clear();
    hash_values_.clear();
    kifu_.clear();
}

void Position::print() const {
    for (int64_t i = 0; i < BOARD_WIDTH; i++) {
        std::cout << (char)('A' + i);
    }
    std::cout << std::endl;
    for (int64_t i = 0; i < BOARD_WIDTH; i++) {
        std::cout << '-';
    }
    std::cout << std::endl;

    for (int32_t r = 0; r < BOARD_WIDTH; r++) {
        for (int32_t f = BOARD_WIDTH - 1; f > 0; f--) {
            std::cout << PieceToSfenStr[board_[xy2square(f, r)]];
        }
        std::cout << "|" << r << std::endl;
    }

    std::cout << "手番:" << (color_ == BLACK ? "先手" : "後手") << std::endl;
    std::cout << "手数:" << turn_number_ << std::endl;
    if (!kifu_.empty()) {
        std::cout << "最後の手:" << lastMove().toPrettyStr() << std::endl;
    }
    std::cout << "ハッシュ値:" << std::hex << hash_value_ << std::dec << std::endl;
}

void Position::doMove(const Move move) {
#if DEBUG
    if (!isLegalMove(move)) {
        printForDebug();
        std::cout << "違法だった手:";
        move.print();
        isLegalMove(move);
        undo();
        printAllMoves();
        assert(false);
    }
#endif

    //現在の状態を残しておく
    //盤面
    board_history_.emplace_back(board_);

    //ハッシュ値
    hash_values_.push_back(hash_value_);

    //手数の更新
    turn_number_++;

    //行動を棋譜に追加
    kifu_.push_back(move);

    //パスならいくつか処理をしてここで終了
    if (move == NULL_MOVE) {
        //手番の更新
        color_ = ~color_;

        //hashの手番要素を更新
        hash_value_ ^= 1;
        return;
    }

    //実際に動かす
    //8方向を一つずつ見ていって反転できる駒があったら反転する
    //石を置く
    Piece p = board_[move.to()] = (Piece)color_;

    //ハッシュ値を変更
    hash_value_ ^= HashSeed[p][move.to()];

    //ここで相手の石が取れる場合がある。結構その処理が面倒そう？

    //手番の更新
    color_ = ~color_;

    //1bit目を反転
    hash_value_ ^= 1;
}

void Position::undo() {
    kifu_.pop_back();

    //手番を戻す
    color_ = ~color_;

    //盤の状態はboard_history_から戻す
    for (int32_t i = 0; i < SQUARE_NUM; i++) {
        board_[i] = board_history_.back()[i];
    }
    board_history_.pop_back();

    //ハッシュの巻き戻し
    hash_value_ = hash_values_.back();
    hash_values_.pop_back();

    //手数
    turn_number_--;
}

bool Position::isLegalMove(const Move move) const {
    if (move == NULL_MOVE) {
        std::vector<Move> moves = generateAllMoves();
        return (moves.size() == 1 && moves[0] == NULL_MOVE);
    }

    //打った先に石がなく、打って自殺になっていなければ良い

    //打った先の確認
    if (board_[move.to()] != EMPTY) {
        return false;
    }

    //自殺判定はまたいつか
    assert(false);
    return true;
}

void Position::initHashSeed() {
    std::mt19937_64 rnd(5981793);
    for (int32_t piece = BLACK_PIECE; piece <= WHITE_PIECE; piece++) {
        for (Square sq = 0; sq < SQUARE_NUM; sq++) {
            HashSeed[piece][sq] = rnd();
        }
    }
}

std::vector<Move> Position::generateAllMoves() const {
    std::vector<Move> moves;
    for (Square sq = 0; sq < SQUARE_NUM; sq++) {
        Move move(sq, color_);
        if (isLegalMove(move)) {
            moves.push_back(move);
        }
    }
    if (moves.empty()) {
        moves.push_back(NULL_MOVE);
    }
    return moves;
}

void Position::initHashValue() {
    hash_value_ = 0;
    for (Square sq = 0; sq < SQUARE_NUM; sq++) {
        hash_value_ ^= HashSeed[board_[sq]][sq];
    }
    hash_value_ &= ~1;     //これで1bit目が0になる
    hash_value_ ^= color_; //先手番なら0、後手番なら1にする
}

std::string Position::toStr() const {
    std::string str;
    for (Square sq = 0; sq < SQUARE_NUM; sq++) {
        str += PieceToSfenStr[board_[sq]];
    }
    str += (color_ == BLACK ? 'b' : 'w');
    return str;
}

void Position::fromStr(const std::string& str) {
    for (uint64_t i = 0; i < SQUARE_NUM; i++) {
        board_[i] = (str[i] == 'x' ? BLACK_PIECE : str[i] == 'o' ? WHITE_PIECE : EMPTY);
    }
    color_ = (str.back() == 'b' ? BLACK : WHITE);
    initHashValue();
}

bool Position::isFinish(float& score, bool check_repeat) const {
    //ゲームが終了するのは
    //(1)盤面が埋まっている場合
    //(2)盤面は埋まっていないが、どちらのプレイヤーも置くところがない場合
    //   これは2回連続パスが続くことで判定できる

    //空マスも含めて石の数を数えていく
    std::array<int64_t, PieceNum> piece_num{};
    for (Square sq = 0; sq < SQUARE_NUM; sq++) {
        piece_num[board_[sq]]++;
    }
    assert(piece_num[WALL] == 0);

    score = (piece_num[color_] == piece_num[~color_] ? (MAX_SCORE + MIN_SCORE) / 2
                                                     : piece_num[color_] > piece_num[~color_] ? MAX_SCORE : MIN_SCORE);

    return (piece_num[EMPTY] == 0) ||
           (kifu_.size() >= 2 && kifu_[kifu_.size() - 1] == NULL_MOVE && kifu_[kifu_.size() - 2] == NULL_MOVE);
}

std::vector<float> Position::makeFeature() const {
    //引数の指定によりデータ拡張する
    //現状はdata_augmentation * 90度回転
    std::vector<float> features(SQUARE_NUM * INPUT_CHANNEL_NUM, 0);

    //駒を見る順番
    static constexpr std::array<Piece, INPUT_CHANNEL_NUM> TARGET_PIECE[ColorNum] = { { BLACK_PIECE, WHITE_PIECE },
                                                                                     { WHITE_PIECE, BLACK_PIECE } };

    //盤上の駒の特徴量
    for (uint64_t i = 0; i < ColorNum; i++) {
        //各マスについて注目している駒がそこにあるなら1,ないなら0とする
        for (Square sq = 0; sq < SQUARE_NUM; sq++) {
            features[i * SQUARE_NUM + sq] = (board_[sq] == TARGET_PIECE[color_][i] ? 1 : 0);
        }
    }

    return features;
}

std::string Position::augmentStr(const std::string& str, int64_t augmentation) {
    return augmentStrMirror(augmentStrRotate(str, augmentation % 4), augmentation / 4);
}

std::string Position::augmentStrRotate(const std::string& str, int64_t augmentation) {
    if (augmentation == 0) {
        //なにもしない
        return str;
    } else if (augmentation == 1) {
        //時計回りに90度回転
        std::string result;
        for (int64_t i = 0; i < BOARD_WIDTH; i++) {
            for (int64_t j = 0; j < BOARD_WIDTH; j++) {
                result += str[i + (BOARD_WIDTH - 1 - j) * BOARD_WIDTH];
            }
        }
        result += str.back();
        return result;
    } else if (augmentation == 2) {
        //時計回りに180度回転
        std::string result = str;
        std::reverse(result.begin(), result.end() - 1);
        return result;
    } else if (augmentation == 3) {
        //時計回りに270度回転
        std::string result;
        for (int64_t i = 0; i < BOARD_WIDTH; i++) {
            for (int64_t j = 0; j < BOARD_WIDTH; j++) {
                result += str[(j + 1) * BOARD_WIDTH - 1 - i];
            }
        }
        result += str.back();
        return result;
    } else {
        std::cout << "in augmentStrRotate, augmentation = " << augmentation << std::endl;
        exit(1);
    }
}

std::string Position::augmentStrMirror(const std::string& str, int64_t augmentation) {
    if (augmentation == 0) {
        //なにもしない
        return str;
    } else if (augmentation == 1) {
        //左右反転
        std::string result;
        for (int64_t i = 0; i < BOARD_WIDTH; i++) {
            for (int64_t j = 0; j < BOARD_WIDTH; j++) {
                result += str[(BOARD_WIDTH - 1 - i) * BOARD_WIDTH + j];
            }
        }
        result += str.back();
        return result;
    } else {
        std::cout << "in augmentStrMirror, augmentation = " << augmentation << std::endl;
        exit(1);
    }
}

} // namespace Go