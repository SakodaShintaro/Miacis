#include"position.hpp"
#include"piece.hpp"
#include"move.hpp"
#include<iostream>
#include<cstdio>
#include<bitset>
#include<algorithm>

int64_t Position::HashSeed[PieceNum][SquareNum];

Position::Position() {
    init();
}

void Position::init() {
    //盤上の初期化
    for (int32_t i = 0; i < SquareNum; i++) {
        board_[i] = WALL;
    }
    for (Square sq : SquareList) {
        board_[sq] = EMPTY;
    }

    //後手の駒
    board_[SQ54] = board_[SQ45] = WHITE_PIECE;

    //先手の駒
    board_[SQ44] = board_[SQ55] = BLACK_PIECE;

    //手番
    color_ = BLACK;

    //手数
    turn_number_ = 0;

    //ハッシュ値の初期化
    initHashValue();

    kifu_.clear();
    kifu_.reserve(512);
}

void Position::print() const {
    //盤上
    std::printf("ABCDEFGH\n");
    std::printf("--------\n");
    for (int r = Rank1; r <= Rank8; r++) {
        for (int f = File8; f >= File1; f--) {
            std::cout << PieceToSfenStr[board_[FRToSquare[f][r]]];
        }
        printf("|%d\n", r);
    }

    //手番
    printf("手番:");
    if (color_ == BLACK) printf("先手\n");
    else printf("後手\n");

    //手数
    printf("手数:%d\n", turn_number_);

    //最後の手
    if (!kifu_.empty()) {
        printf("最後の手:");
        lastMove().print();
    }

    printf("ハッシュ値:%lld\n", static_cast<long long int>(hash_value_));
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

    if (move == NULL_MOVE) {
        doNullMove();
        return;
    }

    //動かす前の状態を残しておく
    std::vector<Piece> board(SquareNum);
    for (int32_t i = 0; i < SquareNum; i++) {
        board[i] = board_[i];
    }
    stack_.emplace_back(board);

    //実際に動かす
    //8方向を一つずつ見ていって反転できる駒があったら反転する
    Piece p = board_[move.to()] = (Piece)color_;

    hash_values_.push_back(hash_value_);

    //ハッシュ値を変更
    hash_value_ ^= HashSeed[p][move.to()];

    for (Dir d : DirList) {
        bool is_there_enemy = false;
        bool isOK = false;
        for (int32_t sq = move.to() + d; board_[sq] != WALL; sq += d) {
            if (board_[sq] == oppositeColor(p)) {
                is_there_enemy = true;
            } else if (board_[sq] == p) {
                if (is_there_enemy) {
                    isOK = true;
                }
                break;
            } else {
                break;
            }
        }
        if (!isOK) {
            continue;
        }

        //実際に駒を変更する
        for (int32_t sq = move.to() + d; board_[sq] != p; sq += d) {
            board_[sq] = p;
            hash_value_ ^= HashSeed[oppositeColor(p)][sq];
            hash_value_ ^= HashSeed[p][sq];
        }
    }

    //手番の更新
    color_ = ~color_;

    //手数の更新
    turn_number_++;

    //棋譜に指し手を追加
    kifu_.push_back(move);

    //1bit目を0にする
    hash_value_ ^= 1;
}

void Position::undo() {
    kifu_.pop_back();

    //手番を戻す(このタイミングでいいかな?)
    color_ = ~color_;

    //盤の状態はstack_から戻して
    for (int32_t i = 0; i < SquareNum; i++) {
        board_[i] = stack_.back()[i];
    }
    stack_.pop_back();

    //ハッシュの更新
    hash_value_ = hash_values_.back();
    hash_values_.pop_back();

    //手数
    turn_number_--;
}

void Position::doNullMove() {
    //スタックの変更
    std::vector<Piece> board(SquareNum);
    for (int32_t i = 0; i < SquareNum; i++) {
        board[i] = board_[i];
    }
    stack_.emplace_back(board);

    hash_values_.push_back(hash_value_);

    //手番の更新
    color_ = ~color_;

    //手数の更新
    turn_number_++;

    //hashの手番要素を更新
    hash_value_ ^= 1;

    //NULL_MOVEを棋譜に追加
    kifu_.push_back(NULL_MOVE);
}

void Position::undoNullMove() {
    kifu_.pop_back();

    //手番を戻す(このタイミングでいいかな?)
    color_ = ~color_;

    //ハッシュの更新(手番分)
    hash_value_ ^= 1;

    //手数
    turn_number_--;
}

bool Position::isLegalMove(const Move move) const {
    if (move == NULL_MOVE) {
        auto moves = generateAllMoves();
        return (moves.size() == 1 && moves[0] == NULL_MOVE);
    }

    //間に敵駒がありその先に自駒があれば良い
    if (board_[move.to()] != EMPTY) {
        return false;
    }
    Piece p = (Piece)color_;
    for (Dir d : DirList) {
        bool ok = false;
        for (int32_t sq = move.to() + d; board_[sq] != WALL; sq += d) {
            if (board_[sq] == oppositeColor(p)) {
                ok = true;
            } else if (board_[sq] == p) {
                if (ok) {
                    return true;
                }
                break;
            } else {
                break;
            }
        }
    }
    return false;
}

bool Position::isFinish() const {
    //パスが2回続いたら終了
    int32_t N = (int32_t)kifu_.size();
    return (N >= 2 && kifu_[N - 1] == NULL_MOVE && kifu_[N - 2] == NULL_MOVE);
}

void Position::initHashSeed() {
    std::mt19937_64 rnd(5981793);
    for (int32_t piece = BLACK_PIECE; piece <= WHITE_PIECE; piece++) {
        for (Square sq : SquareList) {
            HashSeed[piece][sq] = rnd();
        }
    }
}

std::vector<Move> Position::generateAllMoves() const {
    std::vector<Move> moves;
    for (Square sq : SquareList) {
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
    for (auto sq : SquareList) {
        hash_value_ ^= HashSeed[board_[sq]][sq];
    }
    hash_value_ &= ~1; //これで1bit目が0になる(先手番を表す)
}

std::string Position::toStr() const {
    std::string str;
    for (Square sq : SquareList) {
        str += PieceToSfenStr[board_[sq]];
    }
    return str;
}

void Position::fromStr(const std::string& str) {
    for (uint64_t i = 0; i < str.size(); i++) {
        board_[SquareList[i]] = (str[i] == 'x' ? BLACK_PIECE : str[i] == 'o' ? WHITE_PIECE : EMPTY);
    }
}