#include "position.hpp"
#include <algorithm>
#include <bitset>
#include <iostream>
#include <queue>

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
    //Iは使わないのでi >= 7のときは+1する
    for (int64_t i = 0; i < BOARD_WIDTH; i++) {
        std::cout << (char)('A' + i + (i >= 8));
    }
    std::cout << std::endl;
    for (int64_t i = 0; i < BOARD_WIDTH; i++) {
        std::cout << '-';
    }
    std::cout << std::endl;

    for (int32_t y = 0; y < BOARD_WIDTH; y++) {
        for (int32_t x = 0; x < BOARD_WIDTH; x++) {
            std::cout << PieceToSfenStr[board_[xy2square(x, y)]];
        }
        std::cout << "|" << y + 1 << std::endl;
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

    //隣接4方向について相手の石が取れる可能性がある
    for (int32_t i = 0; i < DIR_NUM; i++) {
        int32_t nx = (move.to() % BOARD_WIDTH) + dx[i];
        int32_t ny = (move.to() / BOARD_WIDTH) + dy[i];
        if (nx < 0 || BOARD_WIDTH <= nx || ny < 0 || BOARD_WIDTH <= ny) {
            continue;
        }
        Square nsq = xy2square(nx, ny);
        if (board_[nsq] == oppositeColor(p) && !isLiving(nsq, oppositeColor(p))) {
            //死んでいるので取り除く
            removeDeadStones(nsq);
        }
    }

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
    return board_[move.to()] == EMPTY && isLiving(move.to(), (Piece)color_);
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
    //ゲームが終了するのは2回連続パスが続いた場合
    if (kifu_.size() >= 2 && kifu_[kifu_.size() - 1] == NULL_MOVE && kifu_[kifu_.size() - 2] == NULL_MOVE) {
        //結果を設定する
        //Tromp-Taylorルール(https://senseis.xmp.net/?TrompTaylorRules)に従う
        std::array<int32_t, 2> count = { 0, 0 };
        for (int32_t sq = 0; sq < SQUARE_NUM; sq++) {
            if (board_[sq] == EMPTY) {
                //dfsして到達可能な石を見る
                bool reach_black = canReach(sq, EMPTY, BLACK_PIECE);
                bool reach_white = canReach(sq, EMPTY, WHITE_PIECE);
                if (reach_black && !reach_white) {
                    count[BLACK]++;
                } else if (!reach_black && reach_white) {
                    count[WHITE]++;
                }
            } else {
                count[board_[sq]]++;
            }
        }

        int32_t own = count[color_];
        int32_t opp = count[~color_];

        score = (own > opp ? MAX_SCORE : own < opp ? MIN_SCORE : (MAX_SCORE + MIN_SCORE) / 2);
        return true;
    } else {
        return false;
    }
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

bool Position::isLiving(Square sq, Piece piece) const {
    //pieceのプレイヤーから見てここにある石、あるいはこの空マスに石を置いた場合にその石が死んでいないか判定
    //合法手の確認としてconst化で使いたい場合があるので、sqの位置にある石自体は問わない(置く前に判定を入れたいということ)
    return canReach(sq, piece, EMPTY);
}

bool Position::canReach(Square start, Piece node, Piece target) const {
    //start位置自体がnodeであるかどうかは問わないことにする

    //訪問フラグ
    std::vector<bool> visit(SQUARE_NUM, false);
    visit[start] = true;

    //とりあえずqueueで。stackの方が良いという可能性ある？
    std::queue<Square> q;
    q.push(start);

    //探索
    while (!q.empty()) {
        Square sq = q.front();
        q.pop();

        //まず段と筋に分解
        int32_t x = sq % BOARD_WIDTH;
        int32_t y = sq / BOARD_WIDTH;

        //周囲4方向を見てtargetがあればそこで終了,nodeがあればそこへ遷移
        for (int32_t i = 0; i < DIR_NUM; i++) {
            int32_t nx = x + dx[i];
            int32_t ny = y + dy[i];
            if (nx < 0 || BOARD_WIDTH <= nx || ny < 0 || BOARD_WIDTH <= ny) {
                continue;
            }

            Square nsq = xy2square(nx, ny);
            if (board_[nsq] == target) {
                return true;
            } else if (board_[nsq] == node && !visit[nsq]) {
                q.push(nsq);
                visit[nsq] = true;
            }
        }
    }
}

void Position::removeDeadStones(Square start) {
    //訪問フラグ
    std::vector<bool> visit(SQUARE_NUM, false);
    visit[start] = true;

    //とりあえずqueueで。stackの方が良いという可能性ある？
    std::queue<Square> q;
    q.push(start);

    Piece target = board_[start];

    //探索
    while (!q.empty()) {
        Square sq = q.front();
        q.pop();

        board_[sq] = EMPTY;

        //まず段と筋に分解
        int32_t x = sq % BOARD_WIDTH;
        int32_t y = sq / BOARD_WIDTH;

        //周囲4方向を見てtargetがあればそこで終了,nodeがあればそこへ遷移
        for (int32_t i = 0; i < DIR_NUM; i++) {
            int32_t nx = x + dx[i];
            int32_t ny = y + dy[i];
            if (nx < 0 || BOARD_WIDTH <= nx || ny < 0 || BOARD_WIDTH <= ny) {
                continue;
            }

            Square nsq = xy2square(nx, ny);
            if (board_[nsq] == target) {
                q.push(nsq);
                visit[nsq] = true;
            }
        }
    }
}

} // namespace Go