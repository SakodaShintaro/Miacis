#include "position.hpp"
#include "../model/model_common.hpp"

namespace Shogi {

uint64_t Position::HashSeed[PieceNum][SquareNum];
uint64_t Position::HandHashSeed[ColorNum][PieceNum][19];

Position::Position() { init(); }

void Position::init() {
    //盤上の初期化
    for (Piece& p : board_) p = WALL;
    for (Square sq : SquareList) board_[sq] = EMPTY;

    //後手の駒
    board_[SQ11] = WHITE_LANCE;
    board_[SQ21] = WHITE_KNIGHT;
    board_[SQ31] = WHITE_SILVER;
    board_[SQ41] = WHITE_GOLD;
    board_[SQ51] = WHITE_KING;
    board_[SQ61] = WHITE_GOLD;
    board_[SQ71] = WHITE_SILVER;
    board_[SQ81] = WHITE_KNIGHT;
    board_[SQ91] = WHITE_LANCE;
    board_[SQ22] = WHITE_BISHOP;
    board_[SQ82] = WHITE_ROOK;
    board_[SQ13] = WHITE_PAWN;
    board_[SQ23] = WHITE_PAWN;
    board_[SQ33] = WHITE_PAWN;
    board_[SQ43] = WHITE_PAWN;
    board_[SQ53] = WHITE_PAWN;
    board_[SQ63] = WHITE_PAWN;
    board_[SQ73] = WHITE_PAWN;
    board_[SQ83] = WHITE_PAWN;
    board_[SQ93] = WHITE_PAWN;

    //先手の駒
    board_[SQ19] = BLACK_LANCE;
    board_[SQ29] = BLACK_KNIGHT;
    board_[SQ39] = BLACK_SILVER;
    board_[SQ49] = BLACK_GOLD;
    board_[SQ59] = BLACK_KING;
    board_[SQ69] = BLACK_GOLD;
    board_[SQ79] = BLACK_SILVER;
    board_[SQ89] = BLACK_KNIGHT;
    board_[SQ99] = BLACK_LANCE;
    board_[SQ88] = BLACK_BISHOP;
    board_[SQ28] = BLACK_ROOK;
    board_[SQ17] = BLACK_PAWN;
    board_[SQ27] = BLACK_PAWN;
    board_[SQ37] = BLACK_PAWN;
    board_[SQ47] = BLACK_PAWN;
    board_[SQ57] = BLACK_PAWN;
    board_[SQ67] = BLACK_PAWN;
    board_[SQ77] = BLACK_PAWN;
    board_[SQ87] = BLACK_PAWN;
    board_[SQ97] = BLACK_PAWN;

    //持ち駒
    hand_[BLACK].clear();
    hand_[WHITE].clear();

    //手番
    color_ = BLACK;

    //手数
    turn_number_ = 1;

    //玉の位置
    king_sq_[BLACK] = SQ59;
    king_sq_[WHITE] = SQ51;

    //ハッシュ値の初期化
    initHashValue();

    stack_.clear();
    stack_.reserve(512);
    kifu_.clear();
    kifu_.reserve(512);

    //Bitboard
    occupied_bb_[BLACK] = Bitboard(0, 0);
    occupied_bb_[WHITE] = Bitboard(0, 0);
    for (Bitboard& bb : pieces_bb_) {
        bb = Bitboard(0, 0);
    }
    for (Square sq : SquareList) {
        if (board_[sq] != EMPTY) {
            pieces_bb_[board_[sq]] |= SQUARE_BB[sq];
            occupied_bb_[pieceToColor(board_[sq])] |= SQUARE_BB[sq];
        }
    }
    occupied_all_ = occupied_bb_[BLACK] | occupied_bb_[WHITE];

    is_checked_ = false;

    //合法手生成のフラグを降ろす
    already_generated_moves_ = false;
}

void Position::print() const {
    //盤上
    std::cout << "９８７６５４３２１" << std::endl;
    std::cout << "------------------" << std::endl;
    for (int32_t r = Rank1; r <= Rank9; r++) {
        for (int32_t f = File9; f >= File1; f--) {
            std::cout << PieceToSfenStr[board_[FRToSquare[f][r]]];
        }
        std::cout << "|" << r << std::endl;
    }

    //持ち駒
    std::cout << "持ち駒" << std::endl;
    std::cout << "先手:";
    hand_[BLACK].print();
    std::cout << "後手:";
    hand_[WHITE].print();

    //その他
    std::cout << "手番:" << (color_ == BLACK ? "先手" : "後手") << std::endl;
    std::cout << "手数:" << turn_number_ << std::endl;
    if (!kifu_.empty()) {
        std::cout << "最後の手:" << lastMove().toPrettyStr() << std::endl;
    }
    std::cout << "ハッシュ値:" << std::hex << hash_value_ << std::dec << std::endl;
}

void Position::doMove(const Move move) {
    //動かす前の状態を残しておく
    stack_.emplace_back(*this);

    //実際に動かす
    if (move.isDrop()) { //持ち駒を打つ手

        //持ち駒を減らす
        hand_[color_].sub(kind(move.subject()));

        //移動先にsubjectを設置
        board_[move.to()] = move.subject();

        //ハッシュ値の更新
        //打つ前のHandHashとXORして消す
        hand_hash_ ^= HandHashSeed[color_][kind(move.subject())][hand_[color_].num(kind(move.subject())) + 1];
        //打った後のHandHashとXOR
        hand_hash_ ^= HandHashSeed[color_][kind(move.subject())][hand_[color_].num(kind(move.subject()))];
        //打った後の分をXOR
        board_hash_ ^= HashSeed[move.subject()][move.to()];

        //Bitboard更新
        pieces_bb_[move.subject()] |= SQUARE_BB[move.to()];
        occupied_bb_[color_] |= SQUARE_BB[move.to()];

    } else { //盤上の駒を動かす手

        //移動する駒を消す
        board_[move.from()] = EMPTY;
        pieces_bb_[move.subject()] &= ~SQUARE_BB[move.from()];
        occupied_bb_[color_] &= ~SQUARE_BB[move.from()];

        //取った駒があるならその駒を消し、持ち駒を増やす
        if (move.capture() != EMPTY) {
            //取った駒を消す
            board_[move.to()] = EMPTY;
            pieces_bb_[move.capture()] &= ~SQUARE_BB[move.to()];
            occupied_bb_[~color_] &= ~SQUARE_BB[move.to()];

            //持ち駒を増やす
            hand_[color_].add(kind(move.capture()));

            //ハッシュ値の更新
            //取った駒分のハッシュをXOR
            board_hash_ ^= HashSeed[move.capture()][move.to()];
            //増える前の持ち駒の分をXORして消す
            hand_hash_ ^= HandHashSeed[color_][kind(move.capture())][hand_[color_].num(kind(move.capture())) - 1];
            //増えた後の持ち駒の分XOR
            hand_hash_ ^= HandHashSeed[color_][kind(move.capture())][hand_[color_].num(kind(move.capture()))];
        }

        //成る手ならsubjectに成りのフラグを立てて,そうでないならsubjectをそのまま移動先に設置
        if (move.isPromote()) {
            board_[move.to()] = promote(move.subject());

            //Bitboard更新
            pieces_bb_[promote(move.subject())] |= SQUARE_BB[move.to()];
        } else {
            board_[move.to()] = move.subject();

            //Bitboard更新
            pieces_bb_[move.subject()] |= SQUARE_BB[move.to()];
        }
        //occupiedは成か不成かによらない
        occupied_bb_[color_] |= SQUARE_BB[move.to()];

        //ハッシュ値の更新
        //移動前の分をXORして消す
        board_hash_ ^= HashSeed[move.subject()][move.from()];
        //移動後の分をXOR
        if (move.isPromote()) {
            board_hash_ ^= HashSeed[promote(move.subject())][move.to()];
        } else {
            board_hash_ ^= HashSeed[move.subject()][move.to()];
        }
    }

    //玉を動かす手ならblack_king_pos,white_king_posに反映
    if (kind(move.subject()) == KING) {
        king_sq_[color_] = move.to();
    }

    //occupied_all_を更新
    occupied_all_ = occupied_bb_[BLACK] | occupied_bb_[WHITE];

    //手番の更新
    color_ = ~color_;

    //手数の更新
    turn_number_++;

    //棋譜に指し手を追加
    kifu_.push_back(move);

    //王手
    //is_checked_ = isThereControl(~color_, king_sq_[color_]);
    is_checked_ = isLastMoveCheck();

    //hashの手番要素を更新
    hash_value_ = board_hash_ ^ hand_hash_;
    //1bit目を0にする
    hash_value_ &= ~1;
    //手番が先手だったら1bitは0のまま,後手だったら1bit目は1になる
    hash_value_ |= color_;

    //合法手生成のフラグを降ろす
    already_generated_moves_ = false;
}

void Position::undo() {
    const Move last_move = kifu_.back();
    kifu_.pop_back();

    //手番を戻す(このタイミングでいいかな?)
    color_ = ~color_;

    //動かした駒を消す
    board_[last_move.to()] = EMPTY;
    occupied_bb_[color_] &= ~SQUARE_BB[last_move.to()];

    //盤の状態を戻す
    if (last_move.isDrop()) { //打つ手

        //持ち駒を増やす
        hand_[color_].add(kind(last_move.subject()));

        //ハッシュ値の巻き戻し
        //戻す前のHandHashとXOR
        hand_hash_ ^= HandHashSeed[color_][kind(last_move.subject())][hand_[color_].num(kind(last_move.subject())) - 1];
        //戻す前の分をXORして消す
        board_hash_ ^= HashSeed[last_move.subject()][last_move.to()];
        //戻した後のHandHashとXOR
        hand_hash_ ^= HandHashSeed[color_][kind(last_move.subject())][hand_[color_].num(kind(last_move.subject()))];

        //Bitboard更新
        pieces_bb_[last_move.subject()] &= ~SQUARE_BB[last_move.to()];
    } else { //盤上の駒を動かす手
        //取る手だったらtoに取った駒を戻し、持ち駒を減らす
        if (last_move.capture() != EMPTY) {
            board_[last_move.to()] = last_move.capture();
            hand_[color_].sub(kind(last_move.capture()));

            //Bitboard更新
            pieces_bb_[last_move.capture()] |= SQUARE_BB[last_move.to()];
            occupied_bb_[~color_] |= SQUARE_BB[last_move.to()];

            //ハッシュ値の巻き戻し
            //取る前の分のハッシュをXOR
            board_hash_ ^= HashSeed[last_move.capture()][last_move.to()];
            //増える前の持ち駒の分
            hand_hash_ ^=
                HandHashSeed[color_][last_move.capture() & PIECE_KIND_MASK][hand_[color_].num(kind(last_move.capture()))];
            //増えた後の持ち駒の分XORして消す
            hand_hash_ ^=
                HandHashSeed[color_][last_move.capture() & PIECE_KIND_MASK][hand_[color_].num(kind(last_move.capture())) + 1];
        }

        //動いた駒をfromに戻す
        board_[last_move.from()] = last_move.subject();
        //Bitboard更新
        pieces_bb_[last_move.subject()] |= SQUARE_BB[last_move.from()];
        occupied_bb_[color_] |= SQUARE_BB[last_move.from()];

        //ハッシュ値の巻き戻し
        //移動前の分をXOR
        board_hash_ ^= HashSeed[last_move.subject()][last_move.from()];
        //移動後の分をXORして消す
        if (last_move.isPromote()) board_hash_ ^= HashSeed[promote(last_move.subject())][last_move.to()];
        else
            board_hash_ ^= HashSeed[last_move.subject()][last_move.to()];

        //評価値とBitboardの更新
        if (last_move.isPromote()) {
            pieces_bb_[promote(last_move.subject())] &= ~SQUARE_BB[last_move.to()];
        } else {
            pieces_bb_[last_move.subject()] &= ~SQUARE_BB[last_move.to()];
        }
    }

    //玉を動かす手ならking_sq_に反映
    if (kind(last_move.subject()) == KING) king_sq_[color_] = last_move.from();

    //occupied_all_を更新
    occupied_all_ = occupied_bb_[BLACK] | occupied_bb_[WHITE];

    //王手は自玉へのattackers
    //checkers_ = attackersTo(~color_, king_sq_[color_]);

    //ハッシュの更新
    hash_value_ = board_hash_ ^ hand_hash_;
    //一番右のbitを0にする
    hash_value_ &= ~1;
    //一番右のbitが先手番だったら0のまま、後手番だったら1になる
    hash_value_ |= color_;

    //手数
    turn_number_--;

    //計算が面倒なものはコピーで戻してみる
    is_checked_ = stack_.back().is_checked;

    //Stack更新
    stack_.pop_back();

    //合法手生成のフラグを降ろす
    already_generated_moves_ = false;
}

bool Position::isLegalMove(const Move move) const {
    //違法の場合だけ早くfalseで返す.合法手は一番最後の行でtrueが返る

    //NULL_MOVE
    if (move == NULL_MOVE) {
        return false;
    }

    //打つ手と動かす手両方に共通するもの
    //移動先が盤面内かチェック
    if (!isOnBoard(move.to())) {
        return false;
    }

    //移動先に味方の駒がないかチェック
    if (pieceToColor(board_[move.to()]) == color_) {
        return false;
    }

    //動かす駒がEMPTYでないかチェック
    if (move.subject() == EMPTY) {
        return false;
    }

    //手番と動かす駒の対応チェック
    if (pieceToColor(move.subject()) != color_) {
        return false;
    }

    //取る駒の対応をチェック
    if (move.capture() != board_[move.to()]) {
        return false;
    }

    //取った駒が玉でないかチェック
    if (move.capture() == KING) {
        return false;
    }

    //成りのチェック
    if (move.isPromote() && color_ == BLACK && SquareToRank[move.to()] > Rank3 && SquareToRank[move.from()] > Rank3) {
        return false;
    }
    if (move.isPromote() && color_ == WHITE && SquareToRank[move.to()] < Rank7 && SquareToRank[move.from()] < Rank7) {
        return false;
    }

    if (move.isDrop()) { //駒を打つ手特有の判定
        //打つ先が空であるかチェック
        if (board_[move.to()] != EMPTY) {
            return false;
        }

        //打つ駒が1枚以上持ち駒にあるかチェック
        if (hand_[color_].num(kind(move.subject())) <= 0) {
            return false;
        }

        //二歩のチェック
        if (kind(move.subject()) == PAWN && !canDropPawn(move.to())) {
            return false;
        }

        //歩を最奥段に打っていないか
        if (move.subject() == BLACK_PAWN && SquareToRank[move.to()] == Rank1) {
            return false;
        }
        if (move.subject() == WHITE_PAWN && SquareToRank[move.to()] == Rank9) {
            return false;
        }
        //香を最奥段に打っていないか
        if (move.subject() == BLACK_LANCE && SquareToRank[move.to()] == Rank1) {
            return false;
        }
        if (move.subject() == WHITE_LANCE && SquareToRank[move.to()] == Rank9) {
            return false;
        }
        //桂を最奥段,二段目に打っていないか
        if (move.subject() == BLACK_KNIGHT && (SquareToRank[move.to()] == Rank1 || SquareToRank[move.to()] == Rank2)) {
            return false;
        }
        if (move.subject() == WHITE_KNIGHT && (SquareToRank[move.to()] == Rank9 || SquareToRank[move.to()] == Rank8)) {
            return false;
        }
    } else { //盤上の駒を動かす手の判定
        //動かす駒がfromにある駒と一致しているかチェック
        if (move.subject() != board_[move.from()]) {
            return false;
        }

        //駒を飛び越えていないかチェック
        if (BETWEEN_BB[move.from()][move.to()] & occupied_all_) {
            return false;
        }
    }

    //自殺手でないかチェック
    if (kind(move.subject()) == KING && isThereControl(~color_, move.to())) {
        return false;
    }

    //ピンのチェック
    bool flag = true;
    Bitboard pinners = computePinners();
    pinners.forEach([&](const Square pinner_sq) {
        if ((BETWEEN_BB[pinner_sq][king_sq_[color_]] & SQUARE_BB[move.from()]) //fromがbetween,すなわちピンされている
            && !((BETWEEN_BB[pinner_sq][king_sq_[color_]] | SQUARE_BB[pinner_sq]) &
                 SQUARE_BB[move.to()]) //toがbetween内及びpinner_sq以外
        ) {
            flag = false;
        }
    });
    return flag;
}

bool Position::canDropPawn(const Square to) const {
    //2歩の判定を入れる
    //打ち歩詰めは探索時点で弾くのでチェック不要
    return !(FILE_BB[SquareToFile[to]] & pieces_bb_[color_ == BLACK ? toBlack(PAWN) : toWhite(PAWN)]);
}

void Position::fromStr(const std::string& sfen) {
    //初期化
    for (Piece& p : board_) p = WALL;
    for (Square sq : SquareList) board_[sq] = EMPTY;

    //コマごとの分岐を簡単に書くためテーブル用意
    static std::unordered_map<char, Piece> CharToPiece = {
        { 'P', BLACK_PAWN }, { 'L', BLACK_LANCE },  { 'N', BLACK_KNIGHT }, { 'S', BLACK_SILVER },
        { 'G', BLACK_GOLD }, { 'B', BLACK_BISHOP }, { 'R', BLACK_ROOK },   { 'K', BLACK_KING },
        { 'p', WHITE_PAWN }, { 'l', WHITE_LANCE },  { 'n', WHITE_KNIGHT }, { 's', WHITE_SILVER },
        { 'g', WHITE_GOLD }, { 'b', WHITE_BISHOP }, { 'r', WHITE_ROOK },   { 'k', WHITE_KING },
    };

    //sfen文字列を走査するイテレータ(ダサいやり方な気がするけどパッと思いつくのはこれくらい)
    uint32_t i = 0;

    //盤上の設定
    int r = Rank1, f = File9;
    for (i = 0; i < sfen.size(); i++) {
        if (sfen[i] == '/') {
            //次の段へ移る
            f = File9;
            r++;
        } else if (sfen[i] == ' ') {
            //手番の設定へ
            break;
        } else if ('1' <= sfen[i] && sfen[i] <= '9') {
            //空マス分飛ばす
            f -= sfen[i] - '0';
        } else if (sfen[i] == '+') {
            //次の文字が示す駒を成らせてboard_に設置
            board_[FRToSquare[f--][r]] = promote(CharToPiece[sfen[++i]]);
        } else {
            //玉だったらking_sq_を設定
            if (CharToPiece[sfen[i]] == BLACK_KING) {
                king_sq_[BLACK] = FRToSquare[f][r];
            } else if (CharToPiece[sfen[i]] == WHITE_KING) {
                king_sq_[WHITE] = FRToSquare[f][r];
            }
            //文字が示す駒をboard_に設置
            board_[FRToSquare[f--][r]] = CharToPiece[sfen[i]];
        }
    }

    //手番の設定
    color_ = (sfen[++i] == 'b' ? BLACK : WHITE);

    //空白を飛ばす
    i += 2;

    //持ち駒
    hand_[BLACK].clear();
    hand_[WHITE].clear();
    int num = 1;
    while (sfen[i] != ' ') {
        if (sfen[i] == '-') {
            i++;
            break;
        }
        if ('1' <= sfen[i] && sfen[i] <= '9') { //数字なら枚数の取得
            if ('0' <= sfen[i + 1] && sfen[i + 1] <= '9') {
                //次の文字も数字の場合が一応あるはず(歩が10枚以上)
                num = 10 * (sfen[i] - '0') + sfen[i + 1] - '0';
                i += 2;
            } else {
                //次が数字じゃないなら普通に取る
                num = sfen[i++] - '0';
            }
        } else { //駒なら持ち駒を変更
            Piece piece = CharToPiece[sfen[i++]];
            hand_[pieceToColor(piece)].set(kind(piece), num);

            //枚数を1に戻す
            num = 1;
        }
    }

    //手数
    turn_number_ = 0;
    while (++i < sfen.size()) {
        turn_number_ *= 10;
        turn_number_ += sfen[i] - '0';
    }

    //ハッシュ値の初期化
    initHashValue();

    //Bitboard
    occupied_bb_[BLACK] = Bitboard(0, 0);
    occupied_bb_[WHITE] = Bitboard(0, 0);
    for (Bitboard& bb : pieces_bb_) {
        bb = Bitboard(0, 0);
    }
    for (Square sq : SquareList) {
        if (board_[sq] != EMPTY) {
            pieces_bb_[board_[sq]] |= SQUARE_BB[sq];
            occupied_bb_[pieceToColor(board_[sq])] |= SQUARE_BB[sq];
        }
    }
    occupied_all_ = occupied_bb_[BLACK] | occupied_bb_[WHITE];

    //pinners
    computePinners();

    already_generated_moves_ = false;

    //王手の確認
    is_checked_ = isThereControl(~color_, king_sq_[color_]);

    stack_.clear();
    stack_.reserve(512);
    kifu_.clear();
    kifu_.reserve(512);
}

std::string Position::toStr() const {
    std::string result;
    for (int64_t r = Rank1; r <= Rank9; r++) {
        int64_t empty_num = 0;
        for (int64_t f = File9; f >= File1; f--) {
            if (board_[FRToSquare[f][r]] == EMPTY) {
                empty_num++;
            } else {
                //まずこのマスまでの空白マスを処理
                result += (empty_num == 0 ? "" : std::to_string(empty_num));

                //駒を処理
                result += PieceToSfenStr2[board_[FRToSquare[f][r]]];

                //空白マスを初期化
                empty_num = 0;
            }
        }

        //段最後の空白マスを処理
        result += (empty_num == 0 ? "" : std::to_string(empty_num));

        if (r < Rank9) {
            result += "/";
        }
    }

    //手番
    result += (color_ == BLACK ? " b " : " w ");

    //持ち駒
    bool all0 = true;
    for (Color c : { BLACK, WHITE }) {
        for (Piece p : { ROOK, BISHOP, GOLD, SILVER, KNIGHT, LANCE, PAWN }) {
            if (hand_[c].num(p) == 1) {
                result += PieceToSfenStr2[coloredPiece(c, p)];
                all0 = false;
            } else if (hand_[c].num(p) >= 2) {
                result += std::to_string(hand_[c].num(p));
                result += PieceToSfenStr2[coloredPiece(c, p)];
                all0 = false;
            }
        }
    }

    if (all0) {
        result += "-";
    }

    result += " " + std::to_string(turn_number_);

    return result;
}

void Position::initHashSeed() {
    std::mt19937_64 rnd(5981793);
    for (int piece = BLACK_PAWN; piece <= WHITE_ROOK_PROMOTE; piece++) {
        for (Square sq : SquareList) {
            HashSeed[piece][sq] = rnd();
        }
    }
    for (int color = BLACK; color <= WHITE; color++) {
        for (int piece = PAWN; piece <= ROOK; piece++) {
            for (int num = 0; num <= 18; num++) {
                HandHashSeed[color][piece][num] = rnd();
            }
        }
    }
}

void Position::generateEvasionMoves(std::vector<Move>& move_buf) const {
    //王手への対応は(a)玉が逃げる、(b)王手してきた駒を取る、(c)飛び利きの王手には合駒

    //手番のほうの玉の位置を設定
    Square evasion_from = king_sq_[color_];

    //(a)逃げる手
    //隣接王手を玉で取り返せる場合はここに含む
    Bitboard king_to_bb = controlBB(evasion_from, board_[evasion_from], occupied_all_);

    //味方の駒がいる位置にはいけないので除く
    king_to_bb &= ~occupied_bb_[color_];

    king_to_bb.forEach([&](const Square to) {
        //長い利きが残ってる方に逃げてしまうのでoccupiedから玉のfromを抜く
        if (!attackersTo(~color_, to, occupied_all_ & ~SQUARE_BB[evasion_from])) {
            move_buf.emplace_back(to, evasion_from, false, false, board_[evasion_from], board_[to]);
        }
    });

    Bitboard checkers = attackersTo(~color_, king_sq_[color_]);

    //(b),(c)は両王手だったら無理
    if (checkers.pop_count() != 1) {
        return;
    }

    //王手してきた駒の位置
    const Square checker_sq = checkers.pop();

    //(b)王手してきた駒を玉以外で取る手
    //ピンされた駒を先に動かす
    Bitboard pinned_piece(0, 0);

    Bitboard pinners = computePinners();

    pinners.forEach([&](const Square pinner_sq) {
        //pinnerと自玉の間にあるのがpinされている駒
        Bitboard pinned = BETWEEN_BB[pinner_sq][king_sq_[color_]] & occupied_bb_[color_];

        pinned.forEach([&](const Square from) {
            //取る手なのでpinnerを取る手、かつそこがchecker_sqでないとしかダメ
            Bitboard to_bb = controlBB(from, board_[from], occupied_all_) & SQUARE_BB[pinner_sq] & SQUARE_BB[checker_sq];

            if (to_bb) {
                pushMove(Move(pinner_sq, from, false, false, board_[from], board_[pinner_sq]), move_buf);
            }

            //使ったマスを記録しておく
            pinned_piece |= SQUARE_BB[from];
        });
    });

    //王手してきた駒を取れる駒の候補
    Bitboard removers = attackersTo(color_, checker_sq) & ~pinned_piece & ~SQUARE_BB[king_sq_[color_]];

    removers.forEach(
        [&](Square from) { pushMove(Move(checker_sq, from, false, false, board_[from], board_[checker_sq]), move_buf); });

    //(c)合駒
    //王手してきた駒と自玉の間を示すBitboard
    Bitboard between = BETWEEN_BB[checker_sq][king_sq_[color_]];

    //(c)-1 移動合
    Bitboard from_bb = occupied_bb_[color_] & ~pinned_piece & ~SQUARE_BB[king_sq_[color_]];
    from_bb.forEach([&](const Square from) {
        Bitboard to_bb = controlBB(from, board_[from], occupied_all_) & between;
        to_bb.forEach([&](const Square to) { pushMove(Move(to, from, false, false, board_[from], board_[to]), move_buf); });
    });

    //(c)-2 打つ合
    //王手されているのでbetweenの示すマスは駒がないはず
    generateDropMoves(between, move_buf);
}

void Position::generateNormalMoves(std::vector<Move>& move_buf) const {
    //よく使うのでエイリアスを取る
    const Square& k_sq = king_sq_[color_];

    //ピンされた駒を先に動かす
    Bitboard pinned_piece(0, 0);
    Bitboard pinners = computePinners();

    pinners.forEach([&](const Square pinner_sq) {
        //pinnerと自玉の間にあるのがpinされている駒
        Bitboard pinned = BETWEEN_BB[pinner_sq][k_sq] & occupied_bb_[color_];

        pinned.forEach([&](const Square from) {
            //pinnerを取る or 間の中で移動するように動ける
            Bitboard to_bb = controlBB(from, board_[from], occupied_all_) & (SQUARE_BB[pinner_sq] | BETWEEN_BB[pinner_sq][k_sq]);

            to_bb.forEach([&](const Square to) { pushMove(Move(to, from, false, false, board_[from], board_[to]), move_buf); });

            //使ったマスを記録しておく
            pinned_piece |= SQUARE_BB[from];
        });
    });

    //王の処理
    //手番側の駒がある場所には行けないので除く
    Bitboard king_to_bb = controlBB(k_sq, board_[k_sq], occupied_all_) & ~occupied_bb_[color_];
    king_to_bb.forEach([&](const Square to) {
        //相手の利きがなければそこへいく動きを生成できる
        if (!isThereControl(~color_, to)) {
            move_buf.emplace_back(to, k_sq, false, false, board_[k_sq], board_[to]);
        }
    });

    //ピンされた駒と玉は先に処理したので除く
    Bitboard from_bb = occupied_bb_[color_] & ~pinned_piece & ~SQUARE_BB[k_sq];
    from_bb.forEach([&](const Square from) {
        //sqにある駒の利き
        Bitboard to_bb = controlBB(from, board_[from], occupied_all_) & ~occupied_bb_[color_];
        to_bb.forEach([&](const Square to) { pushMove(Move(to, from, false, false, board_[from], board_[to]), move_buf); });
    });

    //駒を打つ手
    Bitboard drop_to_bb = (~occupied_all_ & BOARD_BB);
    generateDropMoves(drop_to_bb, move_buf);
}

inline bool Position::canPromote(Move move) const {
    if (move.isDrop()                        //打つ手だったらダメ
        || board_[move.from()] & PROMOTE     //すでに成っている駒を動かす手だったらダメ
        || kind(board_[move.from()]) == GOLD //動かす駒が金だったらダメ
        || kind(board_[move.from()]) == KING //動かす駒が玉だったらダメ
    ) {
        return false;
    }

    //位置関係
    if (color_ == BLACK) {
        return ((Rank1 <= SquareToRank[move.to()] && SquareToRank[move.to()] <= Rank3) ||
                (Rank1 <= SquareToRank[move.from()] && SquareToRank[move.from()] <= Rank3));
    } else {
        return ((Rank7 <= SquareToRank[move.to()] && SquareToRank[move.to()] <= Rank9) ||
                (Rank7 <= SquareToRank[move.from()] && SquareToRank[move.from()] <= Rank9));
    }
}

void Position::pushMove(const Move move, std::vector<Move>& move_buf) const {
    //成る手が可能だったら先に生成
    if (canPromote(move)) {
        move_buf.push_back(promotiveMove(move));
        switch (kind(board_[move.from()])) {
            //歩、角、飛は成る手しか生成しなくていい
        case PAWN:
        case BISHOP:
        case ROOK:
            return;
            //香、桂は位置によっては成る手しか不可
            //香の2段目への不成は可能だけど意味がないので生成しない
        case LANCE:
        case KNIGHT:
            if (color_ == BLACK && SquareToRank[move.to()] <= Rank2) return;
            if (color_ == WHITE && SquareToRank[move.to()] >= Rank8) return;
        default:
            break;
        }
    }
    //成らない手を追加
    move_buf.push_back(move);
}

void Position::generateDropMoves(const Bitboard& to_bb, std::vector<Move>& move_buf) const {
    static const Piece ColorToFlag[ColorNum] = { BLACK_FLAG, WHITE_FLAG };

    //歩を打つ手
    //最奥の段は除外する
    if (hand_[color_].num(PAWN) > 0) {
        (to_bb & ~farRank1FromColor(color_)).forEach([&](Square to) {
            if (canDropPawn(to)) {
                move_buf.push_back(dropMove(to, Piece(PAWN | ColorToFlag[color_])));
            }
        });
    }

    //香車
    //最奥の段は除外する
    if (hand_[color_].num(LANCE) > 0) {
        (to_bb & ~farRank1FromColor(color_)).forEach([&](Square to) {
            move_buf.push_back(dropMove(to, Piece(LANCE | ColorToFlag[color_])));
        });
    }

    //桂馬
    //奥の2段は除外する
    if (hand_[color_].num(KNIGHT) > 0) {
        (to_bb & ~(farRank1FromColor(color_) | farRank2FromColor(color_))).forEach([&](Square to) {
            move_buf.push_back(dropMove(to, Piece(KNIGHT | ColorToFlag[color_])));
        });
    }

    //その他
    for (Piece p : { SILVER, GOLD, BISHOP, ROOK }) {
        if (hand_[color_].num(p) > 0) {
            (to_bb).forEach([&](Square to) { move_buf.push_back(dropMove(to, Piece(p | ColorToFlag[color_]))); });
        }
    }
}

Bitboard Position::attackersTo(const Color c, const Square sq) const { return attackersTo(c, sq, occupied_all_); }

Bitboard Position::attackersTo(const Color c, const Square sq, const Bitboard& occupied) const {
    //sqへ利きを持っている駒の位置を示すbitboardを返す
    //sqから相手側の駒として利きを求めてみて、その範囲にc側の駒があるなら利きがある
    Bitboard result(0, 0);
    //歩
    Piece pawn = coloredPiece(c, PAWN);
    result |= controlBB(sq, oppositeColor(pawn), occupied) & pieces_bb_[pawn];

    //桂馬
    Piece knight = coloredPiece(c, KNIGHT);
    result |= controlBB(sq, oppositeColor(knight), occupied) & pieces_bb_[knight];

    //銀
    Piece silver = coloredPiece(c, SILVER);
    Bitboard silver_control = controlBB(sq, oppositeColor(silver), occupied);
    result |= silver_control & pieces_bb_[silver];

    //金
    Piece gold = coloredPiece(c, GOLD);
    Bitboard gold_control = controlBB(sq, oppositeColor(gold), occupied);
    result |= gold_control &
              (pieces_bb_[gold] | pieces_bb_[coloredPiece(c, PAWN_PROMOTE)] | pieces_bb_[coloredPiece(c, LANCE_PROMOTE)] |
               pieces_bb_[coloredPiece(c, KNIGHT_PROMOTE)] | pieces_bb_[coloredPiece(c, SILVER_PROMOTE)]);

    //角
    Piece bishop = coloredPiece(c, BISHOP);
    Bitboard bishop_control = controlBB(sq, oppositeColor(bishop), occupied);
    result |= bishop_control & pieces_bb_[bishop];

    //飛車
    Piece rook = coloredPiece(c, ROOK);
    Bitboard rook_control = controlBB(sq, oppositeColor(rook), occupied);
    result |= rook_control & pieces_bb_[rook];

    //香車(飛車の利きを利用する)
    Piece lance = coloredPiece(c, LANCE);
    Bitboard lance_control = rook_control & FRONT_BB[~c][SquareToRank[sq]];
    result |= lance_control & pieces_bb_[lance];

    //馬(角と金の利きを利用する)
    Piece horse = coloredPiece(c, BISHOP_PROMOTE);
    Bitboard horse_control = bishop_control | gold_control;
    result |= horse_control & pieces_bb_[horse];

    //竜(飛車と銀の利きを利用する)
    Piece dragon = coloredPiece(c, ROOK_PROMOTE);
    Bitboard dragon_control = rook_control | silver_control;
    result |= dragon_control & pieces_bb_[dragon];

    //玉
    Piece king = coloredPiece(c, KING);
    Bitboard king_control = silver_control | gold_control;
    result |= king_control & pieces_bb_[king];

    return result;
}

Bitboard Position::computePinners() const {
    //pinners
    Bitboard pinners(0, 0);
    //香・角(馬)・飛車(竜)に対してピンのチェック
    for (Piece jumper : ColoredJumpPieceList[~color_]) {
        //自玉からこっちの香・角・飛として利きを駒を最大まで伸ばして
        //そこに相手の香・角(馬)・飛(竜)があったらそれはpinnerになりうる
        Bitboard jumper_bb = (kind(jumper) == LANCE ? pieces_bb_[jumper] : pieces_bb_[jumper] | pieces_bb_[promote(jumper)]);
        Bitboard pinner_candidate = controlBB(king_sq_[color_], oppositeColor(jumper), Bitboard(0, 0)) & jumper_bb;

        //各pinnerの候補についてking_sq_との間を見ていく
        pinner_candidate.forEach([&](const Square jump_piece_sq) {
            Bitboard between = BETWEEN_BB[king_sq_[color_]][jump_piece_sq] & occupied_all_;
            if (between.pop_count() == 1 &&
                (between & occupied_bb_[color_])) { //betweenに駒が1個だけかつ駒が手番側のものだったらピンされている
                //ピンしている駒を記録しておけばピンされた駒は自玉との間でbetween見れば復元できる
                pinners |= SQUARE_BB[jump_piece_sq];
            }
        });
    }
    return pinners;
}

std::vector<Move> Position::generateAllMoves() {
    constexpr int32_t MAX_MOVE_LIST_SIZE = 593;

    if (already_generated_moves_) {
        return moves_;
    }
    already_generated_moves_ = true;

    moves_.clear();
    moves_.reserve(MAX_MOVE_LIST_SIZE);

    //手番側に王手がかかっていたら逃れる手だけを生成
    if (is_checked_) {
        generateEvasionMoves(moves_);
    } else {
        generateNormalMoves(moves_);
    }

    return moves_;
}

inline bool Position::isLastMoveCheck() {
    Move move = lastMove();
    if (controlBB(move.to(), board_[move.to()], occupied_all_) & SQUARE_BB[king_sq_[color_]]) {
        //直接王手だったら即返す
        return true;
    }

    //開き王手になっているか
    auto dir = directionAtoB(king_sq_[color_], move.from());
    if (dir == RU || dir == RD || dir == LD || dir == LU) {
        //角の利きで開き王手になっている可能性がある
        if (kind(move.subject()) != KNIGHT &&
            (directionAtoB(move.from(), move.to()) == directionAtoB(move.to(), king_sq_[color_]) ||
             directionAtoB(move.from(), move.to()) == directionAtoB(king_sq_[color_], move.to()))) {
            //同じあるいは逆方向だったら関係ない
            return false;
        }

        //自玉の位置から角の利きを伸ばしてそこに相手の角や馬があったら王手になっている
        auto bishop_control = controlBB(king_sq_[color_], coloredPiece(~color_, BISHOP), occupied_all_);
        if (bishop_control & (pieces_bb_[coloredPiece(~color_, BISHOP)] | pieces_bb_[coloredPiece(~color_, BISHOP_PROMOTE)])) {
            return true;
        }
    } else if (dir == U || dir == R || dir == D || dir == L) {
        //飛車あるいは香車によって開き王手となっている可能性がある
        if (kind(move.subject()) != KNIGHT &&
            (directionAtoB(move.from(), move.to()) == directionAtoB(move.to(), king_sq_[color_]) ||
             directionAtoB(move.from(), move.to()) == directionAtoB(king_sq_[color_], move.to()))) {
            //同じあるいは逆方向だったら関係ない
            return false;
        }

        //元の位置が自玉の上下左右方向なので飛車と香車の開き王手を考える
        auto rook_control = controlBB(king_sq_[color_], coloredPiece(~color_, ROOK), occupied_all_);
        if (rook_control & (pieces_bb_[coloredPiece(~color_, ROOK)] | pieces_bb_[coloredPiece(~color_, ROOK_PROMOTE)])) {
            return true;
        }

        if ((color_ == BLACK && dir == U) || (color_ == WHITE && dir == D)) {
            //香車も考える
            if (rook_control & FRONT_BB[color_][SquareToRank[king_sq_[color_]]] & pieces_bb_[coloredPiece(~color_, LANCE)]) {
                return true;
            }
        }
    }

    return false;
}

bool Position::isRepeating(float& score) const {
    //千日手or連続王手の千日手だったらtrueを返してscoreに適切な値を入れる(技巧と似た実装)
    for (int32_t index = (int32_t)stack_.size() - 4; index > 0 && (index > ((int32_t)stack_.size() - 32)); index -= 2) {
        if (board_hash_ != stack_[index].board_hash) {
            //局面のハッシュ値が一致しなかったら関係ない
            continue;
        }

        //局面のハッシュ値が一致
        if (hand_hash_ == stack_[index].hand_hash) { //手駒のハッシュ値も一致
            if ((index == (int32_t)stack_.size() - 4) &&
                (stack_[index].is_checked && stack_[index + 2].is_checked)) { //手番側が連続王手された
                score = MAX_SCORE;
            } else { //普通の千日手
                score = (MAX_SCORE + MIN_SCORE) / 2;
            }
        } else {                                                      //局面のハッシュ値だけが一致
            if (hand_[color_].superior(stack_[index].hand[color_])) { //優等局面
                score = MAX_SCORE;
            } else if (hand_[color_].inferior(stack_[index].hand[color_])) { //劣等局面
                score = MIN_SCORE;
            } else {
                //その他は繰り返しではない
                return false;
            }
        }
        return true;
    }
    return false;
}

Move Position::transformValidMove(const Move move) {
    //stringToMoveではどっちの手番かがわからない
    //つまりsubjectが完全には入っていないので手番付きの駒を入れる
    return (move.isDrop() ? dropMove(move.to(), (color_ == BLACK ? toBlack(move.subject()) : toWhite(move.subject())))
                          : Move(move.to(), move.from(), false, move.isPromote(), board_[move.from()], board_[move.to()]));
}

void Position::initHashValue() {
    hash_value_ = 0;
    board_hash_ = 0;
    hand_hash_ = 0;
    for (Square sq : SquareList) {
        board_hash_ ^= HashSeed[board_[sq]][sq];
    }
    for (Piece piece = PAWN; piece <= ROOK; piece++) {
        hand_hash_ ^= HandHashSeed[BLACK][piece][hand_[BLACK].num(piece)];
        hand_hash_ ^= HandHashSeed[WHITE][piece][hand_[WHITE].num(piece)];
    }
    hash_value_ = board_hash_ ^ hand_hash_;
    hash_value_ &= ~1; //これで1bit目が0になる(先手番を表す)
}

std::vector<float> Position::makeFeature() const {
#ifdef DLSHOGI
    return makeDLShogiFeature();
#endif
    std::vector<float> features(SQUARE_NUM * INPUT_CHANNEL_NUM, 0);

    uint64_t i = 0;

    //盤上の駒の特徴量
    for (i = 0; i < PieceList.size(); i++) {
        //いま考慮している駒
        Piece t = (color_ == BLACK ? PieceList[i] : oppositeColor(PieceList[i]));

        //各マスについてそこにあるなら1,ないなら0とする
        for (Square sq : SquareList) {
            //後手のときは盤面を180度反転させる
            Piece p = (color_ == BLACK ? board_[sq] : board_[InvSquare[sq]]);
            features[i * SQUARE_NUM + SquareToNum[sq]] = (t == p ? 1 : 0);
        }
    }

    //持ち駒の特徴量:最大枚数で割って正規化する
    static constexpr std::array<Color, ColorNum> colors[2] = { { BLACK, WHITE }, { WHITE, BLACK } };
    static constexpr int32_t HAND_PIECE_NUM = 7;
    static constexpr std::array<Piece, HAND_PIECE_NUM> HAND_PIECES = { PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK };
    static constexpr std::array<float, HAND_PIECE_NUM> MAX_NUMS = { 18.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0 };
    for (int32_t c : colors[color_]) {
        for (int32_t j = 0; j < HAND_PIECE_NUM; j++) {
            for (Square sq : SquareList) {
                features[i * SQUARE_NUM + SquareToNum[sq]] = hand_[c].num(HAND_PIECES[j]) / MAX_NUMS[j];
            }
            i++;
        }
    }

    return features;
}

std::vector<float> Position::makeDLShogiFeature() const {
    std::vector<float> result((DLSHOGI_FEATURES1_NUM + DLSHOGI_FEATURES2_NUM) * SQUARE_NUM, 0);

    auto index = [](int64_t feature, int64_t color, int64_t ch, int64_t sq) {
        return feature * (DLSHOGI_FEATURES1_NUM * SQUARE_NUM) +
               color * (feature == 0 ? DLSHOGI_FEATURES1_NUM / 2 : DLSHOGI_FEATURES2_NUM / 2) * SQUARE_NUM + ch * SQUARE_NUM + sq;
    };

    constexpr unsigned long MAX_ATTACK_NUM = 3; // 利き数の最大値
    const int32_t MAX_PIECES_IN_HAND[HAND_PIECE_KIND_NUM + 1] = { 0, 8, 4, 4, 4, 4, 2, 2 };

    const Bitboard occupied_bb = occupied_all_;

    // dlshogiはEMPTY=0を含めて数えているのでPIECE_KIND_NUM + 1
    constexpr int64_t PieceTypeNum = PIECE_KIND_NUM + 1;

    // 駒の利き(駒種でマージ)
    std::vector<std::vector<Bitboard>> attacks(ColorNum, std::vector<Bitboard>(PieceTypeNum, { 0, 0 }));
    for (Square sq : SquareList) {
        const Piece p = board_[sq];
        if (p != EMPTY) {
            const Color pc = pieceToColor(p);
            const Piece pt = kindWithPromotion(p);
            const int64_t ind = DLShogiPieceToIndex[pt];
            attacks[pc][ind] |= controlBB(sq, p, occupied_bb);
        }
    }

    for (Color c : { BLACK, WHITE }) {
        // 白の場合、色を反転
        const Color c2 = (color_ == BLACK ? c : ~c);

        for (Square sq : SquareList) {
            // 白の場合、盤面を180度回転
            const Square sq2 = (color_ == BLACK ? sq : InvSquare[sq]);
            const int64_t sq_index = SquareToNum[sq2];

            for (int64_t i = 0; i < DLShogiPieceKindList.size(); i++) {
                Piece pt = DLShogiPieceKindList[i];
                // 駒の配置
                if (pieces_bb_[coloredPiece(c, pt)] & SQUARE_BB[sq]) {
                    result[index(0, c2, i, sq_index)] = 1;
                }

                // 駒の利き
                if (attacks[c][i + 1] & SQUARE_BB[sq]) {
                    result[index(0, c2, PIECE_KIND_NUM + i, sq_index)] = 1;
                }
            }

            // 利き数
            const int64_t num = std::min(MAX_ATTACK_NUM, attackersTo(c, sq, occupied_bb).pop_count());
            for (int64_t k = 0; k < num; k++) {
                result[index(0, c2, PIECE_KIND_NUM * 2 + k, sq_index)] = 1;
            }
        }

        // hand
        const Hand hand = hand_[c];
        int64_t p = 0;
        for (Piece hp : DLShogiHandPieceKindList) {
            int64_t num = std::min(hand.num(hp), MAX_PIECES_IN_HAND[hp]);
            std::fill_n(result.begin() + index(1, c2, p, 0), SQUARE_NUM * num, 1);
            p += MAX_PIECES_IN_HAND[hp];
        }
    }

    // is check
    if (is_checked_) {
        std::fill_n(result.end() - SQUARE_NUM, SQUARE_NUM, 1);
    }

    return result;
}

bool Position::isLastMoveDropPawn() const { return (lastMove().isDrop() && kind(lastMove().subject()) == PAWN); }

bool Position::isFinish(float& score, bool check_repeat) {
    //詰みの確認
    if (is_checked_) {
        std::vector<Move> moves = generateAllMoves();
        if (moves.empty()) {
            //打ち歩詰めなら手番側（詰まされた側）が勝ち、そうでないなら手番側が負け
            score = isLastMoveDropPawn() ? MAX_SCORE : MIN_SCORE;
            return true;
        }
    }

    //千日手の確認
    if (check_repeat && isRepeating(score)) {
        return true;
    }

    //宣言勝ち
    if (canWinDeclare()) {
        score = MAX_SCORE;
        return true;
    }

    //長手数による打ち切りだけはsearch_optionが関わるので外部で判定する
    //falseのときのデフォルト値として互角の値を設定
    score = (MAX_SCORE + MIN_SCORE) / 2;

    return false;
}

bool Position::canWinDeclare() const {
    //手番側が入玉宣言できるかどうか
    //WCSC29のルールに準拠
    //1. 宣言側の手番である
    //   手番側で考えるのでこれは自明
    //2. 宣言側の玉が敵陣三段目以内に入っている
    Rank rank = SquareToRank[king_sq_[color_]];
    if ((color_ == BLACK && rank > Rank3) || (color_ == WHITE && rank < Rank7)) {
        return false;
    }

    //5. 宣言側の玉に王手がかかっていない
    if (is_checked_) {
        return false;
    }
    //6. 宣言側の持ち時間が残っている
    //   これは自明なものとする

    //3. 宣言側が大駒5点小駒1点で計算して
    //   ・先手の場合28点以上の持点がある
    //   ・後手の場合27点以上の持点がある
    //   ・点数の対象となるのは、宣言側の持駒と敵陣三段目以内に存在する玉を除く宣言側の駒のみである
    //4. 宣言側の敵陣三段目以内の駒は、玉を除いて10枚以上存在する
    constexpr int64_t SCORE_TABLE[KING] = { 0, 1, 1, 1, 1, 1, 5, 5 };
    constexpr int64_t THRESHOLD[ColorNum] = { 28, 27 };
    constexpr int64_t LOWER_BOUND[ColorNum] = { Rank1, Rank7 };
    constexpr int64_t UPPER_BOUND[ColorNum] = { Rank3, Rank9 };
    int64_t score = 0, num = 0;
    for (int64_t r = LOWER_BOUND[color_]; r <= UPPER_BOUND[color_]; r++) {
        for (int64_t f = File1; f <= File9; f++) {
            Piece p = board_[FRToSquare[f][r]];
            if (pieceToColor(p) != color_ || kind(p) == KING) {
                continue;
            }
            score += SCORE_TABLE[kind(p)];
            num++;
        }
    }

    //持ち駒
    for (int64_t p = PAWN; p < KING; p++) {
        score += SCORE_TABLE[p] * hand_[color_].num(Piece(p));
    }

    return (score >= THRESHOLD[color_] && num >= 10);
}

std::string Position::augmentStr(const std::string& str, int64_t augmentation) {
    if (augmentation == 0) {
        //何もしない
        return str;
    } else if (augmentation == 1) {
        //左右反転

        //まず盤面部分とそれ以外の部分に分ける
        uint64_t space_pos = str.find(' ');
        std::string board = str.substr(0, space_pos);
        std::string others = str.substr(space_pos);

        //board部分について'/'で囲まれた各段を反転していく
        //9段目の末尾には'/'がないので予め付与しておく
        board += '/';
        uint64_t pre_pos = 0;
        for (int64_t i = 0; i < 9; i++) {
            uint64_t next_slash = board.find('/', pre_pos);
            std::reverse(board.begin() + pre_pos, board.begin() + next_slash);
            pre_pos = next_slash + 1;
        }

        //反転させると成を示す+が駒の後に来てしまうので入れ替える
        for (uint64_t i = 0; i < board.size(); i++) {
            if (board[i] == '+') {
                //一個前の前に+を挿入
                board.insert(board.begin() + i - 1, '+');
                board.erase(i + 1, 1);
            }
        }

        //余計に与えた'/'を削除
        board.erase(board.size() - 1);

        //結合して返す
        return board + others;
    } else {
        std::cout << "augmentation = " << augmentation << std::endl;
        exit(1);
    }
}

} // namespace Shogi