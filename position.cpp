#include"position.hpp"
#include"common.hpp"
#include"usi_options.hpp"
#include"neural_network.hpp"
#include<iostream>
#include<cstdio>
#include<ctime>
#include<bitset>
#include<cassert>
#include<iterator>
#include<algorithm>
#include<set>

int64_t Position::HashSeed[PieceNum][SquareNum];
int64_t Position::HandHashSeed[ColorNum][PieceNum][19];

Position::Position(){
    init();
}

void Position::init() {
    //盤上の初期化
    for (int i = 0; i < SquareNum; i++) board_[i] = WALL;
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
    turn_number_ = 0;

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
    for (int p = 0; p < PieceNum; ++p) {
        pieces_bb_[p] = Bitboard(0, 0);
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

    isChecked_ = false;
}

void Position::print(bool with_score) const {
    //盤上
    std::printf("９８７６５４３２１\n");
    std::printf("------------------\n");
    for (int r = Rank1; r <= Rank9; r++) {
        for (int f = File9; f >= File1; f--) {
            std::cout << PieceToSfenStr[board_[FRToSquare[f][r]]];
        }
        printf("|%d\n", r);
    }

    //先手の持ち駒
    printf("持ち駒\n");
    printf("先手:");
    hand_[BLACK].print();

    //後手の持ち駒
    printf("後手:");
    hand_[WHITE].print();

    //手番
    printf("手番:%s\n", (color_ == BLACK ? "先手" : "後手"));

    //手数
    printf("手数:%d\n", turn_number_);

    //最後の手
    if (!kifu_.empty()) {
        printf("最後の手:");
        lastMove().printWithScore();
    }

    //評価値
    if (with_score) {
#ifdef USE_CATEGORICAL
        auto y = nn->policyAndValueBatch(makeFeature());
        auto categorical = y.second.front();

        CalcType value = 0.0;
        for (int32_t i = 0; i < BIN_SIZE; i++) {
            printf("p[%f] = %f\n", MIN_SCORE + VALUE_WIDTH * (0.5 + i), categorical[i]);
            value += (CalcType)((MIN_SCORE + VALUE_WIDTH * (0.5 + i)) * categorical[i]);
        }
        printf("value = %f\n", value);
#else
        auto y = nn->policyAndValueBatch(makeFeature());
        std::cout << "value = " << y.second.front() << std::endl;
#endif
    }

    printf("ハッシュ値:%llx\n", (unsigned long long)hash_value_);
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

    //pinners
    computePinners();

    //手数の更新
    turn_number_++;

    //棋譜に指し手を追加
    kifu_.push_back(move);

    //王手
    //isChecked_ = isThereControl(~color_, king_sq_[color_]);
    isChecked_ = isLastMoveCheck();

    //hashの手番要素を更新
    hash_value_ = board_hash_ ^ hand_hash_;
    //1bit目を0にする
    hash_value_ &= ~1;
    //手番が先手だったら1bitは0のまま,後手だったら1bit目は1になる
    hash_value_ |= color_;
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
            hand_hash_ ^= HandHashSeed[color_][last_move.capture() & PIECE_KIND_MASK][hand_[color_].num(kind(last_move.capture()))];
            //増えた後の持ち駒の分XORして消す
            hand_hash_ ^= HandHashSeed[color_][last_move.capture() & PIECE_KIND_MASK][hand_[color_].num(kind(last_move.capture())) + 1];
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
        if (last_move.isPromote())
            board_hash_ ^= HashSeed[promote(last_move.subject())][last_move.to()];
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
    pinners_ = stack_.back().pinners;
    isChecked_ = stack_.back().isChecked;

    //Stack更新
    stack_.pop_back();
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
    pinners_.forEach([&](const Square pinner_sq) {
        if ((BETWEEN_BB[pinner_sq][king_sq_[color_]] & SQUARE_BB[move.from()]) //fromがbetween,すなわちピンされている
            && !((BETWEEN_BB[pinner_sq][king_sq_[color_]] | SQUARE_BB[pinner_sq]) & SQUARE_BB[move.to()]) //toがbetween内及びpinner_sq以外
            ) {
            flag = false;
        }
    });
    if (!flag) {
        return false;
    }

    return true;
}

bool Position::canDropPawn(const Square to) const {
    //2歩の判定を入れる
    if (FILE_BB[SquareToFile[to]] & pieces_bb_[color_ == BLACK ? toBlack(PAWN) : toWhite(PAWN)]) {
        return false;
    }

    //打ち歩詰めは探索時点で弾くのでチェック不要
    return true;
}

void Position::loadSFEN(std::string sfen) {
    //初期化
    for (int i = 0; i < SquareNum; i++) board_[i] = WALL;
    for (Square sq : SquareList) board_[sq] = EMPTY;

    //テーブル用意しておいたほうがシュッと書ける
    static std::unordered_map<char, Piece> CharToPiece = {
        { 'P', BLACK_PAWN },
        { 'L', BLACK_LANCE },
        { 'N', BLACK_KNIGHT },
        { 'S', BLACK_SILVER },
        { 'G', BLACK_GOLD },
        { 'B', BLACK_BISHOP },
        { 'R', BLACK_ROOK },
        { 'K', BLACK_KING },
        { 'p', WHITE_PAWN },
        { 'l', WHITE_LANCE },
        { 'n', WHITE_KNIGHT },
        { 's', WHITE_SILVER },
        { 'g', WHITE_GOLD },
        { 'b', WHITE_BISHOP },
        { 'r', WHITE_ROOK },
        { 'k', WHITE_KING },
    };

    //sfen文字列を走査するイテレータ(ダサいやり方な気がするけどパッと思いつくのはこれくらい)
    uint32_t i;

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
            if (CharToPiece[sfen[i]] == BLACK_KING) king_sq_[BLACK] = FRToSquare[f][r];
            else if (CharToPiece[sfen[i]] == WHITE_KING) king_sq_[WHITE] = FRToSquare[f][r];
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

    stack_.reserve(256);

    //Bitboard
    occupied_bb_[BLACK] = Bitboard(0, 0);
    occupied_bb_[WHITE] = Bitboard(0, 0);
    for (int p = 0; p < PieceNum; ++p) {
        pieces_bb_[p] = Bitboard(0, 0);
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

    //王手の確認
    isChecked_ = isThereControl(~color_, king_sq_[color_]);

    stack_.clear();
    stack_.reserve(512);
    kifu_.clear();
    kifu_.reserve(512);
}

std::string Position::toSFEN() const {
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

void Position::generateEvasionMoves(Move *& move_ptr) const {
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
            *(move_ptr++) = Move(to, evasion_from, false, false, board_[evasion_from], board_[to]);
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

    pinners_.forEach([&](const Square pinner_sq) {
        //pinnerと自玉の間にあるのがpinされている駒
        Bitboard pinned = BETWEEN_BB[pinner_sq][king_sq_[color_]] & occupied_bb_[color_];

        pinned.forEach([&](const Square from) {
            //取る手なのでpinnerを取る手、かつそこがchecker_sqでないとしかダメ
            Bitboard to_bb = controlBB(from, board_[from], occupied_all_) & SQUARE_BB[pinner_sq] & SQUARE_BB[checker_sq];

            if (to_bb) {
                pushMove(Move(pinner_sq, from, false, false, board_[from], board_[pinner_sq]), move_ptr);
            }

            //使ったマスを記録しておく
            pinned_piece |= SQUARE_BB[from];
        });
    });

    //王手してきた駒を取れる駒の候補
    Bitboard removers = attackersTo(color_, checker_sq) & ~pinned_piece & ~SQUARE_BB[king_sq_[color_]];

    removers.forEach([&](Square from) {
        pushMove(Move(checker_sq, from, false, false, board_[from], board_[checker_sq]), move_ptr);
    });

    //(c)合駒
    //王手してきた駒と自玉の間を示すBitboard
    Bitboard between = BETWEEN_BB[checker_sq][king_sq_[color_]];

    //(c)-1 移動合
    Bitboard from_bb = occupied_bb_[color_] & ~pinned_piece & ~SQUARE_BB[king_sq_[color_]];
    from_bb.forEach([&](const Square from) {
        Bitboard to_bb = controlBB(from, board_[from], occupied_all_) & between;
        to_bb.forEach([&](const Square to) {
            pushMove(Move(to, from, false, false, board_[from], board_[to]), move_ptr);
        });
    });

    //(c)-2 打つ合
    //王手されているのでbetweenの示すマスは駒がないはず
    generateDropMoves(between, move_ptr);
}

void Position::generateCaptureMoves(Move *& move_ptr) const {
    //ピンされた駒を先に動かす
    Bitboard pinned_piece(0, 0);

    pinners_.forEach([&](const Square pinner_sq) {
        //pinnerと自玉の間にあるのがpinされている駒
        Bitboard pinned = BETWEEN_BB[pinner_sq][king_sq_[color_]] & occupied_bb_[color_];

        pinned.forEach([&](const Square from) {
            //取る手なのでpinnerを取る手しかダメ
            Bitboard to_bb = controlBB(from, board_[from], occupied_all_) & SQUARE_BB[pinner_sq];

            if (to_bb) {
                pushMove(Move(pinner_sq, from, false, false, board_[from], board_[pinner_sq]), move_ptr);
            }

            //使ったマスを記録しておく
            pinned_piece |= SQUARE_BB[from];
        });
    });

    //玉は別に処理する
    Bitboard king_to_bb = controlBB(king_sq_[color_], board_[king_sq_[color_]], occupied_all_)
        & occupied_bb_[~color_];
    king_to_bb.forEach([&](const Square to) {
        if (!isThereControl(~color_, to)) {
            *(move_ptr++) = Move(to, king_sq_[color_], false, false, board_[king_sq_[color_]], board_[to]);
        }
    });

    //自分の駒がいる位置から、ピンされた駒と玉は先に処理したので除く
    Bitboard from_bb = occupied_bb_[color_] & ~pinned_piece & ~SQUARE_BB[king_sq_[color_]];

    from_bb.forEach([&](const Square from) {
        //駒を取る手なので利きの先に相手の駒がないといけない
        Bitboard to_bb = controlBB(from, board_[from], occupied_all_) & occupied_bb_[~color_];

        to_bb.forEach([&](const Square to) {
            pushMove(Move(to, from, false, false, board_[from], board_[to]), move_ptr);
        });
    });
}

void Position::generateNonCaptureMoves(Move *& move_ptr) const {
    //ピンされた駒を先に動かす
    Bitboard pinned_piece(0, 0);

    pinners_.forEach([&](const Square pinner_sq) {
        //pinnerと自玉の間にあるのがpinされている駒
        Bitboard pinned = BETWEEN_BB[pinner_sq][king_sq_[color_]] & occupied_bb_[color_];

        pinned.forEach([&](const Square from) {
            //取らない手なのでbetween上を動く手しかダメ(ピンされているのでbetween上に他の駒はない)
            Bitboard to_bb = controlBB(from, board_[from], occupied_all_) & BETWEEN_BB[pinner_sq][king_sq_[color_]];

            to_bb.forEach([&](const Square to) {
                pushMove(Move(to, from, false, false, board_[from], board_[to]), move_ptr);
            });

            //使ったマスを記録しておく
            pinned_piece |= SQUARE_BB[from];
        });
    });

    //王の処理
    Bitboard king_to_bb = controlBB(king_sq_[color_], board_[king_sq_[color_]], occupied_all_)
        & ~occupied_all_;
    king_to_bb.forEach([&](const Square to) {
        //相手の利きがなければそこへいく動きを生成できる
        if (!isThereControl(~color_, to)) {
            *(move_ptr++) = Move(to, king_sq_[color_], false, false, board_[king_sq_[color_]], board_[to]);
        }
    });

    //ピンされた駒と玉は先に処理したので除く
    Bitboard from_bb = occupied_bb_[color_] & ~pinned_piece & ~SQUARE_BB[king_sq_[color_]];
    from_bb.forEach([&](const Square from) {
        //sqにある駒の利き
        Bitboard to_bb = controlBB(from, board_[from], occupied_all_) & ~occupied_all_;
        to_bb.forEach([&](const Square to) {
            pushMove(Move(to, from, false, false, board_[from], board_[to]), move_ptr);
        });
    });

    //駒を打つ手
    Bitboard drop_to_bb = (~occupied_all_ & BOARD_BB);
    generateDropMoves(drop_to_bb, move_ptr);
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
        return ((Rank1 <= SquareToRank[move.to()] && SquareToRank[move.to()] <= Rank3) || (Rank1 <= SquareToRank[move.from()] && SquareToRank[move.from()] <= Rank3));
    } else {
        return ((Rank7 <= SquareToRank[move.to()] && SquareToRank[move.to()] <= Rank9) || (Rank7 <= SquareToRank[move.from()] && SquareToRank[move.from()] <= Rank9));
    }
}

void Position::pushMove(const Move move, Move*& move_ptr) const {
    //成る手が可能だったら先に生成
    if (canPromote(move)) {
        *(move_ptr++) = promotiveMove(move);
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
    //成らない手
    *(move_ptr++) = move;
}

void Position::generateDropMoves(const Bitboard& to_bb, Move *& move_ptr) const {
    //歩を打つ手
    //最奥の段は除外する
    if (hand_[color_].num(PAWN) > 0) {
        (to_bb & ~farRank1FromColor(color_)).forEach([&](Square to) {
            if (canDropPawn(to)) {
                *(move_ptr++) = dropMove(to, Piece(PAWN | ColorToFlag[color_]));
            }
        });
    }

    //香車
    //最奥の段は除外する
    if (hand_[color_].num(LANCE) > 0) {
        (to_bb & ~farRank1FromColor(color_)).forEach([&](Square to) {
            *(move_ptr++) = dropMove(to, Piece(LANCE | ColorToFlag[color_]));
        });
    }

    //桂馬
    //奥の2段は除外する
    if (hand_[color_].num(KNIGHT) > 0) {
        (to_bb & ~(farRank1FromColor(color_) | farRank2FromColor(color_))).forEach([&](Square to) {
            *(move_ptr++) = dropMove(to, Piece(KNIGHT | ColorToFlag[color_]));
        });
    }

    //その他
    for (Piece p : { SILVER, GOLD, BISHOP, ROOK }) {
        if (hand_[color_].num(p) > 0) {
            (to_bb).forEach([&](Square to) {
                *(move_ptr++) = dropMove(to, Piece(p | ColorToFlag[color_]));
            });
        }
    }
}

Bitboard Position::attackersTo(const Color c, const Square sq) const {
    return attackersTo(c, sq, occupied_all_);
}

Bitboard Position::attackersTo(const Color c, const Square sq, const Bitboard & occupied) const {
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
    result |= gold_control & (pieces_bb_[gold]
        | pieces_bb_[coloredPiece(c, PAWN_PROMOTE)]
        | pieces_bb_[coloredPiece(c, LANCE_PROMOTE)]
        | pieces_bb_[coloredPiece(c, KNIGHT_PROMOTE)]
        | pieces_bb_[coloredPiece(c, SILVER_PROMOTE)]);

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

void Position::computePinners() {
    //pinners
    pinners_ = Bitboard(0, 0);
    //香・角(馬)・飛車(竜)に対してピンのチェック
    for (Piece jumper : ColoredJumpPieceList[~color_]) {
        //自玉からこっちの香・角・飛として利きを駒を最大まで伸ばして
        //そこに相手の香・角(馬)・飛(竜)があったらそれはpinnerになりうる
        //香だったらそれだけ,角飛だったらなったものについて存在位置を取る
        Bitboard jumper_bb = (kind(jumper) == LANCE ? pieces_bb_[jumper] : pieces_bb_[jumper] | pieces_bb_[promote(jumper)]);
        Bitboard pinner_candidate = controlBB(king_sq_[color_], oppositeColor(jumper), Bitboard(0, 0)) & jumper_bb;

        //各pinnerの候補についてking_sq_との間を見ていく
        pinner_candidate.forEach([&](const Square jump_piece_sq) {
            Bitboard between = BETWEEN_BB[king_sq_[color_]][jump_piece_sq] & occupied_all_;
            if (between.pop_count() == 1 && (between & occupied_bb_[color_])) { //betweenに駒が1個だけかつ駒が手番側のものだったらピンされている
                                                                                //ピンしている駒を記録しておけばピンされた駒は自玉との間でbetween見れば復元できる
                pinners_ |= SQUARE_BB[jump_piece_sq];
            }
        });
    }
}

std::vector<Move> Position::generateAllMoves() const {
    constexpr int32_t MAX_MOVE_LIST_SIZE = 593;
    Move move_buf[MAX_MOVE_LIST_SIZE];
    Move* move_ptr = move_buf;
    //手番側に王手がかかっていたら逃れる手だけを生成
    if (isChecked_) {
        generateEvasionMoves(move_ptr);
    } else {
        generateCaptureMoves(move_ptr);
        generateNonCaptureMoves(move_ptr);
    }

    std::vector<Move> move_list;
    move_list.reserve(MAX_MOVE_LIST_SIZE);
    for (move_ptr--; move_buf <= move_ptr; move_ptr--) {
        move_list.push_back(*move_ptr);
    }
    return move_list;
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
        if (kind(move.subject()) != KNIGHT && (directionAtoB(move.from(), move.to()) == directionAtoB(move.to(), king_sq_[color_])
            || directionAtoB(move.from(), move.to()) == directionAtoB(king_sq_[color_], move.to()))) {
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
        if (kind(move.subject()) != KNIGHT && (directionAtoB(move.from(), move.to()) == directionAtoB(move.to(), king_sq_[color_])
            || directionAtoB(move.from(), move.to()) == directionAtoB(king_sq_[color_], move.to()))) {
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

bool Position::isRepeating(Score& score) const {
    //千日手or連続王手の千日手だったらtrueを返してscoreに適切な値を入れる(技巧と似た実装)
    for (int32_t index = (int32_t)stack_.size() - 4; index > 0 && (index > ((int32_t)stack_.size() - 32)); index -= 2) {
        if (board_hash_ != stack_[index].board_hash) {
            //局面のハッシュ値が一致しなかったら関係ない
            continue;
        }

        //局面のハッシュ値が一致
        if (hand_hash_ == stack_[index].hand_hash) { //手駒のハッシュ値も一致
            if ((index == (int32_t) stack_.size() - 4) &&
                (stack_[index].isChecked && stack_[index + 2].isChecked)) { //手番側が連続王手された
                score = MAX_SCORE;
            } else { //普通の千日手
                score = (MAX_SCORE + MIN_SCORE) / 2;
            }
        } else { //局面のハッシュ値だけが一致
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
    return (move.isDrop() ?
            dropMove(move.to(), (color_ == BLACK ? toBlack(move.subject()) : toWhite(move.subject()))) :
            Move(move.to(), move.from(), false, move.isPromote(), board_[move.from()], board_[move.to()]));
}

void Position::initHashValue() {
    hash_value_ = 0;
    board_hash_ = 0;
    hand_hash_ = 0;
    for (auto sq : SquareList) {
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
    std::vector<float> features(SQUARE_NUM * INPUT_CHANNEL_NUM, 0);

    int32_t i;

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

    //持ち駒の特徴量:最大枚数で割って正規化するべきか？
    static constexpr Color colors[2][2] = { { BLACK, WHITE }, { WHITE, BLACK} };
    static constexpr int32_t HAND_PIECE_NUM = 7;
    static constexpr Piece HAND_PIECES[HAND_PIECE_NUM] = { PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK };
    static constexpr float MAX_NUM[HAND_PIECE_NUM] = { 18.0f, 4.0f, 4.0f, 4.0f, 4.0f, 2.0f, 2.0f };
    for (int32_t c : colors[color_]) {
        for (int32_t j = 0; j < HAND_PIECE_NUM; j++) {
            for (Square sq : SquareList) {
                features[i * SQUARE_NUM + SquareToNum[sq]] = hand_[c].num(HAND_PIECES[i]) / MAX_NUM[i];
            }
            i++;
        }
    }

    return features;
}

bool Position::isLastMoveDropPawn() const {
    return (lastMove().isDrop() && kind(lastMove().subject()) == PAWN);
}