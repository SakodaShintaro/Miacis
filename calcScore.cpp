#include"position.hpp"
#include"piece.hpp"
#include"common.hpp"
#include"piece_state.hpp"
#include"eval_params.hpp"
#include"usi_options.hpp"
#include"network.hpp"
#include<iostream>
#include<fstream>

//enum {
//    PAWN_VALUE = 100,
//    LANCE_VALUE = 267,
//    KNIGHT_VALUE = 295,
//    SILVER_VALUE = 424,
//    GOLD_VALUE = 510,
//    BISHOP_VALUE = 654,
//    ROOK_VALUE = 738,
//    PAWN_PROMOTE_VALUE = 614,
//    LANCE_PROMOTE_VALUE = 562,
//    KNIGHT_PROMOTE_VALUE = 586,
//    SILVER_PROMOTE_VALUE = 569,
//    BISHOP_PROMOTE_VALUE = 951,
//    ROOK_PROMOTE_VALUE = 1086,
//};
//
//int piece_value[] = {
//    0, static_cast<int>(PAWN_VALUE * 1.05), static_cast<int>(LANCE_VALUE * 1.05), static_cast<int>(KNIGHT_VALUE * 1.05), static_cast<int>(SILVER_VALUE * 1.05),
//    static_cast<int>(GOLD_VALUE * 1.05), static_cast<int>(BISHOP_VALUE * 1.05), static_cast<int>(ROOK_VALUE * 1.05), 0, 0, //0~9
//    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //10~19
//    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //20~29
//    0, 0, 0, PAWN_VALUE, LANCE_VALUE, KNIGHT_VALUE, SILVER_VALUE, GOLD_VALUE, BISHOP_VALUE, ROOK_VALUE, //30~39
//    0, 0, 0, 0, 0, 0, 0, 0, 0, PAWN_PROMOTE_VALUE, //40~49
//    LANCE_PROMOTE_VALUE, KNIGHT_PROMOTE_VALUE,  SILVER_PROMOTE_VALUE, 0, BISHOP_PROMOTE_VALUE, ROOK_PROMOTE_VALUE, 0, 0, 0, 0, //50~59
//    0, 0, 0, 0, 0, -PAWN_VALUE, -LANCE_VALUE, -KNIGHT_VALUE, -SILVER_VALUE, -GOLD_VALUE, //60~69
//    -BISHOP_VALUE, -ROOK_VALUE, 0, 0, 0, 0, 0, 0, 0, 0, //70~79
//    0, -PAWN_PROMOTE_VALUE, -LANCE_PROMOTE_VALUE, -KNIGHT_PROMOTE_VALUE, -SILVER_PROMOTE_VALUE, 0, -BISHOP_PROMOTE_VALUE, -ROOK_PROMOTE_VALUE, 0, 0 //80~89
//};

//Aperyの駒割り
enum {
    PAWN_VALUE = 100 * 9 / 10,
    LANCE_VALUE = 350 * 9 / 10,
    KNIGHT_VALUE = 450 * 9 / 10,
    SILVER_VALUE = 550 * 9 / 10,
    GOLD_VALUE = 600 * 9 / 10,
    BISHOP_VALUE = 950 * 9 / 10,
    ROOK_VALUE = 1100 * 9 / 10,
    PAWN_PROMOTE_VALUE = 600 * 9 / 10,
    LANCE_PROMOTE_VALUE = 600 * 9 / 10,
    KNIGHT_PROMOTE_VALUE = 600 * 9 / 10,
    SILVER_PROMOTE_VALUE = 600 * 9 / 10,
    BISHOP_PROMOTE_VALUE = 1050 * 9 / 10,
    ROOK_PROMOTE_VALUE = 1550 * 9 / 10,
};

int32_t piece_value[] = {
    0, PAWN_VALUE, LANCE_VALUE, KNIGHT_VALUE, SILVER_VALUE, GOLD_VALUE, BISHOP_VALUE, ROOK_VALUE, 0, 0, //0~9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //10~19
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //20~29
    0, 0, 0, PAWN_VALUE, LANCE_VALUE, KNIGHT_VALUE, SILVER_VALUE, GOLD_VALUE, BISHOP_VALUE, ROOK_VALUE, //30~39
    0, 0, 0, 0, 0, 0, 0, 0, 0, PAWN_PROMOTE_VALUE, //40~49
    LANCE_PROMOTE_VALUE, KNIGHT_PROMOTE_VALUE,  SILVER_PROMOTE_VALUE, 0, BISHOP_PROMOTE_VALUE, ROOK_PROMOTE_VALUE, 0, 0, 0, 0, //50~59
    0, 0, 0, 0, 0, -PAWN_VALUE, -LANCE_VALUE, -KNIGHT_VALUE, -SILVER_VALUE, -GOLD_VALUE, //60~69
    -BISHOP_VALUE, -ROOK_VALUE, 0, 0, 0, 0, 0, 0, 0, 0, //70~79
    0, -PAWN_PROMOTE_VALUE, -LANCE_PROMOTE_VALUE, -KNIGHT_PROMOTE_VALUE, -SILVER_PROMOTE_VALUE, 0, -BISHOP_PROMOTE_VALUE, -ROOK_PROMOTE_VALUE, 0, 0 //80~89
};

//KPPTとNNで共通するもの
void Position::initPieceScore() {
    piece_score_ = Score(0);

    //盤上にある駒の価値
    for (Square sq : SquareList) {
        if (0 > board_[sq] || PieceNum < board_[sq]) {
            print();
            printHistory();
            assert(false);
        }
        piece_score_ += piece_value[board_[sq]];
    }

    //持ち駒の価値
    for (Piece p = PAWN; p <= ROOK; p++) {
        piece_score_ += piece_value[p] * hand_[BLACK].num(p);
        piece_score_ -= piece_value[p] * hand_[WHITE].num(p);
    }
}

#ifdef USE_NN

void Position::initScore() {
    initPieceScore();

    std::vector<CalcType> input = makeFeature();
    Vec input_vec = Eigen::Map<const Vec>(input.data(), input.size());
    Vec x[LAYER_NUM], u[LAYER_NUM];
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        x[i] = (i == 0 ? input_vec : Network::activationFunction(u[i - 1]));
        u[i] = eval_params_.w[i] * x[i] + eval_params_.b[i];
    }
    output_ = u[LAYER_NUM - 1];
    already_calc_ = true;
}

void Position::calcScoreDiff() {
    initScore();
}

Vec Position::makeOutput() const{
    std::vector<CalcType> input = makeFeature();
    Vec input_vec = Eigen::Map<const Vec>(input.data(), input.size());
    Vec x[LAYER_NUM], u[LAYER_NUM];
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        x[i] = (i == 0 ? input_vec : Network::activationFunction(u[i - 1]));
        u[i] = eval_params_.w[i] * x[i] + eval_params_.b[i];
    }
    return u[LAYER_NUM - 1];
}

std::vector<CalcType> Position::policy() {
    if (!already_calc_) {
        initScore();
    }
    std::vector<CalcType> policy(POLICY_DIM);
    for (int32_t i = 0; i < POLICY_DIM; i++) {
        policy[i] = output_(i);
    }
    return policy;
}

std::vector<CalcType> Position::maskedPolicy() {
    if (!already_calc_) {
        initScore();
    }
    std::vector<CalcType> policy(POLICY_DIM);
    for (int32_t i = 0; i < POLICY_DIM; i++) {
        policy[i] = output_(i);
    }
    const auto moves = generateAllMoves();
    std::vector<CalcType> dist(moves.size()), y(POLICY_DIM, 0.0);
    for (int32_t i = 0; i < moves.size(); i++) {
        dist[i] = policy[moves[i].toLabel()];
    }
    dist = softmax(dist);
    for (int32_t i = 0; i < moves.size(); i++) {
        y[moves[i].toLabel()] = dist[i];
    }

    return y;
}

Score Position::score() {
    if (!already_calc_) {
        initScore();
    }
    Score score = Score(0);
    int32_t nn_score = (int32_t)output_(POLICY_DIM);
    score += (color_ == BLACK ? nn_score : -nn_score);
    return score;
}

Score Position::scoreForTurn() {
    return (color_ == BLACK ? score() : -score());
}

double Position::valueForTurn() {
    if (!already_calc_) {
        initScore();
    }
#ifdef USE_CATEGORICAL
    std::vector<CalcType> categorical_distribution(BIN_SIZE);
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        categorical_distribution[i] = output_[POLICY_DIM + i];
    }
    categorical_distribution = softmax(categorical_distribution);
    CalcType value = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        value += (CalcType)(VALUE_WIDTH * (0.5 + i) * categorical_distribution[i]);
    }
    return value;
#else
    return standardSigmoid(output_[POLICY_DIM]);
#endif
}

void Position::resetCalc() {
    already_calc_ = false;
}

#ifdef USE_CATEGORICAL
std::array<CalcType, BIN_SIZE> Position::valueDist() {
    if (!already_calc_) {
        initScore();
    }
    std::vector<CalcType> categorical_distribution(BIN_SIZE);
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        categorical_distribution[i] = output_[POLICY_DIM + i];
    }
    categorical_distribution = softmax(categorical_distribution);
    std::array<CalcType, BIN_SIZE> result;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        result[i] =  categorical_distribution[i];
    }
    return result;
}
#endif

std::vector<float> Position::makeFeature() const {
    std::vector<float> features(INPUT_DIM, 0);
    int32_t index = 81;
    float scale = 1.0;
    if (color_ == BLACK) {
        for (auto sq : SquareList) {
            features[SquareToNum[sq]] = scale * (float)pieceToIndex[board_[sq]];
        }
        for (int c : {BLACK, WHITE}) {
            for (Piece p : {PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK}) {
                features[index++] = scale * (float)hand_[c].num(p);
            }
        }
    } else {
        for (auto sq : SquareList) {
            auto inv_sq = InvSquare[sq];
            features[SquareToNum[sq]] = -scale * (float)pieceToIndex[board_[inv_sq]];
        }
        for (int c : {WHITE, BLACK}) {
            for (Piece p : {PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK}) {
                features[index++] = scale * (float)hand_[c].num(p);
            }
        }
    }

    assert(index == INPUT_DIM);
    return features;
}

#else

constexpr int EVAL_SCALE = 32;

#define USE_SIMD

void Position::initScore() {
    initPieceScore();
#ifdef USE_SIMD
    initKKP_KPPScoreBySIMD();
#else
    initKKP_KPPScore();
#endif // USE_SIMD
}

void Position::initKKP_KPPScore() {
    for (int i = 0; i < KKP_KPP_END; i++) {
        for (int j = 0; j < RAW_TURN_END; j++) {
            score_state_[i][j] = 0;
        }
    }

    const int bk_sq = SquareToNum[ee_.king_sq[BLACK]];
    const int wk_sq = SquareToNum[ee_.king_sq[WHITE]];
    const int wk_sqr = SquareToNum[InvSquare[ee_.king_sq[WHITE]]];

    for (uint32_t i = 0; i < PIECE_STATE_LIST_SIZE; i++) {
        score_state_[KKP] += eval_params_.kkp[bk_sq][wk_sq][ee_.piece_state_list[0][i]];

        for (uint32_t j = i + 1; j < PIECE_STATE_LIST_SIZE; j++) {
            score_state_[KPP_BLACK] += eval_params_.kpp[bk_sq][ee_.piece_state_list[0][i]][ee_.piece_state_list[0][j]];
            score_state_[KPP_WHITE] += eval_params_.kpp[wk_sqr][ee_.piece_state_list[1][i]][ee_.piece_state_list[1][j]];
        }
    }

#if DEBUG
    auto copy = score_state_;
    initKKP_KPPScoreBySIMD();
    checkScoreState(copy);
#endif
}

void Position::initKKP_KPPScoreBySIMD() {
    for (int i = 0; i < KKP_KPP_END; i++) {
        for (int j = 0; j < RAW_TURN_END; j++) {
            score_state_[i][j] = 0;
        }
    }

    const int bk_sq = SquareToNum[ee_.king_sq[BLACK]];
    const int wk_sq = SquareToNum[ee_.king_sq[WHITE]];
    const int wk_sqr = SquareToNum[InvSquare[ee_.king_sq[WHITE]]];

    for (uint32_t i = 0; i < PIECE_STATE_LIST_SIZE; i++) {
        PieceState ps = ee_.piece_state_list[0][i];
        PieceState inv_ps = ee_.piece_state_list[1][i];

        score_state_[KKP] += eval_params_.kkp[bk_sq][wk_sq][ps];
        //以下ずっと~0は先手についてのkpp, ~1は後手についてのkpp

        //まずは0初期化したものを準備
        __m256i sum256_0 = _mm256_setzero_si256();
        __m256i sum256_1 = _mm256_setzero_si256();

        uint32_t j = 0;
        for (; j + 8 < i; j += 8) {
            //list[0],[1]から8要素ロード
            __m256i indexes0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&ee_.piece_state_list[0][j]));
            __m256i indexes1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&ee_.piece_state_list[1][j]));

            //indexesをオフセットとしてkppから8要素ギャザー
            __m256i gathered0 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(&(eval_params_.kpp[bk_sq][ps])), indexes0, 4);
            __m256i gathered1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(&(eval_params_.kpp[wk_sqr][inv_ps])), indexes1, 4);

            //16bitのペアになっているので32bitに拡張
            //まずは下位128bitについて
            __m256i low0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gathered0, 0));
            __m256i low1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gathered1, 0));

            //足し合わせる
            sum256_0 = _mm256_add_epi32(sum256_0, low0);
            sum256_1 = _mm256_add_epi32(sum256_1, low1);

            //上位128bitを拡張
            __m256i high0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gathered0, 1));
            __m256i high1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gathered1, 1));

            //足し合わせる
            sum256_0 = _mm256_add_epi32(sum256_0, high0);
            sum256_1 = _mm256_add_epi32(sum256_1, high1);
        }
        for (; j + 4 < i; j += 4) {
            // list[0], [1]から 4 要素ロードする
            __m128i indexes0 = _mm_load_si128(reinterpret_cast<const __m128i*>(&ee_.piece_state_list[0][j]));
            __m128i indexes1 = _mm_load_si128(reinterpret_cast<const __m128i*>(&ee_.piece_state_list[1][j]));

            //indexesをオフセットとしてkppから4要素ギャザー
            __m128i gathred0 = _mm_i32gather_epi32(reinterpret_cast<const int*>(&eval_params_.kpp[bk_sq][ps]), indexes0, 4);
            __m128i gathred1 = _mm_i32gather_epi32(reinterpret_cast<const int*>(&eval_params_.kpp[wk_sqr][inv_ps]), indexes1, 4);

            // 16 ビット整数→32 ビット整数に変換する
            __m256i expanded0 = _mm256_cvtepi16_epi32(gathred0);
            __m256i expanded1 = _mm256_cvtepi16_epi32(gathred1);

            //足し合わせる
            sum256_0 = _mm256_add_epi32(sum256_0, expanded0);
            sum256_1 = _mm256_add_epi32(sum256_1, expanded1);
        }
        for (; j < i; j++) {
            score_state_[KPP_BLACK] += eval_params_.kpp[bk_sq][ps][ee_.piece_state_list[0][j]];
            score_state_[KPP_WHITE] += eval_params_.kpp[wk_sqr][inv_ps][ee_.piece_state_list[1][j]];
        }
        //上位 128 ビットと下位 128 ビットを独立して 8 バイトシフトしたものを足し合わせる
        sum256_0 = _mm256_add_epi32(sum256_0, _mm256_srli_si256(sum256_0, 8));
        sum256_1 = _mm256_add_epi32(sum256_1, _mm256_srli_si256(sum256_1, 8));

        //上位 128 ビットと下位 128 ビットを足しあわせる
        __m128i sum128_0 = _mm_add_epi32(_mm256_extracti128_si256(sum256_0, 0), _mm256_extracti128_si256(sum256_0, 1));
        __m128i sum128_1 = _mm_add_epi32(_mm256_extracti128_si256(sum256_1, 0), _mm256_extracti128_si256(sum256_1, 1));

        //下位 64 ビットをストアする
        std::array<int32_t, 2> sum0{}, sum1{};
        _mm_storel_epi64(reinterpret_cast<__m128i*>(sum0.data()), sum128_0);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(sum1.data()), sum128_1);

        score_state_[KPP_BLACK] += sum0;
        score_state_[KPP_WHITE] += sum1;
    }
}

Score Position::score() const {
    //先手から見た点数を返す
    if (usi_option.draw_turn <= turn_number_) {
        return (Score)usi_option.draw_score;
    }
    int32_t sum = score_state_[KKP][RAW] + score_state_[KPP_BLACK][RAW] - score_state_[KPP_WHITE][RAW];
    int32_t turn_bonus = score_state_[KKP][TURN_BONUS] + score_state_[KPP_BLACK][TURN_BONUS] + score_state_[KPP_WHITE][TURN_BONUS];
    sum += (color_ == BLACK ? turn_bonus : -turn_bonus);
    sum /= EVAL_SCALE;
    sum += piece_score_;
    sum = sum * 10 / 9;
    return Score(sum);
}

Score Position::scoreForTurn() const {
    return (color_ == BLACK ? score() : -score());
}

void Position::initFeature() {
    ee_.king_sq[BLACK] = king_sq_[BLACK];
    ee_.king_sq[WHITE] = king_sq_[WHITE];
    ee_.color = color_;

    int32_t index = 0;
    for (Square sq : SquareList) {
        if (board_[sq] != EMPTY && kind(board_[sq]) != KING) {
            ee_.piece_state_list[0][index] = pieceState(board_[sq], sq);
            ee_.piece_state_list[1][index++] = invPieceState(pieceState(board_[sq], sq));
        }
    }
    for (Piece p = PAWN; p <= ROOK; p++) {
        for (Color c : {BLACK, WHITE}) {
            for (int32_t i = 0; i < hand_[c].num(p); i++) {
                ee_.piece_state_list[0][index] = pieceState(kind(p), i + 1, c);
                ee_.piece_state_list[1][index++] = invPieceState(pieceState(kind(p), i + 1, c));
            }
        }
    }
    assert(index == PIECE_STATE_LIST_SIZE);

    for (int32_t i = 0; i < PieceStateNum; i++) {
        piece_state_to_index_[i] = -1;
    }
    for (int32_t i = 0; i < PIECE_STATE_LIST_SIZE; i++) {
        piece_state_to_index_[ee_.piece_state_list[0][i]] = (int16_t)i;
    }
}

void Position::updatePieceStateList(PieceState before, PieceState after) {
    int index = piece_state_to_index_[before];
    if (index < 0 || PIECE_STATE_LIST_SIZE <= index) {
        std::cout << before << std::endl;

        std::cout << "今のリスト" << std::endl;
        for (int i = 0; i < PIECE_STATE_LIST_SIZE; i++) {
            std::cout << ee_.piece_state_list[0][i] << "->" << piece_state_to_index_[ee_.piece_state_list[0][i]] << std::endl;
        }

        for (int i = 0; i < PieceStateNum; i++) {
            if (piece_state_to_index_[i] != -1) {
                std::cout << "piece_state_to_index_[" << PieceState(i) << "] = " << piece_state_to_index_[i] << std::endl;
            }
        }

        assert(false);
    }
    ee_.piece_state_list[0][index] = after;
    ee_.piece_state_list[1][index] = invPieceState(after); //[1]の方には後手から見たものを入れる

    piece_state_to_index_[before] = -1;
    piece_state_to_index_[after] = (int16_t)index;
}

void Position::changeOnePS(PieceState ps, int c) {
    //cには1 or -1が入る
    const int bk_sq = SquareToNum[ee_.king_sq[BLACK]];
    const int wk_sq = SquareToNum[ee_.king_sq[WHITE]];
    const int wk_sqr = SquareToNum[InvSquare[ee_.king_sq[WHITE]]];

    //KKP
    score_state_[KKP] += c * eval_params_.kkp[bk_sq][wk_sq][ps];

    //KPP
    for (int i = 0; i < PIECE_STATE_LIST_SIZE; i++) {
        score_state_[KPP_BLACK] += c * eval_params_.kpp[bk_sq][ps][ee_.piece_state_list[0][i]];
        score_state_[KPP_WHITE] += c * eval_params_.kpp[wk_sqr][invPieceState(ps)][ee_.piece_state_list[1][i]];
    }
}

void Position::changeOnePSBySIMD(PieceState ps, int c) {
    const int bk_sq = SquareToNum[ee_.king_sq[BLACK]];
    const int wk_sq = SquareToNum[ee_.king_sq[WHITE]];
    const int wk_sqr = SquareToNum[InvSquare[ee_.king_sq[WHITE]]];

    //KKP
    score_state_[KKP] += c * eval_params_.kkp[bk_sq][wk_sq][ps];

    //KPPTは差分計算する
    std::array<int32_t, ColorNum> diff[KKP_KPP_END]{ {0, 0}, {0, 0}, {0, 0} };

    //以下ずっと~0は先手についてのkpp, ~1は後手についてのkpp
    //まずは0初期化したものを準備
    __m256i sum256_0 = _mm256_setzero_si256();
    __m256i sum256_1 = _mm256_setzero_si256();

    int i = 0;
    for (; i + 8 < PIECE_STATE_LIST_SIZE; i += 8) {
        //list[0],[1]から8要素ロード
        __m256i indexes0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&ee_.piece_state_list[0][i]));
        __m256i indexes1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&ee_.piece_state_list[1][i]));

        //indexesをオフセットとしてkppから8要素ギャザー
        __m256i gathered0 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(&(eval_params_.kpp[bk_sq][ps])), indexes0, 4);
        __m256i gathered1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(&(eval_params_.kpp[wk_sqr][invPieceState(ps)])), indexes1, 4);

        //16bitのペアになっているので32bitに拡張
        //まずは下位128bitについて
        __m256i low0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gathered0, 0));
        __m256i low1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gathered1, 0));

        //足し合わせる
        sum256_0 = _mm256_add_epi32(sum256_0, low0);
        sum256_1 = _mm256_add_epi32(sum256_1, low1);

        //16bitのペアになっているので32bitに拡張
        __m256i high0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gathered0, 1));
        __m256i high1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gathered1, 1));

        //足し合わせる
        sum256_0 = _mm256_add_epi32(sum256_0, high0);
        sum256_1 = _mm256_add_epi32(sum256_1, high1);
    }
    for (; i + 4 < PIECE_STATE_LIST_SIZE; i += 4) {
        // list[0], [1]から 4 要素ロードする
        __m128i indexes0 = _mm_load_si128(reinterpret_cast<const __m128i*>(&ee_.piece_state_list[0][i]));
        __m128i indexes1 = _mm_load_si128(reinterpret_cast<const __m128i*>(&ee_.piece_state_list[1][i]));

        //indexesをオフセットとしてkppから4要素ギャザー
        __m128i gathered0 = _mm_i32gather_epi32(reinterpret_cast<const int*>(&eval_params_.kpp[bk_sq][ps]), indexes0, 4);
        __m128i gathered1 = _mm_i32gather_epi32(reinterpret_cast<const int*>(&eval_params_.kpp[wk_sqr][invPieceState(ps)]), indexes1, 4);

        // 16 ビット整数→32 ビット整数に変換する
        __m256i expanded0 = _mm256_cvtepi16_epi32(gathered0);
        __m256i expanded1 = _mm256_cvtepi16_epi32(gathered1);

        //足し合わせる
        sum256_0 = _mm256_add_epi32(sum256_0, expanded0);
        sum256_1 = _mm256_add_epi32(sum256_1, expanded1);
    }
    for (; i < PIECE_STATE_LIST_SIZE; i++) {
        diff[KPP_BLACK] += eval_params_.kpp[bk_sq][ps][ee_.piece_state_list[0][i]];
        diff[KPP_WHITE] += eval_params_.kpp[wk_sqr][invPieceState(ps)][ee_.piece_state_list[1][i]];
    }
    //上位 128 ビットと下位 128 ビットを独立して 8 バイトシフトしたものを足し合わせる
    sum256_0 = _mm256_add_epi32(sum256_0, _mm256_srli_si256(sum256_0, 8));
    sum256_1 = _mm256_add_epi32(sum256_1, _mm256_srli_si256(sum256_1, 8));

    // diffp0 の上位 128 ビットと下位 128 ビットを足しあわせて diffp0_128 に代入する
    __m128i sum128_0 = _mm_add_epi32(_mm256_extracti128_si256(sum256_0, 0), _mm256_extracti128_si256(sum256_0, 1));
    __m128i sum128_1 = _mm_add_epi32(_mm256_extracti128_si256(sum256_1, 0), _mm256_extracti128_si256(sum256_1, 1));

    // diffp0_128 の下位 64 ビットを diff.p[1]にストアする
    std::array<int32_t, ColorNum> sum0, sum1;
    _mm_storel_epi64(reinterpret_cast<__m128i*>(sum0.data()), sum128_0);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(sum1.data()), sum128_1);

    diff[KPP_BLACK] += sum0;
    diff[KPP_WHITE] += sum1;

    score_state_[KPP_BLACK] += c * diff[KPP_BLACK];
    score_state_[KPP_WHITE] += c * diff[KPP_WHITE];
}

Score Position::SEE() {
    Score best_score = (color_ == BLACK ? score() : -score());
    Move move_buf[MAX_MOVE_LIST_SIZE];
    Move* end = move_buf;
    generateRecaptureMovesTo(lastMove().to(), end);
    for (Move* start = move_buf; start != end; start++) {
        doMoveWithoutCalcDiff(*start);
        Score curr_score = -SEE();
        best_score = std::max(best_score, curr_score);
        undo();
    }
    return best_score;
}

Score Position::SEEwithoutDoMove() {
    Score curr_score = (color_ == BLACK ? score() : -score());

    auto ally_attackers = attackersTo(color_, lastMove().to());
    auto enemy_attackers = attackersTo(~color_, lastMove().to());

    int ally_num[ROOK_PROMOTE + 1] = {}, enemy_num[ROOK_PROMOTE + 1] = {};

    Piece subject = board_[lastMove().to()];
    
    while (ally_attackers) {
        auto from = ally_attackers.pop();
        ally_num[kind(board_[from])]++;
    }
    while (enemy_attackers) {
        auto from = enemy_attackers.pop();
        enemy_num[kind(board_[from])]++;
    }

    static constexpr Piece piece_order[] = {
        PAWN, LANCE, KNIGHT, SILVER, GOLD, SILVER_PROMOTE, KNIGHT_PROMOTE, LANCE_PROMOTE, PAWN_PROMOTE, BISHOP, ROOK, BISHOP_PROMOTE, ROOK_PROMOTE
    };

    int ally_index = 0, enemy_index = 0;
    while (true) {
        while (ally_num[piece_order[ally_index]] == 0) {
            if (++ally_index == 13) {
                return curr_score;
            }
        }
        ally_num[piece_order[ally_index]]--;
        curr_score += std::abs(piece_value[subject]);
        subject = piece_order[ally_index];

        while (enemy_num[piece_order[enemy_index]] == 0) {
            if (++enemy_index == 13) {
                return curr_score;
            }
        }
        enemy_num[piece_order[enemy_index]]--;
        curr_score -= std::abs(piece_value[subject]);
        subject = piece_order[enemy_index];
    }

    assert(false);
    return Score();
}

void Position::calcScoreDiff() {
    Move move = lastMove();
    const int bk_sq = SquareToNum[ee_.king_sq[BLACK]];
    const int wk_sq = SquareToNum[ee_.king_sq[WHITE]];
    const int wk_sqr = SquareToNum[InvSquare[ee_.king_sq[WHITE]]];

    //移動する駒が玉かどうか、captureかどうかの4通りで場合わけ
    if (kind(move.subject()) == KING) {
        if (move.capture() == EMPTY) { //玉を動かし、駒は取らない手
            Color move_color = pieceToColor(move.subject());
            int32_t king_sq_num = (move_color == BLACK ? bk_sq : wk_sqr);
            const auto& list = (move_color == BLACK ? ee_.piece_state_list[0] : ee_.piece_state_list[1]);
            //計算し直す
            score_state_[KKP][RAW] = 0;
            score_state_[KKP][TURN_BONUS] = 0;
            score_state_[move_color + 1][RAW] = 0;
            score_state_[move_color + 1][TURN_BONUS] = 0;
#ifdef USE_SIMD
            for (int j = 0; j < PIECE_STATE_LIST_SIZE; j++) {
                score_state_[KKP] += eval_params_.kkp[bk_sq][wk_sq][ee_.piece_state_list[0][j]];

                __m256i sum256 = _mm256_setzero_si256();

                int i = 0;
                for (; i + 8 < j; i += 8) {
                    //listから8要素ロード
                    __m256i indexes = _mm256_load_si256(reinterpret_cast<const __m256i*>(&list[i]));

                    //indexesをオフセットとしてkppから8要素ギャザー
                    __m256i gathered = _mm256_i32gather_epi32(reinterpret_cast<const int*>(&(eval_params_.kpp[king_sq_num][list[j]])), indexes, 4);

                    //16bitのペアになっているので32bitに拡張
                    //まずは下位128bitについて
                    __m256i low = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gathered, 0));

                    //足し合わせる
                    sum256 = _mm256_add_epi32(sum256, low);

                    //16bitのペアになっているので32bitに拡張
                    __m256i high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gathered, 1));

                    //足し合わせる
                    sum256 = _mm256_add_epi32(sum256, high);
                }
                for (; i + 4 < j; i += 4) {
                    // list[0], [1]から 4 要素ロードする
                    __m128i indexes = _mm_load_si128(reinterpret_cast<const __m128i*>(&list[i]));

                    //indexesをオフセットとしてkppから4要素ギャザー
                    __m128i gathered = _mm_i32gather_epi32(reinterpret_cast<const int*>(&eval_params_.kpp[king_sq_num][list[j]]), indexes, 4);

                    // 16 ビット整数→32 ビット整数に変換する
                    __m256i expanded = _mm256_cvtepi16_epi32(gathered);

                    //足し合わせる
                    sum256 = _mm256_add_epi32(sum256, expanded);
                }
                for (; i < j; i++) {
                    score_state_[move_color + 1] += eval_params_.kpp[king_sq_num][list[j]][list[i]];
                }
                //上位 128 ビットと下位 128 ビットを独立して 8 バイトシフトしたものを足し合わせる
                sum256 = _mm256_add_epi32(sum256, _mm256_srli_si256(sum256, 8));

                // diffp0 の上位 128 ビットと下位 128 ビットを足しあわせて diffp0_128 に代入する
                __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(sum256, 0), _mm256_extracti128_si256(sum256, 1));

                // diffp0_128 の下位 64 ビットを diff.p[1]にストアする
                std::array<int32_t, ColorNum> sum{};
                _mm_storel_epi64(reinterpret_cast<__m128i*>(sum.data()), sum128);

                score_state_[move_color + 1] += sum;
            }
#else
            for (int i = 0; i < PIECE_STATE_LIST_SIZE; i++) {
                score_state_[KKP] += eval_params_.kkp[bk_sq][wk_sq][ee_.piece_state_list[0][i]];

                for (int j = i + 1; j < PIECE_STATE_LIST_SIZE; j++) {
                    score_state_[move_color + 1] += eval_params_.kpp[king_sq_num][list[i]][list[j]];
                }
            }
#endif
        } else { //玉を動かし、駒を取る手
            //頭が悪いので全計算で勘弁
            Color move_color = pieceToColor(move.subject());
            PieceState captured = pieceState(move.capture(), move.to(), ~move_color);
            PieceState add = pieceState(kind(move.capture()), hand_[move_color].num(kind(move.capture())), move_color);
            //リストを更新
            updatePieceStateList(captured, add);

            //全計算
#ifdef USE_SIMD
            initKKP_KPPScoreBySIMD();
#else
            initPieceScore();
#endif // USE_SIMD
        }
    } else if (move.isDrop()) { //駒を打つ手
        PieceState dropped_from = pieceState(kind(move.subject()), hand_[pieceToColor(move.subject())].num(move.subject()) + 1, pieceToColor(move.subject()));
        PieceState dropped_to = pieceState(move.subject(), move.to(), pieceToColor(move.subject()));

        //持ち駒の分を削除
#ifdef USE_SIMD
        changeOnePSBySIMD(dropped_from, -1);
#else
        changeOnePS(dropped_from, -1);
#endif // USE_SIMD

        //listを更新
        updatePieceStateList(dropped_from, dropped_to);

        //打った分を追加
#ifdef USE_SIMD
        changeOnePSBySIMD(dropped_to, 1);
#else
        changeOnePS(dropped_to, 1);
#endif // USE_SIMD

    } else {
        if (move.capture() == EMPTY) {
            PieceState removed_from = pieceState(move.subject(), move.from(), pieceToColor(move.subject()));
            PieceState added_to = pieceState(move.isPromote() ? promote(move.subject()) : move.subject(), move.to(), pieceToColor(move.subject()));

            //移動前を引く
#ifdef USE_SIMD
            changeOnePSBySIMD(removed_from, -1);
#else
            changeOnePS(removed_from, -1);
#endif // USE_SIMD

            //listを更新
            updatePieceStateList(removed_from, added_to);

            //移動後を足す
#ifdef USE_SIMD
            changeOnePSBySIMD(added_to, 1);
#else
            changeOnePS(added_to, 1);
#endif // USE_SIMD
        } else {
            //2つPが消えて2つPが増える.厄介
            PieceState removed1 = pieceState(move.subject(), move.from(), pieceToColor(move.subject()));
            PieceState removed2 = pieceState(move.capture(), move.to(), pieceToColor(move.capture()));
            PieceState added1 = pieceState(move.isPromote() ? promote(move.subject()) : move.subject(), move.to(), pieceToColor(move.subject()));

#ifdef DEBUG
            if (kind(move.capture()) == KING) {
                move.print();
                print();
                printHistory();
                undo();
                print();
                assert(false);
        }
#endif // DEBUG

            PieceState added2 = pieceState(kind(move.capture()), hand_[pieceToColor(move.subject())].num(move.capture()), pieceToColor(move.subject()));

            //移動前の駒,取られた駒の分を引く
#ifdef USE_SIMD
            changeOnePSBySIMD(removed1, -1);
            changeOnePSBySIMD(removed2, -1);
#else
            changeOnePS(removed1, -1);
            changeOnePS(removed2, -1);
#endif // USE_SIMD
            //KPPTの方では同じものを2回引いているので補正する
            score_state_[KPP_BLACK] += eval_params_.kpp[bk_sq][removed1][removed2];
            score_state_[KPP_WHITE] += eval_params_.kpp[wk_sqr][invPieceState(removed1)][invPieceState(removed2)];

            //listを更新
            updatePieceStateList(removed1, added1);
            updatePieceStateList(removed2, added2);

#ifdef USE_SIMD
            changeOnePSBySIMD(added1, 1);
            changeOnePSBySIMD(added2, 1);
#else
            changeOnePS(added1, 1);
            changeOnePS(added2, 1);
#endif // USE_SIMD
            //KPPTの方では同じものを2回足しているので補正する
            score_state_[KPP_BLACK] -= eval_params_.kpp[bk_sq][added1][added2];
            score_state_[KPP_WHITE] -= eval_params_.kpp[wk_sqr][invPieceState(added1)][invPieceState(added2)];
    }
}

#ifdef DEBUG
    auto copy = score_state_;
#ifdef USE_SIMD
    initKKP_KPPScoreBySIMD();
#else
    initPieceScore();
#endif // USE_SIMD
    checkScoreState(copy);
#endif
}

inline void Position::checkScoreState(std::array<int32_t, ColorNum> copy[3]) {
    for (int i = 0; i < KKP_KPP_END; i++) {
        for (int j = 0; j < RAW_TURN_END; j++) {
            if (copy[i][j] != score_state_[i][j]) {
                std::cout << "copy[" << i << "][" << j << "] = " << copy[i][j] << "!= score_state_[" << i << "][" << j << "] = " << score_state_[i][j] << std::endl;
                print();
                for (int k = 0; k < PIECE_STATE_LIST_SIZE; k++) {
                    std::cout << ee_.piece_state_list[0][k] << " " << ee_.piece_state_list[1][k] << std::endl;
                }
                assert(false);
            }
        }
    }
}

#endif