#ifndef POSITION_HPP
#define POSITION_HPP

#include"square.hpp"
#include"piece.hpp"
#include"move.hpp"
#include"hand.hpp"
#include"eval_elements.hpp"
#include"types.hpp"
#include"bitboard.hpp"
#include"eval_params.hpp"
#include<random>
#include<cstdint>
#include<unordered_map>

constexpr int MAX_MOVE_LIST_SIZE = 593;

class Position {
public:
    //コンストラクタ
    Position(const EvalParams<DefaultEvalType>& eval_params);

    //初期化
    void init();

    //内部の状態等を表示する関数
    void print() const;
    void printHistory() const;
    void printForDebug() const;

    //一手進める・戻す関数
    void doMove(const Move move);
    void undo();
    void doNullMove();
    void undoNullMove();
    void doMoveWithoutCalcDiff(const Move move);

    //合法性に関する関数
    bool isLegalMove(const Move move) const;
    bool canDropPawn(const Square to) const;
    bool isPsuedoLegalMove(const Move move) const;

    //王手,利き関連
    inline bool isKingChecked() const { return isChecked_; }
    Bitboard attackersTo(const Color c, const Square sq) const;
    Bitboard attackersTo(const Color c, const Square sq, const Bitboard& occupied) const;
    inline bool isThereControl(const Color c, const Square sq) const { return attackersTo(c, sq); }
    inline bool isLastMoveCheck();
    void computePinners();

    //千日手の判定
    bool isRepeating(Score& score) const;

    //評価値計算
    void calcScoreDiff();
#ifdef USE_NN
    Vec makeOutput() const;
    std::vector<CalcType> policy();
    std::vector<CalcType> maskedPolicy();
    Score score();
    Score scoreForTurn();
    double valueForTurn();

    void resetCalc();
#ifdef USE_CATEGORICAL
    std::array<CalcType, BIN_SIZE> valueDist();
#endif
#else
    Score SEE();
    Score SEEwithoutDoMove();
    Score score() const;
    Score scoreForTurn() const;
#endif

    //特徴量作成
#ifdef USE_NN
    Features makeFeature() const;
#endif

    //toとfromしか与えられない状態から完全なMoveに変換する関数
    Move transformValidMove(const Move move);

    //合法手生成
    std::vector<Move> generateAllMoves() const;
    void generateCheckMoves(Move*& move_ptr) const;
    void generateEvasionMoves(Move*& move_ptr) const;
    void generateCaptureMoves(Move*& move_ptr) const;
    void generateNonCaptureMoves(Move*& move_ptr) const;
    void generateRecaptureMovesTo(const Square to, Move*& move_ptr) const;

    //sfenの入出力
    void loadSFEN(std::string sfen);
    std::string toSFEN();

    //ハッシュ
    static void initHashSeed();

    //getter
    Move lastMove() const { return (kifu_.empty() ? NULL_MOVE : kifu_.back()); }
    uint32_t turn_number() const { return turn_number_; }
    Color color() const { return color_; }
    int64_t hash_value() const { return hash_value_; }
    Piece on(const Square sq) const { return board_[sq]; }
    Features features() { return ee_; }
    const EvalParams<DefaultEvalType>& evalParams() { return eval_params_; }
    bool isChecked() { return isChecked_; }
private:
    //--------------------
	//    内部メソッド
    //--------------------
    //合法手生成で用いる関数
    bool canPromote(const Move move) const;
    void pushMove(const Move move, Move*& move_ptr) const;
    void generateDropMoves(const Bitboard& to_bb, Move*& move_ptr) const;

    //評価値計算
    void initScore();
    void initPieceScore();
    void initKKP_KPPScore();
    void initKKP_KPPScoreBySIMD();
    void changeOnePS(PieceState ps, int c);
    void changeOnePSBySIMD(PieceState ps, int c);

    //ハッシュ値の初期化
    void initHashValue();

#ifndef USE_NN
    //初期化
    void initFeature();

    void updatePieceStateList(PieceState before, PieceState after);
#endif

    inline void checkScoreState(std::array<int32_t, ColorNum> copy[3]);

    enum {
        KKP, KPP_BLACK, KPP_WHITE, KKP_KPP_END,
        RAW = 0, TURN_BONUS, RAW_TURN_END
    };

    //------------------
    //    クラス変数
    //------------------
    //ハッシュの各駒・位置に対する決められた値
    static int64_t HashSeed[PieceNum][SquareNum];
    static int64_t HandHashSeed[ColorNum][PieceNum][19];

    //------------------------
    //    インスタンス変数
    //------------------------
    //手番
    Color color_;

    //盤面
    Piece board_[SquareNum];

    //持ち駒
    Hand hand_[ColorNum];

    //手数
    uint32_t turn_number_;

    //駒割による評価値
    Score piece_score_;

    //KKP,KPPによる評価値
    std::array<int32_t, ColorNum> score_state_[3];

    //玉の位置
    Square king_sq_[ColorNum];

    //現局面までの指し手履歴
    std::vector<Move> kifu_;

    //現局面のハッシュ値
    int64_t hash_value_, board_hash_, hand_hash_;
    bool isChecked_;

    //あるpiece_stateがlistの中の何番目にあるか
    //-1(存在しない)または0~37となる
    int16_t piece_state_to_index_[PieceStateNum];

    struct StateInfo {
        //千日手判定用に必要な情報
        int64_t board_hash, hand_hash;
        Hand hand[ColorNum];
        bool isChecked;

        //undoでコピーするための情報
        Score piece_score;
        std::array<int32_t, ColorNum> score_state[3];
        Features features;

        Bitboard pinners;

        StateInfo(Position& pos) :
            board_hash(pos.board_hash_), hand_hash(pos.hand_hash_), isChecked(pos.isChecked_), piece_score(pos.piece_score_), features(pos.ee_), pinners(pos.pinners_) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 2; j++) {
                    score_state[i][j] = pos.score_state_[i][j];
                }
            }
            hand[BLACK] = pos.hand_[BLACK];
            hand[WHITE] = pos.hand_[WHITE];
        }
    };
    std::vector<StateInfo> stack_;

    //Bitboard類
    Bitboard occupied_all_;
    Bitboard occupied_bb_[ColorNum];
    Bitboard pieces_bb_[PieceNum];
    Bitboard pinners_;

    //特徴量
    Features ee_;

#ifdef USE_NN
    Vec output_;
    bool already_calc_;
#endif

    //評価パラメータへの参照
    const EvalParams<DefaultEvalType>& eval_params_;
};

#endif