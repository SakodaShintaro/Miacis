#ifndef POSITION_HPP
#define POSITION_HPP

#include"move.hpp"
#include"hand.hpp"
#include"bitboard.hpp"

class Position {
public:
    //コンストラクタ
    Position();

    //初期化
    void init();

    //内部の状態等を表示する関数
    void print() const;

    //一手進める・戻す関数
    void doMove(Move move);
    void undo();

    //合法性に関する関数
    bool isLegalMove(Move move) const;
    bool isLastMoveDropPawn() const;
    bool canWinDeclare() const;

    //詰み探索中に枝刈りして良いかを判定
    //王手がかかっていないときは枝刈りして良いだろう
    bool canSkipMateSearch() const { return !is_checked_; }

    //この局面が詰み、千日手等で終わっているか確認する関数
    //終わっている場合は手番側から見た点数を引数に書き込んでtrueを返す
    bool isFinish(float& score);

    //特徴量作成
    std::vector<float> makeFeature() const;

    //toとfromしか与えられない状態から完全なMoveに変換する関数
    Move transformValidMove(Move move);

    //合法手生成
    std::vector<Move> generateAllMoves();

    //sfenの入出力
    void fromStr(const std::string& sfen);
    std::string toStr() const;

    //ハッシュ
    static void initHashSeed();

    //getter
    uint32_t turnNumber() const { return turn_number_; }
    Color color() const { return color_; }
    int64_t hashValue() const { return hash_value_; }
    Piece on(const Square sq) const { return board_[sq]; }
    bool isChecked() { return is_checked_; }

    //左右反転のみに対応。将棋はこれ以上の拡張はできない
    static constexpr int64_t DATA_AUGMENTATION_PATTERN_NUM = 2;
    static std::string augmentStr(const std::string& str, int64_t augmentation);
private:
    //--------------------
    //    内部メソッド
    //--------------------
    //合法手生成で用いる関数
    bool canPromote(Move move) const;
    bool canDropPawn(Square to) const;
    void pushMove(Move move, std::vector<Move>& move_buf) const;
    void generateNormalMoves(std::vector<Move>& move_buf) const;
    void generateEvasionMoves(std::vector<Move>& move_buf) const;
    void generateDropMoves(const Bitboard& to_bb, std::vector<Move>& move_buf) const;

    //王手,利き関連
    Bitboard attackersTo(Color c, Square sq) const;
    Bitboard attackersTo(Color c, Square sq, const Bitboard& occupied) const;
    inline bool isThereControl(const Color c, const Square sq) const { return (bool)attackersTo(c, sq); }
    inline bool isLastMoveCheck();
    Bitboard computePinners() const;

    //ハッシュ値の初期化
    void initHashValue();

    //千日手の判定
    bool isRepeating(float& score) const;

    //emptyの条件分けをいちいち書かないための補助関数
    Move lastMove() const { return (kifu_.empty() ? NULL_MOVE : kifu_.back()); }

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

    //玉の位置
    Square king_sq_[ColorNum];

    //現局面までの指し手履歴
    std::vector<Move> kifu_;

    //現局面の合法手
    std::vector<Move> moves_;

    bool already_generated_moves_;

    //現局面のハッシュ値
    int64_t hash_value_, board_hash_, hand_hash_;
    bool is_checked_;

    struct StateInfo {
        //千日手判定用に必要な情報
        int64_t board_hash, hand_hash;
        Hand hand[ColorNum];
        bool is_checked;

        StateInfo(Position& pos) :
            board_hash(pos.board_hash_), hand_hash(pos.hand_hash_), is_checked(pos.is_checked_) {
            hand[BLACK] = pos.hand_[BLACK];
            hand[WHITE] = pos.hand_[WHITE];
        }
    };
    std::vector<StateInfo> stack_;

    //Bitboard類
    Bitboard occupied_all_;
    Bitboard occupied_bb_[ColorNum];
    Bitboard pieces_bb_[PieceNum];
};

#endif