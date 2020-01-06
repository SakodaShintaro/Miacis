#ifndef MIACIS_POSITION_HPP
#define MIACIS_POSITION_HPP

#include"square.hpp"
#include"piece.hpp"
#include"move.hpp"
#include"../types.hpp"
#include<random>
#include<cstdint>
#include<unordered_map>

class Position {
public:
    //コンストラクタ
    Position();

    //初期化
    void init();

    //内部の状態等を表示する関数
    void print() const;

    //一手進める・戻す関数
    void doMove(const Move move);
    void undo();
    void doNullMove();
    void undoNullMove();

    //合法性に関する関数
    bool isLegalMove(const Move move) const;

    //特徴量作成
    std::vector<float> makeFeature(int64_t data_augmentation) const;

    //合法手生成
    std::vector<Move> generateAllMoves() const;

    //終了判定
    bool isFinish(float& score) const;

    //ハッシュ
    static void initHashSeed();

    //stringとの相互変換
    std::string toStr() const;
    void fromStr(const std::string& str);

    //getter
    Move lastMove() const { return (kifu_.empty() ? NULL_MOVE : kifu_.back()); }
    uint32_t turnNumber() const { return turn_number_; }
    Color color() const { return color_; }
    int64_t hashValue() const { return hash_value_; }
    Piece on(const Square sq) const { return board_[sq]; }

    //現状は90度ごとの回転のみに対応
    //原理的にはそれに左右反転まで含めた8通りに拡張できるがまだ未実装
    static constexpr int64_t DATA_AUGMENTATION_PATTERN_NUM = 4;
private:
    //--------------------
    //    内部メソッド
    //--------------------
    //ハッシュ値の初期化
    void initHashValue();

    //------------------
    //    クラス変数
    //------------------
    //ハッシュの各駒・位置に対する決められた値
    static int64_t HashSeed[PieceNum][SquareNum];

    //------------------------
    //    インスタンス変数
    //------------------------
    //手番
    Color color_;

    //盤面
    Piece board_[SquareNum];

    //盤面の履歴をスタックで管理
    std::vector<std::vector<Piece>> stack_;

    //手数
    uint32_t turn_number_;

    //現局面までの指し手履歴
    std::vector<Move> kifu_;

    //現局面のハッシュ値
    int64_t hash_value_;

    //ハッシュ値の履歴
    std::vector<int64_t> hash_values_;
};

#endif //MIACIS_POSITION_HPP