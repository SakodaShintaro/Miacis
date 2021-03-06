#ifndef MIACIS_OTHELLO_POSITION_HPP
#define MIACIS_OTHELLO_POSITION_HPP

#include "../types.hpp"
#include "move.hpp"
#include "piece.hpp"
#include "square.hpp"
#include <cstdint>
#include <random>
#include <unordered_map>

namespace Othello {

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

    //詰み探索を飛ばしても良いか
    //将棋で王手がかかってないときは枝刈りしたいのでこれが必要
    //オセロでは特に飛ばすべき局面はないと思われる
    static bool canSkipMateSearch() { return false; }

    //特徴量作成
    std::vector<float> makeFeature() const;

    //合法手生成
    std::vector<Move> generateAllMoves() const;

    //終了判定
    bool isFinish(float& score, bool check_repeat = true) const;

    //ハッシュ
    static void initHashSeed();

    //stringとの相互変換
    std::string toStr() const;
    void fromStr(const std::string& str);

    //getter
    Move lastMove() const { return (kifu_.empty() ? NULL_MOVE : kifu_.back()); }
    uint32_t turnNumber() const { return turn_number_; }
    Color color() const { return color_; }
    uint64_t hashValue() const { return hash_value_; }

    //回転、左右反転の8通りに拡張可能
    static constexpr int64_t DATA_AUGMENTATION_PATTERN_NUM = 8;
    static std::string augmentStr(const std::string& str, int64_t augmentation);

private:
    //--------------------
    //    内部メソッド
    //--------------------
    //ハッシュ値の初期化
    void initHashValue();

    static std::string augmentStrRotate(const std::string& str, int64_t augmentation);
    static std::string augmentStrMirror(const std::string& str, int64_t augmentation);

    //------------------
    //    クラス変数
    //------------------
    //ハッシュの各駒・位置に対する決められた値
    static uint64_t HashSeed[PieceNum][SquareNum];

    //------------------------
    //    インスタンス変数
    //------------------------
    //手番
    Color color_;

    //盤面
    std::array<Piece, SquareNum> board_;

    //盤面の履歴をスタックで管理
    std::vector<std::array<Piece, SquareNum>> board_history_;

    //手数
    uint32_t turn_number_;

    //現局面までの指し手履歴
    std::vector<Move> kifu_;

    //現局面のハッシュ値
    uint64_t hash_value_;

    //ハッシュ値の履歴
    std::vector<uint64_t> hash_values_;
};

} // namespace Othello

#endif //MIACIS_OTHELLO_POSITION_HPP