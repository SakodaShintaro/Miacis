# C++版Miacisと同じように定数の定義をする
import cshogi

BOARD_SIZE = 9
INPUT_CHANNEL_NUM = 42
SQUARE_NUM = BOARD_SIZE ** 2
POLICY_CHANNEL_NUM = 27

MIN_SCORE = -1.0
MAX_SCORE = 1.0

BIN_SIZE = 51
VALUE_WIDTH = (MAX_SCORE - MIN_SCORE) / BIN_SIZE

PIECE_LIST = [cshogi.BPAWN,
              cshogi.BLANCE,
              cshogi.BKNIGHT,
              cshogi.BSILVER,
              cshogi.BGOLD,
              cshogi.BBISHOP,
              cshogi.BROOK,
              cshogi.BKING,
              cshogi.BPROM_PAWN,
              cshogi.BPROM_LANCE,
              cshogi.BPROM_KNIGHT,
              cshogi.BPROM_SILVER,
              cshogi.BPROM_BISHOP,
              cshogi.BPROM_ROOK,
              cshogi.WPAWN,
              cshogi.WLANCE,
              cshogi.WKNIGHT,
              cshogi.WSILVER,
              cshogi.WGOLD,
              cshogi.WBISHOP,
              cshogi.WROOK,
              cshogi.WKING,
              cshogi.WPROM_PAWN,
              cshogi.WPROM_LANCE,
              cshogi.WPROM_KNIGHT,
              cshogi.WPROM_SILVER,
              cshogi.WPROM_BISHOP,
              cshogi.WPROM_ROOK]


# 移動の定数
MOVE_DIRECTION = [
    UP,
    UP_LEFT,
    UP_RIGHT,
    LEFT,
    RIGHT,
    DOWN,
    DOWN_LEFT,
    DOWN_RIGHT,
    UP2_LEFT,
    UP2_RIGHT,
    UP_PROMOTE,
    UP_LEFT_PROMOTE,
    UP_RIGHT_PROMOTE,
    LEFT_PROMOTE,
    RIGHT_PROMOTE,
    DOWN_PROMOTE,
    DOWN_LEFT_PROMOTE,
    DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE,
    UP2_RIGHT_PROMOTE] = range(20)

# 成り変換テーブル
MOVE_DIRECTION_PROMOTED = [
    UP_PROMOTE,
    UP_LEFT_PROMOTE,
    UP_RIGHT_PROMOTE,
    LEFT_PROMOTE,
    RIGHT_PROMOTE,
    DOWN_PROMOTE,
    DOWN_LEFT_PROMOTE,
    DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE,
    UP2_RIGHT_PROMOTE]

# 指し手を表すラベルの数
MOVE_DIRECTION_LABEL_NUM = len(MOVE_DIRECTION) + 7  # 7は持ち駒の種類
