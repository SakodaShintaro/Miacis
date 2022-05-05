# C++版Miacisと同じように定数の定義をする
import cshogi

BOARD_SIZE = 9
INPUT_CHANNEL_NUM = 42
SQUARE_NUM = BOARD_SIZE ** 2
POLICY_CHANNEL_NUM = 27

MIN_SCORE = -1
MAX_SCORE = 1

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

# rotate 180degree
SQUARES_R180 = [
    cshogi.I1, cshogi.I2, cshogi.I3, cshogi.I4, cshogi.I5, cshogi.I6, cshogi.I7, cshogi.I8, cshogi.I9,
    cshogi.H1, cshogi.H2, cshogi.H3, cshogi.H4, cshogi.H5, cshogi.H6, cshogi.H7, cshogi.H8, cshogi.H9,
    cshogi.G1, cshogi.G2, cshogi.G3, cshogi.G4, cshogi.G5, cshogi.G6, cshogi.G7, cshogi.G8, cshogi.G9,
    cshogi.F1, cshogi.F2, cshogi.F3, cshogi.F4, cshogi.F5, cshogi.F6, cshogi.F7, cshogi.F8, cshogi.F9,
    cshogi.E1, cshogi.E2, cshogi.E3, cshogi.E4, cshogi.E5, cshogi.E6, cshogi.E7, cshogi.E8, cshogi.E9,
    cshogi.D1, cshogi.D2, cshogi.D3, cshogi.D4, cshogi.D5, cshogi.D6, cshogi.D7, cshogi.D8, cshogi.D9,
    cshogi.C1, cshogi.C2, cshogi.C3, cshogi.C4, cshogi.C5, cshogi.C6, cshogi.C7, cshogi.C8, cshogi.C9,
    cshogi.B1, cshogi.B2, cshogi.B3, cshogi.B4, cshogi.B5, cshogi.B6, cshogi.B7, cshogi.B8, cshogi.B9,
    cshogi.A1, cshogi.A2, cshogi.A3, cshogi.A4, cshogi.A5, cshogi.A6, cshogi.A7, cshogi.A8, cshogi.A9,
]
