import math


# 勝率からelo_rateを計算する関数
def calc_elo_rate(winning_rate):
    assert 0 <= winning_rate <= 1
    if winning_rate == 1.0:
        return 10000.0
    elif winning_rate == 0.0:
        return -10000.0
    else:
        return 400 * math.log10(winning_rate / (1 - winning_rate))
