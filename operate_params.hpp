﻿#ifndef OPERATE_PARAMS_HPP
#define OPERATE_PARAMS_HPP

#ifdef USE_CATEGORICAL
inline int32_t valueToIndex(double value) {
    return std::min((int32_t)((value - MIN_SCORE) / VALUE_WIDTH), BIN_SIZE - 1);
}

inline ValueType onehotDist(double value) {
    //valueForBlackのところだけ1.0, 他は0.0とした分布を返す
    //valueForBlack / (1.0 / BIN_SIZE) = valueForBlack * BIN_SIZE のところだけ1.0
    //valueForBlack = 1.0だとちょうどBIN_SIZEになってしまうからminを取る
    int32_t index = std::min((int32_t)(value * BIN_SIZE - 0.01), BIN_SIZE - 1);
    ValueType result;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        result[i] = (CalcType)(i == index ? 1.0 : 0.0);
    }
    return result;
}

inline double expOfValueDist(ValueType dist) {
    double exp = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        //i番目の要素が示す値はMIN_SCORE + (i + 0.5) * VALUE_WIDTH
        exp += (MIN_SCORE + (i + 0.5) * VALUE_WIDTH) * dist[i];
    }
    return exp;
}

#endif

#endif