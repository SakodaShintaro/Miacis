#ifndef OPERATE_PARAMS_HPP
#define OPERATE_PARAMS_HPP

#ifdef USE_CATEGORICAL
inline std::array<CalcType, BIN_SIZE> onehotDist(double value) {
    //valueForBlackのところだけ1.0, 他は0.0とした分布を返す
    //valueForBlack / (1.0 / BIN_SIZE) = valueForBlack * BIN_SIZE のところだけ1.0
    //valueForBlack = 1.0だとちょうどBIN_SIZEになってしまうからminを取る
    int32_t index = std::min((int32_t)(value * BIN_SIZE - 0.01), BIN_SIZE - 1);
    std::array<CalcType, BIN_SIZE> result;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        result[i] = (CalcType)(i == index ? 1.0 : 0.0);
    }
    return result;
}

#endif

#endif