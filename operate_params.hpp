#pragma once

#ifndef OPERATE_PARAMS_HPP
#define OPERATE_PARAMS_HPP

#include"eval_params.hpp"

#ifdef USE_NN

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

#else
inline void regularize(EvalParams<DefaultEvalType>& eval_params) {
    for (int k = 0; k < SqNum; k++) {
        for (int p1 = 0; p1 < PieceStateNum; p1++) {
            for (int p2 = 0; p2 < PieceStateNum; p2++) {
                if (p1 == p2) {    //kpptのp1 == p2については0が入るようにする
                    eval_params.kpp[k][p1][p2][0] = eval_params.kpp[k][p1][p2][1] = 0;
                } else {    //kppt[k][p1][p2] == kppt[k][p2][p1]となるようにする
                    auto mid0 = (DefaultEvalType)((eval_params.kpp[k][p1][p2][0] + eval_params.kpp[k][p2][p1][0]) / 2);
                    auto mid1 = (DefaultEvalType)((eval_params.kpp[k][p1][p2][1] + eval_params.kpp[k][p2][p1][1]) / 2);
                    eval_params.kpp[k][p1][p2][0] = eval_params.kpp[k][p2][p1][0] = mid0;
                    eval_params.kpp[k][p1][p2][1] = eval_params.kpp[k][p2][p1][1] = mid1;
                }
            }
        }
    }
}

inline void checkSymmetry(EvalParams<DefaultEvalType>& eval_params) {
    for (int k = 0; k < SqNum; k++) {
        for (int p1 = 0; p1 < PieceStateNum; p1++) {
            for (int p2 = 0; p2 < PieceStateNum; p2++) {
                for (int t = 0; t < 2; t++) {
                    if (p1 == p2) {    //kpptのp1 == p2については0が入るようにする
                        assert(eval_params.kpp[k][p1][p2][t] == 0);
                    } else {    //kppt[k][p1][p2] == kppt[k][p2][p1]となるようにする
                        assert(eval_params.kpp[k][p1][p2][t] == eval_params.kpp[k][p2][p1][t]);
                    }
                }
            }
        }
    }
}

inline void readAnotherFormatFile(std::string path, std::string kkp_suffix, std::string kpp_suffix) {
    constexpr uint64_t ExPieceStateNum = 1548;
    using EvalType = std::array<DefaultEvalType, ColorNum>;
    std::vector<EvalType> kkp_tmp(SqNum * SqNum * ExPieceStateNum);
    std::vector<EvalType> kpp_tmp(SqNum * ExPieceStateNum * ExPieceStateNum);
    std::string kkp_file_name = path + kkp_suffix;
    std::ifstream kkp_ifs(kkp_file_name, std::ios::binary);
    if (kkp_ifs.fail()) {
        std::cerr << kkp_file_name + " cannot open." << std::endl;
        eval_params->clear();
        return;
    }
    kkp_ifs.read(reinterpret_cast<char*>(kkp_tmp.data()), sizeof(EvalType) * kkp_tmp.size());
    kkp_ifs.close();

    std::string kpp_file_name = path + kpp_suffix;
    std::ifstream kpp_ifs(kpp_file_name, std::ios::binary);
    if (kpp_ifs.fail()) {
        std::cerr << kpp_file_name + " cannot open." << std::endl;
        eval_params->clear();
        return;
    }
    kpp_ifs.read(reinterpret_cast<char*>(kpp_tmp.data()), sizeof(EvalType) * kpp_tmp.size());
    kpp_ifs.close();

    auto changePieceState = [&](int64_t p) {
        if (p >= white_hand_rook) {
            if (black_rook <= p && p < black_horse) {
                p += 81 * 2;
            } else if (black_horse <= p && p < black_dragon) {
                p -= 81 * 2;
            }
            return p + 14;
        } else if (p >= black_hand_rook) {
            return p + 13;
        } else if (p >= white_hand_bishop) {
            return p + 12;
        } else if (p >= black_hand_bishop) {
            return p + 11;
        } else if (p >= white_hand_gold) {
            return p + 10;
        } else if (p >= black_hand_gold) {
            return p + 9;
        } else if (p >= white_hand_silver) {
            return p + 8;
        } else if (p >= black_hand_silver) {
            return p + 7;
        } else if (p >= white_hand_knight) {
            return p + 6;
        } else if (p >= black_hand_knight) {
            return p + 5;
        } else if (p >= white_hand_lance) {
            return p + 4;
        } else if (p >= black_hand_lance) {
            return p + 3;
        } else if (p >= white_hand_pawn) {
            return p + 2;
        } else if (p >= black_hand_pawn) {
            return p + 1;
        }
        assert(false);
        return p;
    };

    for (int64_t k1 = 0; k1 < SqNum; k1++) {
        for (int64_t p1 = 0; p1 < PieceStateNum; p1++) {
            int64_t ex_p1 = changePieceState(p1);
            for (int64_t k2 = 0; k2 < SqNum; k2++) {
                auto key = k1 * SqNum * ExPieceStateNum + k2 * ExPieceStateNum + ex_p1;
                eval_params->kkp[k1][k2][p1] = kkp_tmp[key];
            }
            for (int64_t p2 = 0; p2 < PieceStateNum; p2++) {
                int64_t ex_p2 = changePieceState(p2);
                auto key = k1 * ExPieceStateNum * ExPieceStateNum + ex_p1 * ExPieceStateNum + ex_p2;
                eval_params->kpp[k1][p1][p2] = kpp_tmp[key];
            }
        }
    }
}

#endif

#endif