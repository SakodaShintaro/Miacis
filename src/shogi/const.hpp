#ifndef MIACIS_CONST_HPP
#define MIACIS_CONST_HPP

#include"piece.hpp"

//行動の次元数
constexpr int64_t POLICY_CHANNEL_NUM = 27;
constexpr int64_t POLICY_DIM = SQUARE_NUM * POLICY_CHANNEL_NUM;

#endif //MIACIS_CONST_HPP