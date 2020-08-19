#ifndef MIACIS_INCLUDE_SWITCH_HPP
#define MIACIS_INCLUDE_SWITCH_HPP

//コンパイルしたいゲームに合わせて切り替える

//将棋をコンパイルする場合
#ifdef SHOGI
#include "shogi/position.hpp"
using namespace Shogi;

//オセロをコンパイルする場合
#elif defined(OTHELLO)
#include "othello/position.hpp"

#endif

#endif //MIACIS_INCLUDE_SWITCH_HPP