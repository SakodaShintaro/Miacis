#ifndef MIACIS_INCLUDE_SWITCH_HPP
#define MIACIS_INCLUDE_SWITCH_HPP

//コンパイルしたいゲームに合わせて切り替える

//将棋をコンパイルする場合
#ifdef SHOGI
#include "shogi/position.hpp"
using namespace Shogi;

//囲碁をコンパイルする場合
#elif defined(GO)
#include "go/position.hpp"
using namespace Go;

#endif

#endif //MIACIS_INCLUDE_SWITCH_HPP