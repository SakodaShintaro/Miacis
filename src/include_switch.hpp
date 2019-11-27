#ifndef MIACIS_INCLUDE_SWITCH_HPP
#define MIACIS_INCLUDE_SWITCH_HPP

#define SHOGI

//コンパイルしたいゲームに合わせて切り替える
#ifdef SHOGI

#include "shogi/position.hpp"
#include "shogi/const.hpp"

#endif

#endif //MIACIS_INCLUDE_SWITCH_HPP