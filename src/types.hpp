﻿#ifndef MIACIS_TYPES_HPP
#define MIACIS_TYPES_HPP

#include <array>

enum Color {
    BLACK,
    WHITE,
    ColorNum,
};
inline static Color operator~(Color c) { return (c == BLACK) ? WHITE : BLACK; }

//Valueの活性化関数をsigmoidにするかtanhにするか
//これをオンにするとValueの範囲が[0, 1]になり勝率に対応するようになる
//強化学習の報酬としては[-1, 1]とした方が自然に思えるので基本はオフ
//#define USE_SIGMOID

constexpr float MAX_SCORE = 1.0;

#ifdef USE_SIGMOID
constexpr float MIN_SCORE = 0.0;
#else
constexpr float MIN_SCORE = -MAX_SCORE;
#endif

template<class T, size_t SIZE> inline std::array<T, SIZE>& operator+=(std::array<T, SIZE>& lhs, std::array<T, SIZE> rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] += rhs[i];
    }
    return lhs;
}

template<class T, size_t SIZE> inline std::array<T, SIZE> operator+(std::array<T, SIZE> lhs, std::array<T, SIZE> rhs) {
    return lhs += rhs;
}

template<class T, size_t SIZE> inline std::array<T, SIZE>& operator-=(std::array<T, SIZE>& lhs, std::array<T, SIZE> rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] -= rhs[i];
    }
    return lhs;
}

template<class T, size_t SIZE> inline std::array<T, SIZE> operator-(std::array<T, SIZE> lhs, std::array<T, SIZE> rhs) {
    return lhs -= rhs;
}

template<class T, class U, size_t SIZE> inline std::array<T, SIZE>& operator/=(std::array<T, SIZE>& lhs, U rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] /= rhs;
    }
    return lhs;
}

template<class T, class U, size_t SIZE> inline std::array<T, SIZE> operator/(std::array<T, SIZE> lhs, U rhs) {
    return lhs /= rhs;
}

template<class T, class U, size_t SIZE> inline std::array<T, SIZE> operator*(U lhs, std::array<T, SIZE> rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        rhs[i] *= lhs;
    }
    return rhs;
}

#endif //MIACIS_TYPES_HPP