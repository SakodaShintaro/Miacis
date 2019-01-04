#pragma once

#ifndef TYPES_HPP
#define TYPES_HPP

#include<iostream>
#include<array>

enum Color {
    BLACK, WHITE, ColorNum,
};

enum Bound {
    EXACT_BOUND, UPPER_BOUND, LOWER_BOUND
};

enum Depth {
    PLY = 128,
    DEPTH_MAX = 64 * PLY,
    MATE_DEPTH_MAX = 128,
};

#ifdef USE_NN
using Score = float;
constexpr Score MAX_SCORE = 1000000;
constexpr Score SCORE_ZERO = 0;
constexpr Score DRAW_SCORE = 0;
constexpr Score MIN_SCORE = -MAX_SCORE;
constexpr Score MATE_SCORE_LOWER_BOUND = MAX_SCORE - static_cast<int>(MATE_DEPTH_MAX);
constexpr Score MATE_SCORE_UPPER_BOUND = MIN_SCORE + static_cast<int>(MATE_DEPTH_MAX);
constexpr Score SCORE_NONE = MAX_SCORE + 1;

#else
enum Score {
    MAX_SCORE = 1000000,
    SCORE_ZERO = 0,
    DRAW_SCORE = 0,
    MIN_SCORE = -MAX_SCORE,
    MATE_SCORE_LOWER_BOUND = MAX_SCORE - static_cast<int>(MATE_DEPTH_MAX),
    MATE_SCORE_UPPER_BOUND = MIN_SCORE + static_cast<int>(MATE_DEPTH_MAX),
    SCORE_NONE = MAX_SCORE + 1
};
#endif

constexpr double CP_GAIN = 1.0 / 600.0;

inline bool isMatedScore(const Score score) {
    return score <= MATE_SCORE_UPPER_BOUND || MATE_SCORE_LOWER_BOUND <= score;
}

//演算子をオーバーロードしておかないと不便
#ifndef USE_NN
//Score
constexpr Score operator-(Score lhs) { return Score(-int(lhs)); }
constexpr Score operator+(Score lhs, Score rhs) { return Score(int(lhs) + int(rhs)); }
constexpr Score operator-(Score lhs, Score rhs) { return Score(int(lhs) - int(rhs)); }
constexpr Score operator+(Score lhs, int rhs) { return Score(int(lhs) + rhs); }
constexpr Score operator-(Score lhs, int rhs) { return Score(int(lhs) - rhs); }
constexpr Score operator+(int lhs, Score rhs) { return Score(lhs + int(rhs)); }
constexpr Score operator-(int lhs, Score rhs) { return Score(lhs - int(rhs)); }
inline Score& operator+=(Score& lhs, Score rhs) { return lhs = lhs + rhs; }
inline Score& operator-=(Score& lhs, Score rhs) { lhs = lhs - rhs;  return lhs; }
inline Score& operator+=(Score& lhs, int rhs) { lhs = lhs + rhs;  return lhs; }
inline Score& operator-=(Score& lhs, int rhs) { lhs = lhs - rhs;  return lhs; }
inline Score& operator++(Score& lhs) { lhs = lhs + 1;  return lhs; }
inline Score& operator--(Score& lhs) { lhs = lhs - 1;  return lhs; }
inline Score operator++(Score& lhs, int) { Score t = lhs; lhs += 1;  return t; }
inline Score operator--(Score& lhs, int) { Score t = lhs; lhs -= 1;  return t; }

constexpr Score operator*(Score lhs, int rhs) { return Score(int(lhs) * rhs); }
constexpr Score operator*(int lhs, Score rhs) { return Score(lhs * int(rhs)); }
constexpr Score operator/(Score lhs, int rhs) { return Score(int(lhs) / rhs); }
inline Score& operator*=(Score& lhs, int rhs) { lhs = lhs * rhs;  return lhs; }
inline Score& operator/=(Score& lhs, int rhs) { lhs = lhs / rhs;  return lhs; }
std::ostream& operator<<(std::ostream& os, Score s);
#endif

//Depth
constexpr Depth operator-(Depth lhs) { return Depth(-int(lhs)); }
constexpr Depth operator+(Depth lhs, Depth rhs) { return Depth(int(lhs) + int(rhs)); }
constexpr Depth operator-(Depth lhs, Depth rhs) { return Depth(int(lhs) - int(rhs)); }
constexpr Depth operator+(Depth lhs, int rhs) { return Depth(int(lhs) + rhs); }
constexpr Depth operator-(Depth lhs, int rhs) { return Depth(int(lhs) - rhs); }
constexpr Depth operator+(int lhs, Depth rhs) { return Depth(lhs + int(rhs)); }
constexpr Depth operator-(int lhs, Depth rhs) { return Depth(lhs - int(rhs)); }
inline Depth& operator+=(Depth& lhs, Depth rhs) { return lhs = lhs + rhs; }
inline Depth& operator-=(Depth& lhs, Depth rhs) { lhs = lhs - rhs;  return lhs; }
inline Depth& operator+=(Depth& lhs, int rhs) { lhs = lhs + rhs;  return lhs; }
inline Depth& operator-=(Depth& lhs, int rhs) { lhs = lhs - rhs;  return lhs; }
inline Depth& operator++(Depth& lhs) { lhs = lhs + 1;  return lhs; }
inline Depth& operator--(Depth& lhs) { lhs = lhs - 1;  return lhs; }
inline Depth operator++(Depth& lhs, int) { Depth t = lhs; lhs += 1;  return t; }
inline Depth operator--(Depth& lhs, int) { Depth t = lhs; lhs -= 1;  return t; }

constexpr Depth operator*(Depth lhs, int rhs) { return Depth(int(lhs) * rhs); }
constexpr Depth operator*(int lhs, Depth rhs) { return Depth(lhs * int(rhs)); }
constexpr Depth operator/(Depth lhs, int rhs) { return Depth(int(lhs) / rhs); }
inline Depth& operator*=(Depth& lhs, int rhs) { lhs = lhs * rhs;  return lhs; }
inline Depth& operator/=(Depth& lhs, int rhs) { lhs = lhs / rhs;  return lhs; }
std::ostream& operator<<(std::ostream& os, Depth d);
std::istream& operator>>(std::istream& is, Depth& d);

//std::arrayに関するオーバーロード
inline std::array<int32_t, 2>& operator+=(std::array<int32_t, 2>& lhs, std::array<int16_t, 2> rhs) {
    lhs[0] += rhs[0];
    lhs[1] += rhs[1];
    return lhs;
}
inline std::array<int32_t, 2>& operator-=(std::array<int32_t, 2>& lhs, std::array<int16_t, 2> rhs) {
    lhs[0] -= rhs[0];
    lhs[1] -= rhs[1];
    return lhs;
}
template<class T>
inline std::array<T, 2>& operator+=(std::array<T, 2>& lhs, std::array<T, 2> rhs) {
    lhs[0] += rhs[0];
    lhs[1] += rhs[1];
    return lhs;
}
template<class T>
inline std::array<T, 2>& operator-=(std::array<T, 2>& lhs, std::array<T, 2> rhs) {
    lhs[0] -= rhs[0];
    lhs[1] -= rhs[1];
    return lhs;
}
inline std::array<int16_t, 2> operator*(int c, std::array<int16_t, 2> rhs) {
    rhs[0] *= c;
    rhs[1] *= c;
    return rhs;
}
inline std::array<int32_t, 2> operator*(int c, std::array<int32_t, 2> rhs) {
    rhs[0] *= c;
    rhs[1] *= c;
    return rhs;
}

//これで上の要らなくなりそうだけど
template<class T, size_t SIZE>
inline std::array<T, SIZE> operator+(std::array<T, SIZE> lhs, std::array<T, SIZE> rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] += rhs[i];
    }
    return lhs;
}

template<class T, size_t SIZE>
inline std::array<T, SIZE>& operator+=(std::array<T, SIZE>& lhs, std::array<T, SIZE> rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] += rhs[i];
    }
    return lhs;
}

template<class T, class U, size_t SIZE>
inline std::array<T, SIZE> operator/(std::array<T, SIZE> lhs, U rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] /= rhs;
    }
    return lhs;
}

template<class T, class U, size_t SIZE>
inline std::array<T, SIZE>& operator/=(std::array<T, SIZE>& lhs, U rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] /= rhs;
    }
    return lhs;
}

#endif // !TYPES_HPP