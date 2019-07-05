#ifndef TYPES_HPP
#define TYPES_HPP

#include<array>

enum Color {
    BLACK, WHITE, ColorNum,
};

//#define USE_SIGMOID

using Score = float;
constexpr Score MAX_SCORE = 1.0;

#ifdef USE_SIGMOID
constexpr Score MIN_SCORE = 0.0;
#else
constexpr Score MIN_SCORE = -MAX_SCORE;
#endif

constexpr double CP_GAIN = 1.0 / 600.0;

template<class T, size_t SIZE>
inline std::array<T, SIZE>& operator+=(std::array<T, SIZE>& lhs, std::array<T, SIZE> rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] += rhs[i];
    }
    return lhs;
}

template<class T, size_t SIZE>
inline std::array<T, SIZE> operator+(std::array<T, SIZE> lhs, std::array<T, SIZE> rhs) {
    return lhs += rhs;
}

template<class T, size_t SIZE>
inline std::array<T, SIZE>& operator-=(std::array<T, SIZE>& lhs, std::array<T, SIZE> rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] -= rhs[i];
    }
    return lhs;
}

template<class T, size_t SIZE>
inline std::array<T, SIZE> operator-(std::array<T, SIZE> lhs, std::array<T, SIZE> rhs) {
    return lhs -= rhs;
}

template<class T, class U, size_t SIZE>
inline std::array<T, SIZE>& operator/=(std::array<T, SIZE>& lhs, U rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        lhs[i] /= rhs;
    }
    return lhs;
}

template<class T, class U, size_t SIZE>
inline std::array<T, SIZE> operator/(std::array<T, SIZE> lhs, U rhs) {
    return lhs /= rhs;
}

template<class T, class U, size_t SIZE>
inline std::array<T, SIZE> operator*(U lhs, std::array<T, SIZE> rhs) {
    for (size_t i = 0; i < SIZE; i++) {
        rhs[i] *= lhs;
    }
    return rhs;
}

#endif // !TYPES_HPP