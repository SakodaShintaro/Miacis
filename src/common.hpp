#ifndef MIACIS_COMMON_HPP
#define MIACIS_COMMON_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#elif __GNUC__
// clang-format off
#include <x86intrin.h>
#include <bmi2intrin.h>
// clang-format on
#endif

extern std::mt19937_64 engine;

#ifdef __GNUC__
//cf : http://blog.jiubao.org/2015/01/gcc-bitscanforward-bitscanreverse-msvc.html
uint16_t inline GNUBitScanForward(uint64_t* Index, uint64_t Mask) {
    if (Mask) {
        *Index = __builtin_ctzll(Mask);
        return 1;
    } else {
        /* 戻り値が0のとき、*Index がどうなるかは未定義。*/
        return 0;
    }
}

uint16_t inline GNUBitScanReverse(uint64_t* Index, uint64_t Mask) {
    if (Mask) {
        *Index = 63 - __builtin_clzll(Mask);
        return 1;
    } else {
        /* 戻り値が0のとき、*Index がどうなるかは未定義。*/
        return 0;
    }
}
#endif

inline int32_t MSB64(uint64_t v) {
    assert(v != 0);
    uint64_t index{};

#ifdef _MSC_VER
    _BitScanReverse64(&index, v);
#elif __GNUC__
    GNUBitScanReverse(&index, v);
#endif

    return (int32_t)index;
}

inline uint64_t PEXT64(uint64_t a, uint64_t b) { return _pext_u64(a, b); }

inline uint64_t POP_CNT64(uint64_t bits) {
#ifdef _MSC_VER
    return __popcnt64(bits);
#elif __GNUC__
    return __builtin_popcountll(bits);
#endif
}

inline int32_t pop_lsb(uint64_t& b) {
    uint64_t index{};

#ifdef _MSC_VER
    _BitScanForward64(&index, b);
#elif __GNUC__
    GNUBitScanForward(&index, b);
#endif

    b = _blsr_u64(b);
    return (int32_t)index;
}

template<class T> inline float sigmoid(T x, float gain) { return 1.0 / (1.0 + exp(-gain * x)); }

template<class T> inline std::vector<T> softmax(std::vector<T> x, T temperature = 1.0) {
    assert(!x.empty());
    T max_value = *std::max_element(x.begin(), x.end());

    T exp_sum = 0.0;
    for (auto& p : x) {
        p = (T)std::exp((p - max_value) / temperature);
        exp_sum += p;
    }
    assert(exp_sum != 0);
    for (auto& p : x) {
        p /= exp_sum;
    }

    return x;
}

template<class T> inline int32_t randomChoose(const std::vector<T>& x) {
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    T prob = dist(engine);
    for (uint64_t i = 0; i < x.size(); i++) {
        if ((prob -= x[i]) < 0) {
            return i;
        }
    }
    return (int32_t)(x.size() - 1);
}

#endif //MIACIS_COMMON_HPP