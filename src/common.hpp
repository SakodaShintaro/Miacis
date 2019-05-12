#ifndef COMMON_HPP
#define COMMON_HPP

#include<cmath>
#include<cassert>
#include<cstdint>
#include<vector>
#include<random>
#include<algorithm>
#ifdef _MSC_VER
#include<intrin.h>
#elif __GNUC__
#include<x86intrin.h>
#include<bmi2intrin.h>
#endif

#ifdef __GNUC__
//cf : http://blog.jiubao.org/2015/01/gcc-bitscanforward-bitscanreverse-msvc.html
unsigned char inline GNUBitScanForward(unsigned long *Index, uint64_t Mask) {
    if (Mask) {
        *Index = __builtin_ctzll(Mask);
        return 1;
    } else {
        /* 戻り値が0のとき、*Index がどうなるかは未定義。*/
        return 0;
    }
}

unsigned char inline GNUBitScanReverse(unsigned long *Index, uint64_t Mask) {
    if (Mask) {
        *Index = 63 - __builtin_clzll(Mask);
        return 1;
    } else {
        /* 戻り値が0のとき、*Index がどうなるかは未定義。*/
        return 0;
    }
}
#endif

inline int MSB64(uint64_t v) {
    assert(v != 0);
    unsigned long index;

#ifdef _MSC_VER
    _BitScanReverse64(&index, v);
#elif __GNUC__
    GNUBitScanReverse(&index, v);
#endif

    return (int)index;
}

inline auto PEXT64(uint64_t a, uint64_t b) {
    return _pext_u64(a, b);
}

inline auto POP_CNT64(uint64_t bits) {
#ifdef _MSC_VER
    return __popcnt64(bits);
#elif __GNUC__
    return __builtin_popcountll (bits);
#endif
}

inline int pop_lsb(uint64_t& b) {
    unsigned long index;

#ifdef _MSC_VER
    _BitScanForward64(&index, b);
#elif __GNUC__
    GNUBitScanForward(&index, b);
#endif

    b = _blsr_u64(b);
    return (int)index;
}

template<class Type>
inline double sigmoid(Type x, double gain) {
    return 1.0 / (1.0 + exp(-gain * x));
}

template<class Type>
inline std::vector<Type> softmax(std::vector<Type> x, Type temperature = 1.0) {
    assert(!x.empty());
    auto max_value = *std::max_element(x.begin(), x.end());
    Type sum = 0.0;
    for (auto& p : x) {
        p = (Type)std::exp((p - max_value) / temperature);
        sum += p;
    }
    assert(sum != 0);
    for (auto& p : x) {
        p /= sum;
    }

    return x;
}

template<class Type>
inline int32_t randomChoose(const std::vector<Type>& x) {
    std::random_device seed;
    std::uniform_real_distribution<Type> dist(0.0, 1.0);
    Type prob = dist(seed);
    for (int32_t i = 0; i < x.size(); i++) {
        if ((prob -= x[i]) < 0) {
            return i;
        }
    }
    return (int32_t)(x.size() - 1);
}

#endif
