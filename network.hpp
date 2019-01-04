#pragma once

#if USE_NN

#include"types.hpp"
#include"move.hpp"
#include"eval_params.hpp"
#include<vector>
#include<random>

namespace Network {
    void scoreByPolicy(std::vector<Move>& moves, const std::vector<CalcType>& u, int32_t scale);
    Vec activationFunction(const Vec& x);
    Vec d_activationFunction(const Vec& x);
#ifdef USE_ACTIVATION_RELU
    Vec relu(const Vec& x);
    Vec d_relu(const Vec& x);
#else
    Vec sigmoid(const Vec& x);
    Vec d_sigmoid(const Vec& x);
#endif
};

#endif