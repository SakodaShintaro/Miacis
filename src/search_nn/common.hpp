#ifndef MIACIS_SEARCH_NN_COMMON_HPP
#define MIACIS_SEARCH_NN_COMMON_HPP

#include "../neural_network.hpp"

torch::Tensor getPolicyTeacher(const std::vector<LearningData>& data, torch::Device device);

#endif //MIACIS_SEARCH_NN_COMMON_HPP