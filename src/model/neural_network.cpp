#include "neural_network.hpp"
#include "../common.hpp"
#include "../include_switch.hpp"

#ifdef USE_CATEGORICAL
#ifdef SHOGI
const std::string MODEL_PREFIX = "shogi_cat_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#elif defined(OTHELLO)
const std::string MODEL_PREFIX = "othello_cat_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#elif defined(GO)
const std::string MODEL_PREFIX = "go_cat_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#endif
#else
#ifdef SHOGI
const std::string MODEL_PREFIX = "shogi_sca_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#elif defined(OTHELLO)
const std::string MODEL_PREFIX = "othello_sca_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#elif defined(GO)
const std::string MODEL_PREFIX = "go_sca_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#endif
#endif
//デフォルトで読み書きするファイル名
const std::string DEFAULT_MODEL_NAME = MODEL_PREFIX + ".model";

torch::Tensor inputVectorToTensor(const std::vector<float>& input) {
    return torch::tensor(input).view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
tensorToVector(const std::tuple<torch::Tensor, torch::Tensor>& output) {
    const auto& [policy, value] = output;
    uint64_t batch_size = policy.size(0);

    std::vector<PolicyType> policies(batch_size);
    std::vector<ValueType> values(batch_size);

    if (policy.dtype() == torch::kHalf) {
        torch::Half* p = policy.data_ptr<torch::Half>();
        for (uint64_t i = 0; i < batch_size; i++) {
            policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
        }
    } else {
        float* p = policy.data_ptr<float>();
        for (uint64_t i = 0; i < batch_size; i++) {
            policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
        }
    }

#ifdef USE_CATEGORICAL
    //valueの方はfp16化してもなぜかHalfではなくFloatとして返ってくる
    //ひょっとしたらTRTorchのバグかも
    float* value_p = value.data_ptr<float>();
    for (uint64_t i = 0; i < batch_size; i++) {
        std::copy(value_p + i * BIN_SIZE, value_p + (i + 1) * BIN_SIZE, values[i].begin());
    }
#else
    std::copy(value.data_ptr<float>(), value.data_ptr<float>() + batch_size, values.begin());
#endif
    return std::make_pair(policies, values);
}