#include "neural_network.hpp"
#include "common.hpp"
#include "include_switch.hpp"

//ネットワークの設定
#ifdef SHOGI
static constexpr int32_t BLOCK_NUM = 10;
static constexpr int32_t CHANNEL_NUM = 128;
#elif defined(OTHELLO)
static constexpr int32_t BLOCK_NUM = 5;
static constexpr int32_t CHANNEL_NUM = 64;
#endif
static constexpr int32_t KERNEL_SIZE = 3;
static constexpr int32_t REDUCTION = 8;
static constexpr int32_t VALUE_HIDDEN_NUM = 256;

#ifdef USE_CATEGORICAL
const std::string MODEL_PREFIX = "cat_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#else
const std::string MODEL_PREFIX = "sca_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#endif
//デフォルトで読み書きするファイル名
const std::string DEFAULT_MODEL_NAME = MODEL_PREFIX + ".model";