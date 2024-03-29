﻿#include "model_common.hpp"
#include "../common.hpp"
#include "../shogi/position.hpp"

//デフォルトで読み書きするファイル名
const std::string DEFAULT_MODEL_PREFIX = "default_model";
const std::string DEFAULT_MODEL_NAME = DEFAULT_MODEL_PREFIX + ".model";
const std::string DEFAULT_ONNX_NAME = DEFAULT_MODEL_PREFIX + ".onnx";
const std::string DEFAULT_ENGINE_NAME = DEFAULT_MODEL_PREFIX + ".engine";

torch::Tensor inputVectorToTensor(const std::vector<float>& input) {
    return torch::tensor(input).view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
}