﻿#include"neural_network.hpp"

static constexpr int32_t STATE_BLOCK_NUM = 10;
static constexpr int32_t ACTION_BLOCK_NUM = 10;
static constexpr int32_t CHANNEL_NUM = 64;
static constexpr int32_t VALUE_HIDDEN_NUM = 256;
static constexpr int32_t KERNEL_SIZE = 3;
static constexpr int32_t ACTION_FEATURE_CHANNEL_NUM = 32;
static constexpr int32_t REDUCTION = 8;

#ifdef USE_CATEGORICAL
const std::string NeuralNetworkImpl::MODEL_PREFIX = "cat_bl" + std::to_string(STATE_BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#else
const std::string NeuralNetworkImpl::MODEL_PREFIX = "sca_bl" + std::to_string(STATE_BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#endif
//デフォルトで読み書きするファイル名
const std::string NeuralNetworkImpl::DEFAULT_MODEL_NAME = NeuralNetworkImpl::MODEL_PREFIX + ".model";

Conv2DwithBatchNormImpl::Conv2DwithBatchNormImpl(int64_t input_ch, int64_t output_ch, int64_t kernel_size) {
    conv_ = register_module("conv_", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_ch, output_ch, kernel_size).with_bias(false).padding(kernel_size / 2)));
    norm_ = register_module("norm_", torch::nn::BatchNorm(output_ch));
}

torch::Tensor Conv2DwithBatchNormImpl::forward(torch::Tensor& x) {
    x = conv_->forward(x);
    x = norm_->forward(x);
    return x;
}

ResidualBlockImpl::ResidualBlockImpl(int64_t channel_num, int64_t kernel_size, int64_t reduction) {
    conv_and_norm0_ = register_module("conv_and_norm0_", Conv2DwithBatchNorm(channel_num, channel_num, kernel_size));
    conv_and_norm1_ = register_module("conv_and_norm1_", Conv2DwithBatchNorm(channel_num, channel_num, kernel_size));
    linear0_ = register_module("linear0_", torch::nn::Linear(torch::nn::LinearOptions(channel_num, channel_num / reduction).with_bias(false)));
    linear1_ = register_module("linear1_", torch::nn::Linear(torch::nn::LinearOptions(channel_num / reduction, channel_num).with_bias(false)));
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor& x) {
    torch::Tensor t = x;

    x = conv_and_norm0_->forward(x);
    x = torch::relu(x);
    x = conv_and_norm1_->forward(x);

    //SENet構造
    auto y = torch::avg_pool2d(x, {9, 9});
    y = y.view({-1, CHANNEL_NUM});
    y = linear0_->forward(y);
    y = torch::relu(y);
    y = linear1_->forward(y);
    y = torch::sigmoid(y);
    y = y.view({-1, CHANNEL_NUM, 1, 1});
    x = x * y;

    x = torch::relu(x + t);
    return x;
}

NeuralNetworkImpl::NeuralNetworkImpl() : device_(torch::kCUDA), state_encoder_blocks_(STATE_BLOCK_NUM, nullptr), action_encoder_blocks_(ACTION_BLOCK_NUM, nullptr) {
    state_encoder_first_conv_ = register_module("state_encoder_first_conv_", Conv2DwithBatchNorm(STATE_FEATURE_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < STATE_BLOCK_NUM; i++) {
        state_encoder_blocks_[i] = register_module("state_encoder_blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }

    action_encoder_first_conv_ = register_module("action_encoder_first_conv_", Conv2DwithBatchNorm(ACTION_FEATURE_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < ACTION_BLOCK_NUM; i++) {
        action_encoder_blocks_[i] = register_module("action_encoder_blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }

    policy_conv_ = register_module("policy_conv_", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, POLICY_CHANNEL_NUM, 1).padding(0).with_bias(true)));
    value_conv_ = register_module("value_conv_", Conv2DwithBatchNorm(CHANNEL_NUM, CHANNEL_NUM, 1));
    value_linear0_ = register_module("value_linear0_", torch::nn::Linear(SQUARE_NUM * CHANNEL_NUM, VALUE_HIDDEN_NUM));
    value_linear1_ = register_module("value_linear1_", torch::nn::Linear(VALUE_HIDDEN_NUM, BIN_SIZE));
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
NeuralNetworkImpl::policyAndValueBatch(const std::vector<float>& inputs) {
    torch::Tensor representation = encodeStates(inputs);

    auto batch_size = inputs.size() / (SQUARE_NUM * STATE_FEATURE_CHANNEL_NUM);

    std::vector<PolicyType> policies(batch_size);
    std::vector<ValueType> values(batch_size);

    //CPUに持ってくる
    torch::Tensor policy = decodePolicy(representation).cpu();
#ifdef USE_HALF_FLOAT
    torch::Half* p = policy.data<torch::Half>();
#else
    float* p = policy.data<float>();
#endif
    for (int32_t i = 0; i < batch_size; i++) {
        policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
    }

#ifdef USE_CATEGORICAL
    torch::Tensor value = torch::softmax(decodeValue(representation), 1).cpu();
#ifdef USE_HALF_FLOAT
    torch::Half* value_p = value.data<torch::Half>();
#else
    float* value_p = value.data<float>();
#endif
    for (int32_t i = 0; i < batch_size; i++) {
        std::copy(value_p + i * BIN_SIZE, value_p + (i + 1) * BIN_SIZE, values[i].begin());
    }
#else
    //CPUに持ってくる
    torch::Tensor value = decodeValue(representation).cpu();
#ifdef USE_HALF_FLOAT
    std::copy(value.data<torch::Half>(), value.data<torch::Half>() + batch_size, values.begin());
#else
    std::copy(value.data<float>(), value.data<float>() + batch_size, values.begin());
#endif
#endif
    return { policies, values };
}

torch::Tensor NeuralNetworkImpl::predictTransition(torch::Tensor& state_representations,
                                                   torch::Tensor& move_representations) {
    return state_representations + move_representations;
}

torch::Tensor NeuralNetworkImpl::encodeStates(const std::vector<float>& inputs) {
#ifdef USE_HALF_FLOAT
    torch::Tensor x = torch::tensor(inputs).to(device_, torch::kHalf);
#else
    torch::Tensor x = torch::tensor(inputs).to(device_);
#endif
    x = x.view({ -1, STATE_FEATURE_CHANNEL_NUM, 9, 9 });
    x = state_encoder_first_conv_->forward(x);
    x = torch::relu(x);

    for (int32_t i = 0; i < STATE_BLOCK_NUM; i++) {
        x = state_encoder_blocks_[i]->forward(x);
    }

    return x;
}

torch::Tensor NeuralNetworkImpl::encodeActions(const std::vector<Move>& moves) {
    std::vector<float> move_features;
    for (Move move : moves) {
        //各moveにつき9×9×MOVE_FEATURE_CHANNEL_NUMの特徴マップを得る
        std::vector<float> curr_move_feature(9 * 9 * ACTION_FEATURE_CHANNEL_NUM, 0.0);

        //1ch:toの位置に1を立てる
        curr_move_feature[SquareToNum[move.to()]] = 1;

        //2ch:fromの位置に1を立てる.持ち駒から打つ手ならなし
        //3ch:持ち駒から打つ手なら全て1
        if (move.isDrop()) {
            for (Square sq : SquareList) {
                curr_move_feature[2 * SQUARE_NUM + SquareToNum[sq]] = 1;
            }
        } else {
            curr_move_feature[SQUARE_NUM + SquareToNum[move.from()]] = 1;
        }

        //4ch:成りなら全て1
        if (move.isPromote()) {
            for (Square sq : SquareList) {
                curr_move_feature[3 * SQUARE_NUM + SquareToNum[sq]] = 1;
            }
        }

        //5ch以降:駒の種類に合わせたところだけ全て1
        for (Square sq : SquareList) {
            curr_move_feature[(4 + PieceToNum[move.subject()]) * SQUARE_NUM + SquareToNum[sq]] = 1;
        }

        move_features.insert(move_features.end(), curr_move_feature.begin(), curr_move_feature.end());
    }
#ifdef USE_HALF_FLOAT
    torch::Tensor move_features_tensor = torch::tensor(move_features).to(device_, torch::kHalf);
#else
    torch::Tensor move_features_tensor = torch::tensor(move_features).to(device_);
#endif
    move_features_tensor = move_features_tensor.view({ -1, ACTION_FEATURE_CHANNEL_NUM, 9, 9 });
    torch::Tensor x = action_encoder_first_conv_->forward(move_features_tensor);
    for (auto& block : action_encoder_blocks_) {
        x = block->forward(x);
    }
    return x;
}

torch::Tensor NeuralNetworkImpl::decodePolicy(torch::Tensor& representation) {
    torch::Tensor policy = policy_conv_->forward(representation);
    return policy.view({ -1, POLICY_DIM });
}

torch::Tensor NeuralNetworkImpl::decodeValue(torch::Tensor& representation) {
    torch::Tensor value = value_conv_->forward(representation);
    value = torch::relu(value);
    value = value.view({ -1, SQUARE_NUM * CHANNEL_NUM });
    value = value_linear0_->forward(value);
    value = torch::relu(value);
    value = value_linear1_->forward(value);
    return value;
}

std::array<torch::Tensor, LOSS_TYPE_NUM> NeuralNetworkImpl::loss(const std::vector<LearningData>& data) {
    static Position pos;

    //局面の特徴量を取得
    std::vector<float> curr_state_features;
    std::vector<float> next_state_features;
    for (const LearningData& datum : data) {
        //現局面の特徴量を取得
        pos.loadSFEN(datum.SFEN);
        std::vector<float> curr_state_feature = pos.makeFeature();
        curr_state_features.insert(curr_state_features.end(), curr_state_feature.begin(), curr_state_feature.end());

        //次局面の特徴量を取得
        pos.doMove(datum.move);
        std::vector<float> next_state_feature = pos.makeFeature();
        next_state_features.insert(next_state_features.end(), next_state_feature.begin(), next_state_feature.end());
    }

    //現局面の特徴を表現に変換
    torch::Tensor state_representation = encodeStates(curr_state_features);

    //---------------------
    //  Policyの損失計算
    //---------------------
    //Policyを取得
    torch::Tensor policy = decodePolicy(state_representation);

    //Policyの教師を構築
    std::vector<int64_t> move_teachers;
    for (const LearningData& d : data) {
        move_teachers.push_back(d.move.toLabel());
    }
    torch::Tensor move_teachers_tensor = torch::tensor(move_teachers).to(device_);

    //損失を計算
    torch::Tensor policy_loss = torch::nll_loss(torch::log_softmax(policy, 1), move_teachers_tensor, {},
                                                Reduction::None);

    //--------------------
    //  Valueの損失計算
    //--------------------
    //Valueを取得
    torch::Tensor value = decodeValue(state_representation);

    //Valueの教師を構築
    std::vector<ValueTeacherType> value_teachers;
    for (const LearningData& d : data) {
        value_teachers.push_back(d.value);
    }

    //損失を計算
#ifdef USE_CATEGORICAL
    torch::Tensor value_teachers_tensor = torch::tensor(value_teachers).to(device_);
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(value, 1), value_teachers_tensor);
#else
#ifdef USE_HALF_FLOAT
    torch::Tensor value_teachers_tensor = torch::tensor(value_teachers).to(device_, torch::kHalf);
#else
    torch::Tensor value_teachers_tensor = torch::tensor(value_teachers).to(device_);
#endif

    value = value.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_t = (fp16_ ? torch::tensor(value_teachers).to(device_, torch::kHalf) :
                                     torch::tensor(value_teachers).to(device_));
    torch::Tensor value_loss = -value_t * torch::log(value) - (1 - value_t) * torch::log(1 - value);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_teachers_tensor, Reduction::None);
#endif
#endif

    //----------------------
    //  遷移予測の損失計算
    //----------------------
    //行動の表現を取得
    std::vector<Move> moves;
    for (const LearningData& d : data) {
        moves.push_back(d.move);
    }
    torch::Tensor action_representation = encodeActions(moves);

    //次状態を予測
    torch::Tensor predicted_state_representation = predictTransition(state_representation, action_representation);

    //次状態の表現を取得
    torch::Tensor next_state_representation = encodeStates(next_state_features);

    //損失を計算
    torch::Tensor diff = predicted_state_representation - next_state_representation;
    torch::Tensor square = torch::pow(diff, 2);
    torch::Tensor sum = torch::sum(square, { 1, 2, 3 });
    torch::Tensor transition_loss = torch::sqrt(sum);

    return { policy_loss, value_loss, transition_loss };
}

void NeuralNetworkImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_));
}

NeuralNetwork nn;