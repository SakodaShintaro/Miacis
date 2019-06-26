#include"neural_network.hpp"

//遅くなるのでオフ
//#define USE_HALF_FLOAT

NeuralNetworkImpl::NeuralNetworkImpl() : device_(torch::kCUDA),
                                         conv(BLOCK_NUM, std::vector<torch::nn::Conv2d>(2, nullptr)),
                                         norm(BLOCK_NUM, std::vector<torch::nn::BatchNorm>(2, nullptr)),
                                         fc(BLOCK_NUM, std::vector<torch::nn::Linear>(2, nullptr)) {
    first_conv = register_module("first_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(INPUT_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE).with_bias(false).padding(1)));
    first_norm = register_module("first_norm", torch::nn::BatchNorm(CHANNEL_NUM));
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        for (int32_t j = 0; j < 2; j++) {
            conv[i][j] = register_module("conv" + std::to_string(i) + "_" + std::to_string(j), torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE).with_bias(false).padding(1)));
            norm[i][j] = register_module("norm" + std::to_string(i) + "_" + std::to_string(j), torch::nn::BatchNorm(CHANNEL_NUM));
        }
        fc[i][0] = register_module("fc" + std::to_string(i) + "_0", torch::nn::Linear(torch::nn::LinearOptions(CHANNEL_NUM, CHANNEL_NUM / REDUCTION).with_bias(false)));
        fc[i][1] = register_module("fc" + std::to_string(i) + "_1", torch::nn::Linear(torch::nn::LinearOptions(CHANNEL_NUM / REDUCTION, CHANNEL_NUM).with_bias(false)));
    }
    policy_fc = register_module("policy_fc", torch::nn::Linear(CHANNEL_NUM, SQUARE_NUM * POLICY_CHANNEL_NUM));
    value_fc1 = register_module("value_fc1", torch::nn::Linear(CHANNEL_NUM, VALUE_HIDDEN_NUM));
    value_fc2 = register_module("value_fc2", torch::nn::Linear(VALUE_HIDDEN_NUM, VALUE_HIDDEN_NUM));
    value_fc3 = register_module("value_fc3", torch::nn::Linear(VALUE_HIDDEN_NUM, BIN_SIZE));

    next_state_rep_predictor = register_module("next_state_rep_predictor",
            torch::nn::Linear(REPRESENTATION_DIM + SQUARE_NUM * POLICY_CHANNEL_NUM, REPRESENTATION_DIM));
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
NeuralNetworkImpl::policyAndValueBatch(const std::vector<float>& inputs) {
    torch::Tensor representation = encodeState(inputs);

    auto batch_size = inputs.size() / (SQUARE_NUM * INPUT_CHANNEL_NUM);

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
        constexpr auto unit_size = POLICY_CHANNEL_NUM * SQUARE_NUM;
        policies[i].assign(p + i * unit_size, p + (i + 1) * unit_size);
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

std::pair<torch::Tensor, torch::Tensor>
NeuralNetworkImpl::loss(const std::vector<float>& input,
                        const std::vector<PolicyTeacherType>& policy_teachers,
                        const std::vector<ValueTeacherType>& value_teachers) {
    torch::Tensor representation = encodeState(input);
    torch::Tensor logits = decodePolicy(representation).view({ -1, SQUARE_NUM * POLICY_CHANNEL_NUM });

    std::vector<float> policy_dist(policy_teachers.size() * SQUARE_NUM * POLICY_CHANNEL_NUM, 0.0);
    for (int64_t i = 0; i < policy_teachers.size(); i++) {
        for (const auto& e : policy_teachers[i]) {
            policy_dist[i * SQUARE_NUM * POLICY_CHANNEL_NUM + e.first] = e.second;
        }
    }

#ifdef USE_HALF_FLOAT
    torch::Tensor policy_target = torch::tensor(policy_dist).to(device_, torch::kHalf).view({ -1, SQUARE_NUM * POLICY_CHANNEL_NUM });
#else
    torch::Tensor policy_target = torch::tensor(policy_dist).to(device_).view({ -1, SQUARE_NUM * POLICY_CHANNEL_NUM });
#endif
    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(logits, 1), 1, false);

#ifdef USE_CATEGORICAL
    torch::Tensor categorical_target = torch::tensor(value_teachers).to(device_);
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(decodeValue(representation), 1), categorical_target);
#else

#ifdef USE_HALF_FLOAT
    torch::Tensor value_t = torch::tensor(value_teachers).to(device_, torch::kHalf);
#else
    torch::Tensor value_t = torch::tensor(value_teachers).to(device_);
#endif
    torch::Tensor value = decodeValue(representation).view(-1);
#ifdef USE_SIGMOID
    Var value_loss = -value_t * F::log(value) -(1 - value_t) * F::log(1 - value);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t, Reduction::None);
#endif
#endif
    
    return { policy_loss, value_loss };
}

void NeuralNetworkImpl::setGPU(int16_t gpu_id) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
#ifdef USE_HALF_FLOAT
    to(device_, torch::kHalf);
    first_conv->to(device_, torch::kHalf);
    first_norm->to(device_);
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        for (int32_t j = 0; j < 2; j++) {
            conv[i][j]->to(device_, torch::kHalf);
            norm[i][j]->to(device_);
            fc[i][j]->to(device_, torch::kHalf);
        }
    }
    policy_conv->to(device_, torch::kHalf);
    value_conv->to(device_, torch::kHalf);
    value_bn->to(device_);
    value_fc2->to(device_, torch::kHalf);
    value_fc3->to(device_, torch::kHalf);
#else
    to(device_);
#endif
}

torch::Tensor NeuralNetworkImpl::predictNextStateRep(torch::Tensor state_representations, torch::Tensor move_representations) {
    torch::Tensor concatenated = torch::cat({state_representations, move_representations});
    return next_state_rep_predictor->forward(concatenated);
}

torch::Tensor NeuralNetworkImpl::encodeState(const std::vector<float>& inputs) {
#ifdef USE_HALF_FLOAT
    torch::Tensor x = torch::tensor(inputs).to(device_, torch::kHalf);
#else
    torch::Tensor x = torch::tensor(inputs).to(device_);
#endif
    x = x.view({ -1, INPUT_CHANNEL_NUM, 9, 9 });
    x = first_conv->forward(x);
    x = first_norm->forward(x);
    x = torch::relu(x);

    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        torch::Tensor t = x;

        x = conv[i][0]->forward(x);
        x = norm[i][0]->forward(x);
        x = torch::relu(x);

        x = conv[i][1]->forward(x);
        x = norm[i][1]->forward(x);

        //SENet構造
        torch::Tensor y = torch::avg_pool2d(x, {9, 9});
        y = y.view({-1, CHANNEL_NUM});
        y = fc[i][0]->forward(y);
        y = torch::relu(y);
        y = fc[i][1]->forward(y);
        y = torch::sigmoid(y);
        y = y.view({-1, CHANNEL_NUM, 1, 1});
        x = x * y;

        x = torch::relu(x + t);
    }

    return torch::avg_pool2d(x, {9, 9}).view({-1, CHANNEL_NUM});
}

torch::Tensor NeuralNetworkImpl::encodeAction(const std::vector<Move>& moves) {
    std::vector<int64_t> move_labels;
    for (Move move : moves) {
        move_labels.push_back(move.toLabel());
    }
    torch::Tensor move_labels_tensor = torch::tensor(move_labels);
    return action_encode_fc->forward(move_labels_tensor);
}

torch::Tensor NeuralNetworkImpl::decodePolicy(torch::Tensor representation) {
    torch::Tensor policy = policy_fc->forward(representation);
    return policy;
}

torch::Tensor NeuralNetworkImpl::decodeValue(torch::Tensor representation) {
    torch::Tensor value = value_fc1->forward(representation);
    value = torch::relu(value);
    value = value_fc2->forward(value);
    value = torch::relu(value);
    value = value_fc3->forward(value);
    return value;
}

std::array<torch::Tensor, 3>
NeuralNetworkImpl::loss(const std::vector<std::string>& SFENs,
                        const std::vector<Move>& moves,
                        const std::vector<ValueTeacherType>& values) {
    static Position pos;
    assert(SFENs.size() == moves.size());

    //局面の特徴量を取得
    std::vector<float> curr_state_features;
    std::vector<float> next_state_features;
    for (int64_t i = 0; i < SFENs.size(); i++) {
        //現局面の特徴量を取得
        pos.loadSFEN(SFENs[i]);
        std::vector<float> curr_state_feature = pos.makeFeature();
        curr_state_features.insert(curr_state_features.end(), curr_state_feature.begin(), curr_state_feature.end());

        //次局面の特徴量を取得
        pos.doMove(moves[i]);
        std::vector<float> next_state_feature = pos.makeFeature();
        next_state_features.insert(next_state_features.end(), next_state_feature.begin(), next_state_feature.end());
    }

    //現局面の特徴を表現に変換
    torch::Tensor state_representation = encodeState(curr_state_features);

    //---------------------
    //  Policyの損失計算
    //---------------------
    //Policyを取得
    torch::Tensor policy = decodePolicy(state_representation);

    //Policyの教師を構築
    std::vector<int64_t> move_teachers;
    for (Move move : moves) {
        move_teachers.push_back(move.toLabel());
    }
    torch::Tensor move_teachers_tensor = torch::tensor(move_teachers).to(device_);

    //損失を計算
    torch::Tensor policy_loss = torch::nll_loss(torch::log_softmax(policy, 1), move_teachers_tensor);

    //--------------------
    //  Valueの損失計算
    //--------------------
    //Valueを取得
    torch::Tensor value = decodeValue(state_representation);

    //Valueの教師を構築
    torch::Tensor value_t = torch::tensor(values).to(device_);

    //損失を計算
#ifdef USE_CATEGORICAL
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(value, 1), value_t);
#else
    //これなんでいるんだっけ？
    value = value.view(-1);
#ifdef USE_SIGMOID
    Var value_loss = -value_t * F::log(value) -(1 - value_t) * F::log(1 - value);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t, Reduction::None);
#endif
#endif

    //---------------------------
    //  次状態表現予測の損失計算
    //---------------------------
    //行動の表現を取得
    torch::Tensor action_representation = encodeAction(moves);

    //次状態を予測
    torch::Tensor predict = predictNextStateRep(state_representation, action_representation);

    //次状態の表現を取得
    torch::Tensor next_state_representation = encodeState(curr_state_features);

    //損失を計算
    torch::Tensor predict_loss = torch::mse_loss(state_representation, next_state_representation);

    return { policy_loss, value_loss, predict_loss };
}

NeuralNetwork nn;