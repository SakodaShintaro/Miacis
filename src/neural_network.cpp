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

    action_encoder = register_module("action_encoder", torch::nn::Linear(POLICY_DIM, REPRESENTATION_DIM));

    policy_fc = register_module("policy_fc", torch::nn::Linear(CHANNEL_NUM, POLICY_DIM));
    value_fc1 = register_module("value_fc1", torch::nn::Linear(CHANNEL_NUM, VALUE_HIDDEN_NUM));
    value_fc2 = register_module("value_fc2", torch::nn::Linear(VALUE_HIDDEN_NUM, VALUE_HIDDEN_NUM));
    value_fc3 = register_module("value_fc3", torch::nn::Linear(VALUE_HIDDEN_NUM, BIN_SIZE));

    transition_predictor = register_module("transition_predictor", torch::nn::Linear(2 * REPRESENTATION_DIM, REPRESENTATION_DIM));
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

torch::Tensor NeuralNetworkImpl::predictTransition(torch::Tensor& state_representations,
                                                   torch::Tensor& move_representations) {
    torch::Tensor concatenated = torch::cat({state_representations, move_representations}, 1);
    return transition_predictor->forward(concatenated);
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
    std::vector<float> onehot_move_labels;
    for (Move move : moves) {
        std::vector<float> curr_onehot_label(POLICY_DIM, 0.0);
        curr_onehot_label[move.toLabel()] = 1.0;
        onehot_move_labels.insert(onehot_move_labels.end(), curr_onehot_label.begin(), curr_onehot_label.end());
    }
    torch::Tensor move_labels_tensor = torch::tensor(onehot_move_labels).view({(int64_t)moves.size(), POLICY_DIM}).to(device_);
    return action_encoder->forward(move_labels_tensor);
}

torch::Tensor NeuralNetworkImpl::decodePolicy(torch::Tensor& representation) {
    torch::Tensor policy = policy_fc->forward(representation);
    return policy;
}

torch::Tensor NeuralNetworkImpl::decodeValue(torch::Tensor& representation) {
    torch::Tensor value = value_fc1->forward(representation);
    value = torch::relu(value);
    value = value_fc2->forward(value);
    value = torch::relu(value);
    value = value_fc3->forward(value);
    return value;
}

std::array<torch::Tensor, LOSS_NUM> NeuralNetworkImpl::loss(const std::vector<LearningData>& data) {
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
    torch::Tensor state_representation = encodeState(curr_state_features);

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
    torch::Tensor policy_loss = torch::nll_loss(torch::log_softmax(policy, 1), move_teachers_tensor);

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
    torch::Tensor value_teachers_tensor = torch::tensor(value_teachers).to(device_);

    //損失を計算
#ifdef USE_CATEGORICAL
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(value, 1), value_teachers_tensor);
#else
    value = value.view(-1);
#ifdef USE_SIGMOID
    Var value_loss = -value_teachers_tensor * F::log(value) -(1 - value_teachers_tensor) * F::log(1 - value);
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
    torch::Tensor action_representation = encodeAction(moves);

    //次状態を予測
    torch::Tensor transition = predictTransition(state_representation, action_representation);

    //次状態の表現を取得
    torch::Tensor next_state_representation = encodeState(curr_state_features);

    //損失を計算
    torch::Tensor transition_loss = torch::pow(transition - next_state_representation, 2);

    return { policy_loss, value_loss, transition_loss };
}

NeuralNetwork nn;