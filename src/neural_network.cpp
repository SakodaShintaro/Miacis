#include"neural_network.hpp"

//大して速くならないわりに性能は落ちるのでとりあえずOFF
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

    action_encoder = register_module("action_encoder", torch::nn::Conv2d(torch::nn::Conv2dOptions(MOVE_FEATURE_CHANNEL_NUM, POLICY_CHANNEL_NUM, 3).padding(1).with_bias(true)));

    policy_conv = register_module("policy_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, POLICY_CHANNEL_NUM, 1).padding(0).with_bias(true)));
    value_conv = register_module("value_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, CHANNEL_NUM, 1).padding(0).with_bias(false)));
    value_norm = register_module("value_norm", torch::nn::BatchNorm(CHANNEL_NUM));
    value_fc1 = register_module("value_fc1", torch::nn::Linear(SQUARE_NUM * CHANNEL_NUM, VALUE_HIDDEN_NUM));
    value_fc2 = register_module("value_fc2", torch::nn::Linear(VALUE_HIDDEN_NUM, BIN_SIZE));

    transition_predictor = register_module("transition_predictor", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM + POLICY_CHANNEL_NUM, CHANNEL_NUM, 3).padding(1).with_bias(true)));
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
NeuralNetworkImpl::policyAndValueBatch(const std::vector<float>& inputs) {
    torch::Tensor representation = encodeStates(inputs);

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
    for (const auto& module : modules()) {
        if (module->name() == "torch::nn::BatchNormImpl") {
            module->to(device_);
        } else {
            module->to(device_, torch::kHalf);
        }
    }
#else
    to(device_);
#endif
}

torch::Tensor NeuralNetworkImpl::predictTransition(torch::Tensor& state_representations,
                                                   torch::Tensor& move_representations) {
    torch::Tensor concatenated = torch::cat({state_representations, move_representations}, 1);
    return transition_predictor->forward(concatenated);
}

torch::Tensor NeuralNetworkImpl::encodeStates(const std::vector<float>& inputs) {
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

    return x;
}

torch::Tensor NeuralNetworkImpl::encodeActions(const std::vector<Move>& moves) {
    std::vector<float> move_features;
    for (Move move : moves) {
        //各moveにつき9×9×MOVE_FEATURE_CHANNEL_NUMの特徴マップを得る
        std::vector<float> curr_move_feature(9 * 9 * MOVE_FEATURE_CHANNEL_NUM, 0.0);

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
    move_features_tensor = move_features_tensor.view({ -1, MOVE_FEATURE_CHANNEL_NUM, 9, 9 });
    return action_encoder->forward(move_features_tensor);
}

torch::Tensor NeuralNetworkImpl::decodePolicy(torch::Tensor& representation) {
    torch::Tensor policy = policy_conv->forward(representation);
    return policy.view({ -1, POLICY_DIM });
}

torch::Tensor NeuralNetworkImpl::decodeValue(torch::Tensor& representation) {
    torch::Tensor value = value_conv->forward(representation);
    value = value_norm->forward(value);
    value = torch::relu(value);
    value = value.view({ -1, SQUARE_NUM * CHANNEL_NUM });
    value = value_fc1->forward(value);
    value = torch::relu(value);
    value = value_fc2->forward(value);
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
    torch::Tensor policy_loss = torch::nll_loss(torch::log_softmax(policy, 1), move_teachers_tensor, {}, Reduction::None);

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
    torch::Tensor action_representation = encodeActions(moves);

    //次状態を予測
    torch::Tensor transition = predictTransition(state_representation, action_representation);

    //次状態の表現を取得
    torch::Tensor next_state_representation = encodeStates(next_state_features);

    //損失を計算
    torch::Tensor square = torch::pow(transition - next_state_representation, 2);
    torch::Tensor transition_loss = torch::sqrt(torch::sum(square, {1, 2, 3}));

    return { policy_loss, value_loss, transition_loss };
}

NeuralNetwork nn;