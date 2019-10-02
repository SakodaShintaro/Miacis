#include"neural_network.hpp"

static constexpr int32_t STATE_BLOCK_NUM = 10;
static constexpr int32_t ACTION_BLOCK_NUM = 4;
static constexpr int32_t PREDICT_TRANSITION_BLOCK_NUM = 4;
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

torch::Tensor Conv2DwithBatchNormImpl::forward(const torch::Tensor& x) {
    torch::Tensor t = x;
    t = conv_->forward(t);
    t = norm_->forward(t);
    return t;
}

ResidualBlockImpl::ResidualBlockImpl(int64_t channel_num, int64_t kernel_size, int64_t reduction) {
    conv_and_norm0_ = register_module("conv_and_norm0_", Conv2DwithBatchNorm(channel_num, channel_num, kernel_size));
    conv_and_norm1_ = register_module("conv_and_norm1_", Conv2DwithBatchNorm(channel_num, channel_num, kernel_size));
    linear0_ = register_module("linear0_", torch::nn::Linear(torch::nn::LinearOptions(channel_num, channel_num / reduction).with_bias(false)));
    linear1_ = register_module("linear1_", torch::nn::Linear(torch::nn::LinearOptions(channel_num / reduction, channel_num).with_bias(false)));
}

torch::Tensor ResidualBlockImpl::forward(const torch::Tensor& x) {
    torch::Tensor t = x;

    t = conv_and_norm0_->forward(t);
    t = torch::relu(t);
    t = conv_and_norm1_->forward(t);

    //SENet構造
    auto y = torch::avg_pool2d(t, {9, 9});
    y = y.view({-1, CHANNEL_NUM});
    y = linear0_->forward(y);
    y = torch::relu(y);
    y = linear1_->forward(y);
    y = torch::sigmoid(y);
    y = y.view({-1, CHANNEL_NUM, 1, 1});
    t = t * y;

    t = torch::relu(x + t);
    return t;
}

NeuralNetworkImpl::NeuralNetworkImpl() : device_(torch::kCUDA), fp16_(false),
                                         state_encoder_blocks_(STATE_BLOCK_NUM, nullptr),
                                         action_encoder_blocks_(ACTION_BLOCK_NUM, nullptr),
                                         predict_transition_blocks_(PREDICT_TRANSITION_BLOCK_NUM, nullptr) {
    state_encoder_first_conv_and_norm_ = register_module("state_encoder_first_conv_and_norm_", Conv2DwithBatchNorm(STATE_FEATURE_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < STATE_BLOCK_NUM; i++) {
        state_encoder_blocks_[i] = register_module("state_encoder_blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }

    action_encoder_first_conv_and_norm_ = register_module("action_encoder_first_conv_and_norm_", Conv2DwithBatchNorm(ACTION_FEATURE_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < ACTION_BLOCK_NUM; i++) {
        action_encoder_blocks_[i] = register_module("action_encoder_blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }

    predict_transition_first_conv_and_norm_ = register_module("predict_transition_first_conv_and_norm_", Conv2DwithBatchNorm(CHANNEL_NUM + CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < PREDICT_TRANSITION_BLOCK_NUM; i++) {
        predict_transition_blocks_[i] = register_module("predict_transition_blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }

    policy_conv_ = register_module("policy_conv_", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, POLICY_CHANNEL_NUM, 1).padding(0).with_bias(true)));
    value_conv_and_norm_ = register_module("value_conv_and_norm_", Conv2DwithBatchNorm(CHANNEL_NUM, CHANNEL_NUM, 1));
    value_linear0_ = register_module("value_linear0_", torch::nn::Linear(SQUARE_NUM * CHANNEL_NUM, VALUE_HIDDEN_NUM));
    value_linear1_ = register_module("value_linear1_", torch::nn::Linear(VALUE_HIDDEN_NUM, BIN_SIZE));

    reconstruct_board_conv_ = register_module("reconstruct_board_conv_", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, PIECE_KIND_NUM * 2 + 1, 3).padding(1).with_bias(true)));
    reconstruct_hand_linear_ = register_module("reconstruct_hand_linear_", torch::nn::Linear(SQUARE_NUM * CHANNEL_NUM, HAND_PIECE_KIND_NUM * 2));
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
    for (uint64_t i = 0; i < batch_size; i++) {
        policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
    }

#ifdef USE_CATEGORICAL
    torch::Tensor value = torch::softmax(decodeValue(representation), 1).cpu();
#ifdef USE_HALF_FLOAT
    torch::Half* value_p = value.data<torch::Half>();
#else
    float* value_p = value.data<float>();
#endif
    for (uint64_t i = 0; i < batch_size; i++) {
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

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
NeuralNetworkImpl::decodePolicyAndValueBatch(const std::vector<float>& state_rep) {
    int64_t batch_size = state_rep.size() / (SQUARE_NUM * CHANNEL_NUM);
    torch::Tensor representation = torch::tensor(state_rep);
    representation = representation.view({ batch_size, CHANNEL_NUM, 9, 9 }).to(device_);

    std::vector<PolicyType> policies(batch_size);
    std::vector<ValueType> values(batch_size);

    //CPUに持ってくる
    torch::Tensor policy = decodePolicy(representation).cpu();
#ifdef USE_HALF_FLOAT
    torch::Half* p = policy.data<torch::Half>();
#else
    float* p = policy.data<float>();
#endif
    for (int64_t i = 0; i < batch_size; i++) {
        policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
    }

#ifdef USE_CATEGORICAL
    torch::Tensor value = torch::softmax(decodeValue(representation), 1).cpu();
#ifdef USE_HALF_FLOAT
    torch::Half* value_p = value.data<torch::Half>();
#else
    float* value_p = value.data<float>();
#endif
    for (int64_t i = 0; i < batch_size; i++) {
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

torch::Tensor NeuralNetworkImpl::predictTransition(const torch::Tensor& state_representations,
                                                   const torch::Tensor& move_representations) {
    torch::Tensor x = torch::cat({state_representations, move_representations}, 1);
    x = predict_transition_first_conv_and_norm_->forward(x);
    for (auto& blocks : predict_transition_blocks_) {
        x = blocks->forward(x);
    }
    return x;
}

std::vector<FloatType> NeuralNetworkImpl::predictTransition(const std::vector<FloatType>& state_rep, Move move) {
    torch::Tensor state_rep_tensor = torch::tensor(state_rep).view({ 1, CHANNEL_NUM, 9, 9 }).to(device_);
    torch::Tensor move_rep_tensor = encodeActions({ move });
    torch::Tensor predicted = predictTransition(state_rep_tensor, move_rep_tensor).cpu();
    return std::vector<FloatType>(predicted.data<FloatType>(), predicted.data<FloatType>() + SQUARE_NUM * CHANNEL_NUM);
}

torch::Tensor NeuralNetworkImpl::encodeStates(const std::vector<float>& inputs) {
#ifdef USE_HALF_FLOAT
    torch::Tensor x = torch::tensor(inputs).to(device_, torch::kHalf);
#else
    torch::Tensor x = torch::tensor(inputs).to(device_);
#endif
    x = x.view({ -1, STATE_FEATURE_CHANNEL_NUM, 9, 9 });
    x = state_encoder_first_conv_and_norm_->forward(x);
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
    torch::Tensor x = action_encoder_first_conv_and_norm_->forward(move_features_tensor);
    for (auto& block : action_encoder_blocks_) {
        x = block->forward(x);
    }
    return x;
}

torch::Tensor NeuralNetworkImpl::decodePolicy(const torch::Tensor& representation) {
    torch::Tensor policy = policy_conv_->forward(representation);
    return policy.view({ -1, POLICY_DIM });
}

torch::Tensor NeuralNetworkImpl::decodeValue(const torch::Tensor& representation) {
    torch::Tensor value = value_conv_and_norm_->forward(representation);
    value = torch::relu(value);
    value = value.view({ -1, SQUARE_NUM * CHANNEL_NUM });
    value = value_linear0_->forward(value);
    value = torch::relu(value);
    value = value_linear1_->forward(value);
#ifdef USE_SIGMOID
    value = torch::sigmoid(value);
#else
    value = torch::tanh(value);
#endif
    return value;
}

std::array<torch::Tensor, LOSS_TYPE_NUM> NeuralNetworkImpl::loss(const std::vector<LearningData>& data) {
    static Position pos;

    //局面の特徴量を取得
    std::vector<float> curr_state_features;
    std::vector<float> next_state_features;

    //再構成損失を計算する際の教師情報
    //入力情報とはやや形式が違うものを使う
    std::vector<FloatType> board_teacher_vec;
    std::vector<FloatType> hand_teacher_vec;

    for (const LearningData& datum : data) {
        //現局面の特徴量を取得
        pos.loadSFEN(datum.SFEN);
        std::vector<float> curr_state_feature = pos.makeFeature();
        curr_state_features.insert(curr_state_features.end(), curr_state_feature.begin(), curr_state_feature.end());

        //現局面の再構成損失の教師を取得
        std::pair<std::vector<float>, std::vector<float>> reconstruct_teacher = pos.makeReconstructTeacher();
        board_teacher_vec.insert(board_teacher_vec.end(), reconstruct_teacher.first.begin(), reconstruct_teacher.first.end());
        hand_teacher_vec.insert(hand_teacher_vec.end(), reconstruct_teacher.second.begin(), reconstruct_teacher.second.end());

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
    torch::Tensor value_loss = -value_teachers_tensor * torch::log(value)
                        - (1 - value_teachers_tensor) * torch::log(1 - value);
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
    torch::Tensor predicted_state_representation = predictTransition(state_representation.detach(), action_representation);

    //次状態の表現を取得
    //勾配を止める
    torch::Tensor next_state_representation = encodeStates(next_state_features).detach();

    //損失を計算
    torch::Tensor transition_loss = transitionLoss(predicted_state_representation, next_state_representation);

    //--------------------
    //  再構成損失の計算
    //--------------------
    //盤面の再構成
    torch::Tensor board = reconstruct_board_conv_->forward(state_representation);
    //各マスについて駒次元の方にSoftmaxを計算
    board = torch::softmax(board, 3);
    //実際の駒の配置と照らし合わせて損失計算
    //教師Tensorの構成
    torch::Tensor board_teacher = torch::tensor(board_teacher_vec);
    board_teacher = board_teacher.view({ -1, 9, 9, PIECE_KIND_NUM * 2 + 1 }).to(device_);
    torch::Tensor board_reconstruct_loss = board_teacher * torch::log(board);
    //駒種方向に和を取ることで各マスについて交差エントロピーを計算したことになる
    board_reconstruct_loss = board_reconstruct_loss.sum(3);
    //各マスについての交差エントロピーを全マスについて平均化する
    board_reconstruct_loss = board_reconstruct_loss.mean({1, 2});
    std::cout << board_reconstruct_loss << std::endl;

    //手駒の再構成
    torch::Tensor hand = reconstruct_hand_linear_->forward(state_representation.flatten(1));
    //シグモイド関数をかけて[0, 1]の範囲に収める
    hand = torch::sigmoid(hand);
    //各持ち駒のあり得る枚数かけて範囲を変える
    //e.g.) 歩なら[0, 18], 銀なら[0, 4], 飛車なら[0, 2]
    torch::Tensor hand_max_coeff = torch::tensor({18, 4, 4, 4, 4, 2, 2, 18, 4, 4, 4, 4, 2, 2});
    hand = hand_max_coeff * hand;
    //自乗誤差
    torch::Tensor hand_teacher = torch::tensor(hand_teacher_vec);
    hand_teacher = hand_teacher.view({ -1, 9, 9, HAND_PIECE_KIND_NUM * 2 }).to(device_);
    torch::Tensor hand_reconstruct_loss = torch::mse_loss(hand, hand_teacher, Reduction::None);
    std::cout << hand_reconstruct_loss << std::endl;

    torch::Tensor reconstruct_loss = board_reconstruct_loss + hand_reconstruct_loss;

    return { policy_loss, value_loss, transition_loss, reconstruct_loss };
}

void NeuralNetworkImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_));
}

torch::Tensor NeuralNetworkImpl::transitionLoss(const torch::Tensor& predict, const torch::Tensor& ground_truth) {
    //要素ごとの自乗誤差の和
    //最後にsqrtを取ればユークリッド距離になるが、取ったほうが良いかどうかは不明
//    torch::Tensor diff = predict - ground_truth;
//    torch::Tensor square = torch::pow(diff, 2);
//    torch::Tensor sum = torch::sum(square, { 1, 2, 3 });
//    return torch::sqrt(sum);

    //平均自乗誤差にする
    torch::Tensor diff = predict - ground_truth;
    torch::Tensor square = torch::pow(diff, 2);
    return torch::mean(square, {1, 2, 3});
}

NeuralNetwork nn;
