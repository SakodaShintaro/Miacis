#include"neural_network.hpp"

static constexpr int32_t STATE_BLOCK_NUM = 10;
static constexpr int32_t ACTION_BLOCK_NUM = 4;
static constexpr int32_t PREDICT_TRANSITION_BLOCK_NUM = 4;
static constexpr int32_t CHANNEL_NUM = 64;
static constexpr int32_t VALUE_HIDDEN_NUM = 256;
static constexpr int32_t KERNEL_SIZE = 3;
static constexpr int32_t ACTION_FEATURE_CHANNEL_NUM = 4 + 18;
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
    reconstruct_hand_conv_and_norm_ = register_module("reconstruct_hand_conv_and_norm_", Conv2DwithBatchNorm(CHANNEL_NUM, 1, 1));
    reconstruct_hand_linear_ = register_module("reconstruct_hand_linear_", torch::nn::Linear(SQUARE_NUM, HAND_PIECE_KIND_NUM * 2));
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

        //この行動の手番
        Color color = pieceToColor(move.subject());

        //1ch:toの位置に1を立てる
        Square to = (color == BLACK ? move.to() : InvSquare[move.to()]);
        curr_move_feature[SquareToNum[to]] = 1;

        //2ch:fromの位置に1を立てる.持ち駒から打つ手ならなし
        //3ch:持ち駒から打つ手なら全て1
        if (move.isDrop()) {
            for (Square sq : SquareList) {
                curr_move_feature[2 * SQUARE_NUM + SquareToNum[sq]] = 1;
            }
        } else {
            Square from = (color == BLACK ? move.from() : InvSquare[move.from()]);
            curr_move_feature[SQUARE_NUM + SquareToNum[from]] = 1;
        }

        //4ch:成りなら全て1
        if (move.isPromote()) {
            for (Square sq : SquareList) {
                curr_move_feature[3 * SQUARE_NUM + SquareToNum[sq]] = 1;
            }
        }

        //5ch以降:駒の種類に合わせたところだけ全て1
        for (Square sq : SquareList) {
            Piece p = (color == BLACK ? move.subject() : oppositeColor(move.subject()));
            curr_move_feature[(4 + PieceToNum[p]) * SQUARE_NUM + SquareToNum[sq]] = 1;
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
    for (ResidualBlock& block : action_encoder_blocks_) {
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
    std::array<std::vector<float>, LEARNING_RANGE> state_features;

    //再構成損失を計算する際の教師情報
    //入力情報とはやや形式が違うものを使う
    std::array<std::vector<FloatType>, LEARNING_RANGE> board_teacher_vec;
    std::array<std::vector<FloatType>, LEARNING_RANGE> hand_teacher_vec;

    //Policyの教師
    std::array<std::vector<int64_t>, LEARNING_RANGE> move_teachers;

    //Valueの教師
    std::array<std::vector<ValueTeacherType>, LEARNING_RANGE> value_teachers;

    //行動の表現
    std::array<std::vector<Move>, LEARNING_RANGE> moves;

    for (const LearningData& datum : data) {
        //現局面の特徴量を取得
        pos.loadSFEN(datum.SFEN);

        for (int64_t i = 0; i < LEARNING_RANGE; i++) {
            std::vector<float> curr_state_feature = pos.makeFeature();
            state_features[i].insert(state_features[i].end(), curr_state_feature.begin(), curr_state_feature.end());

            //現局面の再構成損失の教師を取得
            std::pair<std::vector<float>, std::vector<float>> reconstruct_teacher = pos.makeReconstructTeacher();
            board_teacher_vec[i].insert(board_teacher_vec[i].end(), reconstruct_teacher.first.begin(),
                                        reconstruct_teacher.first.end());
            hand_teacher_vec[i].insert(hand_teacher_vec[i].end(), reconstruct_teacher.second.begin(),
                                       reconstruct_teacher.second.end());

            //Policyの教師
            move_teachers[i].push_back(datum.moves[i].toLabel());

            //Valueの教師
            value_teachers[i].push_back(datum.value[i]);

            //行動の表現取得のため
            moves[i].push_back(datum.moves[i]);

            //次の局面へ遷移
            pos.doMove(datum.moves[i]);
        }
    }

    std::array<torch::Tensor, LOSS_TYPE_NUM> losses{};

    //まずは現局面で初期化
    torch::Tensor next_state_representation = encodeStates(state_features[0]);

    for (int64_t i = 0; i < LEARNING_RANGE; i++) {
        //i == 0のときはすぐ上の行で、それ以外のときは下のループ中で状態をエンコードしているのでそれを再利用
        torch::Tensor state_representation = next_state_representation;

        losses[i * STANDARD_LOSS_TYPE_NUM + POLICY_LOSS_INDEX] = policyLoss(state_representation, move_teachers[i]);
        losses[i * STANDARD_LOSS_TYPE_NUM + VALUE_LOSS_INDEX] = valueLoss(state_representation, value_teachers[i]);
        losses[i * STANDARD_LOSS_TYPE_NUM + RECONSTRUCT_LOSS_INDEX] = reconstructLoss(state_representation, board_teacher_vec[i], hand_teacher_vec[i]);

        if (i < LEARNING_RANGE - 1) {
            //----------------------
            //  遷移予測の損失計算
            //----------------------
            //行動の表現を取得
            torch::Tensor action_representation = encodeActions(moves[i]);

            //次状態を予測
            torch::Tensor predicted_state_representation = predictTransition(state_representation.detach(), action_representation);

            //次状態の表現を取得
            next_state_representation = encodeStates(state_features[i + 1]);

            //損失を計算
            losses[i * STANDARD_LOSS_TYPE_NUM + TRANS_LOSS_INDEX] = transitionLoss(predicted_state_representation, next_state_representation.detach());

            //--------------------------------------
            //  遷移後の表現から予測するPolicyの損失
            //--------------------------------------
            losses[i * STANDARD_LOSS_TYPE_NUM + NEXT_POLICY_LOSS_INDEX] = policyLoss(predicted_state_representation, move_teachers[i + 1]);
            losses[i * STANDARD_LOSS_TYPE_NUM + NEXT_VALUE_LOSS_INDEX] = valueLoss(predicted_state_representation, value_teachers[i + 1]);
            losses[i * STANDARD_LOSS_TYPE_NUM + NEXT_RECONSTRUCT_LOSS_INDEX] = reconstructLoss(predicted_state_representation, board_teacher_vec[i + 1], hand_teacher_vec[i + 1]);
        }
    }

    return losses;
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

void NeuralNetworkImpl::reconstruct(const torch::Tensor& representation, Color color) {
    //盤面の再構成
    torch::Tensor board = reconstruct_board_conv_->forward(representation);

    //最大のものを取得
    torch::Tensor piece = torch::argmax(board, 1);

    for (int64_t rank = 0; rank < 9; rank++) {
        for (int64_t file = 8; file >= 0; file--) {
            //後手番のときは反転されているので逆側を参照
            int64_t value = (color == BLACK ? piece[0][file][rank].item<int64_t>()
                                            : piece[0][8 - file][8 - rank].item<int64_t>());
            Piece p = (value == PieceList.size() ? EMPTY : PieceList[value]);

            if (color == WHITE) {
                p = oppositeColor(p);
            }
            std::cout << PieceToSfenStr[p];
        }
        std::cout << std::endl;
    }

    //手駒の再構成
    torch::Tensor hand = reconstruct_hand_conv_and_norm_->forward(representation);
    hand = reconstruct_hand_linear_->forward(hand.flatten(1));

    std::cout << std::fixed;
    for (int64_t c : { BLACK, WHITE }) {
        std::cout << (c == BLACK ? "先手: " : "後手: ");
        for (int64_t i = 0; i < HAND_PIECE_KIND_NUM; i++) {
            std::cout << PieceToStr[i + 1] << std::setw(5) << std::setprecision(2) << hand[0][(c != color) * HAND_PIECE_KIND_NUM + i].item<float>() << " ";
        }
        std::cout << std::endl;
    }
}

torch::Tensor NeuralNetworkImpl::policyLoss(const torch::Tensor& state_representation, const std::vector<int64_t>& policy_teacher) {
    torch::Tensor policy = decodePolicy(state_representation);
    torch::Tensor policy_teacher_tensor = torch::tensor(policy_teacher).to(device_);
    return torch::nll_loss(torch::log_softmax(policy, 1), policy_teacher_tensor, {}, Reduction::None);
}

torch::Tensor NeuralNetworkImpl::valueLoss(const torch::Tensor& state_representation, const std::vector<ValueTeacherType>& value_teacher) {
    //Valueを取得
    torch::Tensor value = decodeValue(state_representation).view(-1);

    //教師の構築
#ifdef USE_CATEGORICAL
    torch::Tensor value_teacher_tensor = torch::tensor(value_teacher).to(device_);
#else
#ifdef USE_HALF_FLOAT
    torch::Tensor value_teacher_tensor = torch::tensor(value_teacher).to(device_, torch::kHalf);
#else
    torch::Tensor value_teacher_tensor = torch::tensor(value_teacher).to(device_);
#endif
#endif

    //損失計算
#ifdef USE_CATEGORICAL
    return torch::nll_loss(torch::log_softmax(value, 1), value_teacher_tensor);
#else
#ifdef USE_SIGMOID
    return -value_teacher_tensor * torch::log(value) - (1 - value_teacher_tensor) * torch::log(1 - value);
#else
    return torch::mse_loss(value, value_teacher_tensor, Reduction::None);
#endif
#endif
}

torch::Tensor NeuralNetworkImpl::reconstructLoss(const torch::Tensor& state_representation,
                                                 const std::vector<FloatType>& board_teacher,
                                                 const std::vector<FloatType>& hand_teacher) {
    //盤面の再構成
    torch::Tensor board = reconstruct_board_conv_->forward(state_representation);
    //教師Tensorの構成
    torch::Tensor board_teacher_tensor = torch::tensor(board_teacher)
            .view({ -1, PIECE_KIND_NUM * 2 + 1, 9, 9 })
            .to(device_);

    torch::Tensor board_reconstruct_loss = -board_teacher_tensor * torch::log_softmax(board, 1);
    //駒種方向に和を取ることで各マスについて交差エントロピーを計算したことになる
    board_reconstruct_loss = board_reconstruct_loss.sum(1);
    //各マスについての交差エントロピーを全マスについて平均化する
    board_reconstruct_loss = board_reconstruct_loss.mean({1, 2});

    //手駒の再構成
    torch::Tensor hand = reconstruct_hand_conv_and_norm_->forward(state_representation);
    hand = reconstruct_hand_linear_->forward(hand.flatten(1));

    //自乗誤差
    torch::Tensor hand_teacher_tensor = torch::tensor(hand_teacher).view({ -1, HAND_PIECE_KIND_NUM * 2 }).to(device_);

    //バッチ以外だけ平均化したいのを上手くやる方法がわからないのでReductionはNoneで手動meanを取っている
    torch::Tensor hand_reconstruct_loss = torch::mse_loss(hand, hand_teacher_tensor, Reduction::None).mean({ 1});

    return board_reconstruct_loss + hand_reconstruct_loss;
}

NeuralNetwork nn;
