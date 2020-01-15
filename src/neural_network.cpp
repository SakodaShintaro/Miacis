#include"neural_network.hpp"
#include"include_switch.hpp"

//ネットワークの設定
#ifdef SHOGI
static constexpr int32_t BLOCK_NUM = 10;
#elif defined(OTHELLO)
static constexpr int32_t BLOCK_NUM = 2;
#endif

static constexpr int32_t KERNEL_SIZE = 3;
static constexpr int32_t CHANNEL_NUM = 64;
static constexpr int32_t REDUCTION = 8;
static constexpr int32_t VALUE_HIDDEN_NUM = 256;

#ifdef USE_CATEGORICAL
const std::string NeuralNetworkImpl::MODEL_PREFIX = "cat_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#else
const std::string NeuralNetworkImpl::MODEL_PREFIX = "sca_bl" + std::to_string(BLOCK_NUM) + "_ch" + std::to_string(CHANNEL_NUM);
#endif
//デフォルトで読み書きするファイル名
const std::string NeuralNetworkImpl::DEFAULT_MODEL_NAME = NeuralNetworkImpl::MODEL_PREFIX + ".model";

NeuralNetworkImpl::NeuralNetworkImpl() : device_(torch::kCUDA), fp16_(false), state_blocks_(BLOCK_NUM, nullptr) {
    state_first_conv_and_norm_ = register_module("state_first_conv_and_norm_", Conv2DwithBatchNorm(INPUT_CHANNEL_NUM, CHANNEL_NUM, KERNEL_SIZE));
    for (int32_t i = 0; i < BLOCK_NUM; i++) {
        state_blocks_[i] = register_module("state_blocks_" + std::to_string(i), ResidualBlock(CHANNEL_NUM, KERNEL_SIZE, REDUCTION));
    }
    policy_conv_ = register_module("policy_conv_", torch::nn::Conv2d(torch::nn::Conv2dOptions(CHANNEL_NUM, POLICY_CHANNEL_NUM, 1).padding(0).with_bias(true)));
    value_conv_and_norm_ = register_module("value_conv_and_norm_", Conv2DwithBatchNorm(CHANNEL_NUM, CHANNEL_NUM, 1));
    value_linear0_ = register_module("value_linear0_", torch::nn::Linear(SQUARE_NUM * CHANNEL_NUM, VALUE_HIDDEN_NUM));
    value_linear1_ = register_module("value_linear1_", torch::nn::Linear(VALUE_HIDDEN_NUM, BIN_SIZE));
}

torch::Tensor NeuralNetworkImpl::encode(const std::vector<float>& inputs) {
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    x = state_first_conv_and_norm_->forward(x);
    x = torch::relu(x);

    for (ResidualBlock& block : state_blocks_) {
        x = block->forward(x);
    }
    return x;
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetworkImpl::decode(const torch::Tensor& representation) {
    //policy
    torch::Tensor policy = policy_conv_->forward(representation);

    //value
    torch::Tensor value = value_conv_and_norm_->forward(representation);
    value = torch::relu(value);
    value = value.view({ -1, SQUARE_NUM * CHANNEL_NUM });
    value = value_linear0_->forward(value);
    value = torch::relu(value);
    value = value_linear1_->forward(value);

#ifndef USE_CATEGORICAL
#ifdef USE_SIGMOID
    value = torch::sigmoid(value);
#else
    value = torch::tanh(value);
#endif
#endif

    return { policy, value };
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetworkImpl::forward(const std::vector<float>& inputs) {
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    x = state_first_conv_and_norm_->forward(x);
    x = torch::relu(x);

    for (ResidualBlock& block : state_blocks_) {
        x = block->forward(x);
    }

    //ここから分岐
    //policy
    torch::Tensor policy = policy_conv_->forward(x);

    //value
    torch::Tensor value = value_conv_and_norm_->forward(x);
    value = torch::relu(value);
    value = value.view({ -1, SQUARE_NUM * CHANNEL_NUM });
    value = value_linear0_->forward(value);
    value = torch::relu(value);
    value = value_linear1_->forward(value);

#ifndef USE_CATEGORICAL
#ifdef USE_SIGMOID
    value = torch::sigmoid(value);
#else
    value = torch::tanh(value);
#endif
#endif

    return { policy, value };
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>>
NeuralNetworkImpl::policyAndValueBatch(const std::vector<float>& inputs) {
    std::pair<torch::Tensor, torch::Tensor> y = forward(inputs);

    uint64_t batch_size = inputs.size() / (SQUARE_NUM * INPUT_CHANNEL_NUM);

    std::vector<PolicyType> policies(batch_size);
    std::vector<ValueType> values(batch_size);

    //CPUに持ってくる
    torch::Tensor policy = y.first.cpu();
    if (fp16_) {
        torch::Half* p = policy.data<torch::Half>();
        for (uint64_t i = 0; i < batch_size; i++) {
            policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
        }
    } else {
        float* p = policy.data<float>();
        for (uint64_t i = 0; i < batch_size; i++) {
            policies[i].assign(p + i * POLICY_DIM, p + (i + 1) * POLICY_DIM);
        }
    }

#ifdef USE_CATEGORICAL
    torch::Tensor value = torch::softmax(y.second, 1).cpu();
    if (fp16_) {
        torch::Half* value_p = value.data<torch::Half>();
        for (uint64_t i = 0; i < batch_size; i++) {
            std::copy(value_p + i * BIN_SIZE, value_p + (i + 1) * BIN_SIZE, values[i].begin());
        }
    } else {
        float* value_p = value.data<float>();
        for (uint64_t i = 0; i < batch_size; i++) {
            std::copy(value_p + i * BIN_SIZE, value_p + (i + 1) * BIN_SIZE, values[i].begin());
        }
    }
#else
    //CPUに持ってくる
    torch::Tensor value = y.second.cpu();
    if (fp16_) {
        std::copy(value.data<torch::Half>(), value.data<torch::Half>() + batch_size, values.begin());
    } else {
        std::copy(value.data<float>(), value.data<float>() + batch_size, values.begin());
    }
#endif
    return { policies, values };
}

std::array<torch::Tensor, LOSS_TYPE_NUM>
NeuralNetworkImpl::loss(const std::vector<LearningData>& data, bool data_augmentation) {
    static Position pos;
    std::vector<FloatType> inputs;
    std::vector<PolicyTeacherType> policy_teachers;
    std::vector<ValueTeacherType> value_teachers;

    //データ拡張用の乱数生成器を準備
    static std::mt19937_64 engine(std::random_device{}());
    static std::uniform_int_distribution<int64_t> dist(0, Position::DATA_AUGMENTATION_PATTERN_NUM - 1);

    for (const LearningData& datum : data) {
        pos.fromStr(datum.position_str);

        if (data_augmentation) {
            int64_t rand = dist(engine);
            const std::vector<float> feature = pos.makeFeature(rand);
            inputs.insert(inputs.end(), feature.begin(), feature.end());
            //policyの教師信号も適切に回転させなければならない
            //まだ未実装なので抜ける
            policy_teachers.push_back(datum.policy);
            exit(1);
        } else {
            const std::vector<float> feature = pos.makeFeature(0);
            inputs.insert(inputs.end(), feature.begin(), feature.end());
            policy_teachers.push_back(datum.policy);
        }
        //valueは拡張しても変わらない
        value_teachers.push_back(datum.value);
    }

    std::pair<torch::Tensor, torch::Tensor> y = forward(inputs);
    torch::Tensor logits = y.first.view({ -1, POLICY_DIM });

    std::vector<float> policy_dist(policy_teachers.size() * POLICY_DIM, 0.0);
    for (uint64_t i = 0; i < policy_teachers.size(); i++) {
        for (const std::pair<int32_t, float>& e : policy_teachers[i]) {
            policy_dist[i * POLICY_DIM + e.first] = e.second;
        }
    }

    torch::Tensor policy_target = (fp16_ ? torch::tensor(policy_dist).to(device_, torch::kHalf) :
                                           torch::tensor(policy_dist).to(device_));

    policy_target = policy_target.view({ -1, POLICY_DIM });

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(logits, 1), 1, false);

#ifdef USE_CATEGORICAL
    torch::Tensor categorical_target = torch::tensor(value_teachers).to(device_);
    torch::Tensor value_loss = torch::nll_loss(torch::log_softmax(y.second, 1), categorical_target);
#else

    torch::Tensor value_t = (fp16_ ? torch::tensor(value_teachers).to(device_, torch::kHalf) :
                                     torch::tensor(value_teachers).to(device_));
    torch::Tensor value = y.second.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_loss = -value_t * torch::log(value) - (1 - value_t) * torch::log(1 - value);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t, Reduction::None);
#endif
#endif
    
    return { policy_loss, value_loss };
}

std::array<torch::Tensor, LOSS_TYPE_NUM>
NeuralNetworkImpl::mixUpLoss(const std::vector<LearningData>& data, float alpha) {
    static Position pos;

    if (data.size() % 2 == 1) {
        std::cout << "データサイズが奇数 in mixUpLoss()" << std::endl;
        exit(1);
    }
    uint64_t actual_batch_size = data.size() / 2;

    constexpr int64_t INPUT_DIM = BOARD_WIDTH * BOARD_WIDTH * INPUT_CHANNEL_NUM;
    std::vector<FloatType> inputs(actual_batch_size * INPUT_DIM);
    std::vector<float> policy_teacher_dist(actual_batch_size * POLICY_DIM, 0.0);
    std::vector<float> value_teacher_dist(actual_batch_size * BIN_SIZE, 0.0);

    static std::mt19937_64 engine(std::random_device{}());
    std::gamma_distribution<float> gamma_dist(alpha);

    for (uint64_t i = 0; i < data.size(); i += 2) {
        //i番目とi+1番目のデータをmix
        //ベータ分布はガンマ分布を組み合わせたものなのでガンマ分布からサンプリングすれば良い
        //cf. https://shoichimidorikawa.github.io/Lec/ProbDistr/gammaDist.pdf
        float gamma1 = gamma_dist(engine), gamma2 = gamma_dist(engine);
        float beta = gamma1 / (gamma1 + gamma2);

        pos.fromStr(data[i].position_str);
        std::vector<float> feature1 = pos.makeFeature(false);
        pos.fromStr(data[i + 1].position_str);
        std::vector<float> feature2 = pos.makeFeature(false);

        //入力を足し合わせる
        for (int64_t j = 0; j < INPUT_DIM; j++) {
            inputs[i / 2 * INPUT_DIM + j] = (beta * feature1[j] + (1 - beta) * feature2[j]);
        }

        //教師信号を足し合わせる
        //Policy
        for (const std::pair<int32_t, float>& pair : data[i].policy) {
            policy_teacher_dist[i / 2 * POLICY_DIM + pair.first] += beta * pair.second;
        }
        for (const std::pair<int32_t, float>& pair : data[i + 1].policy) {
            policy_teacher_dist[i / 2 * POLICY_DIM + pair.first] += (1 - beta) * pair.second;
        }

        //Value
#ifdef USE_CATEGORICAL
        value_teacher_dist[i / 2 * BIN_SIZE + data[i].value] += beta;
        value_teacher_dist[i / 2 * BIN_SIZE + data[i + 1].value] += (1 - beta);
#else
        value_teacher_dist[i / 2] += beta * data[i].value;
        value_teacher_dist[i / 2] += (1 - beta) * data[i + 1].value;
#endif
    }

    //順伝播
    std::pair<torch::Tensor, torch::Tensor> y = forward(inputs);
    torch::Tensor logits = y.first.view({ -1, POLICY_DIM });

    //Policyの損失計算
    torch::Tensor policy_target = (fp16_ ? torch::tensor(policy_teacher_dist).to(device_, torch::kHalf) :
                                           torch::tensor(policy_teacher_dist).to(device_)).view({ -1, POLICY_DIM });

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(logits, 1), 1, false);

#ifdef USE_CATEGORICAL
    torch::Tensor categorical_target = (fp16_ ? torch::tensor(value_teacher_dist).to(device_, torch::kHalf) :
                                                torch::tensor(value_teacher_dist).to(device_)).view({ -1, BIN_SIZE });
    torch::Tensor value_loss = torch::sum(-categorical_target * torch::log_softmax(y.second, 1), 1, false);;
#else

    torch::Tensor value_t = (fp16_ ? torch::tensor(value_teacher_dist).to(device_, torch::kHalf) :
                                     torch::tensor(value_teacher_dist).to(device_));
    torch::Tensor value = y.second.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_loss = -value_t * torch::log(value) - (1 - value_t) * torch::log(1 - value);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t, Reduction::None);
#endif
#endif

    return { policy_loss, value_loss };
}

std::array<torch::Tensor, LOSS_TYPE_NUM>
NeuralNetworkImpl::mixUpLossFinalLayer(const std::vector<LearningData>& data, float alpha) {
    static Position pos;

    if (data.size() % 2 == 1) {
        std::cout << "データサイズが奇数 in mixUpLossFinalLayer()" << std::endl;
        exit(1);
    }
    uint64_t actual_batch_size = data.size() / 2;

    constexpr int64_t INPUT_DIM = BOARD_WIDTH * BOARD_WIDTH * INPUT_CHANNEL_NUM;
    std::vector<FloatType> inputs1(actual_batch_size * INPUT_DIM);
    std::vector<FloatType> inputs2(actual_batch_size * INPUT_DIM);
    std::vector<FloatType> betas(actual_batch_size);
    std::vector<float> policy_teacher_dist(actual_batch_size * POLICY_DIM, 0.0);
    std::vector<float> value_teacher_dist(actual_batch_size * BIN_SIZE, 0.0);

    static std::mt19937_64 engine(std::random_device{}());
    std::gamma_distribution<float> gamma_dist(alpha);

    for (uint64_t i = 0; i < data.size(); i += 2) {
        //i番目とi+1番目のデータをmix
        //ベータ分布はガンマ分布を組み合わせたものなのでガンマ分布からサンプリングすれば良い
        //cf. https://shoichimidorikawa.github.io/Lec/ProbDistr/gammaDist.pdf
        float gamma1 = gamma_dist(engine), gamma2 = gamma_dist(engine);
        float beta = gamma1 / (gamma1 + gamma2);
        betas[i / 2] = beta;

        pos.fromStr(data[i].position_str);
        std::vector<float> feature1 = pos.makeFeature(false);
        pos.fromStr(data[i + 1].position_str);
        std::vector<float> feature2 = pos.makeFeature(false);

        //入力を準備する
        for (int64_t j = 0; j < INPUT_DIM; j++) {
            inputs1[i / 2 * INPUT_DIM + j] = feature1[j];
            inputs2[i / 2 * INPUT_DIM + j] = feature2[j];
        }

        //教師信号を足し合わせる
        //Policy
        for (const std::pair<int32_t, float>& pair : data[i].policy) {
            policy_teacher_dist[i / 2 * POLICY_DIM + pair.first] += beta * pair.second;
        }
        for (const std::pair<int32_t, float>& pair : data[i + 1].policy) {
            policy_teacher_dist[i / 2 * POLICY_DIM + pair.first] += (1 - beta) * pair.second;
        }

        //Value
#ifdef USE_CATEGORICAL
        value_teacher_dist[i / 2 * BIN_SIZE + data[i].value] += beta;
        value_teacher_dist[i / 2 * BIN_SIZE + data[i + 1].value] += (1 - beta);
#else
        value_teacher_dist[i / 2] += beta * data[i].value;
        value_teacher_dist[i / 2] += (1 - beta) * data[i + 1].value;
#endif
    }

    //順伝播
    torch::Tensor representation1 = encode(inputs1);
    torch::Tensor representation2 = encode(inputs2);
    torch::Tensor betas_tensor = torch::tensor(betas).to(device_).view({ -1, 1, 1, 1 });

    torch::Tensor mixed_representation = betas_tensor * representation1 + (1 - betas_tensor) * representation2;
    std::pair<torch::Tensor, torch::Tensor> y = decode(mixed_representation);
    torch::Tensor logits = y.first.view({ -1, POLICY_DIM });

    //Policyの損失計算
    torch::Tensor policy_target = (fp16_ ? torch::tensor(policy_teacher_dist).to(device_, torch::kHalf) :
                                           torch::tensor(policy_teacher_dist).to(device_)).view({ -1, POLICY_DIM });

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(logits, 1), 1, false);

#ifdef USE_CATEGORICAL
    torch::Tensor categorical_target = (fp16_ ? torch::tensor(value_teacher_dist).to(device_, torch::kHalf) :
                                                torch::tensor(value_teacher_dist).to(device_)).view({ -1, BIN_SIZE });
    torch::Tensor value_loss = torch::sum(-categorical_target * torch::log_softmax(y.second, 1), 1, false);;
#else

    torch::Tensor value_t = (fp16_ ? torch::tensor(value_teacher_dist).to(device_, torch::kHalf) :
                             torch::tensor(value_teacher_dist).to(device_));
    torch::Tensor value = y.second.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_loss = -value_t * torch::log(value) - (1 - value_t) * torch::log(1 - value);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_t, Reduction::None);
#endif
#endif

    return { policy_loss, value_loss };
}

void NeuralNetworkImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_));
}