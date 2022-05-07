#include "infer_model.hpp"
#include "../common.hpp"
#include "../learn/learn.hpp"
#include "../shogi/position.hpp"
#include "calibrator.hpp"
#include <torch/torch.h>
using namespace Shogi;

class Logger : public nvinfer1::ILogger {
    const char* error_type(Severity severity) {
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
            return "[F] ";
        case Severity::kERROR:
            return "[E] ";
        case Severity::kWARNING:
            return "[W] ";
        case Severity::kINFO:
            return "[I] ";
        case Severity::kVERBOSE:
            return "[V] ";
        default:
            assert(0);
            return "";
        }
    }
    void log(Severity severity, const char* msg) noexcept {
        if (severity == Severity::kINTERNAL_ERROR) {
            std::cerr << error_type(severity) << msg << std::endl;
        }
    }
} gLogger;

void InferModel::convertOnnxToEngine(const std::string& onnx_path, const FP_MODE fp_mode, const int64_t opt_batch_size,
                                     const std::string& calibration_data_path) {
    // 最大バッチサイズは目的バッチサイズの2倍で決め打ち
    const int64_t max_batch_size = opt_batch_size * 2;

    // build
    auto builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        throw std::runtime_error("createInferBuilder");
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        throw std::runtime_error("createNetworkV2");
    }

    auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        throw std::runtime_error("createBuilderConfig");
    }

    auto parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        throw std::runtime_error("createParser");
    }

    auto parsed = parser->parseFromFile(onnx_path.c_str(), (int)nvinfer1::ILogger::Severity::kINTERNAL_ERROR);
    if (!parsed) {
        throw std::runtime_error("parseFromFile");
    }

    builder->setMaxBatchSize(max_batch_size);

    config->setMaxWorkspaceSize(1ull << 31);

    std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
    if (builder->platformHasFastInt8() && fp_mode == INT8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        calibrator.reset(new Int8EntropyCalibrator2(onnx_path, max_batch_size, calibration_data_path));
        config->setInt8Calibrator(calibrator.get());
    } else if (builder->platformHasFastFp16() && fp_mode == FP16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
        std::cout << "Fail to decide precision" << std::endl;
        std::exit(1);
    }

    assert(network->getNbInputs() == 1);
    assert(network->getNbOutputs() == 2);

    // Optimization Profiles
    auto profile = builder->createOptimizationProfile();
    nvinfer1::Dims dims = network->getInput(0)->getDimensions();
    dims.d[0] = 1;
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, dims);
    dims.d[0] = opt_batch_size;
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, dims);
    dims.d[0] = max_batch_size;
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, dims);
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        throw std::runtime_error("buildEngineWithConfig");
    }

    // serializing a model
    auto serialized_engine = InferUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
    if (!serialized_engine) {
        throw std::runtime_error("Engine serialization failed");
    }

    // 拡張子を.onnxから.engineに変えたものとして保存
    assert(onnx_path.substr(onnx_path.size() - 5) == ".onnx");
    const std::string engine_path = onnx_path.substr(0, onnx_path.size() - 5) + ".engine";
    std::ofstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
        throw std::runtime_error("Cannot open engine file");
    }
    engine_file.write(static_cast<char*>(serialized_engine->data()), serialized_engine->size());
    if (engine_file.fail()) {
        throw std::runtime_error("Cannot open engine file");
    }
}

InferModel::~InferModel() {
    checkCudaErrors(cudaFree(x1_dev_));
    checkCudaErrors(cudaFree(y1_dev_));
    checkCudaErrors(cudaFree(y2_dev_));

    //destroyを入れるとむしろSegmentation Faultが発生するのでコメントアウト
    //しかし何もしていないとリークしていそうだが、それは良いのか？
    //engine_->destroy();
    //context_->destroy();
}

void InferModel::load(int64_t gpu_id, const SearchOptions& search_option) {
    gpu_id_ = gpu_id;
    opt_batch_size_ = search_option.search_batch_size;
    max_batch_size_ = search_option.search_batch_size * 2;
    // Create host and device buffers
    if (x1_dev_ == nullptr) {
        checkCudaErrors(cudaMalloc((void**)&x1_dev_, max_batch_size_ * sizeof(float) * INPUT_CHANNEL_NUM * SQUARE_NUM));
    }
    if (y1_dev_ == nullptr) {
        checkCudaErrors(cudaMalloc((void**)&y1_dev_, max_batch_size_ * sizeof(float) * POLICY_DIM));
    }
    if (y2_dev_ == nullptr) {
        checkCudaErrors(cudaMalloc((void**)&y2_dev_, max_batch_size_ * sizeof(float) * BIN_SIZE));
    }
    if (context_ != nullptr) {
        context_->destroy();
    }
    if (engine_ != nullptr) {
        engine_->destroy();
    }

    input_bindings_ = { x1_dev_, y1_dev_, y2_dev_ };

    const std::string engine_path = search_option.model_name;

    if (engine_path.substr(engine_path.size() - 7) != ".engine") {
        std::cerr << "エンジンパスは拡張子.engineである必要があります: " << engine_path << std::endl;
        std::exit(1);
    }
    std::ifstream serialized_file(engine_path, std::ios::binary);
    if (!serialized_file.is_open()) {
        std::cerr << "!serialized_file.is_open(): " << engine_path << std::endl;
        std::exit(1);
    }

    // deserializing a model
    serialized_file.seekg(0, std::ios_base::end);
    const size_t modelSize = serialized_file.tellg();
    serialized_file.seekg(0, std::ios_base::beg);
    std::unique_ptr<char[]> blob(new char[modelSize]);
    serialized_file.read(blob.get(), modelSize);
    auto runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    engine_ = runtime->deserializeCudaEngine(blob.get(), modelSize, nullptr);

    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("createExecutionContext");
    }
}

void InferModel::forward(const int64_t batch_size, const float* x1, void* y1, void* y2) {
    checkCudaErrors(cudaMemcpy(x1_dev_, x1, batch_size * sizeof(float) * INPUT_CHANNEL_NUM * SQUARE_NUM, cudaMemcpyHostToDevice));

    nvinfer1::Dims dims = engine_->getBindingDimensions(0);
    dims.d[0] = batch_size;
    context_->setBindingDimensions(0, dims);
    context_->executeV2(input_bindings_.data());

    checkCudaErrors(cudaMemcpy(y1, y1_dev_, batch_size * sizeof(float) * POLICY_DIM, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(y2, y2_dev_, batch_size * sizeof(float) * BIN_SIZE, cudaMemcpyDeviceToHost));
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>> InferModel::policyAndValueBatch(const std::vector<float>& inputs) {
    constexpr int64_t element_num = INPUT_CHANNEL_NUM * SQUARE_NUM;
    const int64_t batch_size = inputs.size() / element_num;

    std::vector<float> policy_buffer(batch_size * POLICY_DIM);
    std::vector<float> value_buffer(batch_size * BIN_SIZE);

    forward(batch_size, inputs.data(), policy_buffer.data(), value_buffer.data());

    std::vector<PolicyType> policy(batch_size);
    std::vector<ValueType> value(batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
        //policyをコピー
        policy[i].insert(policy[i].begin(), policy_buffer.begin() + i * POLICY_DIM, policy_buffer.begin() + (i + 1) * POLICY_DIM);

#ifdef USE_CATEGORICAL
        std::vector<float> softmaxed(value_buffer.begin() + i * BIN_SIZE, value_buffer.begin() + (i + 1) * BIN_SIZE);
        softmaxed = softmax(softmaxed);
        std::copy_n(softmaxed.begin(), BIN_SIZE, value[i].begin());
#else
#ifdef USE_SIGMOID
        //Sigmoidのときはそのままで良い
#else
        //tanh想定のときはvalueの規格を修正
        value_buffer[i] = value_buffer[i] * 2 - 1;
#endif
#endif
    }

    return std::make_pair(policy, value);
}

std::array<torch::Tensor, LOSS_TYPE_NUM> InferModel::validLoss(const std::vector<LearningData>& data) {
    static Position pos;
    std::vector<float> inputs;
    std::vector<float> policy_teachers(data.size() * POLICY_DIM, 0.0);
    std::vector<ValueTeacherType> value_teachers;

    for (uint64_t i = 0; i < data.size(); i++) {
        pos.fromStr(data[i].position_str);

        //入力
        const std::vector<float> feature = pos.makeFeature();
        inputs.insert(inputs.end(), feature.begin(), feature.end());

        //policyの教師信号
        for (const std::pair<int32_t, float>& e : data[i].policy) {
            policy_teachers[i * POLICY_DIM + e.first] = e.second;
        }

        //valueの教師信号
        value_teachers.push_back(data[i].value);
    }

    const int64_t batch_size = data.size();
    std::vector<float> policy_buffer(batch_size * POLICY_DIM);
    std::vector<float> value_buffer(batch_size * BIN_SIZE);
    forward(batch_size, inputs.data(), policy_buffer.data(), value_buffer.data());

    torch::Tensor policy_target = torch::tensor(policy_teachers).view({ batch_size, POLICY_DIM });
    torch::Tensor value_target = torch::tensor(value_teachers);

    torch::Tensor policy_logit = torch::tensor(policy_buffer).view({ batch_size, POLICY_DIM });
    torch::Tensor value = torch::tensor(value_buffer).view({ batch_size, BIN_SIZE });
    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(policy_logit, 1), 1, false);

#ifdef USE_CATEGORICAL
    //Valueの分布を取得
    torch::Tensor value_cat = torch::softmax(value, 1);

    //i番目の要素が示す値はMIN_SCORE + (i + 0.5) * VALUE_WIDTH
    std::vector<float> each_value;
    for (int64_t i = 0; i < BIN_SIZE; i++) {
        each_value.emplace_back(MIN_SCORE + (i + 0.5) * VALUE_WIDTH);
    }
    torch::Tensor each_value_tensor = torch::tensor(each_value);

    //Categorical分布と内積を取ることで期待値を求める
    value = (each_value_tensor * value_cat).sum(1);

    //target側も数値に変換
    value_target = MIN_SCORE + (value_target + 0.5f) * VALUE_WIDTH;

#else //Scalarモデルの場合
    value = value.view(-1);
#endif

    //Sigmoidのときはbce, tanhのときはmse
#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_target, {}, torch::Reduction::None);
#else
    torch::Tensor value_loss = torch::mse_loss(value, value_target, torch::Reduction::None);
#endif

    return { policy_loss, value_loss };
}