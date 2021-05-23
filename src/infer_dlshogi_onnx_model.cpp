#include "infer_dlshogi_onnx_model.hpp"

#ifdef DLSHOGI

#include "learn.hpp"

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
    void log(Severity severity, const char* msg) {
        if (severity == Severity::kINTERNAL_ERROR) {
            std::cerr << error_type(severity) << msg << std::endl;
        }
    }
} gLogger;

InferDLShogiOnnxModel::~InferDLShogiOnnxModel() {
    checkCudaErrors(cudaFree(x1_dev_));
    checkCudaErrors(cudaFree(x2_dev_));
    checkCudaErrors(cudaFree(y1_dev_));
    checkCudaErrors(cudaFree(y2_dev_));

    //destroyを入れるとむしろSegmentation Faultが発生するのでコメントアウト
    //しかし何もしていないとリークしていそうだが、それは良いのか？
    //engine_->destroy();
    //context_->destroy();
}

void InferDLShogiOnnxModel::build(const std::string& onnx_filename) {
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

    auto parsed = parser->parseFromFile(onnx_filename.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING);
    if (!parsed) {
        throw std::runtime_error("parseFromFile");
    }

    builder->setMaxBatchSize(max_batch_size_);

    //MByte単位で指定
    config->setMaxWorkspaceSize(64 * (1ull << 20));

    std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
    if (builder->platformHasFastInt8()) {
        std::string calibration_cache_filename = std::string(onnx_filename) + ".calibcache";
        std::ifstream calibcache(calibration_cache_filename);
        if (calibcache.is_open()) {
            calibcache.close();

            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            calibrator.reset(new Int8EntropyCalibrator2(onnx_filename.c_str(), 1));
            config->setInt8Calibrator(calibrator.get());
        } else if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
    } else if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    assert(network->getNbInputs() == 2);
    nvinfer1::Dims inputDims[] = { network->getInput(0)->getDimensions(), network->getInput(1)->getDimensions() };
    assert(inputDims[0].nbDims == 4);
    assert(inputDims[1].nbDims == 4);

    assert(network->getNbOutputs() == 2);

    // Optimization Profiles
    auto profile = builder->createOptimizationProfile();
    const auto dims1 = inputDims[0].d;
    profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims1[1], dims1[2], dims1[3]));
    profile->setDimensions("input1", nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4(opt_batch_size_, dims1[1], dims1[2], dims1[3]));
    profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(max_batch_size_, dims1[1], dims1[2], dims1[3]));
    const auto dims2 = inputDims[1].d;
    profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims2[1], dims2[2], dims2[3]));
    profile->setDimensions("input2", nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4(opt_batch_size_, dims2[1], dims2[2], dims2[3]));
    profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(max_batch_size_, dims2[1], dims2[2], dims2[3]));
    config->addOptimizationProfile(profile);

    engine_ = builder->buildEngineWithConfig(*network, *config);
    if (!engine_) {
        throw std::runtime_error("buildEngineWithConfig");
    }
}

void InferDLShogiOnnxModel::load_model(const char* filename) {
    std::string serialized_filename =
        std::string(filename) + "." + std::to_string(gpu_id_) + "." + std::to_string(max_batch_size_) + ".serialized";
    std::ifstream serializedFile(serialized_filename, std::ios::binary);
    if (serializedFile.is_open()) {
        // deserializing a model
        serializedFile.seekg(0, std::ios_base::end);
        const size_t modelSize = serializedFile.tellg();
        serializedFile.seekg(0, std::ios_base::beg);
        std::unique_ptr<char[]> blob(new char[modelSize]);
        serializedFile.read(blob.get(), modelSize);
        auto runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        engine_ = runtime->deserializeCudaEngine(blob.get(), modelSize, nullptr);
    } else {

        // build
        build(filename);

        // serializing a model
        auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(engine_->serialize());
        if (!serializedEngine) {
            throw std::runtime_error("Engine serialization failed");
        }
        std::ofstream engineFile(serialized_filename, std::ios::binary);
        if (!engineFile) {
            throw std::runtime_error("Cannot open engine file");
        }
        engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
        if (engineFile.fail()) {
            throw std::runtime_error("Cannot open engine file");
        }
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("createExecutionContext");
    }

    input_dims1_ = engine_->getBindingDimensions(0);
    input_dims2_ = engine_->getBindingDimensions(1);
}

void InferDLShogiOnnxModel::load(int64_t gpu_id, const SearchOptions& search_option) {
    gpu_id_ = gpu_id;
    opt_batch_size_ = search_option.search_batch_size;
    max_batch_size_ = search_option.search_batch_size * 2;
    // Create host and device buffers
    checkCudaErrors(cudaMalloc((void**)&x1_dev_, max_batch_size_ * sizeof(features1_t)));
    checkCudaErrors(cudaMalloc((void**)&x2_dev_, max_batch_size_ * sizeof(features2_t)));
    checkCudaErrors(cudaMalloc((void**)&y1_dev_, max_batch_size_ * sizeof(DType) * POLICY_DIM));
    checkCudaErrors(cudaMalloc((void**)&y2_dev_, max_batch_size_ * sizeof(DType)));

    input_bindings_ = { x1_dev_, x2_dev_, y1_dev_, y2_dev_ };

    load_model(search_option.model_name.c_str());
}

void InferDLShogiOnnxModel::forward(const int64_t batch_size, void* x1, void* x2, DType* y1, DType* y2) {
    input_dims1_.d[0] = batch_size;
    input_dims2_.d[0] = batch_size;

    context_->setBindingDimensions(0, input_dims1_);
    context_->setBindingDimensions(1, input_dims2_);

    checkCudaErrors(cudaMemcpy(x1_dev_, x1, batch_size * sizeof(features1_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(x2_dev_, x2, batch_size * sizeof(features2_t), cudaMemcpyHostToDevice));

    const bool status = context_->executeV2(input_bindings_.data());
    assert(status);

    checkCudaErrors(cudaMemcpy(y1, y1_dev_, batch_size * sizeof(DType) * POLICY_DIM, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(y2, y2_dev_, batch_size * sizeof(DType), cudaMemcpyDeviceToHost));
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>> InferDLShogiOnnxModel::policyAndValueBatch(const std::vector<float>& inputs) {
    constexpr int64_t element_num = INPUT_CHANNEL_NUM * SQUARE_NUM;
    const int64_t batch_size = inputs.size() / element_num;

    torch::Tensor x = inputVectorToTensor(inputs);
    std::vector<torch::Tensor> xs = x.split(DLSHOGI_FEATURES1_NUM, 1);
    torch::Tensor x1 = xs[0].contiguous();
    torch::Tensor x2 = xs[1].contiguous();

    std::vector<DType> policy_buffer(batch_size * POLICY_DIM);
    std::vector<DType> value_buffer(batch_size);

    forward(batch_size, x1.data_ptr(), x2.data_ptr(), policy_buffer.data(), value_buffer.data());

    std::vector<PolicyType> policy(batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
        //policyをコピー
        policy[i].insert(policy[i].begin(), policy_buffer.begin() + i * POLICY_DIM, policy_buffer.begin() + (i + 1) * POLICY_DIM);

#ifdef USE_SIGMOID
        //Sigmoidのときはそのままで良い
#else
        //tanh想定のときはvalueの規格を修正
        value_buffer[i] = value_buffer[i] * 2 - 1;
#endif
    }

    return std::make_pair(policy, value_buffer);
}

std::array<torch::Tensor, LOSS_TYPE_NUM> InferDLShogiOnnxModel::validLoss(const std::vector<LearningData>& data) {
#ifdef USE_CATEGORICAL
    std::cout << "dlshogiモデルはCategoricalモードに対応していない" << std::endl;
    std::exit(1);
#else
    auto [input, policy_target, value_target] = learningDataToTensor(data, torch::Device(torch::kCPU), true);

    std::vector<torch::Tensor> xs = input.split(DLSHOGI_FEATURES1_NUM, 1);
    torch::Tensor x1_tensor = xs[0].contiguous();
    torch::Tensor x2_tensor = xs[1].contiguous();

    const int64_t batch_size = x1_tensor.size(0);

    std::vector<DType> policy_buffer(batch_size * POLICY_DIM, -1);
    std::vector<DType> value_buffer(batch_size, -1);

    forward(batch_size, x1_tensor.data_ptr(), x2_tensor.data_ptr(), policy_buffer.data(), value_buffer.data());

    torch::Tensor policy_logits = torch::tensor(policy_buffer).view({ -1, POLICY_DIM });
    torch::Tensor value = torch::tensor(value_buffer);

    torch::Tensor policy_loss = torch::sum(-policy_target * torch::log_softmax(policy_logits, 1), 1, false);

    value = value.view(-1);
#ifdef USE_SIGMOID
    torch::Tensor value_loss = torch::binary_cross_entropy(value, value_target, {}, torch::Reduction::None);
#else
    value = value * 2 - 1;
    torch::Tensor value_loss = torch::mse_loss(value, value_target, torch::Reduction::None);
#endif

    return { policy_loss, value_loss };
#endif
}

#endif