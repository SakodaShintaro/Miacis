#include "infer_dlshogi_onnx_model.hpp"

#ifdef DLSHOGI

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

constexpr int MAX_MOVE_LABEL_NUM = MOVE_DIRECTION_NUM + HAND_PIECE_KIND_NUM;

constexpr long long int operator"" _MiB(long long unsigned int val) { return val * (1 << 20); }

InferDLShogiOnnxModel::~InferDLShogiOnnxModel() {
    checkCudaErrors(cudaFree(x1_dev));
    checkCudaErrors(cudaFree(x2_dev));
    checkCudaErrors(cudaFree(y1_dev));
    checkCudaErrors(cudaFree(y2_dev));
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

    builder->setMaxBatchSize(max_batch_size);
    config->setMaxWorkspaceSize(64_MiB);

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
                           nvinfer1::Dims4(max_batch_size, dims1[1], dims1[2], dims1[3]));
    profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(max_batch_size, dims1[1], dims1[2], dims1[3]));
    const auto dims2 = inputDims[1].d;
    profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims2[1], dims2[2], dims2[3]));
    profile->setDimensions("input2", nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4(max_batch_size, dims2[1], dims2[2], dims2[3]));
    profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(max_batch_size, dims2[1], dims2[2], dims2[3]));
    config->addOptimizationProfile(profile);

    engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        throw std::runtime_error("buildEngineWithConfig");
    }
}

void InferDLShogiOnnxModel::load_model(const char* filename) {
    std::string serialized_filename =
        std::string(filename) + "." + std::to_string(gpu_id) + "." + std::to_string(max_batch_size) + ".serialized";
    std::ifstream seriarizedFile(serialized_filename, std::ios::binary);
    if (seriarizedFile.is_open()) {
        // deserializing a model
        seriarizedFile.seekg(0, std::ios_base::end);
        const size_t modelSize = seriarizedFile.tellg();
        seriarizedFile.seekg(0, std::ios_base::beg);
        std::unique_ptr<char[]> blob(new char[modelSize]);
        seriarizedFile.read(blob.get(), modelSize);
        auto runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        engine = runtime->deserializeCudaEngine(blob.get(), modelSize, nullptr);
    } else {

        // build
        build(filename);

        // serializing a model
        auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
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

    context = engine->createExecutionContext();
    if (!context) {
        throw std::runtime_error("createExecutionContext");
    }

    inputDims1 = engine->getBindingDimensions(0);
    inputDims2 = engine->getBindingDimensions(1);
}

void InferDLShogiOnnxModel::forward(const int batch_size, void* x1, void* x2, DType* y1, DType* y2) {
    inputDims1.d[0] = batch_size;
    inputDims2.d[0] = batch_size;

    context->setBindingDimensions(0, inputDims1);
    context->setBindingDimensions(1, inputDims2);

    checkCudaErrors(cudaMemcpy(x1_dev, x1, sizeof(features1_t) * batch_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(x2_dev, x2, sizeof(features2_t) * batch_size, cudaMemcpyHostToDevice));

    const bool status = context->executeV2(inputBindings.data());
    assert(status);

    checkCudaErrors(
        cudaMemcpy(y1, y1_dev, MAX_MOVE_LABEL_NUM * (size_t)SquareNum * batch_size * sizeof(DType), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(y2, y2_dev, batch_size * sizeof(DType), cudaMemcpyDeviceToHost));
}

void InferDLShogiOnnxModel::load(int64_t gpu_id, const SearchOptions& search_option) {
    gpu_id = gpu_id;
    max_batch_size = search_option.search_batch_size * 2;
    // Create host and device buffers
    checkCudaErrors(cudaMalloc((void**)&x1_dev, sizeof(features1_t) * max_batch_size));
    checkCudaErrors(cudaMalloc((void**)&x2_dev, sizeof(features2_t) * max_batch_size));
    checkCudaErrors(cudaMalloc((void**)&y1_dev, MAX_MOVE_LABEL_NUM * (size_t)SquareNum * max_batch_size * sizeof(DType)));
    checkCudaErrors(cudaMalloc((void**)&y2_dev, max_batch_size * sizeof(DType)));

    inputBindings = { x1_dev, x2_dev, y1_dev, y2_dev };

    load_model(search_option.model_name.c_str());
}

std::pair<std::vector<PolicyType>, std::vector<ValueType>> InferDLShogiOnnxModel::policyAndValueBatch(const std::vector<float>& inputs) {
    constexpr int64_t element_num = INPUT_CHANNEL_NUM * SQUARE_NUM;
    const int64_t batch_size = inputs.size() / element_num;

    torch::Tensor x = inputVectorToTensor(inputs);
    std::vector<torch::Tensor> xs = x.split(DLSHOGI_FEATURES1_NUM, 1);
    torch::Tensor x1_tensor = xs[0].view({ -1, ColorNum, MAX_FEATURES1_NUM, SQUARE_NUM });
    torch::Tensor x2_tensor = xs[1].view({ -1, MAX_FEATURES2_NUM, SQUARE_NUM });

    // std::cout << x1_tensor.sizes() << ", " << x2_tensor.sizes() << std::endl;

    std::vector<DType> policy_buffer(batch_size * POLICY_DIM, -1);
    std::vector<DType> value_buffer(batch_size, -1);

    forward(batch_size, x1_tensor.data_ptr(), x2_tensor.data_ptr(), policy_buffer.data(), value_buffer.data());

    std::vector<PolicyType> policy(batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
        policy[i].insert(policy[i].begin(), policy_buffer.begin() + i * POLICY_DIM, policy_buffer.begin() + (i + 1) * POLICY_DIM);
    }

    return std::make_pair(policy, value_buffer);
}

std::array<torch::Tensor, LOSS_TYPE_NUM> InferDLShogiOnnxModel::validLoss(const std::vector<LearningData>& data) {
    return std::array<torch::Tensor, LOSS_TYPE_NUM>{};
}


#endif