#ifndef INFER_MODEL_HPP
#define INFER_MODEL_HPP

#include "../search/search_options.hpp"
#include "model_common.hpp"
#include <torch/script.h>

#ifdef DLSHOGI

#include "infer_dlshogi_model.hpp"
#include "infer_dlshogi_onnx_model.hpp"
using InferModel = InferDLShogiOnnxModel;

#else

#include "../include_switch.hpp"
#include "../search/search_options.hpp"
#include "model_common.hpp"
#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <NvOnnxParser.h>
#include <cctype>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

inline void checkCudaErrors(cudaError_t status) {
    if (status != 0) {
        std::stringstream _error;
        _error << "Cuda failure\nError: " << cudaGetErrorString(status);
        std::cerr << _error.str() << "\nAborting...\n";
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

struct InferDeleter {
    template<typename T> void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

template<typename T> using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

class InferModel {
public:
    InferModel() = default;
    ~InferModel();
    void load(int64_t gpu_id, const SearchOptions& search_option);
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);
    std::array<torch::Tensor, LOSS_TYPE_NUM> validLoss(const std::vector<LearningData>& data);

private:
    int64_t gpu_id_;
    int64_t opt_batch_size_;
    int64_t max_batch_size_;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    std::vector<void*> input_bindings_;
    void* x1_dev_ = nullptr;
    void* y1_dev_ = nullptr;
    void* y2_dev_ = nullptr;

    void forward(const int64_t batch_size, const float* x1, void* y1, void* y2);
};

#endif

#endif