#ifndef TENSORRT_MODEL_HPP
#define TENSORRT_MODEL_HPP

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

enum FP_MODE { FP16, INT8, FP_MODE_NUM };

class TensorRTModel {
public:
    TensorRTModel() = default;
    ~TensorRTModel();
    void load(int64_t gpu_id, const SearchOptions& search_option);
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);
    std::array<std::vector<float>, LOSS_TYPE_NUM> validLoss(const std::vector<LearningData>& data);
    static void convertOnnxToEngine(const std::string& onnx_path, const FP_MODE fp_mode, const int64_t opt_batch_size,
                                    const std::string& calibration_data_path);

private:
    int64_t gpu_id_;
    int64_t opt_batch_size_;
    int64_t max_batch_size_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<void*> input_bindings_;
    void* x1_dev_ = nullptr;
    void* y1_dev_ = nullptr;
    void* y2_dev_ = nullptr;

    void forward(const int64_t batch_size, const float* x1, void* y1, void* y2);
};

#endif