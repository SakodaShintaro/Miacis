#ifndef MIACIS_INFER_DLSHOGI_ONNX_MODEL_HPP
#define MIACIS_INFER_DLSHOGI_ONNX_MODEL_HPP

#ifdef DLSHOGI

#include "include_switch.hpp"
#include "neural_network.hpp"
#include "search_options.hpp"
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

typedef float DType;
typedef DType features1_t[ColorNum][DLSHOGI_FEATURES1_NUM / 2][SQUARE_NUM];
typedef DType features2_t[DLSHOGI_FEATURES2_NUM][SQUARE_NUM];

inline void FatalError(const std::string& s) {
    std::cerr << s << "\nAborting...\n";
    cudaDeviceReset();
    exit(EXIT_FAILURE);
}

inline void checkCudaErrors(cudaError_t status) {
    if (status != 0) {
        std::stringstream _error;
        _error << "Cuda failure\nError: " << cudaGetErrorString(status);
        FatalError(_error.str());
    }
}
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator2(const char* model_filename, const int batch_size, const char* hcpe_filename,
                           const size_t int8_calibration_data_size)
        : batch_size(batch_size), int8_calibration_data_size(int8_calibration_data_size), features1(new features1_t[batch_size]),
          features2(new features2_t[batch_size]) {
        calibration_cache_filename = std::string(model_filename) + ".calibcache";
        checkCudaErrors(cudaMalloc(&input1, sizeof(features1_t) * batch_size));
        checkCudaErrors(cudaMalloc(&input2, sizeof(features2_t) * batch_size));

        // キャリブレーションキャッシュがあるかチェック
        std::ifstream calibcache(calibration_cache_filename);
        if (!calibcache.is_open()) {
            if (hcpe_filename != "") {
                // hcpeからランダムに選ぶ
                //                std::ifstream ifs;
                //                ifs.open(hcpe_filename, std::ifstream::in | std::ifstream::binary | std::ios::ate);
                //                const auto entryNum = ifs.tellg() / sizeof(HuffmanCodedPosAndEval);
                //                std::uniform_int_distribution<s64> inputFileDist(0, entryNum - 1);
                //                std::mt19937_64 mt_64(std::chrono::system_clock::now().time_since_epoch().count());
                //                HuffmanCodedPosAndEval hcpe;
                //                for (size_t i = 0; i < int8_calibration_data_size; ++i) {
                //                    ifs.seekg(inputFileDist(mt_64) * sizeof(HuffmanCodedPosAndEval), std::ios_base::beg);
                //                    ifs.read(reinterpret_cast<char*>(&hcpe), sizeof(hcpe));
                //                    int8_calibration_data.emplace_back(hcpe.hcp);
                //                }
            } else {
                // キャリブレーションキャッシュがない場合エラー
                std::cerr << "missing calibration cache" << std::endl;
                throw std::runtime_error("missing calibration cache");
            }
        }
    }
    Int8EntropyCalibrator2(const char* model_filename, const int batch_size)
        : Int8EntropyCalibrator2(model_filename, batch_size, "", 0) {}

    ~Int8EntropyCalibrator2() {
        checkCudaErrors(cudaFree(input1));
        checkCudaErrors(cudaFree(input2));
    }

    int getBatchSize() const override { return batch_size; }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override {
        assert(nbBindings == 2);
        if (current_pos > int8_calibration_data_size - batch_size) {
            return false;
        }

        std::fill_n((float*)features1.get(), sizeof(features1_t) / sizeof(float) * batch_size, 0);
        std::fill_n((float*)features2.get(), sizeof(features2_t) / sizeof(float) * batch_size, 0);

        for (int i = 0; i < batch_size; ++i, ++current_pos) {
            pos.fromStr(int8_calibration_data[current_pos].position_str);
            //make_input_features(pos, features1.get() + i, features2.get() + i);
        }

        checkCudaErrors(cudaMemcpy(input1, features1.get(), sizeof(features1_t) * batch_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(input2, features2.get(), sizeof(features2_t) * batch_size, cudaMemcpyHostToDevice));
        bindings[0] = input1;
        bindings[1] = input2;

        return true;
    }

    const void* readCalibrationCache(size_t& length) override {
        calibration_cache.clear();
        std::ifstream input(calibration_cache_filename, std::ios::binary);
        input >> std::noskipws;
        if (input.good()) {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calibration_cache));
        }
        length = calibration_cache.size();
        return length ? calibration_cache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) override {
        std::ofstream output(calibration_cache_filename, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    std::vector<LearningData> int8_calibration_data;
    Position pos;
    const int batch_size;
    size_t current_pos = 0;
    std::unique_ptr<features1_t[]> features1;
    std::unique_ptr<features2_t[]> features2;
    void* input1;
    void* input2;
    std::string calibration_cache_filename;
    size_t int8_calibration_data_size;
    std::vector<char> calibration_cache;
};
struct InferDeleter {
    template<typename T> void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

template<typename T> using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

class InferDLShogiOnnxModel {
public:
    InferDLShogiOnnxModel() = default;
    ~InferDLShogiOnnxModel();
    void load(int64_t gpu_id, const SearchOptions& search_option);
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> policyAndValueBatch(const std::vector<float>& inputs);
    std::array<torch::Tensor, LOSS_TYPE_NUM> validLoss(const std::vector<LearningData>& data);

private:
    int64_t gpu_id_;
    int64_t opt_batch_size_;
    int64_t max_batch_size_;
    nvinfer1::ICudaEngine* engine_;
    features1_t* x1_dev_;
    features2_t* x2_dev_;
    DType* y1_dev_;
    DType* y2_dev_;
    std::vector<void*> input_bindings_;
    nvinfer1::IExecutionContext* context_;
    nvinfer1::Dims input_dims1_;
    nvinfer1::Dims input_dims2_;

    void load_model(const char* filename);
    void build(const std::string& onnx_filename);
    void forward(const int64_t batch_size, void* x1, void* x2, DType* y1, DType* y2);
};

#endif

#endif //MIACIS_INFER_DLSHOGI_ONNX_MODEL_HPP