#ifndef CALIBRATOR_HPP
#define CALIBRATOR_HPP

#include "../learn/learn.hpp"

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator2(const std::string& model_filename, const int64_t batch_size, const std::string& kifu_dir)
        : batch_size_(batch_size) {
        std::cout << "Int8EntropyCalibrator2 Constructor" << std::endl;
        calibration_cache_filename = model_filename + ".calibcache";
        checkCudaErrors(cudaMalloc(&input_dev_, sizeof(float) * batch_size * INPUT_CHANNEL_NUM * SQUARE_NUM));
        data_ = loadData(kifu_dir, false, rate_threshold_);
    }

    ~Int8EntropyCalibrator2() { checkCudaErrors(cudaFree(input_dev_)); }

    int getBatchSize() const override { return batch_size_; }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override {
        if (batch_num_ >= BATCH_NUM) {
            return false;
        }
        int64_t curr_batch_size = 0;
        std::vector<float> curr_input;
        curr_input.reserve(batch_size_ * INPUT_CHANNEL_NUM * SQUARE_NUM);
        for (; index_ < data_.size() && curr_batch_size < batch_size_; index_++) {
            const LearningData& curr_data = data_[index_];
            pos_.fromStr(curr_data.position_str);
            std::vector<float> f = pos_.makeFeature();
            curr_input.insert(curr_input.end(), f.begin(), f.end());
            curr_batch_size++;
        }
        checkCudaErrors(cudaMemcpy(input_dev_, curr_input.data(), sizeof(float) * batch_size_ * INPUT_CHANNEL_NUM * SQUARE_NUM,
                                   cudaMemcpyHostToDevice));
        bindings[0] = input_dev_;
        batch_num_++;
        return (curr_batch_size == batch_size_);
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
    static constexpr int64_t BATCH_NUM = 10;
    std::vector<LearningData> int8_calibration_data;
    Position pos_;
    const int64_t batch_size_;
    int64_t batch_num_ = 0;
    void* input_dev_;
    std::vector<LearningData> data_;
    int64_t index_ = 0;
    const float rate_threshold_ = 3200;
    std::string calibration_cache_filename;
    std::vector<char> calibration_cache;
};

#endif
