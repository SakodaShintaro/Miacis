#ifndef MIACIS_LEARN_HPP
#define MIACIS_LEARN_HPP

#include "../model/infer_model.hpp"
#include "../model/learning_model.hpp"
#include "../model/model_common.hpp"
#include "../timer.hpp"

//標準出力とファイルストリームに同時に出力するためのクラス
//参考)https://aki-yam.hatenablog.com/entry/20080630/1214801872
class dout {
private:
    std::ostream &os1, &os2;

public:
    explicit dout(std::ostream& _os1, std::ostream& _os2) : os1(_os1), os2(_os2){};
    template<typename T> dout& operator<<(const T& rhs) {
        os1 << rhs;
        os2 << rhs;
        return *this;
    };
    dout& operator<<(std::ostream& (*__pf)(std::ostream&)) {
        __pf(os1);
        __pf(os2);
        return *this;
    };
    /*!<  Interface for manipulators, such as \c std::endl and \c std::setw
      For more information, see ostream header */
};

class tout {
private:
    std::ostream &os1, &os2, &os3;

public:
    explicit tout(std::ostream& _os1, std::ostream& _os2, std::ostream& _os3) : os1(_os1), os2(_os2), os3(_os3){};
    template<typename T> tout& operator<<(const T& rhs) {
        os1 << rhs;
        os2 << rhs;
        os3 << rhs;
        return *this;
    };
    tout& operator<<(std::ostream& (*__pf)(std::ostream&)) {
        __pf(os1);
        __pf(os2);
        __pf(os3);
        return *this;
    };
    /*!<  Interface for manipulators, such as \c std::endl and \c std::setw
      For more information, see ostream header */
};

//教師データを読み込む関数
std::vector<LearningData> loadData(const std::string& file_path, bool data_augmentation, float rate_threshold);
std::vector<LearningData> loadHCPE(const std::string& file_path, bool data_augmentation);

//validationを行う関数
std::array<float, LOSS_TYPE_NUM> validation(InferModel& model, const std::vector<LearningData>& valid_data, uint64_t batch_size);

//validationを行い、各局面の損失をtsvで出力する関数
std::array<float, LOSS_TYPE_NUM> validationWithSave(InferModel& model, const std::vector<LearningData>& valid_data, uint64_t batch_size);

//学習データをtensorへ変換する関数
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> learningDataToTensor(const std::vector<LearningData>& data,
                                                                             torch::Device device);

#endif //MIACIS_LEARN_HPP