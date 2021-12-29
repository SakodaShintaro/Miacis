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

template<class LearningClass> class LearnManager {
public:
    explicit LearnManager(const std::string& learn_name, int64_t initial_step_num);
    torch::Tensor learnOneStep(const std::vector<LearningData>& curr_data, int64_t step_num);
    void saveModelAsDefaultName();

private:
    void setLearnRate(int64_t step_num);

    //学習するモデル
    LearningClass neural_network_;

    //学習するモデルの名前
    std::string model_prefix_;

    //Optimizer
    std::unique_ptr<torch::optim::SGD> optimizer_;

    //検証データ
    std::vector<LearningData> valid_data_;

    //検証を行う間隔
    int64_t validation_interval_;

    //パラメータを保存する間隔
    int64_t save_interval_;

    //各ロスを足し合わせる比
    std::array<float, LOSS_TYPE_NUM> coefficients_{};

    //学習,検証のログファイル
    std::ofstream train_log_, valid_log_;

    //学習時間計測器
    Timer timer_;

    //初期学習率
    float learn_rate_;

    //学習率のスケジューリングモード指定
    int64_t learn_rate_decay_mode_;

    //学習率スケジューリングの動作周期
    int64_t learn_rate_decay_period_;

    //warmupのステップ数
    int64_t warm_up_step_;

    //mixupを行う場合の混合比を決定する値
    float mixup_alpha_;

    //Sharpness-Aware Minimizationを行うかどうか
    bool use_sam_optim_;

    //勾配クリッピングの値
    float clip_grad_norm_;
};

//教師データを読み込む関数
std::vector<LearningData> loadData(const std::string& file_path, bool data_augmentation, float rate_threshold);
std::vector<LearningData> loadHCPE(const std::string& file_path, bool data_augmentation);

//validationを行う関数
template<class ModelType>
std::array<float, LOSS_TYPE_NUM> validation(ModelType& model, const std::vector<LearningData>& valid_data, uint64_t batch_size);

//学習データをtensorへ変換する関数
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> learningDataToTensor(const std::vector<LearningData>& data,
                                                                             torch::Device device);

//valid_logから最終ステップ数を読み込む関数
int64_t loadStepNumFromValidLog(const std::string& valid_log_name);

//棋譜からの教師あり学習
void supervisedLearn();

//AlphaZero式の強化学習
void reinforcementLearn();

//ランダムに自己対局したラベルなしデータから教師なしで対比学習
void contrastiveLearn();

//実験的にLibTorchモデルを教師あり学習する関数
void experimentalSupervisedLearn();

#endif //MIACIS_LEARN_HPP