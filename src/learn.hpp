#ifndef MIACIS_LEARN_HPP
#define MIACIS_LEARN_HPP

#include "neural_network.hpp"
#include <chrono>
#include <string>

//経過時間を取得する関数
std::string elapsedTime(const std::chrono::steady_clock::time_point& start);
float elapsedHours(const std::chrono::steady_clock::time_point& start);

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

class LearnManager {
public:
    explicit LearnManager(const std::string& learn_name);
    torch::Tensor learnOneStep(const std::vector<LearningData>& curr_data, int64_t stem_num);

    //学習するモデル。強化学習時に定期的な同期を挟むためにpublicに置く
    NeuralNetwork neural_network;

private:
    //Optimizer
    std::unique_ptr<torch::optim::SGD> optimizer_;

    //mixupを行う場合の混合比
    float mixup_alpha_;

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

    //学習開始時点の時刻
    std::chrono::steady_clock::time_point start_time_;

    //初期学習率
    float learn_rate_;

    //学習率のスケジューリングモード指定
    int64_t learn_rate_decay_mode_;

    //ステップdecayのタイミング
    int64_t learn_rate_decay_step1_;
    int64_t learn_rate_decay_step2_;
    int64_t learn_rate_decay_step3_;
    int64_t learn_rate_decay_step4_;

    //その他周期的なスケジューリングの周期
    int64_t learn_rate_decay_period_;

    //Cosine annealing時の最小値
    float min_learn_rate_;
};

//教師データを読み込む関数
std::vector<LearningData> loadData(const std::string& file_path, bool data_augmentation, float rate_threshold);

//validationを行う関数
std::array<float, LOSS_TYPE_NUM> validation(NeuralNetwork nn, const std::vector<LearningData>& validation_data,
                                            uint64_t batch_size);

//子階層にあるディレクトリ名を取得する関数
std::vector<std::string> childFiles(const std::string& file_path);

//パラメータを初期化
void initParams();

//棋譜からの教師あり学習
void supervisedLearn();

//AlphaZero式の強化学習
void reinforcementLearn();

#endif //MIACIS_LEARN_HPP