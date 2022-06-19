#ifndef MIACIS_LEARN_MANAGER_HPP
#define MIACIS_LEARN_MANAGER_HPP

#include "../model/infer_model.hpp"
#include "../model/learning_model.hpp"
#include "../model/model_common.hpp"
#include "../timer.hpp"

class LearnManager {
public:
    explicit LearnManager(const std::string& learn_name, int64_t initial_step_num);
    torch::Tensor learnOneStep(const std::vector<LearningData>& curr_data, int64_t step_num);
    void saveModelAsDefaultName();

private:
    void setLearnRate(int64_t step_num);

    //学習するモデル
    LearningModel neural_network_;

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

//log_fileから最終ステップ数を読み込む関数
int64_t loadStepNumFromLog(const std::string& log_file_path);

//AlphaZero式の強化学習
void reinforcementLearn();

#endif //MIACIS_LEARN_MANAGER_HPP