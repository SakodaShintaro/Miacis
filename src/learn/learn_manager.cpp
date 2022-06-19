#include "learn_manager.hpp"
#include "../shogi/game.hpp"
#include "../shogi/position.hpp"
#include "hyperparameter_loader.hpp"
#include "learn.hpp"
#include "multi_out.hpp"
#include <iomanip>
#include <random>
#include <sstream>

// optimizerの保存名
static const std::string optimizer_file_name = "optimizer.pt";

// モデルの拡張子 .ptの方が普通そうだが……
static const std::string MODEL_SUFFIX = ".model";

LearnManager::LearnManager(const std::string& learn_name, int64_t initial_step_num) {
    assert(learn_name == "supervised" || learn_name == "reinforcement" || learn_name == "contrastive");
    HyperparameterLoader settings(learn_name + "_learn_settings.txt");

    //検証を行う間隔
    validation_interval_ = settings.get<int64_t>("validation_interval");

    //各損失の重み
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        coefficients_[i] = settings.get<float>(LOSS_TYPE_NAME[i] + "_loss_coeff");
    }

    //検証用データの読み込み
    std::string valid_kifu_path = settings.get<std::string>("valid_kifu_path");
    float valid_rate_threshold = settings.get<float>("valid_rate_threshold");
    valid_data_ = loadData(valid_kifu_path, false, valid_rate_threshold);
    std::cout << "valid_data.size() = " << valid_data_.size() << std::endl;

    //学習推移のログファイル
    train_log_.open(learn_name + "_train_log.txt", std::ios::app);
    valid_log_.open(learn_name + "_valid_log.txt", std::ios::app);
    if (initial_step_num == 0) {
        //ヘッダを書き込む
        tout(std::cout, train_log_, valid_log_) << std::fixed << "time\tstep\t";
        for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
            tout(std::cout, train_log_, valid_log_) << LOSS_TYPE_NAME[i] + "_loss\t";
        }
        tout(std::cout, train_log_, valid_log_) << "learn_rate" << std::endl;
    }

    model_prefix_ = settings.get<std::string>("model_prefix");

    //評価関数読み込み
    neural_network_.load(model_prefix_ + MODEL_SUFFIX, 0);

    //学習前のパラメータを出力
    neural_network_.save(model_prefix_ + "_before_learn.model");

    //optimizerの準備
    learn_rate_ = settings.get<float>("learn_rate");
    torch::optim::SGDOptions sgd_option(learn_rate_);
    sgd_option.momentum(settings.get<float>("momentum"));
    sgd_option.weight_decay(settings.get<float>("weight_decay"));
    std::vector<torch::Tensor> parameters;
    optimizer_ = std::make_unique<torch::optim::SGD>(neural_network_.parameters(), sgd_option);
    if (std::ifstream(optimizer_file_name).is_open()) {
        torch::load(*optimizer_, optimizer_file_name);
    }

    //パラメータの保存間隔
    save_interval_ = settings.get<int64_t>("save_interval");

    //学習率のスケジューリングについての変数
    learn_rate_decay_mode_ = settings.get<int64_t>("learn_rate_decay_mode");
    learn_rate_decay_period_ = settings.get<int64_t>("learn_rate_decay_period");

    //warm-upにかけるステップ数
    warm_up_step_ = settings.get<int64_t>("warm_up_step");

    //学習率の設定
    setLearnRate(initial_step_num);

    //mixupの混合比を決定する値
    mixup_alpha_ = settings.get<float>("mixup_alpha");

    //SAM
    use_sam_optim_ = settings.get<int64_t>("use_sam_optim");

    //clip_grad_norm_の値
    clip_grad_norm_ = settings.get<float>("clip_grad_norm");

    //学習開始時間の設定
    timer_.start();
}

torch::Tensor LearnManager::learnOneStep(const std::vector<LearningData>& curr_data, int64_t step_num) {
    //バッチサイズを取得
    const int64_t batch_size = curr_data.size();

    //学習
    optimizer_->zero_grad();
    std::array<torch::Tensor, LOSS_TYPE_NUM> loss =
        (mixup_alpha_ == 0 ? neural_network_.loss(curr_data) : neural_network_.mixUpLoss(curr_data, mixup_alpha_));
    torch::Tensor loss_sum = torch::zeros({ batch_size });
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        loss_sum += coefficients_[i] * loss[i].cpu();
    }
    loss_sum.mean().backward();

    if (use_sam_optim_) {
        auto& param_groups = optimizer_->param_groups();
        std::vector<std::vector<torch::Tensor>> diff(param_groups.size());
        {
            torch::NoGradGuard no_grad_guard;
            //normを計算する
            std::vector<torch::Tensor> norms;
            for (const auto& group : optimizer_->param_groups()) {
                for (const auto& p : group.params()) {
                    if (p.requires_grad()) {
                        norms.push_back(p.grad().norm(2));
                    }
                }
            }
            torch::Tensor norm = torch::norm(torch::stack(norms), 2);
            const float rho = 0.05;
            torch::Tensor scale = rho / (norm + 1e-12);

            //勾配を上昇
            for (uint64_t i = 0; i < param_groups.size(); i++) {
                auto& params = param_groups[i].params();
                diff[i].resize(params.size());
                for (int64_t j = 0; j < params.size(); j++) {
                    if (!params[j].requires_grad()) {
                        continue;
                    }
                    torch::Tensor e_w = params[j].grad() * scale;
                    diff[i][j] = e_w;
                    params[j].add_(e_w);
                }
            }
        }

        //勾配を初期化
        optimizer_->zero_grad();

        //再計算
        loss = neural_network_.loss(curr_data);
        loss_sum = torch::zeros({ batch_size });
        for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
            loss_sum += coefficients_[i] * loss[i].cpu();
        }
        loss_sum.mean().backward();

        {
            torch::NoGradGuard no_grad_guard;
            //パラメータ変化を打ち消す
            for (uint64_t i = 0; i < param_groups.size(); i++) {
                auto& params = param_groups[i].params();
                for (int64_t j = 0; j < params.size(); j++) {
                    if (!params[j].requires_grad()) {
                        continue;
                    }
                    params[j].sub_(diff[i][j]);
                }
            }
        }
    }

    //勾配をクリップ
    torch::nn::utils::clip_grad_norm_(neural_network_.parameters(), clip_grad_norm_);

    //パラメータを更新
    optimizer_->step();

    //表示
    if (step_num % std::max(validation_interval_ / 1000, (int64_t)1) == 0) {
        dout(std::cout, train_log_) << timer_.elapsedTimeStr() << "\t" << step_num << "\t";
        for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
            dout(std::cout, train_log_) << loss[i].mean().item<float>() << "\t";
        }
        dout(std::cout, train_log_)
            << (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() << "\r" << std::flush;
    }

    if (step_num % validation_interval_ == 0) {
        //validation_lossを計算
        neural_network_.eval();
        std::array<float, LOSS_TYPE_NUM> valid_loss = validation(neural_network_, valid_data_, batch_size);
        neural_network_.train();
        float sum_loss = 0;
        for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
            sum_loss += coefficients_[i] * valid_loss[i];
        }

        //表示
        dout(std::cout, valid_log_) << timer_.elapsedTimeStr() << "\t" << step_num << "\t";
        for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
            dout(std::cout, valid_log_) << valid_loss[i] << "\t";
        }
        dout(std::cout, valid_log_)
            << (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() << std::endl;

        neural_network_.save(model_prefix_ + MODEL_SUFFIX);
        torch::save(*optimizer_, optimizer_file_name);
    }

    //パラメータをステップ付きで保存
    if (step_num % save_interval_ == 0) {
        neural_network_.save(model_prefix_ + "_" + std::to_string(step_num) + MODEL_SUFFIX);
    }

    //学習率の更新
    setLearnRate(step_num);

    return loss_sum.detach();
}

void LearnManager::saveModelAsDefaultName() { neural_network_.save(model_prefix_ + MODEL_SUFFIX); }

void LearnManager::setLearnRate(int64_t step_num) {
    if (warm_up_step_ > 0 && step_num <= warm_up_step_) {
        (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() =
            learn_rate_ * step_num / warm_up_step_;
        return;
    }

    if (learn_rate_decay_mode_ == 0) {
        // なにもしない
    } else if (learn_rate_decay_mode_ == 1) {
        //指数的な減衰(0.1)
        const int64_t div_num = step_num / learn_rate_decay_period_;
        (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() =
            learn_rate_ * std::pow(0.1, div_num);
    } else if (learn_rate_decay_mode_ == 2) {
        //Cosine annealing
        const int64_t curr_step = step_num % learn_rate_decay_period_;
        (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() =
            0.5 * learn_rate_ * (1 + cos(acos(-1) * curr_step / learn_rate_decay_period_));
    } else if (learn_rate_decay_mode_ == 3) {
        //指数的な減衰(0.9)
        const int64_t div_num = step_num / learn_rate_decay_period_;
        (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() =
            learn_rate_ * std::pow(0.9, div_num);
    } else if (learn_rate_decay_mode_ == 4) {
        //線形な減衰
        const int64_t curr_step = step_num % learn_rate_decay_period_;
        const int64_t rem_step = (learn_rate_decay_period_ - curr_step);
        (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() =
            learn_rate_ * rem_step / learn_rate_decay_period_;
    } else if (learn_rate_decay_mode_ == 5) {
        //ルートの逆数で減衰
        (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() =
            learn_rate_ * std::sqrt((double)warm_up_step_ / step_num);
    } else {
        std::cout << "Invalid learn_rate_decay_mode_: " << learn_rate_decay_mode_ << std::endl;
        std::exit(1);
    }
}

int64_t loadStepNumFromLog(const std::string& log_file_path) {
    std::ifstream log_file(log_file_path);
    if (!log_file.is_open()) {
        return 0;
    }

    // 最終行から読み込む
    std::string line, final_line;
    while (getline(log_file, line, '\r')) {
        final_line = line;
    }
    int64_t first_tab = final_line.find('\t');
    int64_t second_tab = final_line.find('\t', first_tab + 1);
    std::string step_num_str = final_line.substr(first_tab + 1, second_tab - first_tab - 1);
    if (step_num_str == "step") {
        // ヘッダだけあって学習結果0行の場合、ステップ0からスタートで良い
        return 0;
    }
    return std::stoll(step_num_str);
}
