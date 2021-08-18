#include "learn.hpp"
#include "../game.hpp"
#include "../include_switch.hpp"
#include "../model/infer_dlshogi_model.hpp"
#include "../model/libtorch_model.hpp"
#include "hyperparameter_loader.hpp"
#include <iomanip>
#include <random>
#include <sstream>

template<class ModelType>
std::array<float, LOSS_TYPE_NUM> validation(ModelType& model, const std::vector<LearningData>& valid_data, uint64_t batch_size) {
    torch::NoGradGuard no_grad_guard;
    std::array<float, LOSS_TYPE_NUM> losses{};
    for (uint64_t index = 0; index < valid_data.size();) {
        std::vector<LearningData> curr_data;
        while (index < valid_data.size() && curr_data.size() < batch_size) {
            curr_data.push_back(valid_data[index++]);
        }

        std::array<torch::Tensor, LOSS_TYPE_NUM> loss = model.validLoss(curr_data);
        for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
            losses[i] += loss[i].sum().item<float>();
        }
    }

    //データサイズで割って平均
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        losses[i] /= valid_data.size();
    }

    return losses;
}

template std::array<float, LOSS_TYPE_NUM> validation<InferModel>(InferModel& model, const std::vector<LearningData>& valid_data,
                                                                 uint64_t batch_size);

std::vector<LearningData> loadData(const std::string& file_path, bool data_augmentation, float rate_threshold) {
    //棋譜を読み込めるだけ読み込む
    std::vector<Game> games = loadGames(file_path, rate_threshold);

    //データを局面単位にバラす
    std::vector<LearningData> data_buffer;
    for (const Game& game : games) {
        Position pos;
        for (const OneTurnElement& e : game.elements) {
            const Move& move = e.move;
            uint32_t label = move.toLabel();
            std::string position_str = pos.toStr();
            for (int64_t i = 0; i < (data_augmentation ? Position::DATA_AUGMENTATION_PATTERN_NUM : 1); i++) {
                LearningData datum{};
                datum.policy.push_back({ Move::augmentLabel(label, i), 1.0 });
#ifdef USE_CATEGORICAL
                datum.value = valueToIndex((pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result));
#else
                datum.value = (float)(pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result);
#endif
                datum.position_str = Position::augmentStr(position_str, i);
                data_buffer.push_back(datum);
            }
            pos.doMove(move);
        }
    }

    return data_buffer;
}

template<class LearningClass> LearnManager<LearningClass>::LearnManager(const std::string& learn_name) {
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
    train_log_.open(learn_name + "_train_log.txt");
    valid_log_.open(learn_name + "_valid_log.txt");
    tout(std::cout, train_log_, valid_log_) << std::fixed << "time\tstep\t";
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        tout(std::cout, train_log_, valid_log_) << LOSS_TYPE_NAME[i] + "_loss"
                                                << "\t\n"[i == LOSS_TYPE_NUM - 1];
    }

    //評価関数読み込み
    neural_network_.load(DEFAULT_MODEL_NAME, 0);

    //学習前のパラメータを出力
    neural_network_.save(MODEL_PREFIX + "_before_learn.model");

    //optimizerの準備
    learn_rate_ = settings.get<float>("learn_rate");
    torch::optim::SGDOptions sgd_option(learn_rate_);
    sgd_option.momentum(settings.get<float>("momentum"));
    sgd_option.weight_decay(settings.get<float>("weight_decay"));
    std::vector<torch::Tensor> parameters;
    optimizer_ = std::make_unique<torch::optim::SGD>(neural_network_.parameters(), sgd_option);

    //パラメータの保存間隔
    save_interval_ = settings.get<int64_t>("save_interval");

    //学習率のスケジューリングについての変数
    learn_rate_decay_mode_ = settings.get<int64_t>("learn_rate_decay_mode");
    learn_rate_decay_step1_ = settings.get<int64_t>("learn_rate_decay_step1");
    learn_rate_decay_step2_ = settings.get<int64_t>("learn_rate_decay_step2");
    learn_rate_decay_step3_ = settings.get<int64_t>("learn_rate_decay_step3");
    learn_rate_decay_step4_ = settings.get<int64_t>("learn_rate_decay_step4");
    learn_rate_decay_period_ = settings.get<int64_t>("learn_rate_decay_period");

    warm_up_step_ = settings.get<int64_t>("warm_up_step");
    if (warm_up_step_ > 0) {
        (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() = 0;
    }

    //mixupの混合比を決定する値
    mixup_alpha_ = settings.get<float>("mixup_alpha");

    //SAM
    use_sam_optim_ = settings.get<int64_t>("use_sam_optim");

    //clip_grad_norm_の値
    clip_grad_norm_ = settings.get<float>("clip_grad_norm");

    //学習開始時間の設定
    timer_.start();
}

template<class LearningClass>
torch::Tensor LearnManager<LearningClass>::learnOneStep(const std::vector<LearningData>& curr_data, int64_t step_num) {
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
            dout(std::cout, train_log_) << loss[i].mean().item<float>() << "\t\r"[i == LOSS_TYPE_NUM - 1];
        }
        dout(std::cout, train_log_) << std::flush;
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
            dout(std::cout, valid_log_) << valid_loss[i] << "\t\n"[i == LOSS_TYPE_NUM - 1];
        }
        dout(std::cout, valid_log_) << std::flush;
    }

    //パラメータをステップ付きで保存
    if (step_num % save_interval_ == 0) {
        neural_network_.save(MODEL_PREFIX + "_" + std::to_string(step_num) + ".model");
    }

    //学習率の変化はoptimizer_->defaults();を使えそうな気がする
    if (learn_rate_decay_mode_ == 1) {
        //特定ステップに達したら1/10にするスケジューリング
        if (step_num == learn_rate_decay_step1_ || step_num == learn_rate_decay_step2_ || step_num == learn_rate_decay_step3_ ||
            step_num == learn_rate_decay_step4_) {
            (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() /= 10;
        }
    } else if (learn_rate_decay_mode_ == 2) {
        //Cosine annealing
        int64_t curr_step = step_num % learn_rate_decay_period_;
        (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() =
            0.5 * learn_rate_ * (1 + cos(acos(-1) * curr_step / learn_rate_decay_period_));
    } else if (learn_rate_decay_mode_ == 3) {
        //指数的な減衰
        if (step_num % learn_rate_decay_period_ == 0) {
            (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() *= 0.9;
        }
    }

    if (step_num <= warm_up_step_) {
        (dynamic_cast<torch::optim::SGDOptions&>(optimizer_->param_groups().front().options())).lr() =
            learn_rate_ * step_num / warm_up_step_;
    }

    return loss_sum.detach();
}

template<class LearningClass> void LearnManager<LearningClass>::saveModelAsDefaultName() {
    neural_network_.save(DEFAULT_MODEL_NAME);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> learningDataToTensor(const std::vector<LearningData>& data,
                                                                             torch::Device device, bool valid) {
    static Position pos;
    std::vector<float> inputs;
    std::vector<float> policy_teachers(data.size() * POLICY_DIM, 0.0);
    std::vector<ValueTeacherType> value_teachers;

    for (uint64_t i = 0; i < data.size(); i++) {
        pos.fromStr(data[i].position_str);

        //入力
        const std::vector<float> feature = pos.makeFeature();
        inputs.insert(inputs.end(), feature.begin(), feature.end());

        //policyの教師信号
        for (const std::pair<int32_t, float>& e : data[i].policy) {
            policy_teachers[i * POLICY_DIM + e.first] = e.second;
        }

        //valueの教師信号
        if (!valid) {
            //trainモードのときはそのまま突っ込めば良い
            value_teachers.push_back(data[i].value);
        } else {
            //validモードのときはCategoricalモデルを使うとしてもfloatのvalueをターゲットにしたい
#ifdef USE_CATEGORICAL
            if (data[i].value != 0 && data[i].value != BIN_SIZE - 1) {
                std::cerr << "Categoricalの検証データは現状のところValueが-1 or 1でないといけない" << std::endl;
                std::exit(1);
            }
            value_teachers.push_back(data[i].value == 0 ? MIN_SCORE : MAX_SCORE);
#else
            value_teachers.push_back(data[i].value);
#endif
        }
    }

    torch::Tensor input_tensor = inputVectorToTensor(inputs).to(device);
    torch::Tensor policy_target = torch::tensor(policy_teachers).view({ -1, POLICY_DIM }).to(device);
    torch::Tensor value_target = torch::tensor(value_teachers).to(device);

    return std::make_tuple(input_tensor, policy_target, value_target);
}

void initLibTorchModel() {
    LibTorchModel model;
    model.save(DEFAULT_MODEL_NAME);
    std::cout << DEFAULT_MODEL_NAME << "にパラメータを保存" << std::endl;
}

template class LearnManager<LearningModel>;
template class LearnManager<LibTorchModel>;