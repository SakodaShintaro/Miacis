#include "learn.hpp"
#include "../game.hpp"
#include "../include_switch.hpp"
#include "hyperparameter_loader.hpp"
#include <iomanip>
#include <random>
#include <sstream>

// optimizerの保存名
static const std::string optimizer_file_name = "optimizer.pt";

// モデルの拡張子 .ptの方が普通そうだが……
static const std::string MODEL_SUFFIX = ".model";

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

std::vector<LearningData> deleteDuplicate(std::vector<LearningData>& data_buffer) {
    std::sort(data_buffer.begin(), data_buffer.end(),
              [](LearningData& lhs, LearningData& rhs) { return lhs.position_str < rhs.position_str; });

    std::vector<LearningData> remain;
    remain.reserve(data_buffer.size());
    //統計情報
    int64_t redundant_num = 0;
    float value_sum = 0;
    std::map<int64_t, float> policy_map;

    for (uint64_t i = 0; i < data_buffer.size(); i++) {
        const LearningData& curr = data_buffer[i];
        redundant_num++;
#ifdef USE_CATEGORICAL
        value_sum += (MIN_SCORE + (curr.value + 0.5) * VALUE_WIDTH);
#else
        value_sum += curr.value;
#endif
        for (auto [label, prob] : curr.policy) {
            policy_map[label] += prob;
        }

        if (i == data_buffer.size() - 1 || curr.position_str != data_buffer[i + 1].position_str) {
            LearningData datum;
            datum.position_str = curr.position_str;
            float v = value_sum / redundant_num;
#ifdef USE_CATEGORICAL
            datum.value = valueToIndex(v);
#else
            datum.value = v;
#endif
            for (auto [label, prob_sum] : policy_map) {
                datum.policy.push_back({ label, prob_sum / redundant_num });
            }

            //            if (redundant_num > 100) {
            //                std::cout << "redundant_num = " << redundant_num << std::endl;
            //                Position pos;
            //                pos.fromStr(datum.position_str);
            //                pos.print();
            //                std::cout << "v = " << v << std::endl;
            //                for (auto [label, prob] : datum.policy) {
            //                    std::cout << label << " " << prob << std::endl;
            //                }
            //            }

            remain.push_back(datum);

            redundant_num = 0;
            value_sum = 0;
            policy_map.clear();
        }
    }

    return remain;
}

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

    //重複を削除して返す
    return deleteDuplicate(data_buffer);
}

// make move
Move make_move_label(const uint16_t move16, const Color color) {
    // xxxxxxxx x1111111  移動先
    // xx111111 1xxxxxxx  移動元。駒打ちの際には、PieceType + SquareNum - 1
    // x1xxxxxx xxxxxxxx  1 なら成り
    uint16_t to_sq = move16 & 0b1111111;
    uint16_t from_sq = (move16 >> 7) & 0b1111111;

    Square to = SquareList[to_sq];

    if (from_sq < SQUARE_NUM) {
        Square from = SquareList[from_sq];
        bool promote = (move16 & 0b100000000000000) > 0;
        return Move(to, from, false, promote);
    } else {
        // 持ち駒の場合
        const int hand_piece = from_sq - (uint16_t)SQUARE_NUM;
        Piece p = coloredPiece(color, DLShogiPieceKindList[hand_piece]);
        return dropMove(to, p);
    }
}

// make result
inline float make_result(const uint8_t result, const Color color) {
    const GameResult gameResult = (GameResult)(result & 0x3);
    if (gameResult == Draw) return (MAX_SCORE + MIN_SCORE) / 2;

    if ((color == BLACK && gameResult == BlackWin) || (color == WHITE && gameResult == WhiteWin)) {
        return MAX_SCORE;
    } else {
        return MIN_SCORE;
    }
}

std::vector<LearningData> __hcpe_decode_with_value(const size_t len, char* ndhcpe, bool data_augmentation) {
    HuffmanCodedPosAndEval* hcpe = reinterpret_cast<HuffmanCodedPosAndEval*>(ndhcpe);

    std::vector<LearningData> data_buffer;

    Position pos;
    for (size_t i = 0; i < len; i++, hcpe++) {
        pos.fromHCP(hcpe->hcp);
        std::string position_str = pos.toStr();

        Move move = make_move_label(hcpe->bestMove16, pos.color());
        move = pos.transformValidMove(move);
        uint32_t label = move.toLabel();

        float score = 1.0f / (1.0f + expf(-(float)hcpe->eval * 0.0013226f)) * (MAX_SCORE - MIN_SCORE) + MIN_SCORE;
        float result = make_result(hcpe->gameResult, pos.color());
        float target_value = (score + result) / 2;

        for (int64_t i = 0; i < (data_augmentation ? Position::DATA_AUGMENTATION_PATTERN_NUM : 1); i++) {
            LearningData datum{};
            datum.policy.push_back({ Move::augmentLabel(label, i), 1.0 });
#ifdef USE_CATEGORICAL
            datum.value = valueToIndex(target_value);
#else
            datum.value = target_value;
#endif
            datum.position_str = Position::augmentStr(position_str, i);
            data_buffer.push_back(datum);
        }
    }

    return deleteDuplicate(data_buffer);
}

std::vector<LearningData> loadHCPE(const std::string& file_path, bool data_augmentation) {
    std::ifstream hcpe_file(file_path, std::ios::binary);
    hcpe_file.seekg(0, std::ios_base::end);
    const size_t file_size = hcpe_file.tellg();
    hcpe_file.seekg(0, std::ios_base::beg);
    std::unique_ptr<char[]> blob(new char[file_size]);
    hcpe_file.read(blob.get(), file_size);
    const size_t len = file_size / sizeof(HuffmanCodedPosAndEval);
    return __hcpe_decode_with_value(len, blob.get(), data_augmentation);
}

template<class LearningClass> LearnManager<LearningClass>::LearnManager(const std::string& learn_name, int64_t initial_step_num) {
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
    setLearnRate(initial_step_num);

    //パラメータの保存間隔
    save_interval_ = settings.get<int64_t>("save_interval");

    //学習率のスケジューリングについての変数
    learn_rate_decay_mode_ = settings.get<int64_t>("learn_rate_decay_mode");
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

template<class LearningClass> void LearnManager<LearningClass>::saveModelAsDefaultName() {
    neural_network_.save(model_prefix_ + MODEL_SUFFIX);
}

template<class LearningClass> void LearnManager<LearningClass>::setLearnRate(int64_t step_num) {
    if (step_num <= warm_up_step_) {
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
    } else {
        std::cout << "Invalid learn_rate_decay_mode_: " << learn_rate_decay_mode_ << std::endl;
        std::exit(1);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> learningDataToTensor(const std::vector<LearningData>& data,
                                                                             torch::Device device) {
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
        value_teachers.push_back(data[i].value);
    }

    torch::Tensor input_tensor = inputVectorToTensor(inputs).to(device);
    torch::Tensor policy_target = torch::tensor(policy_teachers).view({ -1, POLICY_DIM }).to(device);
    torch::Tensor value_target = torch::tensor(value_teachers).to(device);

    return std::make_tuple(input_tensor, policy_target, value_target);
}

int64_t loadStepNumFromValidLog(const std::string& valid_log_name) {
    std::ifstream valid_log(valid_log_name);
    if (!valid_log.is_open()) {
        return 0;
    }

    //validの最終行から読み込む
    std::string line, final_line;
    while (getline(valid_log, line)) {
        final_line = line;
    }
    int64_t first_tab = final_line.find('\t');
    int64_t second_tab = final_line.find('\t', first_tab + 1);
    std::string step_num_str = final_line.substr(first_tab + 1, second_tab - first_tab - 1);
    return std::stoll(step_num_str);
}

template class LearnManager<LearningModel>;