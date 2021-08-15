#include "../model/infer_model.hpp"
#include "hyperparameter_loader.hpp"
#include "learn.hpp"
#include "random_game_generator.hpp"
#include <torch/torch.h>

void contrastiveLearn() {
    // clang-format off
    SearchOptions search_options;
    HyperparameterLoader settings("contrastive_learn_settings.txt");
    float lambda                      = 0;
    float per_alpha                   = 0;
    search_options.draw_turn          = settings.get<int64_t>("draw_turn");
    int64_t batch_size                = settings.get<int64_t>("batch_size");
    int64_t max_step_num              = settings.get<int64_t>("max_step_num");
    int64_t batch_size_per_gen        = settings.get<int64_t>("batch_size_per_gen");
    int64_t max_stack_size            = settings.get<int64_t>("max_stack_size");
    int64_t first_wait                = settings.get<int64_t>("first_wait");
    int64_t output_interval           = LLONG_MAX;
    int64_t sleep_msec                = settings.get<int64_t>("sleep_msec");
    bool data_augmentation            = settings.get<bool>("data_augmentation");
    search_options.calibration_kifu_path = settings.get<std::string>("calibration_kifu_path");
    // clang-format on

    //学習推移のログファイル
    std::ofstream train_log_("contrastive_train_log.txt");
    dout(std::cout, train_log_) << std::fixed << "time\tstep\tloss" << std::endl;

    //評価関数読み込み
    LearningModel neural_network;
    neural_network.load(DEFAULT_MODEL_NAME, 0);

    //学習前のパラメータを出力
    neural_network.save(MODEL_PREFIX + "_before_learn.model");

    //optimizerの準備
    float learn_rate_ = settings.get<float>("learn_rate");
    torch::optim::SGDOptions sgd_option(learn_rate_);
    sgd_option.momentum(settings.get<float>("momentum"));
    sgd_option.weight_decay(settings.get<float>("weight_decay"));
    std::vector<torch::Tensor> parameters;
    torch::optim::SGD optimizer_(neural_network.parameters(), sgd_option);

    //パラメータの保存間隔
    int64_t save_interval_ = settings.get<int64_t>("save_interval");

    //リプレイバッファの生成
    ReplayBuffer replay_buffer(first_wait, max_stack_size, output_interval, lambda, per_alpha, data_augmentation);

    //時間計測開始
    Timer timer;
    timer.start();

    //自己対局生成器を生成
    //ReplayBufferに入れるタイミングで排他制御があるので2スレッドは用意した方が良い
    constexpr int64_t thread_num = 2;
    std::vector<std::unique_ptr<RandomGameGenerator>> generators(thread_num);
    std::vector<std::thread> gen_threads;
    for (uint64_t i = 0; i < thread_num; i++) {
        generators[i] = std::make_unique<RandomGameGenerator>(search_options, replay_buffer);
        gen_threads.emplace_back([&generators, i]() { generators[i]->start(); });
    }

    //学習ループ
    for (int64_t step_num = 1; step_num <= max_step_num; step_num++) {
        //バッチサイズ分データを選択
        std::vector<LearningData> curr_data = replay_buffer.makeBatch(batch_size);

        //1回目はmakeBatch内で十分棋譜が貯まるまで待ち時間が発生する.その生成速度を計算
        if (step_num == 1) {
            float gen_speed = (float)first_wait / timer.elapsedSeconds();
            if (sleep_msec == -1) {
                sleep_msec = (int64_t)(batch_size * 1000 / (batch_size_per_gen * gen_speed));
            }
            std::ofstream ofs("gen_speed.txt");
            dout(std::cout, ofs) << "gen_speed = " << gen_speed << " pos / sec, sleep_msec = " << sleep_msec << std::endl;
        }

        //1ステップ学習し、損失を取得
        torch::Tensor loss_sum = neural_network.contrastiveLoss(curr_data);

        optimizer_.zero_grad();

        loss_sum.mean().backward();

        //パラメータを更新
        optimizer_.step();

        //表示
        if (step_num % 100 == 0) {
            dout(std::cout, train_log_) << timer.elapsedTimeStr() << "\t" << step_num << "\t" << loss_sum.mean().item<float>()
                                        << "\r" << std::flush;
        }

        //Cosine annealing
        (dynamic_cast<torch::optim::SGDOptions&>(optimizer_.param_groups().front().options())).lr() =
            0.5 * learn_rate_ * (1 + cos(acos(-1) * step_num / max_step_num));

        //学習スレッドを眠らせることで擬似的にActorの数を増やす
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_msec));
    }

    //生成スレッドを止める
    for (uint64_t i = 0; i < thread_num; i++) {
        generators[i]->stop_signal = true;
    }
    for (std::thread& th : gen_threads) {
        th.join();
    }

    std::cout << "finish contrastiveLearn" << std::endl;
}