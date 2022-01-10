#include "../common.hpp"
#include "../game.hpp"
#include "hyperparameter_loader.hpp"
#include "learn.hpp"

void supervisedLearn() {
    // clang-format off
    HyperparameterLoader settings("supervised_learn_settings.txt");
    float train_rate_threshold  = settings.get<float>("train_rate_threshold");
    bool data_augmentation      = settings.get<bool>("data_augmentation");
    bool load_multi_dir         = settings.get<bool>("load_multi_dir");
    bool break_near_24h         = settings.get<bool>("break_near_24h");
    int64_t batch_size          = settings.get<int64_t>("batch_size");
    int64_t max_step            = settings.get<int64_t>("max_step");
    std::string train_kifu_path = settings.get<std::string>("train_kifu_path");
    // clang-format on

    //ディレクトリを逐次的に展開していく場合、まず展開するパス名を取得する
    std::vector<std::string> dir_paths;
    if (load_multi_dir) {
#ifdef _MSC_VER
        namespace sys = std::filesystem;
#elif __GNUC__
        namespace sys = std::experimental::filesystem;
#endif
        const sys::path dir(train_kifu_path);
        for (sys::directory_iterator p(dir); p != sys::directory_iterator(); p++) {
            const std::string path = p->path().string();
            if (path.find("dlshogi_with_gct") == std::string::npos) {
                continue;
            }
            dir_paths.push_back(p->path().string());
        }
        std::shuffle(dir_paths.begin(), dir_paths.end(), engine);
        train_kifu_path = dir_paths[0];
    }

    //データを取得
    std::vector<LearningData> train_data = loadHCPE(train_kifu_path, data_augmentation);

    //どのEpochでどのデータを使っているかを記録する
    std::ofstream epoch_log("epoch_log.txt");
    dout(std::cout, epoch_log) << "dir_path.size() = " << dir_paths.size() << std::endl;
    epoch_log << "0 0 " << train_data.size() << std::endl;

    const std::string prefix = "supervised";

    //エポックを超えたステップ数を初期化
    int64_t global_step = loadStepNumFromLog(prefix + "_train_log.txt");

    //学習クラスを生成
    LearnManager<LearningModel> learn_manager(prefix, global_step);

    //24時間超えそうになったら切るためのタイマー
    constexpr int64_t TIME_LIMIT = 23.50 * 3600;
    Timer timer;

    //学習開始
    for (int64_t epoch = 1; global_step < max_step; epoch++) {
        //データをシャッフル
        std::shuffle(train_data.begin(), train_data.end(), engine);

        for (uint64_t step = 0; batch_size * (step + 1) <= train_data.size() && global_step < max_step; step++) {
            //バッチサイズ分データを確保
            std::vector<LearningData> curr_data(train_data.begin() + batch_size * step,
                                                train_data.begin() + batch_size * (step + 1));

            learn_manager.learnOneStep(curr_data, ++global_step);

            if (break_near_24h && timer.elapsedSeconds() >= TIME_LIMIT) {
                break;
            }
        }

        if (break_near_24h && timer.elapsedSeconds() >= TIME_LIMIT) {
            break;
        }

        if (load_multi_dir) {
            train_data = loadHCPE(dir_paths[epoch % dir_paths.size()], data_augmentation);
            epoch_log << epoch << " " << global_step << " " << train_data.size() << std::endl;
        }
    }

    //学習パラメータを保存
    learn_manager.saveModelAsDefaultName();

    std::cout << "finish supervisedLearn" << std::endl;
}