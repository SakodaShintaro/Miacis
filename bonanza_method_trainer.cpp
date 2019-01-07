#include"bonanza_method_trainer.hpp"
#include"position.hpp"
#include"MCTSearcher.hpp"
#include"usi_options.hpp"
#include<chrono>
#include<vector>
#include<algorithm>
#include<utility>
#include<functional>
#include<iostream>
#include<iomanip>
#include<thread>

static std::mutex MUTEX;

BonanzaMethodTrainer::BonanzaMethodTrainer(std::string settings_file_path) {
    std::ifstream ifs(settings_file_path);
    if (!ifs) {
        std::cerr << "fail to open setting_file(" << settings_file_path << ")" << std::endl;
        assert(false);
    }

    std::string name;
    while (ifs >> name) {
        if (name == "kifu_path") {
            ifs >> KIFU_PATH;
        } else if (name == "game_num") {
            ifs >> game_num_;
        } else if (name == "batch_size") {
            ifs >> BATCH_SIZE;
        } else if (name == "L1_grad_coefficient") {
            ifs >> L1_GRAD_COEFFICIENT;
        } else if (name == "optimizer") {
            ifs >> OPTIMIZER_NAME;
            if (!isLegalOptimizer()) {
                std::cerr << "Optimizerは[SGD, AdaGrad, RMSProp, AdaDelta]から選択" << std::endl;
                assert(false);
            }
        } else if (name == "learn_rate") {
            ifs >> LEARN_RATE;
        } else if (name == "loss_function_mode") {
            ifs >> LOSS_FUNCTION_MODE;
        } else if (name == "thread_num") {
            ifs >> THREAD_NUM;
        } else {
            std::cerr << "Error! There is no such setting." << std::endl;
            assert(false);
        }
    }
}

void BonanzaMethodTrainer::train() {
    std::cout << "start BonanzaMethod" << std::endl;

    //学習開始時間の設定
    start_time_ = std::chrono::steady_clock::now();

    //探索の設定を学習用に変更
    usi_option.draw_turn = 1024;
    usi_option.draw_score = -100;
    usi_option.stop_signal = false;

    //評価関数ロード
    learning_model_.load("cnn.model");

    //棋譜を読み込む
    std::cout << "start loadGames ...";
    games_ = loadGames(KIFU_PATH, game_num_);
    std::cout << " done" << std::endl;
    std::cout << "games.size() = " << games_.size() << std::endl;

    //棋譜シャッフル
    std::random_device seed;
    std::default_random_engine engine(seed());
    std::shuffle(games_.begin(), games_.end(), engine);

    //学習情報の初期化
    learned_games_num = 0;
    learned_position_num = 0;
    succeeded_position_num = 0;
    game_index_ = 0;
    ordering_num_.assign(MAX_MOVE_LIST_SIZE, 0);

    //log_file_の設定
    log_file_.open("bonanza_method_log.txt", std::ios::out);
    log_file_ << "elapsed\tgame_num\tposition_num\tloss_average\ttop1\ttop3\ttop5\ttop10\ttop20\ttop50" << std::endl;
    log_file_ << std::fixed;

    THREAD_NUM = std::min(std::max(1u, THREAD_NUM), std::thread::hardware_concurrency());
    std::cout << "使用するスレッド数 : " << THREAD_NUM << std::endl;
    std::cout << std::fixed;

    //学習開始
    std::vector<std::thread> learn_threads(THREAD_NUM);
    for (uint32_t i = 0; i < THREAD_NUM; ++i) {
        learn_threads[i] = std::thread(&BonanzaMethodTrainer::trainSlave, this, i);
    }
    for (auto& t : learn_threads) {
        t.join();
    }

    log_file_.close();
    std::cout << "finish BonanzaMethod" << std::endl;
}

void BonanzaMethodTrainer::trainSlave(uint32_t thread_id) {
    //探索クラスを準備
    auto searcher = std::make_unique<MCTSearcher<Node>>(usi_option.USI_Hash, usi_option.thread_num, learning_model_);

    //使用する棋譜のidは各スレッドで共有(std::atomic<int>)
    while (true) {
        //局面を初期化(初期局面でいいはず)
        Position position;

        const int game_id = game_index_++;

        if (game_id >= games_.size()) {
            break;
        }

        //棋譜
        const Game& game = games_[game_id];

        //最後の方は王手ラッシュで意味がないと判断して0.8をかけた手数で打ち切り
        const int num_moves = static_cast<int>(game.moves.size() * 0.8);

        //局面を1手ごと進めながら各局面について損失,勾配を計算する
        for (int j = 0; j < num_moves; j++) {
            //全指し手生成
            std::vector<Move> move_list = position.generateAllMoves();

            if (move_list.empty()) {
                //おそらく詰みの局面には行かないはずだけど……
                break;
            }

            if (move_list.size() == 1) {
                //可能な指し手が1つしかないなら学習に適してないのでスキップ
                //一手進める
                position.doMove(game.moves[j]);
                continue;
            }

            //教師となる指し手
            const Move teacher_move = game.moves[j];

            //一致率の計算
//            Network::scoreByPolicy(move_list, position.policy(), 100000);
//            sort(move_list.begin(), move_list.end(), std::greater<>());
//            for (int32_t i = 0; i < move_list.size(); i++) {
//                if (move_list[i] == teacher_move) {
//                    ordering_num_[i]++;
//                    break;
//                }
//            }
//            std::vector<CalcType> teacher(OUTPUT_DIM, 0.0);
//            teacher[teacher_move.toLabel()] = 1.0;
//            teacher[POLICY_DIM] = (CalcType)(position.color() == BLACK ? game.result : 1.0 - game.result);
//            MUTEX.lock();
//            loss_ = loss_ + addGrad(*grad_, position, teacher);
//            MUTEX.unlock();

            //学習した局面を増やす
            learned_position_num++;

            //一手進める
            position.doMove(game.moves[j]);

        } //ここで1局分が終わる

        //必要局数に達していたらパラメータ更新
        if (++learned_games_num == BATCH_SIZE) {
            update();
        }
    }

    //最後にあまりが残っていると思う
    std::cout << "learned_games_num = " << learned_games_num << std::endl;

    //必要局数の8割くらいやってたらもったいないからパラメータ更新する
    if (learned_games_num >= BATCH_SIZE * 0.8) {
        update();
    }
    std::cout << "BonanzaMethod finish" << std::endl;
}

Features BonanzaMethodTrainer::getLeafFeatures(Position& pos,const std::vector<Move>& pv) {
    //pvがうまくとれているとは限らないので局面を動かした回数を数えておく
    int counter = 0;
    for (Move move : pv) {
        if (pos.isLegalMove(move)) {
            pos.doMove(move);
            counter++;
        } else {
            break;
        }
    }
    Features ee = pos.features();
    for (int i = 0; i < counter; i++) {
        pos.undo();
    }
    return ee;
}

void BonanzaMethodTrainer::update() {
    std::unique_lock<std::mutex> lock(MUTEX);
    //学習局面、局数を合算
    learned_games_num_sum += learned_games_num;
    learned_position_num_sum += learned_position_num;

    auto time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(time - start_time_);
    int r_minutes = (int)((double)(games_.size() - learned_games_num_sum) / learned_games_num_sum * elapsed.count());
    printf("finish %5d games, 経過時間:%3ld時間%3ld分, 残り時間:%3d時間%3d分 ",
        (int)learned_games_num_sum, elapsed.count() / 60, elapsed.count() % 60, r_minutes / 60, r_minutes % 60);

    //更新して書き出し
    learning_model_.save(MODEL_PATH);

    //学習情報を出力
    std::cout << "loss_average = " << (loss_[0] + loss_[1]) / learned_position_num << std::endl;

    //指し手の一致率
    for (int32_t i = 0; i < 50; i++) {
        if (i == 0 || i == 2 || i == 4 || i == 9 || i == 19 || i == 49) {
            printf("%3d個目以内に入った割合:%6.2f\n", i + 1, 100.0 * ordering_num_[i] / learned_position_num);
        }
        ordering_num_[i + 1] += ordering_num_[i];
    }

    timestamp();
    log_file_ << learned_games_num_sum << "\t" << learned_position_num_sum << "\t" << (loss_[0]  + loss_[1]) / learned_position_num;
    for (int32_t i = 0; i < 50; i++) {
        if (i == 0 || i == 2 || i == 4 || i == 9 || i == 19 || i == 49) {
            log_file_ << "\t" << (double)ordering_num_[i] / learned_position_num;
        }
    }
    log_file_ << std::endl;

    ordering_num_.assign(MAX_MOVE_LIST_SIZE, 0);
    loss_[0] = loss_[1] = 0.0;

    //学習情報を0に戻す
    learned_games_num = 0;
    learned_position_num = 0;
    succeeded_position_num = 0;
}