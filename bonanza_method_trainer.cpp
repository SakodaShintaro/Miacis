#include"bonanza_method_trainer.hpp"
#include"position.hpp"
#include"searcher.hpp"
#include"eval_params.hpp"
#include"usi_options.hpp"
#include"network.hpp"
#include<chrono>
#include<vector>
#include<algorithm>
#include<utility>
#include<functional>
#include<iostream>
#include<iomanip>
#include<thread>

extern USIOption usi_option;

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
#ifndef USE_MCTS
        } else if (name == "search_depth") {
            ifs >> usi_option.depth_limit;
            usi_option.depth_limit *= PLY;
#endif
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
    Searcher::stop_signal = false;

    //評価関数ロード
    eval_params->readFile();
    std::cout << "eval_params->printHistgram()" << std::endl;
    eval_params->printHistgram();
    eval_params->writeFile("before_learn.bin");
#ifndef USE_NN
    learning_parameters = std::make_unique<EvalParams<LearnEvalType>>();
    learning_parameters->copy(*eval_params);
#endif

    //勾配初期化
    grad_ = std::make_unique<EvalParams<LearnEvalType>>();
    grad_->clear();
    if (OPTIMIZER_NAME == "ADAGRAD" || OPTIMIZER_NAME == "RMSPROP") {
        //AdaGrad, RMSPropならRMSgrad_だけ準備する
        RMSgrad_ = std::make_unique<EvalParams<LearnEvalType>>();
        RMSgrad_->clear();
        RMSgrad_->readFile("RMSgrad.bin");
        std::cout << "RMSgrad->printHistgram()" << std::endl;
        RMSgrad_->printHistgram();
    } else if (OPTIMIZER_NAME == "ADADELTA") {
        //AdaDeltaならRMSgrad_とRMSdelta_を準備する
        RMSgrad_ = std::make_unique<EvalParams<LearnEvalType>>();
        RMSgrad_->clear();
        RMSgrad_->readFile("RMSgrad.bin");
        std::cout << "RMSgrad->printHistgram()" << std::endl;
        RMSgrad_->printHistgram();

        RMSdelta_ = std::make_unique<EvalParams<LearnEvalType>>();
        RMSdelta_->clear();
        RMSdelta_->readFile("RMSdelta.bin");
        std::cout << "RMSdelta->printHistgram()" << std::endl;
        RMSdelta_->printHistgram();
    }

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
#ifdef USE_NN
    loss_[0] = L1_GRAD_COEFFICIENT * eval_params->sumAbs();
#else
    loss_ = L1_GRAD_COEFFICIENT * eval_params->sumAbs();
#endif
    learned_games_num = 0;
    learned_position_num = 0;
    succeeded_position_num = 0;
    game_index_ = 0;
#ifdef USE_NN
    ordering_num_.assign(MAX_MOVE_LIST_SIZE, 0);
#endif

    //log_file_の設定
    log_file_.open("bonanza_method_log.txt", std::ios::out);
#ifdef USE_NN
    log_file_ << "elapsed\tgame_num\tposition_num\tloss_average\ttop1\ttop3\ttop5\ttop10\ttop20\ttop50" << std::endl;
#else
    log_file_ << "elapsed\tgame_num\tposition_num\tloss_average\tAccuracy\teval_params.abs()" << std::endl;
#endif
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
    auto searcher = std::make_unique<Searcher>(usi_option.USI_Hash, usi_option.thread_num);

    //使用する棋譜のidは各スレッドで共有(std::atomic<int>)
    while (true) {
        //局面を初期化(初期局面でいいはず)
        Position position(*eval_params);

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

            if (move_list.size() == 0) {
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

#ifdef USE_NN
            //一致率の計算
            Network::scoreByPolicy(move_list, position.policy(), 100000);
            sort(move_list.begin(), move_list.end(), std::greater<Move>());
            for (int32_t i = 0; i < move_list.size(); i++) {
                if (move_list[i] == teacher_move) {
                    ordering_num_[i]++;
                    break;
                }
            }
            std::vector<CalcType> teacher(OUTPUT_DIM, 0.0);
            teacher[teacher_move.toLabel()] = 1.0;
            teacher[POLICY_DIM] = (CalcType)(position.color() == BLACK ? game.result : 1.0 - game.result);
            MUTEX.lock();
            loss_ = loss_ + addGrad(*grad_, position, teacher);
            MUTEX.unlock();
#else
            //全ての指し手に評価値をつけてPVを設定する
            //この評価値は手番側から見たものだから、教師手が最大になってほしい
            std::vector<std::pair<Move, std::vector<Move>>> move_and_pv;
            searcher->clearHistory();
            for (Move& move : move_list) {
                position.doMove(move);
                searcher->resetPVTable();
                move.score = -searcher->search<true>(position, MIN_SCORE, MAX_SCORE, usi_option.depth_limit, 0);
                position.undo();
                std::vector<Move> pv = searcher->pv();
                pv.insert(pv.begin(), move);
                move_and_pv.emplace_back(move, pv);
            }

            //ソート
            std::sort(move_and_pv.begin(), move_and_pv.end(), std::greater<std::pair<Move, std::vector<Move>>>());

            //教師手を探す
            int teacher_index = -1;
            for (uint32_t k = 0; k < move_and_pv.size(); k++) {
                if (move_and_pv[k].first == teacher_move) {
                    teacher_index = k;
                    break;
                }
            }
            if (teacher_index == -1) {
                //生成された合法手の中に教師手が含まれなかったということ
                //飛・角・歩とかの不成はここに来る
                //学習には適さないのでスキップする
                position.doMove(teacher_move);
                continue;
            } else if (teacher_index == 0) {
                //一致したということ
                learned_position_num++;
                succeeded_position_num++;
                position.doMove(teacher_move);
                continue;
            }

            //教師手を先頭に持ってくる
            std::swap(move_and_pv[0], move_and_pv[teacher_index]);

            //教師手のpvを棋譜のものに変更する
            //これやらないほうがいいのでは
            //for (auto l = 0; l < move_and_pv[0].second.size(); ++l) {
            //    move_and_pv[0].second[l] = game.moves[j + l];
            //}

            //教師手以外をスコアでソート
            std::sort(move_and_pv.begin() + 1, move_and_pv.end(), std::greater<std::pair<Move, std::vector<Move>>>());

            //Lossの本体:第1項 教師手との差
            //教師手のリーフノードの特徴量を用意
            Features teacher_leaf_element = getLeafFeatures(position, move_and_pv[0].second);

            uint32_t fail_move_num = 0;
            double progress = std::min(static_cast<double>(j) / 120.0, 0.85);
            Score margin = Score(10) + (int)(256 * progress);
            //各パラメータについて勾配を計算
            //各指し手についてリーフに出てくる特徴量に対応する勾配を増やし,その分だけ教師手リーフに出てくる特徴量に対応する勾配を減らす
            for (uint32_t m = 1; m < move_and_pv.size(); m++) {
                if (move_and_pv[m].first.score + margin >= move_and_pv[0].first.score) {
                    //どれだけの手が教師手より良いと誤って判断されたのかカウントする
                    fail_move_num++;
                } else {
                    break;
                }
            }
            for (uint32_t m = 1; m < 1 + fail_move_num; m++) {
                auto diff = move_and_pv[m].first.score + margin - move_and_pv[0].first.score;
                assert(diff >= 0);

                //atomicだと+=演算子使えないっぽい
                loss_ = loss_ + loss_function(diff) / fail_move_num;
                //勾配を変化させる量は損失を微分した値 / ダメだった手の数
                double delta = d_loss_function(diff) / fail_move_num;
                if (position.color() == WHITE) {
                    delta = -delta;
                }

                //リーフノードの特徴量
                Features leaf_element = getLeafFeatures(position, move_and_pv[m].second);

                MUTEX.lock();
                //高く評価しすぎた手について勾配を増やす
                updateGradient(*grad_, leaf_element, static_cast<LearnEvalType>(delta));
                //教師手について勾配を減らす
                updateGradient(*grad_, teacher_leaf_element, static_cast<LearnEvalType>(-delta));
                MUTEX.unlock();
            }
#endif

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

#ifdef USE_NN
    //更新して書き出し
    updateParams(*eval_params, *grad_);
    eval_params->writeFile();

    //学習情報を出力
    std::cout << "loss_average = " << (loss_[0] + loss_[1]) / learned_position_num << " 最大変化量 = " << LEARN_RATE * grad_->maxAbs() << std::endl;

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
#else
    //double型の学習用重みをアップデート
    if (OPTIMIZER_NAME== "SGD") {
        updateParamsSGD(*learning_parameters, *grad_);
    }

    //通常型のパラメータ配列にコピーして書き出し
    eval_params->copy(*learning_parameters);
    eval_params->writeFile();

    //学習情報を出力
    std::cout << "SumLoss = " << loss_ << ", L1_PENALTY = " << L1_GRAD_COEFFICIENT * eval_params->sumAbs() << ", loss_average = " << loss_ / learned_position_num
        << ", Accuracy = " << (double)succeeded_position_num / learned_position_num * 100 << ", grad.maxAbs() = " << grad_->maxAbs()
        << ", eval_params->abs() = " << eval_params->sumAbs() << std::endl;
    eval_params->printHistgram(500);

    timestamp();
    log_file_ << learned_games_num_sum << "\t" << learned_position_num_sum << "\t" << loss_ / learned_position_num
        << "\t" << (double)succeeded_position_num / learned_position_num * 100 << "\t" << eval_params->sumAbs() << std::endl;

    loss_ = L1_GRAD_COEFFICIENT * eval_params->sumAbs();
#endif

    //勾配初期化
    grad_->clear();

    //学習情報を0に戻す
    learned_games_num = 0;
    learned_position_num = 0;
    succeeded_position_num = 0;
}

double BonanzaMethodTrainer::loss_function(int score_diff) {
    //どれかを利用する
    if (LOSS_FUNCTION_MODE == 0) {
        //sigmoid
        return sigmoid(score_diff, BonanzaMethodTrainer::gain);
    } else if (LOSS_FUNCTION_MODE == 1) {
        //線形
        return linear_coefficient * score_diff;
    } else {
        //ここには来ないはず
        assert(false);
        return 0.0;
    }
}

double BonanzaMethodTrainer::d_loss_function(int score_diff) {
    //どれかを利用する
    if (LOSS_FUNCTION_MODE == 0) {
        //sigmoid
        return d_sigmoid(score_diff, BonanzaMethodTrainer::gain);
    } else if (LOSS_FUNCTION_MODE == 1) {
        //線形
        return linear_coefficient;
    } else {
        //ここには来ないはず
        assert(false);
        return 0.0;
    }
}