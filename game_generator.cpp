#include <atomic>
#include <thread>
#include "game_generator.hpp"
#include "usi_options.hpp"

#ifdef USE_PARALLEL_SEARCHER

void GameGenerator::genGames(int64_t game_num) {
    //キューのクリア
    clearEvalQueue();

    game_num_ = game_num;

    //GPUスレッドを生成
    running_ = true;
    std::thread gpu_thread(&GameGenerator::gpuFunc, this);

    //先に探索クラスを生成しておくと上手く動く
    for (int64_t i = 0; i < parallel_num_; i++) {
        searchers_.emplace_back(usi_option.USI_Hash, i, *this);
    }

    //生成スレッドを生成
    std::vector<std::thread> threads;
    for (int64_t i = 0; i < parallel_num_; i++) {
        threads.emplace_back(&GameGenerator::genSlave, this, i);
    }

    //生成スレッドが終わるのを待つ
    for (auto& th : threads) {
        th.join();
    }

    //GPUスレッドが終わるのを待つ
    running_ = false;
    gpu_thread.join();
}

void GameGenerator::genSlave(int64_t id) {
    //生成
    while (true) {
        int64_t num = game_num_--;
        if (num % 10 == 0) {
            std::cout << "num = " << num << std::endl;
        }
        if (num <= 0) {
            break;
        }

        Game game;
        Position pos;

        while (true) {
            //id番目のsearcherを使って探索
            auto result = searchers_[id].think(pos);
            if (result.first == NULL_MOVE) {
                game.result = (pos.color() == BLACK ? Game::RESULT_WHITE_WIN : Game::RESULT_BLACK_WIN);
                break;
            }
            pos.doMove(result.first);
            game.moves.push_back(result.first);
            game.teachers.push_back(result.second);

            if (pos.turn_number() >= usi_option.draw_turn) {
                game.result = Game::RESULT_DRAW_OVER_LIMIT;
                break;
            }
        }

        rb_.push(game);
    }
}

void GameGenerator::gpuFunc() {
    bool enough_batch_size = true;

    while (running_) {
        lock_expand_.lock();
        if (current_features_.empty()) {
            lock_expand_.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            continue;
        }

        if (!enough_batch_size && current_features_.size() < parallel_num_ / 2) {
            lock_expand_.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            enough_batch_size = true;
            continue;
        }

        enough_batch_size = false;

        //現在のキューを保存
        auto eval_features = current_features_;
        auto eval_hash_index_queue = current_hash_index_queue_;
        auto eval_thread_ids = current_thread_ids_;

        //カレントキューを入れ替える
        current_queue_index_ ^= 1;
        current_features_ = features_[current_queue_index_];
        current_features_.clear();
        current_hash_index_queue_ = hash_index_queues_[current_queue_index_];
        current_hash_index_queue_.clear();
        current_thread_ids_ = thread_ids_[current_queue_index_];
        current_thread_ids_.clear();
        lock_expand_.unlock();

        auto result = evaluator_.policyAndValueBatch(eval_features);
        auto policies = result.first;
        auto values = result.second;

        for (int32_t i = 0; i < eval_hash_index_queue.size(); i++) {
            //何番目のスレッドのどの位置に書き込むかを取得
            auto& current_node = searchers_[eval_thread_ids[i]].hash_table_[eval_hash_index_queue[i]];

            //policyを設定
            std::vector<float> legal_moves_policy(static_cast<unsigned long>(current_node.child_num));
            for (int32_t j = 0; j < current_node.child_num; j++) {
                legal_moves_policy[j] = policies[i][current_node.legal_moves[j].toLabel()];
            }
            current_node.nn_rates = softmax(legal_moves_policy);

            //valueを設定
            current_node.value = values[i];
            current_node.evaled = true;
        }
    }
}

void GameGenerator::clearEvalQueue() {
    current_queue_index_ = 0;
    for (int32_t i = 0; i < 2; i++) {
        features_[i].clear();
        hash_index_queues_[i].clear();
    }
    current_features_ = features_[current_queue_index_];
    current_hash_index_queue_ = hash_index_queues_[current_queue_index_];
}

#else

void GameGenerator::genGames(int64_t game_num) {
    //生成対局数を設定
    game_num_ = game_num;

    //生成スレッドを生成
    std::vector<std::thread> threads;
    for (int64_t i = 0; i < THREAD_NUM; i++) {
        threads.emplace_back(&GameGenerator::genSlave, this, i);
    }

    //生成スレッドが終わるのを待つ
    for (auto& th : threads) {
        th.join();
    }
}

void GameGenerator::genSlave(int64_t id) {
    //キュー
    std::vector<float> features;
    std::vector<std::stack<int32_t>> hash_indices;
    std::vector<std::stack<int32_t>> actions;
    std::vector<int32_t> ids;

    //並列化する総数をスレッド数で割ったものがこのスレッドで管理するべき数
    assert(parallel_num_ % THREAD_NUM == 0);
    const int64_t parallel_num = parallel_num_ / THREAD_NUM;

    //このスレッドが管理するデータら
    std::vector<Game> games(parallel_num);
    std::vector<Position> positions(parallel_num);

    //探索クラスの生成,初期局面を探索する準備
    std::vector<SearcherForGen> searchers;
    for (int32_t i = 0; i < parallel_num; i++) {
        searchers.emplace_back(usi_option.USI_Hash, i, features, hash_indices, actions, ids);
        searchers[i].prepareForCurrPos(positions[i]);
    }

    //今からi番目のものが担当する番号を初期化
    std::vector<int64_t> nums(parallel_num);
    for (int32_t i = 0; i < parallel_num; i++) {
        nums[i] = game_num_--;
    }

    //GPUで評価する関数
    auto evalWithGPU = [&](){
        gpu_mutex_.lock();
        auto result = evaluator_.policyAndValueBatch(features);
        gpu_mutex_.unlock();
        auto policies = result.first;
        auto values = result.second;

        for (int32_t i = 0; i < hash_indices.size(); i++) {
            //何番目のsearcherが持つハッシュテーブルのどの位置に書き込むかを取得
            auto &current_node = searchers[ids[i]].hash_table_[hash_indices[i].top()];

            //policyを設定
            std::vector<float> legal_moves_policy(static_cast<unsigned long>(current_node.child_num));
            for (int32_t j = 0; j < current_node.child_num; j++) {
                legal_moves_policy[j] = policies[i][current_node.legal_moves[j].toLabel()];
            }
            current_node.nn_rates = softmax(legal_moves_policy);

            //valueを設定
            current_node.value = values[i];

            //これいらない気がするけどハッシュテーブル自体の構造は変えたくないので念の為
            current_node.evaled = true;

            searchers[ids[i]].backup(hash_indices[i], actions[i]);
        }
    };

    //初期局面をまず1回評価
    evalWithGPU();

    while (true) {
        features.clear();
        hash_indices.clear();
        actions.clear();
        ids.clear();

        for (int32_t i = 0; i < parallel_num; i++) {
            if (nums[i] <= 0) {
                //担当するゲームのidが0以下だったらスキップ
                continue;
            }

            if (searchers[i].shouldStop()) {
                //探索結果を取得して次の局面へ遷移
                auto result = searchers[i].resultForCurrPos(positions[i]);
                positions[i].doMove(result.first);
                games[i].moves.push_back(result.first);
                games[i].teachers.push_back(result.second);

                bool curr_game_finish = false;
                if (positions[i].turn_number() >= usi_option.draw_turn) {
                    //長手数による引き分け
                    games[i].result = (MAX_SCORE + MIN_SCORE) / 2;
                    curr_game_finish = true;
                } else if (!searchers[i].prepareForCurrPos(positions[i])) { //次局面を展開
                    //投了
                    games[i].result = (positions[i].color() == BLACK ? Game::RESULT_WHITE_WIN : Game::RESULT_BLACK_WIN);
                    curr_game_finish = true;
                }

                if (curr_game_finish) {
                    //データを送る
                    rb_.push(games[i]);

                    //次の対局へ向かう
                    nums[i] = game_num_--;
                    positions[i].init();
                    games[i].moves.clear();
                    games[i].teachers.clear();

                    if (nums[i] > 0) {
                        searchers[i].prepareForCurrPos(positions[i]);
                    }
                }
            } else {
                //引き続き同じ局面について探索
                //1回分探索してGPUキューへ評価要求を貯める
                searchers[i].onePlay(positions[i]);
            }
        }

        //numsが正であるものが一つでもあるかどうかを確認
        bool all_finish = true;
        for (int32_t i = 0; i < parallel_num; i++) {
            if (nums[i] > 0) {
                all_finish = false;
                break;
            }
        }
        if (all_finish) {
            break;
        }

        //GPUで評価
        //探索結果が既存ノードへの合流,あるいは詰みのときには計算要求がないので一応空かどうかを確認
        //複数のsearcherが同時にそうなる確率はかなり低そうだが
        if (!ids.empty()) {
            evalWithGPU();
        }
    }
}

#endif