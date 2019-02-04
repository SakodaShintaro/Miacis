#include <atomic>
#include <thread>
#include "game_generator2.hpp"
#include "usi_options.hpp"

void GameGenerator2::genGames() {
    //生成スレッドを生成
    std::vector<std::thread> threads;
    for (int64_t i = 0; i < 1; i++) {
        threads.emplace_back(&GameGenerator2::genSlave, this, i);
    }

    //生成スレッドが終わるのを待つ
    for (auto& th : threads) {
        th.join();
    }
}

void GameGenerator2::genSlave(int64_t id) {
    //キュー
    std::vector<float> features;
    std::vector<std::stack<int32_t>> hash_indices;
    std::vector<std::stack<int32_t>> actions;
    std::vector<int32_t> ids;

    //このスレッドが管理するデータら
    std::vector<Game> games(BATCH_SIZE);
    std::vector<Position> positions(BATCH_SIZE);

    //探索クラスの生成
    std::vector<SearcherForGen2> searchers;
    for (int32_t i = 0; i < BATCH_SIZE; i++) {
        searchers.emplace_back(usi_option.USI_Hash, i, features, hash_indices, actions, ids);
        searchers[i].prepareForCurrPos(positions[i]);
    }

    //今からi番目のものが担当する番号を初期化
    std::vector<int64_t> nums(BATCH_SIZE);
    for (int32_t i = 0; i < BATCH_SIZE; i++) {
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

        for (int32_t i = 0; i < BATCH_SIZE; i++) {
            if (nums[i] <= 0) {
                //担当するゲームのidが0以下だったらスキップ
                continue;
            }

            if (searchers[i].shouldGoNextPosition()) {
                //探索結果を取得して次の局面へ遷移
                auto result = searchers[i].resultForCurrPos(positions[i]);
                positions[i].doMove(result.first);
                games[i].moves.push_back(result.first);
                games[i].teachers.push_back(result.second);

                std::cout << "i = " << i << std::endl;
                positions[i].print();

                bool curr_game_finish = false;
                if (positions[i].turn_number() >= usi_option.draw_turn) {
                    //長手数による引き分けとしてリプレイバッファにデータを送る
                    games[i].result = (MAX_SCORE + MIN_SCORE) / 2;
                    curr_game_finish = true;
                } else if (!searchers[i].prepareForCurrPos(positions[i])) { //次局面を展開
                    //投了なのでまずリプレイバッファにデータを送る
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
        for (int32_t i = 0; i < BATCH_SIZE; i++) {
            if (nums[i] > 0) {
                all_finish = false;
            }
        }
        if (all_finish) {
            break;
        }

        //GPUで評価
        evalWithGPU();
    }
}