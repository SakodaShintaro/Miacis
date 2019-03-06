#include "game_generator.hpp"
#include "usi_options.hpp"
#include <thread>

void GameGenerator::genGames(int64_t game_num) {
    //生成対局数を設定
    game_num_ = game_num;

    //生成スレッドを生成
    std::vector<std::thread> threads;
    for (int64_t i = 0; i < usi_option.thread_num; i++) {
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

    //容量の確保
    features.reserve(usi_option.search_batch_size * INPUT_CHANNEL_NUM * SQUARE_NUM);
    hash_indices.reserve(usi_option.search_batch_size);
    actions.reserve(usi_option.search_batch_size);
    ids.reserve(usi_option.search_batch_size);

    //このスレッドが管理するデータら
    std::vector<Game> games(usi_option.search_batch_size);
    std::vector<Position> positions(usi_option.search_batch_size);

    //探索クラスの生成,初期局面を探索する準備
    std::vector<SearcherForGenerate> searchers;
    for (int32_t i = 0; i < usi_option.search_batch_size; i++) {
        searchers.emplace_back(usi_option.search_limit, i, features, hash_indices, actions, ids);
        searchers[i].prepareForCurrPos(positions[i]);
    }

    //今からi番目のものが担当する番号を初期化
    std::vector<int64_t> nums(usi_option.search_batch_size);
    for (int32_t i = 0; i < usi_option.search_batch_size; i++) {
        nums[i] = game_num_--;
    }

    //GPUで評価する関数
    auto evalWithGPU = [&](){
        gpu_mutex.lock();
        auto result = evaluator_->policyAndValueBatch(features);
        gpu_mutex.unlock();
        auto policies = result.first;
        auto values = result.second;

        for (int32_t i = 0; i < hash_indices.size(); i++) {
            //何番目のsearcherが持つハッシュテーブルのどの位置に書き込むかを取得
            auto& current_node = searchers[ids[i]].hash_table_[hash_indices[i].top()];

            //policyを設定
            std::vector<float> legal_moves_policy(current_node.moves.size());
            for (int32_t j = 0; j < current_node.moves.size(); j++) {
                legal_moves_policy[j] = policies[i][current_node.moves[j].toLabel()];
                assert(!std::isnan(legal_moves_policy[j]));
            }
            current_node.nn_policy = softmax(legal_moves_policy);

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
        //キューのリセット
        features.clear();
        hash_indices.clear();
        actions.clear();
        ids.clear();

        for (int32_t i = 0; i < usi_option.search_batch_size; i++) {
            if (nums[i] <= 0) {
                //担当するゲームのidが0以下だったらスキップ
                continue;
            }

            if (searchers[i].shouldStop()) {
                if (usi_option.stop_signal) {
                    return;
                }
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
                searchers[i].select(positions[i]);
            }
        }

        //numsが正であるものが一つでもあるかどうかを確認
        if (std::find_if(nums.begin(), nums.end(), [](const auto& e){ return e > 0; }) == nums.end()) {
            //一つもなかったら終了
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