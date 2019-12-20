#include "game_generator.hpp"
#include <thread>

void GameGenerator::genGames(int64_t game_num) {
    //生成対局数を設定
    game_num_ = game_num;

    //生成スレッドを生成
    std::vector<std::thread> threads;
    for (int64_t i = 0; i < usi_options_.thread_num; i++) {
        threads.emplace_back(&GameGenerator::genSlave, this);
    }

    //生成スレッドが終わるのを待つ
    for (auto& th : threads) {
        th.join();
    }
}

void GameGenerator::genSlave() {
    //キュー
    std::vector<float> features;
    std::vector<std::stack<int32_t>> hash_indices;
    std::vector<std::stack<int32_t>> actions;
    std::vector<int32_t> ids;

    //容量の確保
    features.reserve(usi_options_.search_batch_size * INPUT_CHANNEL_NUM * SQUARE_NUM);
    hash_indices.reserve(usi_options_.search_batch_size);
    actions.reserve(usi_options_.search_batch_size);
    ids.reserve(usi_options_.search_batch_size);

    //このスレッドが管理するデータら
    std::vector<Game> games(usi_options_.search_batch_size);
    std::vector<Position> positions(usi_options_.search_batch_size);

    //探索クラスの生成,初期局面を探索する準備
    std::vector<std::unique_ptr<SearcherForGenerate>> searchers;
    for (int32_t i = 0; i < usi_options_.search_batch_size; i++) {
        searchers.emplace_back(std::make_unique<SearcherForGenerate>( i, usi_options_, Q_dist_lambda_, features, hash_indices, actions, ids));
        searchers[i]->prepareForCurrPos(positions[i]);
    }

    //今からi番目のものが担当する番号を初期化
    std::vector<int64_t> nums(usi_options_.search_batch_size);
    for (int32_t i = 0; i < usi_options_.search_batch_size; i++) {
        nums[i] = game_num_--;
    }

    //GPUで評価する関数
    auto evalWithGPU = [&](){
        gpu_mutex.lock();
        torch::NoGradGuard no_grad_guard;
        std::pair<std::vector<PolicyType>, std::vector<ValueType>> result = evaluator_->policyAndValueBatch(features);
        gpu_mutex.unlock();
        const std::vector<PolicyType>& policies = result.first;
        const std::vector<ValueType>& values = result.second;

        for (uint64_t i = 0; i < hash_indices.size(); i++) {
            //何番目のsearcherが持つハッシュテーブルのどの位置に書き込むかを取得
            auto& current_node = searchers[ids[i]]->hash_table_[hash_indices[i].top()];

            //policyを設定
            std::vector<float> legal_moves_policy(current_node.moves.size());
            for (uint64_t j = 0; j < current_node.moves.size(); j++) {
                legal_moves_policy[j] = policies[i][current_node.moves[j].toLabel()];
                assert(!std::isnan(legal_moves_policy[j]));
            }
            current_node.nn_policy = softmax(legal_moves_policy);

            //valueを設定
            current_node.value = values[i];

            //これいらない気がするけどハッシュテーブル自体の構造は変えたくないので念の為
            current_node.evaled = true;

            searchers[ids[i]]->backup(hash_indices[i], actions[i]);
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

        for (int32_t i = 0; i < usi_options_.search_batch_size; i++) {
            if (nums[i] <= 0) {
                //担当するゲームのidが0以下だったらスキップ
                continue;
            }

            if (searchers[i]->shouldStop()) {
                if (Searcher::stop_signal) {
                    return;
                }
                //探索結果を取得して次の局面へ遷移
                OneTurnElement result = searchers[i]->resultForCurrPos(positions[i]);
                positions[i].doMove(result.move);
                games[i].elements.push_back(result);

                float score;
                if (positions[i].isFinish(score)) {
                    //決着したので最終結果を設定
                    games[i].result = (positions[i].color() == BLACK ? score : -score);

                    //データを送る
                    rb_.push(games[i]);

                    //次の対局へ向かう
                    //まず担当するゲームのidを取得
                    nums[i] = game_num_--;
                    if (nums[i] > 0) {
                        //0より大きい場合のみ継続
                        //局面の初期化
                        positions[i].init();

                        //初期化。内部のvector<OneTurnElement>がどうなるのかよくわからない
                        games[i] = Game();

                        //次のルート局面の展開
                        searchers[i]->prepareForCurrPos(positions[i]);
                    }
                } else {
                    //次のルート局面を展開
                    searchers[i]->prepareForCurrPos(positions[i]);
                }
            } else {
                //引き続き同じ局面について探索
                //1回分探索してGPUキューへ評価要求を貯める
                searchers[i]->select(positions[i]);
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