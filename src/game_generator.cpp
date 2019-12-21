﻿#include "game_generator.hpp"
#include <thread>

void GameGenerator::genGames() {
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
    constexpr int64_t WORKER_NUM = 1;

    //容量の確保
    gpu_queue_.inputs.reserve(WORKER_NUM * usi_options_.search_batch_size * INPUT_CHANNEL_NUM * SQUARE_NUM);
    gpu_queue_.hash_tables.reserve(WORKER_NUM * usi_options_.search_batch_size);
    gpu_queue_.indices.reserve(WORKER_NUM * usi_options_.search_batch_size);

    std::vector<std::unique_ptr<GenerateWorker>> workers(WORKER_NUM);

    for (int32_t i = 0; i < WORKER_NUM; i++) {
        workers[i] = std::make_unique<GenerateWorker>(usi_options_, gpu_queue_, Q_dist_lambda_, replay_buffer_);
        workers[i]->prepareForCurrPos();
    }

    //初期局面をまず1回評価
    evalWithGPU();

    while (!stop_signal) {
        //キューのリセット
        gpu_queue_.inputs.clear();
        gpu_queue_.hash_tables.clear();
        gpu_queue_.indices.clear();

        for (int32_t i = 0; i < WORKER_NUM; i++) {
            workers[i]->select();
        }

        //GPUで評価
        //探索結果が既存ノードへの合流,あるいは詰みのときには計算要求がないので一応空かどうかを確認
        //複数のsearcherが同時にそうなる確率はかなり低そうだが
        if (!gpu_queue_.inputs.empty()) {
            evalWithGPU();
        }

        for (int32_t i = 0; i < WORKER_NUM; i++) {
            workers[i]->backup();
        }
    }
}

std::vector<FloatType> GameGenerator::dirichletDistribution(uint64_t k, FloatType alpha) {
    static std::default_random_engine engine(std::random_device{}());
    std::gamma_distribution<FloatType> gamma(alpha, 1.0);
    std::vector<FloatType> dirichlet(k);

    //kが小さく、不運が重なるとsum = 0となり0除算が発生してしまうことがあるので小さい値で初期化
    FloatType sum = 1e-6;
    for (uint64_t i = 0; i < k; i++) {
        sum += (dirichlet[i] = gamma(engine));
    }
    for (uint64_t i = 0; i < k; i++) {
        dirichlet[i] /= sum;
    }
    return dirichlet;
}

void GameGenerator::evalWithGPU() {
    gpu_mutex.lock();
    torch::NoGradGuard no_grad_guard;
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> result = neural_network_->policyAndValueBatch(gpu_queue_.inputs);
    gpu_mutex.unlock();
    const std::vector<PolicyType>& policies = result.first;
    const std::vector<ValueType>& values = result.second;

    for (uint64_t i = 0; i < gpu_queue_.hash_tables.size(); i++) {
        //何番目のsearcherが持つハッシュテーブルのどの位置に書き込むかを取得
        UctHashEntry& curr_node = gpu_queue_.hash_tables[i].get()[gpu_queue_.indices[i]];

        //policyを設定
        std::vector<float> legal_moves_policy(curr_node.moves.size());
        for (uint64_t j = 0; j < curr_node.moves.size(); j++) {
            legal_moves_policy[j] = policies[i][curr_node.moves[j].toLabel()];
            assert(!std::isnan(legal_moves_policy[j]));
        }
        curr_node.nn_policy = softmax(legal_moves_policy);

        //policyにディリクレノイズを付与
        constexpr FloatType epsilon = 0.25;
        std::vector<FloatType> dirichlet = dirichletDistribution(curr_node.moves.size(), 0.15);
        for (uint64_t j = 0; j < curr_node.moves.size(); j++) {
            curr_node.nn_policy[j] = (FloatType) ((1.0 - epsilon) * curr_node.nn_policy[j] + epsilon * dirichlet[j]);
        }

        //valueを設定
        curr_node.value = values[i];

        curr_node.evaled = true;
    }
}

GenerateWorker::GenerateWorker(const SearchOptions& usi_options, GPUQueue& gpu_queue, FloatType Q_dist_lambda,
                               ReplayBuffer& rb)
: usi_options_(usi_options),
  gpu_queue_(gpu_queue),
  Q_dist_lambda_(Q_dist_lambda),
  replay_buffer_(rb),
  hash_table_(usi_options.USI_Hash),
  searcher_(usi_options, hash_table_, gpu_queue),
  root_raw_value_{}
{}

void GenerateWorker::prepareForCurrPos() {
    //古いハッシュを削除
    hash_table_.deleteOldHash(position_, false);

    //ルートノードの展開
    std::stack<int32_t> indices;
    std::stack<int32_t> actions;
    hash_table_.root_index = searcher_.expand(position_, indices, actions);
}

OneTurnElement GenerateWorker::resultForCurrPos() {
    const UctHashEntry& root_node = hash_table_[hash_table_.root_index];
    if (root_node.moves.empty()) {
        position_.print();
        std::cout << "in resultForCurrPos(), root_node.moves.empty()" << std::endl;
        std::exit(1);
    }
    if (root_node.sum_N == 0) {
        position_.print();
        std::cout << "in resultForCurrPos(), root_node.sum_N == 0" << std::endl;
        std::exit(1);
    }

    const std::vector<int32_t>& N = root_node.N;
    if (root_node.sum_N != std::accumulate(N.begin(), N.end(), 0)) {
        std::cout << "root_node.sum_N != std::accumulate(N.begin(), N.end(), 0)" << std::endl;
        std::exit(1);
    }

    //探索回数最大の手を選択する
    int32_t best_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());

    //選択した着手の勝率の算出
    //詰みのときは未展開であることに注意する
    FloatType best_value = (root_node.child_indices[best_index] == UctHashTable::NOT_EXPANDED ? MAX_SCORE :
                            hash_table_.expQfromNext(root_node, best_index));

    //教師データを作成
    OneTurnElement element;

    //valueのセット
    element.score = best_value;

    //policyのセット
    if (position_.turnNumber() < usi_options_.random_turn) {
        //分布に従ってランダムに行動選択
        //探索回数を正規化した分布
        //探索回数のsoftmaxを取ることを検討したほうが良いかもしれない
        std::vector<FloatType> N_dist(root_node.moves.size());
        //行動価値のsoftmaxを取った分布
        std::vector<FloatType> Q_dist(root_node.moves.size());
        for (uint64_t i = 0; i < root_node.moves.size(); i++) {
            if (N[i] < 0 || N[i] > root_node.sum_N) {
                std::cout << "N[i] < 0 || N[i] > root_node.sum_N" << std::endl;
                std::exit(1);
            }

            //探索回数を正規化
            N_dist[i] = (FloatType)N[i] / root_node.sum_N;

            //選択回数が0ならMIN_SCORE
            //選択回数が0ではないのに未展開なら詰み探索が詰みを発見したということなのでMAX_SCORE
            //その他は普通に計算
            Q_dist[i] = (N[i] == 0 ? MIN_SCORE : root_node.child_indices[i] == UctHashTable::NOT_EXPANDED ? MAX_SCORE : hash_table_.expQfromNext(root_node, i));
        }
        Q_dist = softmax(Q_dist, usi_options_.temperature_x1000 / 1000.0f);

        //教師分布のセット
        //(1)どちらの分布を使うべきか
        //(2)実際に行動選択をする分布と一致しているべきか
        //など要検討
        for (uint64_t i = 0; i < root_node.moves.size(); i++) {
            //N_distにQ_distの値を混ぜ込む
            N_dist[i] = (1 - Q_dist_lambda_) * N_dist[i] + Q_dist_lambda_ * Q_dist[i];

            //N_distを教師分布とする
            element.policy_teacher.push_back({ root_node.moves[i].toLabel(), N_dist[i] });
        }

        //N_distに従って行動選択
        element.move = root_node.moves[randomChoose(N_dist)];
    } else {
        //最良の行動を選択
        element.policy_teacher.push_back({ root_node.moves[best_index].toLabel(), 1.0f });
        element.move = root_node.moves[best_index];
    }

    //priorityを計算する用にNNの出力をセットする
    element.nn_output_policy.resize(POLICY_DIM, 0.0);
    for (uint64_t i = 0; i < root_node.moves.size(); i++) {
        element.nn_output_policy[root_node.moves[i].toLabel()] = root_node.nn_policy[i];
    }
    element.nn_output_value = root_raw_value_;

    return element;
}

void GenerateWorker::select() {
    if (hash_table_[hash_table_.root_index].sum_N >= usi_options_.search_limit) {
        //探索結果を取得して次の局面へ遷移
        OneTurnElement result = resultForCurrPos();
        position_.doMove(result.move);
        game_.elements.push_back(result);

        float score;
        if (position_.isFinish(score)) {
            //決着したので最終結果を設定
            game_.result = (position_.color() == BLACK ? score : -score);

            //データを送る
            replay_buffer_.push(game_);

            //次の対局へ向かう
            //まず担当するゲームのidを取得
            //0より大きい場合のみ継続
            //局面の初期化
            position_.init();

            //ゲームの初期化
            game_.elements.clear();
        }

        //次のルート局面を展開
        prepareForCurrPos();
    } else {
        //引き続き同じ局面について探索
        //探索してGPUキューへ評価要求を貯める
        //探索する前にニューラルネットワークの出力を保存しておく
        if (hash_table_[hash_table_.root_index].sum_N == 0) {
            root_raw_value_ = hash_table_[hash_table_.root_index].value;
        }
        for (int64_t j = 0; j < usi_options_.search_batch_size; j++) {
            searcher_.select(position_);
        }
    }
}

void GenerateWorker::backup() {
    searcher_.backupAll();
}