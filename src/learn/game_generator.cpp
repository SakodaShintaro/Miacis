#include "game_generator.hpp"
#include <thread>
#include <torch_tensorrt/torch_tensorrt.h>

void GameGenerator::genGames() {
    //まず最初のロード
    loadNeuralNetwork();
    need_load = false;

    //生成スレッドを生成
    std::vector<std::thread> threads;
    for (int64_t i = 0; i < search_options_.thread_num_per_gpu; i++) {
        threads.emplace_back(&GameGenerator::genSlave, this, i);
    }

    //生成スレッドが終わるのを待つ
    for (std::thread& th : threads) {
        th.join();
    }
}

void GameGenerator::genSlave(int64_t thread_id) {
    //スレッドごとにCUDAをセットしておかないとエラーが出る
    torch_tensorrt::set_device(gpu_id_);

    //Workerを準備
    std::vector<std::unique_ptr<GenerateWorker>> workers(worker_num_);
    for (int32_t i = 0; i < worker_num_; i++) {
        workers[i] = std::make_unique<GenerateWorker>(search_options_, gpu_queues_[thread_id], Q_dist_lambda_, replay_buffer_);
        workers[i]->prepareForCurrPos();
    }

    //初期局面をまず1回評価
    evalWithGPU(thread_id);

    //停止信号が来るまで生成し続けるループ
    while (!stop_signal) {
        //キューのリセット
        gpu_queues_[thread_id].inputs.clear();
        gpu_queues_[thread_id].hash_tables.clear();
        gpu_queues_[thread_id].indices.clear();

        //各Workerについて選択ステップを実行し評価要求を溜める
        for (int32_t i = 0; i < worker_num_; i++) {
            workers[i]->select();
        }

        //GPUで評価
        //探索結果が既存ノードへの合流,あるいは詰みのときには計算要求がないので一応空かどうかを確認
        //複数のWorkerが同時にそうなる確率はかなり低そうだが理論上はあり得るので
        if (!gpu_queues_[thread_id].inputs.empty()) {
            evalWithGPU(thread_id);
        }

        //各Workerについてbackup
        for (int32_t i = 0; i < worker_num_; i++) {
            workers[i]->backup();
        }

        //パラメータをロードし直す必要があれば実行
        //全スレッドが読み込もうとする必要はないので代表してid=0のスレッドに任せる
        if (need_load && thread_id == 0) {
            gpu_mutex.lock();
            loadNeuralNetwork();
            need_load = false;
            gpu_mutex.unlock();
        }
    }
}

std::vector<float> GameGenerator::dirichletDistribution(uint64_t k, float alpha) {
    std::gamma_distribution<float> gamma(alpha, 1.0);
    std::vector<float> dirichlet(k);

    //kが小さく、不運が重なるとsum = 0となり0除算が発生してしまうことがあるので小さい値で初期化
    float sum = 1e-6;
    for (uint64_t i = 0; i < k; i++) {
        sum += (dirichlet[i] = gamma(engine));
    }
    for (uint64_t i = 0; i < k; i++) {
        dirichlet[i] /= sum;
    }
    return dirichlet;
}

std::vector<float> GameGenerator::onehotNoise(uint64_t k) {
    std::uniform_int_distribution<uint64_t> dist(0, k - 1);
    std::vector<float> onehot(k, 0.0);
    onehot[dist(engine)] = 1.0;
    return onehot;
}

void GameGenerator::evalWithGPU(int64_t thread_id) {
    //順伝播計算
    gpu_mutex.lock();
    torch::NoGradGuard no_grad_guard;
    std::pair<std::vector<PolicyType>, std::vector<ValueType>> result = neural_network_.policyAndValueBatch(gpu_queues_[thread_id].inputs);
    gpu_mutex.unlock();

    const std::vector<PolicyType>& policies = result.first;
    const std::vector<ValueType>& values = result.second;

    //各入力が対応する置換表の適切なエントリーに計算結果を書き込んでいく
    for (uint64_t i = 0; i < gpu_queues_[thread_id].hash_tables.size(); i++) {
        //何番目のworkerが持つハッシュテーブルのどの位置に書き込むかを取得
        HashTable& curr_table = gpu_queues_[thread_id].hash_tables[i].get();
        Index curr_index = gpu_queues_[thread_id].indices[i];
        HashEntry& curr_node = curr_table[curr_index];

        //policyを設定
        //合法手だけ取ってからsoftmax関数にかける
        curr_node.nn_policy.resize(curr_node.moves.size());
        for (uint64_t j = 0; j < curr_node.moves.size(); j++) {
            curr_node.nn_policy[j] = policies[i][curr_node.moves[j].toLabel()];
        }
        curr_node.nn_policy = softmax(curr_node.nn_policy);

        //今回の位置がroot_indexだった場合のみpolicyにノイズを付与
        if (curr_index == curr_table.root_index) {
            std::vector<float> noise = (noise_mode_ == DIRICHLET ? dirichletDistribution(curr_node.moves.size(), noise_alpha_)
                                                                 : onehotNoise(curr_node.moves.size()));
            for (uint64_t j = 0; j < curr_node.moves.size(); j++) {
                curr_node.nn_policy[j] = (float)((1.0 - noise_epsilon_) * curr_node.nn_policy[j] + noise_epsilon_ * noise[j]);
            }
        }

        //valueを設定
        curr_node.value = values[i];

        //フラグを設定
        curr_node.evaled = true;
    }
}

void GameGenerator::loadNeuralNetwork() {
    //探索バッチサイズのworker_num_倍が実際の推論バッチサイズなので無理やり変更する
    SearchOptions tmp_option = search_options_;
    tmp_option.search_batch_size *= worker_num_;
    tmp_option.use_calibration_cache = false;
    neural_network_.load(gpu_id_, tmp_option);
}

GenerateWorker::GenerateWorker(const SearchOptions& search_options, GPUQueue& gpu_queue, float Q_dist_lambda, ReplayBuffer& rb)
    : search_options_(search_options), gpu_queue_(gpu_queue), Q_dist_lambda_(Q_dist_lambda), replay_buffer_(rb),
      hash_table_(search_options.search_limit * 3), searcher_(search_options, hash_table_, gpu_queue), root_raw_value_{},
      mate_searcher_(hash_table_, search_options) {}

void GenerateWorker::prepareForCurrPos() {
    //古いハッシュを削除
    hash_table_.deleteOldHash(position_, false);

    //ルートノードの展開
    std::stack<int32_t> indices;
    std::stack<int32_t> actions;
    hash_table_.root_index = searcher_.expand(position_, indices, actions);

    //千日手模様のときは評価済みなのにPolicyが展開されていないということが起こり得るので対処
    HashEntry& node = hash_table_[hash_table_.root_index];
    if (node.evaled && (node.moves.size() != node.nn_policy.size())) {
        //GPUへの計算要求を追加
        std::vector<float> this_feature = position_.makeFeature();
        gpu_queue_.inputs.insert(gpu_queue_.inputs.end(), this_feature.begin(), this_feature.end());
        gpu_queue_.hash_tables.emplace_back(hash_table_);
        gpu_queue_.indices.push_back(hash_table_.root_index);
    }

    //50手以降のときだけ詰み探索。本当はこの閾値も制御できるようにした方が良い気はするが……
    if (position_.turnNumber() >= 50) {
        mate_searcher_.mateSearch(position_, 5);
    }
}

OneTurnElement GenerateWorker::resultForCurrPos() {
    const HashEntry& root_node = hash_table_[hash_table_.root_index];
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

    //探索回数最大の手を選択する
    int32_t best_index = (int32_t)(std::max_element(N.begin(), N.end()) - N.begin());

    //選択した着手の勝率の算出
    //詰みのときは未展開であることに注意する
    float best_value =
        (root_node.child_indices[best_index] == HashTable::NOT_EXPANDED ? MAX_SCORE
                                                                        : hash_table_.expQfromNext(root_node, best_index));

    //教師データを作成
    OneTurnElement element;

    //valueのセット
    element.score = best_value;

    //policyのセット
    if (position_.turnNumber() <= search_options_.random_turn) {
        //分布に従ってランダムに行動選択
        //探索回数を正規化した分布
        std::vector<float> N_dist(root_node.moves.size());

        //行動価値のsoftmaxを取った分布
        std::vector<float> Q_dist(root_node.moves.size());

        for (uint64_t i = 0; i < root_node.moves.size(); i++) {
            //探索回数を正規化
            N_dist[i] = (float)N[i] / root_node.sum_N;

            //選択回数が0ならMIN_SCORE
            //選択回数が0ではないのに未展開なら詰み探索が詰みを発見したということなのでMAX_SCORE
            //その他は普通に計算
            Q_dist[i] = (N[i] == 0                                               ? MIN_SCORE
                         : root_node.child_indices[i] == HashTable::NOT_EXPANDED ? MAX_SCORE
                                                                                 : hash_table_.expQfromNext(root_node, i));
        }
        Q_dist = softmax(Q_dist, std::max(search_options_.temperature_x1000 / 1000.0f, 1e-4f));

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

    //あとでpriorityを計算するためにNNの出力をセットする
    element.nn_output_policy.resize(POLICY_DIM, 0.0);
    for (uint64_t i = 0; i < root_node.moves.size(); i++) {
        element.nn_output_policy[root_node.moves[i].toLabel()] = root_node.nn_policy[i];
    }
    element.nn_output_value = root_raw_value_;

    return element;
}

void GenerateWorker::select() {
    if (shouldStop()) {
        //探索結果を取得して次の局面へ遷移
        OneTurnElement result = resultForCurrPos();
        position_.doMove(result.move);
        game_.elements.push_back(result);

        float score{};
        if (position_.isFinish(score, false) || position_.turnNumber() > search_options_.draw_turn) {
            //決着したので最終結果を設定
            game_.result = (position_.color() == BLACK ? score : MAX_SCORE + MIN_SCORE - score);

            //データを送る
            replay_buffer_.push(game_);

            //次の対局へ向かう
            position_.init();
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
        for (int64_t j = 0; j < search_options_.search_batch_size && !shouldStop(); j++) {
            searcher_.select(position_);
        }
    }
}

void GenerateWorker::backup() { searcher_.backupAll(); }

bool GenerateWorker::shouldStop() {
    //ハッシュテーブルの容量チェック
    if (!hash_table_.hasEmptyEntries(1)) {
        return true;
    }

    const HashEntry& root = hash_table_[hash_table_.root_index];

    //探索回数のチェック
    int32_t max1 = 0, max2 = 0;
    for (uint64_t i = 0; i < root.moves.size(); i++) {
        int32_t num = root.N[i];
        if (num > max1) {
            max2 = max1;
            max1 = num;
        } else if (num > max2) {
            max2 = num;
        }
    }
    return max1 - max2 >= search_options_.search_limit - root.sum_N;
}