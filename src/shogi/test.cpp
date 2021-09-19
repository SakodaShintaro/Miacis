#include "test.hpp"
#include "../learn/game_generator.hpp"
#include "../model/infer_dlshogi_model.hpp"
#include "../model/infer_dlshogi_onnx_model.hpp"
#include "../model/infer_model.hpp"
#include "../model/libtorch_model.hpp"
#include "../search/searcher_for_play.hpp"
#include "book.hpp"

namespace Shogi {

void checkGenSpeed() {
    constexpr int64_t buffer_size = 1048576;
    SearchOptions search_options;
    search_options.search_limit = 800;
    search_options.draw_turn = 320;
    search_options.random_turn = 320;
    search_options.temperature_x1000 = 10;
    constexpr float Q_dist_lambda = 1.0;
    constexpr int64_t noise_mode = 0;
    constexpr float noise_epsilon = 0.25;
    constexpr float noise_alpha = 0.15;

    int64_t total_worker_num = 0;
    std::cout << "total_worker_num(デフォルトは128): ";
    std::cin >> total_worker_num;

    std::cout << std::fixed;

    std::ofstream ofs("check_gen_speed.txt", std::ios::app);
    ofs << "thread batch_size worker pos sec speed(pos/sec)" << std::fixed << std::endl;

    constexpr int64_t sec = 1200;
    constexpr int64_t num = 10;

    for (search_options.thread_num_per_gpu = 2; search_options.thread_num_per_gpu <= 3; search_options.thread_num_per_gpu++) {
        int64_t worker_num = total_worker_num / search_options.thread_num_per_gpu;
        for (search_options.search_batch_size = 2; search_options.search_batch_size <= 4; search_options.search_batch_size *= 2) {
            ReplayBuffer buffer(0, buffer_size, 1, 1.0, 1.0, false);
            auto start = std::chrono::steady_clock::now();
            GameGenerator generator(search_options, worker_num, Q_dist_lambda, noise_mode, noise_epsilon, noise_alpha, buffer, 0);
            std::thread t(&GameGenerator::genGames, &generator);
            for (int64_t i = 0; i < num; i++) {
                std::this_thread::sleep_for(std::chrono::seconds(sec));
                auto curr_time = std::chrono::steady_clock::now();
                auto ela = std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - start);
                float gen_speed_per_sec = (buffer.totalNum() * 1000.0) / ela.count();
                // clang-format off
                std::cout << "thread = " << search_options.thread_num_per_gpu
                          << ",  batch_size = " << std::setw(2) << search_options.search_batch_size
                          << ",  worker = " << std::setw(3) << worker_num
                          << ",  pos = " << std::setw(7) << buffer.totalNum()
                          << ",  min = " << std::setw(4) << ela.count() / 60000
                          << ",  speed = " << std::setprecision(3) << gen_speed_per_sec << " pos / sec"
                          << std::endl;
                // clang-format on
            }
            ofs << search_options.thread_num_per_gpu << " " << std::setw(2) << search_options.search_batch_size << " "
                << std::setw(4) << worker_num << " " << std::setw(7) << buffer.totalNum() << " " << std::setw(9) << num * sec
                << " " << std::setprecision(3) << (float)buffer.totalNum() / (num * sec) << std::endl;
            generator.stop_signal = true;
            t.join();
        }
    }
    exit(0);
}

void checkSearchSpeed() {
    constexpr int64_t time_limit = 10000;
    constexpr int64_t trial_num = 10;
    SearchOptions search_options;
    search_options.print_interval = time_limit * 2;
    search_options.print_info = false;
    while (true) {
        std::string input;
        std::cin >> input;
        if (input == "go") {
            break;
        }
        assert(input == "setoption");
        std::cin >> input;
        assert(input == "name");
        std::cin >> input;

        for (auto& pair : search_options.check_options) {
            if (input == pair.first) {
                std::cin >> input;
                std::cin >> input;
                pair.second.value = (input == "true");
            }
        }
        for (auto& pair : search_options.spin_options) {
            if (input == pair.first) {
                std::cin >> input;
                std::cin >> pair.second.value;
            }
        }
        for (auto& pair : search_options.filename_options) {
            if (input == pair.first) {
                std::cin >> input;
                std::cin >> pair.second.value;
            }
        }
    }

    std::cout << std::fixed << std::setprecision(1);

    Position pos;
    std::cout << "初期局面" << std::endl;
    for (int64_t _ = 0; _ < trial_num; _++) {
        SearcherForPlay searcher(search_options);
        Move best_move = searcher.think(pos, time_limit);
        const HashTable& hash_table = searcher.hashTable();
        const HashEntry& root_entry = hash_table[hash_table.root_index];
        std::cout << root_entry.sum_N / (time_limit / 1000.0) << "\t" << best_move << std::endl;
    }

    pos.fromStr("l2+P4l/7s1/p2ppkngp/9/2p6/PG7/K2PP+r+b1P/1S5P1/L7L w RBGS2N5Pgsn2p 82");
    std::cout << "中盤の局面" << std::endl;
    for (int64_t _ = 0; _ < trial_num; _++) {
        SearcherForPlay searcher(search_options);
        Move best_move = searcher.think(pos, time_limit);
        const HashTable& hash_table = searcher.hashTable();
        const HashEntry& root_entry = hash_table[hash_table.root_index];
        std::cout << root_entry.sum_N / (time_limit / 1000.0) << "\t" << best_move << std::endl;
    }

    std::cout << "finish checkSearchSpeed" << std::endl;
}

void checkSearchSpeed2() {
    constexpr int64_t time_limit = 10000;
    constexpr int64_t trial_num = 10;
    SearchOptions search_options;
    search_options.print_interval = time_limit * 2;
    search_options.print_info = false;
    search_options.USI_Hash = 8192;

    std::cout << std::fixed << std::setprecision(1);

    for (search_options.search_batch_size = 1; search_options.search_batch_size <= 256; search_options.search_batch_size++) {
        Position pos;
        float sum_init = 0.0, sum_mid = 0.0;
        for (int64_t _ = 0; _ < trial_num; _++) {
            SearcherForPlay searcher(search_options);
            Move best_move = searcher.think(pos, time_limit);
            const HashTable& hash_table = searcher.hashTable();
            const HashEntry& root_entry = hash_table[hash_table.root_index];
            float curr_nps = root_entry.sum_N / (time_limit / 1000.0);
            std::cout << "s:" << _ << " " << curr_nps << "\t" << best_move << "  \r" << std::flush;
            sum_init += curr_nps;
        }

        pos.fromStr("l2+P4l/7s1/p2ppkngp/9/2p6/PG7/K2PP+r+b1P/1S5P1/L7L w RBGS2N5Pgsn2p 82");
        for (int64_t _ = 0; _ < trial_num; _++) {
            SearcherForPlay searcher(search_options);
            Move best_move = searcher.think(pos, time_limit);
            const HashTable& hash_table = searcher.hashTable();
            const HashEntry& root_entry = hash_table[hash_table.root_index];
            float curr_nps = root_entry.sum_N / (time_limit / 1000.0);
            std::cout << "m:" << _ << " " << curr_nps << "\t" << best_move << "  \r" << std::flush;
            sum_mid += curr_nps;
        }

        std::cout << search_options.search_batch_size << " " << sum_init / trial_num << " " << sum_mid / trial_num << std::endl;
    }

    std::cout << "finish checkSearchSpeed" << std::endl;
}

void checkVal() {
    //データを取得
    std::string path;
    std::cout << "validation kifu path : ";
    std::cin >> path;
    int64_t batch_size;
    std::cout << "batch_size : ";
    std::cin >> batch_size;
    std::string model_file;
    std::cout << "model_file : ";
    std::cin >> model_file;
    float rate_threshold;
    std::cout << "rate_threshold : ";
    std::cin >> rate_threshold;

    std::vector<LearningData> data = loadData(path, false, rate_threshold);
    std::cout << "data.size() = " << data.size() << std::endl;

    //ネットワークの準備
    LearningModel nn;
    nn.load(model_file, 0);
    nn.eval();

    std::array<float, LOSS_TYPE_NUM> v = validation(nn, data, batch_size);
    std::cout << std::fixed << std::setprecision(4);
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        std::cout << v[i] << " \n"[i == LOSS_TYPE_NUM - 1];
    }
}

void checkValInfer() {
    //データを取得
    SearchOptions search_options;

    std::string path;
    std::cout << "validation kifu path : ";
    std::cin >> path;
    std::cout << "batch_size : ";
    std::cin >> search_options.search_batch_size;
    std::cout << "model_file : ";
    std::cin >> search_options.model_name;
    std::cout << "calibration_kifu_path : ";
    std::cin >> search_options.calibration_kifu_path;
    std::cout << "fp16 : ";
    std::cin >> search_options.use_fp16;
    float rate_threshold;
    std::cout << "rate_threshold : ";
    std::cin >> rate_threshold;

    std::vector<LearningData> data = loadData(path, false, rate_threshold);
    std::cout << "data.size() = " << data.size() << std::endl;

    //ネットワークの準備
    InferModel nn;
    nn.load(0, search_options);

    std::array<float, LOSS_TYPE_NUM> v = validation(nn, data, search_options.search_batch_size);
    std::cout << std::fixed << std::setprecision(4);
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        std::cout << v[i] << " \n"[i == LOSS_TYPE_NUM - 1];
    }
    std::cout << "finish checkValInfer" << std::endl;
}

void checkPredictSpeed() {
    Position pos;
    constexpr int64_t REPEAT_NUM = 1000;
    constexpr int64_t BATCH_SIZE = 512;
    std::cout << std::fixed;

    SearchOptions search_options;

    InferModel nn;
    nn.load(0, search_options);

    for (int64_t batch_size = 1; batch_size <= BATCH_SIZE; batch_size *= 2) {
        //バッチサイズ分入力を取得
        std::vector<float> input;
        for (int64_t k = 0; k < batch_size; k++) {
            auto f = pos.makeFeature();
            input.insert(input.end(), f.begin(), f.end());

            auto moves = pos.generateAllMoves();
            if (moves.empty()) {
                pos.init();
            } else {
                std::uniform_int_distribution<> dist(0, moves.size() - 1);
                pos.doMove(moves[dist(engine)]);
            }
        }

        float time = 0.0;
        for (int64_t i = 0; i < REPEAT_NUM; i++) {
            auto start = std::chrono::steady_clock::now();
            torch::NoGradGuard no_grad_guard;
            nn.policyAndValueBatch(input);
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            time += elapsed.count();
        }

        std::cout << std::setw(5) << batch_size << "\t" << time / REPEAT_NUM << "\tmicrosec/batch" << std::endl;
    }
}

void checkDoAndUndo() {
    for (int64_t i = 0; i < 1000000000000; i++) {
        Position pos;
        float score{};
        while (!pos.isFinish(score)) {
            std::vector<Move> moves = pos.generateAllMoves();
            std::uniform_int_distribution<int64_t> dist(0, moves.size() - 1);
            int64_t index = dist(engine);
            pos.doMove(moves[index]);
            std::uniform_real_distribution<float> dist2(0, 100);
            float p = dist2(engine);
            if (p < 0.5) {
                pos.undo();
            }
        }
        while (pos.turnNumber() > 1) {
            pos.undo();
        }
        std::cout << i << std::endl;
    }
}

void checkMirror() {
    for (int64_t i = 0; i < 1; i++) {
        Position pos;
        float score{};
        while (!pos.isFinish(score)) {
            std::vector<Move> moves = pos.generateAllMoves();
            std::uniform_int_distribution<int64_t> dist(0, moves.size() - 1);
            int64_t index = dist(engine);
            pos.doMove(moves[index]);

            std::string str = pos.toStr();
            std::cout << str << std::endl;
            std::cout << Position::augmentStr(str, 1) << std::endl;

            uint32_t label = moves[index].toLabel();
            uint32_t mirror_label = Move::augmentLabel(label, 1);
            std::cout << moves[index].toPrettyStr() << std::endl;
            std::cout << label % SQUARE_NUM << ", " << label / SQUARE_NUM << std::endl;
            std::cout << mirror_label % SQUARE_NUM << ", " << mirror_label / SQUARE_NUM << std::endl;
        }
        while (pos.turnNumber() > 1) {
            pos.undo();
        }
        std::cout << i << std::endl;
    }
}

void checkBook() {
    Book book;
    book.open("./book.txt");
    Position pos;
    float score{};
    while (!pos.isFinish(score)) {
        pos.print();
        if (book.hasEntry(pos)) {
            Move best_move = book.pickOne(pos, 1);
            std::cout << best_move << std::endl;
            pos.doMove(pos.transformValidMove(best_move));
        } else {
            std::cout << "定跡なし" << std::endl;
            std::cout << pos.toStr() << std::endl;
            break;
        }
    }
    std::cout << "finish checkBook" << std::endl;
}

void makeBook() {
    int64_t search_num = 0, think_sec = 0;
    std::cout << "定跡に追加するノード数: ";
    std::cin >> search_num;
    std::cout << "一局面の思考時間(秒): ";
    std::cin >> think_sec;

    SearchOptions search_options;
    search_options.print_interval = think_sec * 2000;
    search_options.USI_Hash = 8096;
    search_options.use_book = false;

    SearcherForPlay searcher(search_options);

    Book book;
    book.open("book.txt");
    for (int64_t _ = 0; _ < search_num; _++) {
        Position pos;
        std::vector<Move> selected_moves;
        while (true) {
            if (!book.hasEntry(pos)) {
                break;
            }

            const BookEntry& book_entry = book.getEntry(pos);

            //選択回数の合計
            int64_t sum = std::accumulate(book_entry.select_num.begin(), book_entry.select_num.end(), (int64_t)0);

            //UCBを計算し、一番高い行動を選択
            int64_t max_index = -1;
            float max_value = -1;
            for (uint64_t i = 0; i < book_entry.moves.size(); i++) {
                float U = std::sqrt(sum + 1) / (book_entry.select_num[i] + 1);
                float ucb = search_options.Q_coeff_x1000 / 1000.0 * book_entry.values[i] +
                            search_options.C_PUCT_x1000 / 1000.0 * book_entry.policies[i] * U;
                if (ucb > max_value) {
                    max_index = i;
                    max_value = ucb;
                }
            }

            Move best_move = pos.transformValidMove(book_entry.moves[max_index]);
            selected_moves.push_back(best_move);
            pos.doMove(best_move);
        }

        //-------------
        //    展開部
        //-------------
        //この局面を探索する
        pos.print();
        searcher.think(pos, think_sec * 1000);

        //結果を取得
        const HashTable& searched = searcher.hashTable();
        const HashEntry& root_node = searched[searched.root_index];

        //展開
        BookEntry& book_entry = book.getEntry(pos);
        book_entry.moves = root_node.moves;
        book_entry.policies = root_node.nn_policy;
        book_entry.values.resize(book_entry.moves.size());
        for (uint64_t i = 0; i < book_entry.moves.size(); i++) {
            book_entry.values[i] = searched.expQfromNext(root_node, i);
        }
        book_entry.select_num.assign(book_entry.moves.size(), 1);
        float value = *max_element(book_entry.values.begin(), book_entry.values.end());

        //この局面を登録
        //backupする
        while (!selected_moves.empty()) {
            //局面を戻し、そこに相当するエントリを取得
            pos.undo();
            BookEntry& curr_entry = book.getEntry(pos);

            //価値を反転
            value = -value;

            //最終手を取得
            Move last_move = selected_moves.back();
            selected_moves.pop_back();

            //更新
            for (uint64_t i = 0; i < curr_entry.moves.size(); i++) {
                if (pos.transformValidMove(curr_entry.moves[i]) != last_move) {
                    continue;
                }

                //この手の価値を更新
                float alpha = 1.0f / (++curr_entry.select_num[i]);
                curr_entry.values[i] += alpha * (value - curr_entry.values[i]);
                break;
            }
        }

        book.write("book.txt");
    }
    std::cout << "finish makeBook" << std::endl;
}

void searchWithLog() {
    SearchOptions search_options;
    search_options.USI_Hash = 8192;
    search_options.random_turn = 30;
    search_options.print_policy_num = 600;
    search_options.print_info = false;
    search_options.output_log_file = true;
    SearcherForPlay searcher(search_options);

    for (int64_t i = 0; i < LLONG_MAX; i++) {
        std::cout << i << std::endl;
        Position pos;

        while (true) {
            Move best_move = searcher.think(pos, 30000);
            std::cout << best_move << " ";
            if (best_move == NULL_MOVE) {
                //終了
                break;
            }

            pos.doMove(best_move);
            float finish_score{};
            if (pos.isFinish(finish_score)) {
                break;
            }
        }
        std::cout << std::endl;
    }
}

void testLoad() {
    constexpr int64_t LOOP_NUM = 20;

    SearchOptions search_options;

    //時間計測開始
    Timer timer;
    timer.start();
    int64_t pre = 0;
    //通常試行
    std::cout << "通常の試行" << std::endl;
    for (int64_t num = 0; num < 0; num++) {
        InferModel model;
        model.load(0, search_options);
        int64_t ela = timer.elapsedSeconds();
        int64_t curr = ela - pre;
        pre = ela;
        std::cout << std::setw(3) << num + 1 << "回目終了, 今回" << curr << "秒, 平均" << ela / (num + 1.0) << "秒" << std::endl;
    }

    //スレッドを作成しての試行
    timer.start();
    pre = 0;
    std::cout << "スレッドを作成しての試行" << std::endl;
    const int64_t gpu_num = torch::getNumGPUs();
    for (int64_t num = 0; num < LOOP_NUM; num++) {
        std::vector<std::thread> threads;
        for (int64_t i = 0; i < gpu_num; i++) {
            threads.emplace_back([i, search_options]() {
                InferModel model;
                model.load(i, search_options);
            });
        }
        for (int64_t i = 0; i < gpu_num; i++) {
            threads[i].join();
        }
        int64_t ela = timer.elapsedSeconds();
        int64_t curr = ela - pre;
        pre = ela;
        std::cout << std::setw(3) << num + 1 << "回目終了, 今回" << curr << "秒, 平均" << ela / (num + 1.0) << "秒" << std::endl;
    }

    std::cout << "finish testLoad" << std::endl;
    std::exit(0);
}

void testModel() {
    //ネットワークの準備
    SearchOptions search_options;
    search_options.use_calibration_cache = false;
    search_options.search_batch_size = 2;

    std::cout << "model_file : ";
    std::cin >> search_options.model_name;

    InferModel nn;
    nn.load(0, search_options);

    Position pos;
    pos.fromStr("l2+P4l/7s1/p2ppkngp/9/2p6/PG7/K2PP+r+b1P/1S5P1/L7L w RBGS2N5Pgsn2p 82");
    //    pos.fromStr("lnsgk4/9/pppp1ppp1/9/8+P/9/PPPP1PPP1/4+p4/LNSGK4 b RBGSNLPrbgsnlp 1");
    std::vector<float> vec;
    for (int64_t i = 0; i < search_options.search_batch_size; i++) {
        auto f = pos.makeDLShogiFeature();
        vec.insert(vec.end(), f.begin(), f.end());
    }
    auto [policy, value] = nn.policyAndValueBatch(vec);

    std::ofstream ofs("policy.txt");
    ofs << std::fixed << std::setprecision(4);
    for (int64_t i = 0; i < policy[0].size(); i++) {
        ofs << policy[0][i] << std::endl;
    }
    std::cout << "finish testModel" << std::endl;
    std::exit(0);
}

void checkValLibTorchModel() {
    //データを取得
    std::string path;
    std::cout << "validation kifu path : ";
    std::cin >> path;
    int64_t batch_size;
    std::cout << "batch_size : ";
    std::cin >> batch_size;
    std::string model_file;
    std::cout << "model_file : ";
    std::cin >> model_file;
    float rate_threshold;
    std::cout << "rate_threshold : ";
    std::cin >> rate_threshold;

    std::vector<LearningData> valid_data = loadData(path, false, rate_threshold);
    std::cout << "valid_data.size() = " << valid_data.size() << std::endl;

    //ネットワークの準備
    LibTorchModel model;
    model.load(model_file, 0);
    model.eval();

    torch::NoGradGuard no_grad_guard;

    constexpr int64_t loop_num = DEFAULT_LOOP_NUM + 5;
    std::vector<std::array<float, LOSS_TYPE_NUM>> losses(loop_num);

    for (uint64_t index = 0; index < valid_data.size();) {
        std::vector<LearningData> curr_data;
        while (index < valid_data.size() && curr_data.size() < batch_size) {
            curr_data.push_back(valid_data[index++]);
        }

        std::cout << index << " / " << valid_data.size() << std::endl;

        std::vector<std::array<torch::Tensor, LOSS_TYPE_NUM>> curr_losses = model.validLosses(curr_data, loop_num);
        for (int64_t l = 0; l < loop_num; l++) {
            for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
                losses[l][i] += curr_losses[l][i].sum().item<float>();
            }
        }
    }

    //データサイズで割って平均
    for (int64_t l = 0; l < loop_num; l++) {
        for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
            losses[l][i] /= valid_data.size();
        }
    }

    for (int64_t l = 0; l < loop_num; l++) {
        std::cout << std::setw(2) << l + 1 << "\t";
        std::cout << std::fixed << std::setprecision(4);
        for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
            std::cout << losses[l][i] << "\t\n"[i == LOSS_TYPE_NUM - 1];
        }
    }
    std::cout << "finish checkLibTorchModel" << std::endl;
}

void checkLibTorchModel() {
    //データを取得
    std::string path;
    std::cout << "validation kifu path : ";
    std::cin >> path;
    int64_t batch_size;
    std::cout << "batch_size : ";
    std::cin >> batch_size;
    std::string model_file;
    std::cout << "model_file : ";
    std::cin >> model_file;
    float rate_threshold;
    std::cout << "rate_threshold : ";
    std::cin >> rate_threshold;

    std::vector<LearningData> valid_data = loadData(path, false, rate_threshold);
    std::cout << "valid_data.size() = " << valid_data.size() << std::endl;

    //ネットワークの準備
    LibTorchModel model;
    model.load(model_file, 0);
    model.eval();

    std::cout << std::fixed;

    torch::NoGradGuard no_grad_guard;
    std::array<float, LOSS_TYPE_NUM> losses{};
    for (uint64_t index = 0; index < valid_data.size();) {
        std::vector<LearningData> curr_data;
        while (index < valid_data.size() && curr_data.size() < batch_size) {
            curr_data.push_back(valid_data[index++]);
        }

        std::vector<torch::Tensor> reps = model.getRepresentations(curr_data);

        std::cout << "表現の平均\t";
        for (int64_t i = 0; i < reps.size(); i++) {
            torch::Tensor r = reps[i];
            r = r.mean({ 0, 2, 3 });
            r = (r * r).mean();
            std::cout << r.item<float>() << "\t\n"[i == reps.size() - 1];
        }

        std::cout << "表現の分散\t";
        for (int64_t i = 0; i < reps.size(); i++) {
            torch::Tensor r = reps[i];
            r = r.var({ 0, 2, 3 });
            r = r.mean();
            std::cout << r.item<float>() << "\t\n"[i == reps.size() - 1];
        }

        for (torch::Tensor& t : reps) {
            t = t.flatten(1);
        }

        std::cout << "1ループ前の表現とのコサイン類似度" << std::endl;
        for (int64_t i = 1; i < reps.size(); i++) {
            torch::Tensor cos_sim = torch::cosine_similarity(reps[i], reps[i - 1]);
            for (int64_t j = 0; j < batch_size; j++) {
                std::cout << cos_sim[j].item<float>() << "\t\n"[j == batch_size - 1];
            }
        }

        // 0と1の表現について見てみよう
        std::cout << "先頭2データの表現について" << std::endl;
        for (int64_t i = 0; i < reps.size(); i++) {
            for (int64_t j = 0; j < reps.size(); j++) {
                torch::Tensor cos_sim = torch::cosine_similarity(reps[i][0], reps[j][1], 0);
                std::cout << cos_sim.item<float>() << "\t\n"[j == reps.size() - 1];
            }
        }

        break;
    }

    std::cout << "finish checkLibTorchModel" << std::endl;
}

void checkInitLibTorchModel() {
    //データを取得
    int64_t batch_size = 512;

    //ネットワークの準備
    LibTorchModel model;
    model.eval();

    std::cout << std::fixed;

    torch::NoGradGuard no_grad_guard;

    torch::Tensor dummy_input = torch::randn({ batch_size, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    std::vector<torch::Tensor> reps = model.getRepresentations(dummy_input);

    std::cout << "表現の平均\t";
    for (int64_t i = 0; i < reps.size(); i++) {
        torch::Tensor r = reps[i];
        r = r.mean({ 0, 2, 3 });
        r = (r * r).mean();
        std::cout << r.item<float>() << "\t\n"[i == reps.size() - 1];
    }

    std::cout << "表現の分散\t";
    for (int64_t i = 0; i < reps.size(); i++) {
        torch::Tensor r = reps[i];
        r = r.var({ 0, 2, 3 });
        r = r.mean();
        std::cout << r.item<float>() << "\t\n"[i == reps.size() - 1];
    }

    std::cout << "finish checkInitLibTorchModel" << std::endl;
}

void checkLearningModel() {
    //データを取得
    std::string path;
    std::cout << "validation kifu path : ";
    std::cin >> path;
    int64_t batch_size;
    std::cout << "batch_size : ";
    std::cin >> batch_size;
    std::string model_file;
    std::cout << "model_file : ";
    std::cin >> model_file;
    float rate_threshold;
    std::cout << "rate_threshold : ";
    std::cin >> rate_threshold;

    std::vector<LearningData> valid_data = loadData(path, false, rate_threshold);
    std::cout << "valid_data.size() = " << valid_data.size() << std::endl;

    //ネットワークの準備
    LearningModel model;
    model.load(model_file, 0);
    model.eval();

    std::cout << std::fixed;

    torch::NoGradGuard no_grad_guard;
    std::array<float, LOSS_TYPE_NUM> losses{};
    for (uint64_t index = 0; index < valid_data.size();) {
        std::vector<LearningData> curr_data;
        while (index < valid_data.size() && curr_data.size() < batch_size) {
            curr_data.push_back(valid_data[index++]);
        }

        std::vector<torch::Tensor> reps = model.getRepresentations(curr_data);

        std::cout << "表現の平均\t";
        for (int64_t i = 0; i < reps.size(); i++) {
            torch::Tensor r = reps[i];
            r = r.mean({ 0, 2, 3 });
            r = (r * r).mean();
            std::cout << r.item<float>() << "\t\n"[i == reps.size() - 1];
        }

        std::cout << "表現の分散\t";
        for (int64_t i = 0; i < reps.size(); i++) {
            torch::Tensor r = reps[i];
            r = r.var({ 0, 2, 3 });
            r = r.mean();
            std::cout << r.item<float>() << "\t\n"[i == reps.size() - 1];
        }

        for (torch::Tensor& t : reps) {
            t = t.flatten(1);
        }

        std::cout << "1ループ前の表現とのコサイン類似度" << std::endl;
        for (int64_t i = 1; i < reps.size(); i++) {
            torch::Tensor cos_sim = torch::cosine_similarity(reps[i], reps[i - 1]);
            for (int64_t j = 0; j < batch_size; j++) {
                std::cout << cos_sim[j].item<float>() << "\t\n"[j == batch_size - 1];
            }
        }

        // 0と1の表現について見てみよう
        std::cout << "先頭2データの表現について" << std::endl;
        for (int64_t i = 0; i < reps.size(); i++) {
            for (int64_t j = 0; j < reps.size(); j++) {
                torch::Tensor cos_sim = torch::cosine_similarity(reps[i][0], reps[j][1], 0);
                std::cout << cos_sim.item<float>() << "\t\n"[j == reps.size() - 1];
            }
        }

        break;
    }

    std::cout << "finish checkLibTorchModel" << std::endl;
}

void checkValidData() {
    //データを取得
    std::string path;
    std::cout << "validation kifu path : ";
    std::cin >> path;
    float rate_threshold;
    std::cout << "rate_threshold : ";
    std::cin >> rate_threshold;

    std::vector<LearningData> valid_data = loadData(path, false, rate_threshold);
    std::cout << "valid_data.size() = " << valid_data.size() << std::endl;

    std::vector<float> sum(SQUARE_NUM * POLICY_CHANNEL_NUM);

    for (const LearningData& data : valid_data) {
        for (const auto [label, prob] : data.policy) {
            sum[label] += prob;
        }
    }

    std::vector<std::pair<int64_t, float>> policy;
    for (int64_t i = 0; i < SQUARE_NUM * POLICY_CHANNEL_NUM; i++) {
        policy.emplace_back(i, sum[i]);
    }

    std::sort(policy.begin(), policy.end(),
              [](std::pair<int64_t, float>& lhs, std::pair<int64_t, float>& rhs) { return lhs.second > rhs.second; });

    std::ofstream ofs("valid_data_total.txt");
    int64_t count = 0;
    int64_t ng_hand = 0;
    int64_t ng_from = 0;
    int64_t ng_promote = 0;
    int64_t ng_knight = 0;
    for (int64_t i = 0; i < SQUARE_NUM * POLICY_CHANNEL_NUM; i++) {
        int64_t sq_num = policy[i].first % SQUARE_NUM;
        Square to_sq = SquareList[sq_num];
        int64_t dir = policy[i].first / SQUARE_NUM;
        Square from_sq = to_sq;
        //enum MOVE_DIRECTION { UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT, MOVE_DIRECTION_NUM };

        bool ok = true;
        if (dir >= 20) {
            //打つ手
            if (dir == 20 || dir == 21) {
                //歩 or 香車
                if (SquareToRank[to_sq] <= Rank1) {
                    ok = false;
                    ng_hand++;
                }
            } else if (dir == 22) {
                if (SquareToRank[to_sq] <= Rank2) {
                    ok = false;
                    ng_hand++;
                }
            }
        } else {
            //移動手
            bool promote = (dir >= 10);
            switch (dir % 10) {
            case 0:
                from_sq = to_sq + D;
                if (isOnBoard(from_sq) && promote && (SquareToRank[to_sq] > Rank3)) {
                    ok = false;
                    ng_promote++;
                }
                break;
            case 1:
                from_sq = to_sq + RD;
                if (isOnBoard(from_sq) && promote && (SquareToRank[to_sq] > Rank3)) {
                    ok = false;
                    ng_promote++;
                }
                break;
            case 2:
                from_sq = to_sq + LD;
                if (isOnBoard(from_sq) && promote && (SquareToRank[to_sq] > Rank3)) {
                    ok = false;
                    ng_promote++;
                }
                break;
            case 3:
                from_sq = to_sq + R;
                if (isOnBoard(from_sq) && promote && (SquareToRank[to_sq] > Rank3)) {
                    ok = false;
                    ng_promote++;
                }
                break;
            case 4:
                from_sq = to_sq + L;
                if (isOnBoard(from_sq) && promote && (SquareToRank[to_sq] > Rank3)) {
                    ok = false;
                    ng_promote++;
                }
                break;
            case 5:
                from_sq = to_sq + U;
                break;
            case 6:
                from_sq = to_sq + RU;
                if (isOnBoard(from_sq) && promote && (SquareToRank[to_sq] > SquareToFile[to_sq] + 2)) {
                    ok = false;
                    ng_promote++;
                }
                break;
            case 7:
                from_sq = to_sq + LU;
                if (isOnBoard(from_sq) && promote && (SquareToRank[to_sq] > 12 - SquareToFile[to_sq])) {
                    ok = false;
                    ng_promote++;
                }
                break;
            case 8:
                from_sq = to_sq + RDD;
                if (isOnBoard(from_sq) && !promote && (SquareToRank[to_sq] <= Rank2)) {
                    ok = false;
                    ng_knight++;
                }
                if (isOnBoard(from_sq) && promote && (SquareToRank[to_sq] > Rank3)) {
                    ok = false;
                    ng_promote++;
                }
                break;
            case 9:
                from_sq = to_sq + LDD;
                if (isOnBoard(from_sq) && !promote && (SquareToRank[to_sq] <= Rank2)) {
                    ok = false;
                    ng_knight++;
                }
                if (isOnBoard(from_sq) && promote && (SquareToRank[to_sq] > Rank3)) {
                    ok = false;
                    ng_promote++;
                }
                break;
            default:
                break;
            }
            if (ok && !isOnBoard(from_sq)) {
                ok = false;
                ng_from++;
            }
        }

        if (!ok) {
            continue;
        }

        ofs << ++count << "\t" << policy[i].first << "\t" << 100 * policy[i].second / valid_data.size() << "\t" << to_sq << "\t"
            << dir << std::endl;
    }

    std::cout << "ng_hand    = " << ng_hand << std::endl;
    std::cout << "ng_from    = " << ng_from << std::endl;
    std::cout << "ng_promote = " << ng_promote << std::endl;
    std::cout << "ng_knight  = " << ng_knight << std::endl;

    std::cout << "finish checkValidData" << std::endl;
}

} // namespace Shogi