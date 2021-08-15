﻿#include "test.hpp"
#include "../learn/game_generator.hpp"
#include "../model/infer_dlshogi_model.hpp"
#include "../model/infer_dlshogi_onnx_model.hpp"
#include "../model/infer_model.hpp"
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

    std::vector<LearningData> data = loadData(path, false, 3000);
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

} // namespace Shogi