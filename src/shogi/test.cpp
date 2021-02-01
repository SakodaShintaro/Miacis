#include "test.hpp"
#include "../game_generator.hpp"
#include "../infer_model.hpp"
#include "../searcher_for_play.hpp"
#include "book.hpp"

namespace Shogi {

void test() {
    SearchOptions search_options;
    search_options.search_limit = 800;
    search_options.print_interval = 100000;
    search_options.thread_num_per_gpu = 1;
    search_options.search_batch_size = 1;
    search_options.output_log_file = true;
    InferModel nn;
    nn.load(DEFAULT_MODEL_NAME, 0);
    SearcherForPlay searcher(search_options);

    Position pos;
    Game game;

    auto start = std::chrono::steady_clock::now();
    while (true) {
        Move best_move = searcher.think(pos, LLONG_MAX);

        if (best_move == NULL_MOVE) {
            //投了
            game.result = (pos.color() == BLACK ? MIN_SCORE : MAX_SCORE);
            break;
        }

        float finish_score{};
        if ((pos.isFinish(finish_score) && finish_score == (MAX_SCORE + MIN_SCORE) / 2) ||
            pos.turnNumber() > search_options.draw_turn) {
            //千日手or持将棋
            game.result = finish_score;
            break;
        }

        pos.doMove(best_move);
        pos.print();
        OneTurnElement element;
        element.move = best_move;
        game.elements.push_back(element);
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << elapsed.count() / pos.turnNumber() << " msec / pos" << std::endl;

    game.writeKifuFile("./");
    std::cout << "finish test" << std::endl;
}

void infiniteTest() {
    SearchOptions search_options;
    search_options.thread_num_per_gpu = 1;
    search_options.search_batch_size = 32;
    search_options.random_turn = 30;
    SearcherForPlay searcher(search_options);

    for (int64_t i = 0; i < LLONG_MAX; i++) {
        std::cout << i << std::endl;
        Position pos;

        while (true) {
            Move best_move = searcher.think(pos, 50);
            if (best_move == NULL_MOVE) {
                //終了
                break;
            }

            pos.doMove(best_move);
            //pos.print();
            float finish_score{};
            if (pos.isFinish(finish_score)) {
                break;
            }
        }
    }
}

void checkGenSpeed() {
    InferModel nn;
    nn.load(DEFAULT_MODEL_NAME, 0);

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
            GameGenerator generator(search_options, worker_num, Q_dist_lambda, noise_mode, noise_epsilon, noise_alpha, buffer,
                                    nn);
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

    std::vector<LearningData> data = loadData(path, false, 3000);
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

void checkPredictSpeed() {
    Position pos;
    constexpr int64_t REPEAT_NUM = 1000;
    std::cout << std::fixed;

    InferModel nn;
    nn.load(DEFAULT_MODEL_NAME, 0);

    for (int64_t batch_size = 1; batch_size <= 4096; batch_size *= 2) {
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

        std::cout << input.size() << std::endl;

        float time = 0.0;
        for (int64_t i = 0; i < REPEAT_NUM; i++) {
            auto start = std::chrono::steady_clock::now();
            torch::NoGradGuard no_grad_guard;
            nn.policyAndValueBatch(pos.makeFeature());
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            time += elapsed.count();
        }

        std::cout << "batch_size = " << std::setw(5) << batch_size << ", " << time / REPEAT_NUM << " microsec / batch"
                  << std::endl;
    }
}

void checkSegmentTree() {
    constexpr int64_t n = 8;
    SegmentTree st(n);
    st.print();
    st.update(0, 100);
    st.update(1, 50);
    st.update(2, 49);
    st.update(3, 1);
    st.update(4, 800);
    st.print();
    float sum = st.getSum();
    std::cout << std::fixed;
    std::cout << "sum = " << sum << std::endl;
    std::uniform_real_distribution<float> dist(0.0, sum);

    constexpr int64_t sample_num = 10000;

    std::vector<int64_t> freq(n, 0);
    for (int64_t i = 0; i < sample_num; i++) {
        auto value = dist(engine);
        auto index = st.getIndex(value);
        std::cout << "value = " << std::setw(10) << value << ", index = " << index << std::endl;
        freq[index]++;
    }

    for (int64_t i = 0; i < n; i++) {
        std::cout << std::setw(5) << i << " " << 1.0 * freq[i] / sample_num << std::endl;
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
    YaneBook book;
    book.open("./standard_book.db");
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

    Book book;
    book.open("book.txt");
    for (int64_t _ = 0; _ < search_num; _++) {
        book.updateOne(think_sec);
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

} // namespace Shogi