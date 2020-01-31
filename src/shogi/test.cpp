#include"test.hpp"
#include"../game_generator.hpp"
#include"../searcher_for_play.hpp"
#include"../learn.hpp"

void test() {
    SearchOptions search_options;
    search_options.search_limit = 800;
    search_options.print_interval = 100000;
    search_options.thread_num_per_gpu = 1;
    search_options.search_batch_size = 1;
    NeuralNetwork nn;
    torch::load(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
    nn->setGPU(0);
    nn->eval();
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

        float finish_score;
        if ((pos.isFinish(finish_score) && finish_score == (MAX_SCORE + MIN_SCORE) / 2)
            || pos.turnNumber() > search_options.draw_turn) {
            //千日手or持将棋
            game.result = finish_score;
            break;
        }

        pos.doMove(best_move);
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
            float finish_score;
            if (pos.isFinish(finish_score)) {
                break;
            }
        }
    }
}

void checkGenSpeed() {
    NeuralNetwork nn;
    torch::load(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);

    constexpr int64_t buffer_size = 1048576;
    constexpr int64_t N = 4;
    SearchOptions search_options;
    search_options.use_fp16 = false;
    search_options.search_limit = 800;
    search_options.draw_turn = 512;
    search_options.random_turn = 512;
    search_options.temperature_x1000 = 100;
    search_options.thread_num_per_gpu = 4;
    constexpr FloatType Q_dist_lambda = 1.0;

    nn->setGPU(0, search_options.use_fp16);
    nn->eval();

    std::cout << std::fixed;

    std::ofstream ofs("check_gen_speed.txt");
    ofs << "thread batch_size worker pos sec speed(pos/sec)" << std::fixed << std::endl;

    for (int64_t worker_num = 32; worker_num <= 64; worker_num *= 2) {
        for (search_options.search_batch_size = 1; search_options.search_batch_size <= search_options.search_limit / 2; search_options.search_batch_size *= 2) {
            ReplayBuffer buffer(0, buffer_size, 100 * buffer_size, 1.0, 1.0);
            auto start = std::chrono::steady_clock::now();
            GameGenerator generator(search_options, worker_num, Q_dist_lambda, buffer, nn);
            std::thread t(&GameGenerator::genGames, &generator);
            std::vector<double> gen_speeds;
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(60));
                auto curr_time = std::chrono::steady_clock::now();
                auto ela = std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - start);
                double gen_speed_per_sec = (buffer.totalNum() * 1000.0) / ela.count();
                std::cout << "thread = " << search_options.thread_num_per_gpu
                          << ",  batch_size = " << std::setw(2) << search_options.search_batch_size
                          << ",  worker = " << std::setw(4) << worker_num
                          << ",  pos = " << std::setw(7) << buffer.totalNum()
                          << ",  sec = " << std::setw(9) << ela.count() / 1000
                          << ",  speed = " << std::setprecision(3) << gen_speed_per_sec << " pos / sec"
                          << std::endl;
                gen_speeds.push_back(gen_speed_per_sec);
                if (gen_speeds.size() < N) {
                    continue;
                }

                //直近N回を見て、その中の最大値と最小値が基準値以下だったら収束したと判定して打ち切る
                double min_value = *std::min_element(gen_speeds.end() - N, gen_speeds.end());
                double max_value = *std::max_element(gen_speeds.end() - N, gen_speeds.end());
                if (min_value != 0 && (max_value - min_value) < 5e-2) {
                    ofs << search_options.thread_num_per_gpu << " "
                        << std::setw(2) << search_options.search_batch_size << " "
                        << std::setw(4) << worker_num << " "
                        << std::setw(7) << buffer.totalNum() << " "
                        << std::setw(9) << ela.count() / 1000 << " "
                        << std::setprecision(3) << gen_speed_per_sec << std::endl;
                    break;
                }
            }
            generator.stop_signal = true;
            t.join();
        }
    }
}

void checkSearchSpeed() {
    constexpr int64_t time_limit = 10000;
    SearchOptions search_options;
    search_options.USI_Hash = 4096;
    search_options.print_interval = time_limit * 2;
    int64_t max_thread_num = std::thread::hardware_concurrency() / search_options.gpu_num;
    Position pos;
    for (search_options.search_batch_size = 64; search_options.search_batch_size <= 512; search_options.search_batch_size *= 2) {
        std::cout << "search_batch_size = " << search_options.search_batch_size << std::endl;
        for (search_options.thread_num_per_gpu = 1; search_options.thread_num_per_gpu <= max_thread_num; search_options.thread_num_per_gpu++) {
            SearcherForPlay searcher(search_options);
            searcher.think(pos, time_limit);
        }
    }
}

void checkVal() {
    //データを取得
    std::string path;
    std::cout << "validation kifu path : ";
    std::cin >> path;
    std::vector<LearningData> data = loadData(path, false);

    //ネットワークの準備
    NeuralNetwork nn;
    torch::load(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
    nn->setGPU(0);
    nn->eval();

    std::array<float, LOSS_TYPE_NUM> v = validation(nn, data, 32);
    std::cout << std::fixed << std::setprecision(4);
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        std::cout << v[i] << " \n"[i == LOSS_TYPE_NUM - 1];
    }
}

void checkPredictSpeed() {
    Position pos;
    constexpr int64_t REPEAT_NUM = 1000;
    std::cout << std::fixed;
    std::mt19937_64 engine(0);

    NeuralNetwork nn;
    torch::load(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
    nn->setGPU(0);
    nn->eval();

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

        long double time = 0.0;
        for (int64_t i = 0; i < REPEAT_NUM; i++) {
            auto start = std::chrono::steady_clock::now();
            torch::NoGradGuard no_grad_guard;
            nn->policyAndValueBatch(pos.makeFeature());
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            time += elapsed.count();
        }

        std::cout << "batch_size = " << std::setw(5) << batch_size << ", " << time / REPEAT_NUM << " microsec / batch" << std::endl;
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
    std::mt19937_64 engine(0);
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
    std::mt19937_64 engine(std::random_device{}());
    for (int64_t i = 0; i < 1000000000000; i++) {
        Position pos;
        float score;
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
    std::mt19937_64 engine(std::random_device{}());
    for (int64_t i = 0; i < 1; i++) {
        Position pos;
        float score;
        while (!pos.isFinish(score)) {
            std::vector<Move> moves = pos.generateAllMoves();
            std::uniform_int_distribution<int64_t> dist(0, moves.size() - 1);
            int64_t index = dist(engine);
            pos.doMove(moves[index]);

            std::string str = pos.toStr();
            std::cout << str << std::endl;
            std::cout << Position::augmentedStr(str, 1) << std::endl;

            uint32_t label = moves[index].toLabel();
            uint32_t mirror_label = Move::augmentedLabel(label, 1);
            moves[index].print();
            std::cout << label % SQUARE_NUM << ", " << label / SQUARE_NUM << std::endl;
            std::cout << mirror_label % SQUARE_NUM << ", " << mirror_label / SQUARE_NUM << std::endl;
        }
        while (pos.turnNumber() > 1) {
            pos.undo();
        }
        std::cout << i << std::endl;
    }
}