#include"test.hpp"
#include"../game_generator.hpp"
#include"../searcher_for_play.hpp"
#include"../learn.hpp"

void test() {
    UsiOptions usi_options;
    usi_options.search_limit = 800;
    usi_options.thread_num = 1;
    usi_options.search_batch_size = 1;
    torch::load(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
    SearcherForPlay searcher(usi_options, nn);

    Position pos;
    Game game;

    auto start = std::chrono::steady_clock::now();
    while (true) {
        Move best_move = searcher.think(pos, LLONG_MAX);

        if (best_move == NULL_MOVE) {
            //投了
            game.result = (pos.color() == BLACK ? Game::RESULT_WHITE_WIN : Game::RESULT_BLACK_WIN);
            break;
        }
        float finish_score;
        if (pos.isFinish(finish_score) && finish_score == (MAX_SCORE + MIN_SCORE) / 2) {
            //千日手
            game.result = Game::RESULT_DRAW_REPEAT;
            break;
        } else if (pos.turnNumber() > usi_options.draw_turn) {
            //長手数
            game.result = Game::RESULT_DRAW_OVER_LIMIT;
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

void checkGenSpeed() {
    torch::load(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);

    constexpr int64_t buffer_size = 200000;
    UsiOptions usi_options;
    usi_options.search_limit = 800;
    usi_options.draw_turn = 256;
    usi_options.thread_num = 2;
    constexpr FloatType Q_dist_lambda = 0.0;
    std::cout << std::fixed;

    for (usi_options.search_batch_size = 2; usi_options.search_batch_size <= 4096; usi_options.search_batch_size *= 2) {
        ReplayBuffer buffer(0, buffer_size, 10 * buffer_size, 1.0, 1.0);
        Searcher::stop_signal = false;
        auto start = std::chrono::steady_clock::now();
        GameGenerator generator(usi_options, Q_dist_lambda, buffer, nn);
        std::thread t(&GameGenerator::genGames, &generator, (int64_t)1e15);
        double pre_gen_speed = 0;
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(200));
            auto curr_time = std::chrono::steady_clock::now();
            auto ela = std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - start);
            double gen_speed_per_sec = (buffer.totalNum() * 1000.0) / ela.count();
            std::cout << "search_batch_size = " << std::setw(4) << usi_options.search_batch_size
                      << ",  elapsed = " << std::setw(12) << ela.count()
                      << ",  totalNum = " << std::setw(7) << buffer.totalNum()
                      << ",  speed = " << std::setprecision(3) << gen_speed_per_sec << " pos / sec" << std::endl;
            if (gen_speed_per_sec != 0 && std::abs(gen_speed_per_sec - pre_gen_speed) < 1e-4) {
                break;
            }
            pre_gen_speed = gen_speed_per_sec;
        }
    }
}

void checkSearchSpeed() {
    UsiOptions usi_options;
    usi_options.USI_Hash = 2048;
    constexpr int64_t time_limit = 10000;
    Position pos;
    for (usi_options.search_batch_size = 64; usi_options.search_batch_size <= 512; usi_options.search_batch_size *= 2) {
        std::cout << "search_batch_size = " << usi_options.search_batch_size << std::endl;
        for (usi_options.thread_num = 1; usi_options.thread_num <= 3; usi_options.thread_num++) {
            SearcherForPlay searcher(usi_options, nn);
            searcher.think(pos, time_limit);
        }
    }
}

void checkVal() {
    //データを取得
    std::string path;
    std::cout << "validation kifu path : ";
    std::cin >> path;
    auto data = loadData(path);

    //データをシャッフルして必要量以外を削除
    std::default_random_engine engine(0);
    std::shuffle(data.begin(), data.end(), engine);
    data.erase(data.begin() + 409600, data.end());
    data.shrink_to_fit();

    for (int32_t i = 1; i <= 100000; i++) {
        auto v = validation(data, 32);
        printf("%5d回目 : %f\t%f\n", i, v[0], v[1]);
    }
}

void checkPredictSpeed() {
    Position pos;
    constexpr int64_t REPEAT_NUM = 1000;
    std::cout << std::fixed;
    std::mt19937_64 engine(0);

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