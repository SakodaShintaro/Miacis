#include"test.hpp"
#include"game_generator.hpp"
#include"searcher_for_play.hpp"
#include"learn.hpp"

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
        Score repeat_score;
        if (pos.isRepeating(repeat_score) && repeat_score == (MAX_SCORE + MIN_SCORE) / 2) {
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

    constexpr int64_t buffer_size = 20000;
    UsiOptions usi_options;
    usi_options.search_limit = 800;
    usi_options.draw_turn = 256;
    usi_options.thread_num = 2;
    constexpr FloatType Q_dist_lambda = 0.0;

    for (usi_options.search_batch_size = 32; usi_options.search_batch_size <= 128; usi_options.search_batch_size *= 2) {
        ReplayBuffer buffer(0, buffer_size, 10 * buffer_size, 1.0, 1.0);
        Searcher::stop_signal = false;
        auto start = std::chrono::steady_clock::now();
        GameGenerator generator(usi_options, Q_dist_lambda, buffer, nn);
        std::thread t(&GameGenerator::genGames, &generator, (int64_t)1e15);
        while (buffer.size() < buffer_size) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        auto end = std::chrono::steady_clock::now();
        Searcher::stop_signal = true;
        t.join();
        auto ela = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "search_batch_size = " << std::setw(4) << usi_options.search_batch_size
                  << ", elapsed = " << ela.count()
                  << ", size = " << buffer.size()
                  << ", speed = " << (buffer.size() * 1000.0) / ela.count() << " pos / sec" << std::endl;
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
    data.erase(data.begin() + 40960, data.end());
    data.shrink_to_fit();

    auto v = validation(data, 32);
    printf("%f\t%f\t%f\n", v[0], v[1], v[2]);
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

void checkTransitionModel() {
    torch::load(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
    torch::NoGradGuard no_grad_guard;

    Position pos;
    //現状態表現
    torch::Tensor seq_predicted_state_rep = nn->encodeStates(pos.makeFeature());

    std::cout << std::fixed;

    std::mt19937_64 engine(1);
    for (int64_t i = 1; ; i++) {
        std::vector<Move> moves = pos.generateAllMoves();
        if (moves.empty()) {
            break;
        }
        //現状態表現
        torch::Tensor curr_state_rep = nn->encodeStates(pos.makeFeature());

        //行動をランダムに選択
        auto y = nn->policyAndValueBatch(pos.makeFeature());
        auto raw_policy = y.first[0];
        std::vector<float> masked_policy(moves.size());
        for (uint64_t j = 0; j < moves.size(); j++) {
            masked_policy[j] = raw_policy[moves[j].toLabel()];
        }
        Move best_move = moves[std::max_element(masked_policy.begin(), masked_policy.end()) - masked_policy.begin()];
        pos.doMove(best_move);

        //次状態表現
        torch::Tensor next_state_rep = nn->encodeStates(pos.makeFeature());

        //行動の表現
        torch::Tensor move_rep = nn->encodeActions({ best_move });

        //次状態表現の予測
        torch::Tensor predicted_state_rep = nn->predictTransition(curr_state_rep, move_rep);

        //連続的に予測している表現
        seq_predicted_state_rep = nn->predictTransition(seq_predicted_state_rep, move_rep);

        //損失を計算
        torch::Tensor loss1 = nn->transitionLoss(predicted_state_rep, next_state_rep);
        torch::Tensor loss2 = nn->transitionLoss(seq_predicted_state_rep, next_state_rep);

        //状態価値を計算
        torch::Tensor value0 = nn->decodeValue(next_state_rep);
        torch::Tensor value1 = nn->decodeValue(predicted_state_rep);
        torch::Tensor value2 = nn->decodeValue(seq_predicted_state_rep);

        std::cout << i << "\t" << loss1.item<float>() << "\t" << loss2.item<float>() << "\t"
            << value0.item<float>() << "\t" << value1.item<float>() << "\t" << value2.item<float>() << "\t";
        best_move.print();
    }
}

void checkActionRepresentations() {
    torch::load(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
    torch::NoGradGuard no_grad_guard;

    std::cout << std::fixed;

    std::vector<Move> moves;
    moves.emplace_back(SQ53, SQ54, false, false, BLACK_GOLD, EMPTY);
    moves.emplace_back(SQ53, SQ54, false, false, BLACK_PAWN_PROMOTE, EMPTY);
    moves.emplace_back(SQ53, SQ54, false, false, BLACK_LANCE_PROMOTE, EMPTY);
    moves.emplace_back(SQ53, SQ54, false, false, BLACK_KNIGHT_PROMOTE, EMPTY);
    moves.emplace_back(SQ53, SQ43, false, false, BLACK_KNIGHT_PROMOTE, EMPTY);
    moves.emplace_back(SQ55, SQ54, false, false, BLACK_KNIGHT_PROMOTE, EMPTY);
    moves.emplace_back(SQ54, SQ55, false, false, BLACK_KNIGHT_PROMOTE, EMPTY);
    moves.emplace_back(SQ24, SQ28, false, false, BLACK_ROOK, EMPTY);
    moves.emplace_back(SQ22, SQ88, false, false, BLACK_BISHOP, EMPTY);

    for (uint64_t i = 0; i < moves.size(); i++) {
        moves[i].print();
        std::cout << "\t\t";
        for (uint64_t j = 0; j < i; j++) {
            torch::Tensor move_reps = nn->encodeActions(moves);
            torch::Tensor diff = move_reps[i] - move_reps[j];

            torch::Tensor loss = torch::pow(diff, 2).mean();
            std::cout << loss.item<float>() << " ";
        }
        std::cout << std::endl;
    }
}

void checkReconstruct() {
    torch::load(nn, NeuralNetworkImpl::DEFAULT_MODEL_NAME);
    torch::NoGradGuard no_grad_guard;

    Position pos;

    while (true) {
        //現状態表現
        torch::Tensor curr_state_rep = nn->encodeStates(pos.makeFeature());

        //局面を表示して再構成と比較
        pos.print();
        nn->reconstruct(curr_state_rep);

        //行動をランダムに選択
        std::vector<Move> moves = pos.generateAllMoves();
        if (moves.empty()) {
            break;
        }

        auto y = nn->policyAndValueBatch(pos.makeFeature());
        auto raw_policy = y.first[0];
        std::vector<float> masked_policy(moves.size());
        for (uint64_t j = 0; j < moves.size(); j++) {
            masked_policy[j] = raw_policy[moves[j].toLabel()];
        }
        Move best_move = moves[std::max_element(masked_policy.begin(), masked_policy.end()) - masked_policy.begin()];
        pos.doMove(best_move);

    }
}