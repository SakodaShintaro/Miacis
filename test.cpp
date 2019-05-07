﻿#include"test.hpp"
#include"usi.hpp"
#include"piece.hpp"
#include"square.hpp"
#include"position.hpp"
#include"hand.hpp"
#include"game.hpp"
#include"neural_network.hpp"
#include"usi_options.hpp"
#include"replay_buffer.hpp"
#include"game_generator.hpp"
#include"searcher_for_play.hpp"
#include"learn.hpp"
#include<cassert>
#include<numeric>
#include<climits>
#include<iomanip>

void test() {
    usi_option.search_limit = 800;
    usi_option.limit_msec = LLONG_MAX;
    usi_option.USI_Hash = 1;
    usi_option.thread_num = 1;
    usi_option.search_batch_size = 1;
    usi_option.random_turn = 512;
    usi_option.draw_turn = 512;
    torch::load(nn, MODEL_PATH);
    SearcherForPlay searcher(usi_option.search_limit, usi_option.thread_num, usi_option.search_batch_size, nn);

    Position pos;
    Game game;

    auto start = std::chrono::steady_clock::now();
    while (true) {
        Move best_move = searcher.think(pos);

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
        } else if (pos.turn_number() >= usi_option.draw_turn) {
            //長手数
            game.result = Game::RESULT_DRAW_OVER_LIMIT;
            break;
        }

        pos.doMove(best_move);
        game.moves.push_back(best_move);
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << elapsed.count() / pos.turn_number() << " msec / pos" << std::endl;

    game.writeKifuFile("./");
    std::cout << "finish test" << std::endl;
}

void checkGenSpeed() {
    torch::load(nn, MODEL_PATH);

    usi_option.limit_msec = LLONG_MAX;
    usi_option.search_limit = 100;
    usi_option.draw_turn = 512;
    usi_option.random_turn = 512;
    usi_option.thread_num = 2;
    constexpr int64_t limit = 20000;

    for (usi_option.search_batch_size = 32; usi_option.search_batch_size <= 128; usi_option.search_batch_size *= 2) {
        ReplayBuffer buffer(0, limit, 1.0);
        usi_option.stop_signal = false;
        auto start = std::chrono::steady_clock::now();
        GameGenerator generator(buffer, nn);
        std::thread t(&GameGenerator::genGames, &generator, (int64_t)1e15);
        while (buffer.size() < limit) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        auto end = std::chrono::steady_clock::now();
        usi_option.stop_signal = true;
        t.join();
        auto ela = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "search_batch_size = " << std::setw(4) << usi_option.search_batch_size
                  << ", elapsed = " << ela.count()
                  << ", size = " << buffer.size()
                  << ", speed = " << (buffer.size() * 1000.0) / ela.count() << " pos / sec" << std::endl;
    }
}

void checkSearchSpeed() {
    usi_option.limit_msec = 10000;
    usi_option.search_limit = static_cast<int64_t>(1e10);
    Position pos;
    for (uint64_t search_batch_size = 64; search_batch_size <= 512; search_batch_size *= 2) {
        std::cout << "search_batch_size = " << search_batch_size << std::endl;
        for (uint64_t thread_num = 1; thread_num <= 3; thread_num++) {
            SearcherForPlay searcher(1000000, thread_num, search_batch_size, nn);
            searcher.think(pos);
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
        auto v = validation(data);
        printf("%5d回目 : %f\t%f\n", i, v[0], v[1]);
    }
}

void checkPredictSpeed() {
    Position pos;
    constexpr int64_t time = 1000;
    for (int64_t j = 0; j < 10; j++) {
        auto start = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < time; i++) {
            torch::NoGradGuard no_grad_guard;
            nn->policyAndValueBatch(pos.makeFeature());
        }
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "elapsed = " << elapsed.count() << std::endl;

        auto moves = pos.generateAllMoves();
        pos.doMove(moves[j]);
    }
}