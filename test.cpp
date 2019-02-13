﻿#include"test.hpp"
#include"usi.hpp"
#include"piece.hpp"
#include"square.hpp"
#include"position.hpp"
#include"hand.hpp"
#include"game.hpp"
#include"neural_network.hpp"
#include"MCTSearcher.hpp"
#include"parallel_MCTSearcher.hpp"
#include"usi_options.hpp"
#include"replay_buffer.hpp"
#include"game_generator.hpp"
#include<cassert>
#include<numeric>
#include<climits>
#include<iomanip>

void test() {
    usi_option.playout_limit = 800;
    usi_option.limit_msec = LLONG_MAX;
    usi_option.USI_Hash = 1;
    usi_option.draw_turn = 512;
#ifdef USE_LIBTORCH
    torch::load(nn, MODEL_PATH);
    //auto searcher = std::make_unique<MCTSearcher>(usi_option.USI_Hash, nn);
    auto searcher = std::make_unique<ParallelMCTSearcher>(usi_option.USI_Hash, 1, nn);
#else
    nn->load(MODEL_PATH);
    //auto searcher = std::make_unique<MCTSearcher>(usi_option.USI_Hash, *nn);
    auto searcher = std::make_unique<ParallelMCTSearcher>(usi_option.USI_Hash, 1, *nn);
#endif

    Position pos;
    Game game;

    auto start = std::chrono::steady_clock::now();
    while (true) {
        pos.print(false);

        Move best_move = searcher->think(pos);
        if (best_move == NULL_MOVE) {
            //投了
            game.result = (pos.color() == BLACK ? Game::RESULT_WHITE_WIN : Game::RESULT_BLACK_WIN);
            break;
        }
        Score repeat_score;
        if (pos.isRepeating(repeat_score)) {
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
    usi_option.USI_Hash = 1;
    usi_option.playout_limit = 800;
    usi_option.draw_turn = 100;

    int64_t game_num;
    ReplayBuffer buffer;

    for (int64_t thread_num = 2; thread_num <= 256; thread_num *= 2) {
        game_num = thread_num;
        buffer.max_size = static_cast<uint64_t>(game_num * usi_option.draw_turn);
        buffer.clear();
        auto start = std::chrono::steady_clock::now();
#ifdef USE_LIBTORCH
        GameGenerator generator(0, thread_num, buffer, nn);
#else
        GameGenerator generator(0, thread_num, buffer, *nn);
#endif
        generator.genGames(game_num);
        auto end = std::chrono::steady_clock::now();
        auto ela = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "thread_num = " << std::setw(4) << thread_num << ", elapsed = " << ela.count() << ", speed = "
                  << (buffer.size() * 1000.0) / ela.count() << " pos / sec" << std::endl;
    }
}