#include"test.hpp"
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
#ifdef USE_LIBTORCH
    torch::load(nn, MODEL_PATH);
#else
    nn->load(MODEL_PATH);
#endif
    SearcherForPlay searcher(usi_option.search_limit, usi_option.thread_num, usi_option.search_batch_size, nn);

    Position pos;
    Game game;

    auto start = std::chrono::steady_clock::now();
    while (true) {
        pos.print(true);

        Move best_move = searcher.think(pos);
        if (pos.turn_number() == 0) {
            best_move = Move(SQ76, SQ77);
            best_move = pos.transformValidMove(best_move);
        } else if (pos.turn_number() == 1) {
            best_move = Move(SQ34, SQ33);
            best_move = pos.transformValidMove(best_move);
        } else if (pos.turn_number() == 2) {
            best_move = Move(SQ33, SQ88, false, true);
            best_move = pos.transformValidMove(best_move);
        }

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
    usi_option.USI_Hash = 1;
    usi_option.search_limit = 800;
    usi_option.draw_turn = 100;

    for (int64_t thread_num = 2; thread_num <= 256; thread_num *= 2) {
        int64_t game_num = thread_num;
        ReplayBuffer buffer(0, game_num * usi_option.draw_turn, 1.0);
        auto start = std::chrono::steady_clock::now();
        GameGenerator generator(buffer, nn);
        generator.genGames(game_num);
        auto end = std::chrono::steady_clock::now();
        auto ela = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "thread_num = " << std::setw(4) << thread_num << ", elapsed = " << ela.count() << ", speed = "
                  << (buffer.size() * 1000.0) / ela.count() << " pos / sec" << std::endl;
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