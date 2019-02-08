#include"test.hpp"
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

void testNN() {
#ifdef USE_LIBTORCH
    torch::load(nn, MODEL_PATH);
    auto searcher = std::make_unique<MCTSearcher>(1, nn);
#else
    nn->load(MODEL_PATH);
    auto searcher = std::make_unique<MCTSearcher>(16, *nn);
#endif

    Position pos;
    usi_option.playout_limit = 800;

    std::random_device rd;

    while (true) {
        pos.print(false);

        auto search_result = searcher->think(pos);
        Move best_move = search_result.first;
        if (best_move == NULL_MOVE) {
            break;
        }

        pos.doMove(best_move);
    }
}

void testKifuOutput() {
    //Game game;
    //eval_params->readFile();
    //Position pos_c(*eval_params), pos_t(*eval_params);
    //auto searcher = std::make_unique<Searcher>(usi_option.USI_Hash, 1);
    //usi_option.draw_turn = 512;
    //usi_option.depth_limit = PLY * 3;

    //while (true) {
    //    //iが偶数のときpos_cが先手
    //    auto move_and_teacher = ((pos_c.turn_number() % 2) == 0 ?
    //        searcher->think(pos_c) :
    //        searcher->think(pos_t));
    //    Move best_move = move_and_teacher.first;
    //    TeacherType teacher = move_and_teacher.second;

    //    if (best_move == NULL_MOVE) { //NULL_MOVEは投了を示す
    //        game.result = (pos_c.color() == BLACK ? Game::RESULT_WHITE_WIN : Game::RESULT_BLACK_WIN);
    //        break;
    //    }

    //    if (!pos_c.isLegalMove(best_move)) {
    //        pos_c.printForDebug();
    //        best_move.printWithScore();
    //        assert(false);
    //    }
    //    pos_c.doMove(best_move);
    //    pos_t.doMove(best_move);
    //    game.moves.push_back(best_move);
    //    game.teachers.push_back(teacher);

    //    Score repeat_score;
    //    if (pos_c.isRepeating(repeat_score)) { //繰り返し
    //        if (isMatedScore(repeat_score)) { //連続王手の千日手だけが怖い
    //            //しかしどうすればいいかわからない
    //        } else {
    //            game.result = Game::RESULT_DRAW_REPEAT;
    //            break;
    //        }
    //    }

    //    if (pos_c.turn_number() >= usi_option.draw_turn) { //長手数
    //        game.result = Game::RESULT_DRAW_OVER_LIMIT;
    //        break;
    //    }
    //}

    //game.writeCSAFile("./");
    //game.writeKifuFile("./");
    //printf("finish testKifuOutput()\n");
}

void checkGenSpeed() {
    usi_option.USI_Hash = 1;
    usi_option.playout_limit = 800;
    usi_option.draw_turn = 512;

    int64_t game_num;
    ReplayBuffer buffer;
    buffer.max_size = game_num * usi_option.draw_turn;
    
    for (int64_t thread_num = 2; thread_num <= 256; thread_num *= 2) {
        game_num = thread_num;
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