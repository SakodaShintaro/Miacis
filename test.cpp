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
#include"game_generator2.hpp"


#include<cassert>
#include<numeric>
#include<climits>
#include<iomanip>

void testHand() {
	Hand h;
	h.set(PAWN, 10);
	assert(h.num(PAWN) == 10);
	h.set(LANCE, 3);
	assert(h.num(LANCE) ==3);
	h.set(KNIGHT, 2);
	assert(h.num(KNIGHT) == 2);
	h.set(SILVER, 1);
	assert(h.num(SILVER) == 1);
}

void testSFEN() {
    std::string sfen1 = "1nsgsk2l/l8/1pppp1+R1n/p4pB1p/9/2P1P4/PP1P1PP1P/1B1S5/LN1GKGSNL w RG3P 1";
    std::cout << sfen1 << std::endl;
    Position p;
    p.loadSFEN(sfen1);
    p.printForDebug();
    std::string sfen2 = "lnsgkgsn1/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
    std::cout << sfen2 << std::endl;
    p.loadSFEN(sfen2);
    p.printForDebug();
}

void testMakeRandomPosition() {
    //auto s = std::make_unique<Searcher>(usi_option.USI_Hash, 1);

    //uint64_t try_num, random_turn, SEARCH_DEPTH;
    //double temperature;
    //std::cout << "何回試行するか:";
    //std::cin >> try_num;
    //std::cout << "何手ランダムの局面か:";
    //std::cin >> random_turn;
    //std::cout << "ランダムムーブ後の局面を調べる深さ:";
    //std::cin >> SEARCH_DEPTH;
    //std::cout << "温度:";
    //std::cin >> temperature;
    //std::cout << std::endl;

    //usi_option.draw_turn = 256;
    //usi_option.depth_limit = PLY * (int32_t)SEARCH_DEPTH;

    ////同一局面の登場回数
    //std::map<uint64_t, uint64_t> appear_num;

    ////評価値
    //std::vector<double> scores(try_num);

    //for (uint64_t i = 0; i < try_num; i++) {
    //    Position p(*eval_params);

    //    int move_count = 0;
    //    for (uint64_t j = 0; j < random_turn; j++) {
    //        Move random_move = s->softmaxChoice(p, temperature);
    //        if (random_move == NULL_MOVE) {
    //            break;
    //        }
    //        p.doMove(random_move);
    //        move_count++;
    //    }

    //    //手数分ランダムに動かした
    //    //p.print();
    //    
    //    if (move_count != random_turn) {
    //        //途中で詰んだ場合もう一度やり直す
    //        for (uint64_t j = 0; j < move_count; j++) {
    //            p.undo();
    //        }
    //        i--;
    //        continue;
    //    }

    //    //探索してランダムムーブ後の局面の評価値を得る
    //    auto result = s->think(p);
    //    auto move = result.first;
    //    if (isMatedScore(move.score)) {
    //        //探索して詰みが見えた場合ももう一度
    //        for (uint64_t j = 0; j < move_count; j++) {
    //            p.undo();
    //        }
    //        i--;
    //        continue;
    //    }
    //    scores[i] = static_cast<double>(move.score);
    //    //std::cout << "今回のスコア : " << scores[i] << std::endl;
    //    appear_num[p.hash_value()]++;
    //}

    ////統計情報を得る
    //double sum = std::accumulate(scores.begin(), scores.end(), 0.0);
    //double average = sum / try_num;
    //double variance = 0;
    //for (double s : scores) {
    //    variance += std::pow(s - average, 2);
    //}
    //variance /= try_num;

    //printf("average = %f, variance = %f\n", average, variance);

    ////局面の重複を確認
    //for (auto p : appear_num) {
    //    if (p.second >= 2) {
    //        std::cout << p.first << "の局面の登場回数:" << p.second << std::endl;
    //    }
    //}

    //printf("終了\n");
}

void testNN() {
    nn->load(MODEL_PATH);
    Position pos;
    auto searcher = std::make_unique<MCTSearcher<Tensor>>(16, 1, *nn);
    usi_option.playout_limit = 800;

    testToLabel();

    std::random_device rd;

    while (true) {
        pos.print(false);

//        auto moves = pos.generateAllMoves();
//        if (moves.empty()) {
//            break;
//        }
//        Move best_move = moves[rd() % moves.size()];

        auto search_result = searcher->think(pos);
        Move best_move = search_result.first;
        if (best_move == NULL_MOVE) {
            break;
        }

        pos.doMove(best_move);
    }
}

void testToLabel() {
    //for (Piece subject : PieceList) {
    //    for (Square to : SquareList) {
    //        for (Square from : SquareList) {
    //            Move move(to, from, false, false, subject);
    //            printf("%4d ", move.toLabel());
    //            move.printWithScore();
    //        }
    //    }
    //}

    Move dragon11b(SQ11, SQ12, false, false, BLACK_ROOK_PROMOTE);
    Move dragon99w(SQ99, SQ98, false, false, WHITE_ROOK_PROMOTE);
    assert(dragon11b.toLabel() == dragon99w.toLabel());
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

void testSFENoutput() {
    std::random_device rd;
    std::default_random_engine engine(rd());
    nn->init();
    usi_option.draw_turn = 512;
    for (int32_t i = 0; i < 100; i++) {
        Position pos1, pos2;

        while (true) {
            auto moves = pos1.generateAllMoves();
            if (moves.empty()) {
                break;
            }

            auto move = moves[engine() % moves.size()];
            pos1.doMove(move);
            pos2.loadSFEN(pos1.toSFEN());
        }
    }
}

void testSpeed() {
    nn->load(MODEL_PATH);

    usi_option.limit_msec = LLONG_MAX;
    usi_option.random_turn = 0;
    usi_option.playout_limit = 800;
    usi_option.USI_Hash = 1024;

    Position pos;

    for (usi_option.thread_num = 1; usi_option.thread_num <= 128; usi_option.thread_num++) {
        ParallelMCTSearcher searcher(usi_option.USI_Hash, usi_option.thread_num, *nn);

        auto start = std::chrono::steady_clock::now();
        for (int32_t i = 0; i < 10; i++) {
            searcher.think(pos);
        }
        auto end = std::chrono::steady_clock::now();
        auto elapsed = end - start;
        std::cout << std::setw(4) << usi_option.thread_num << " " << elapsed.count() << std::endl;
    }
}

void checkGenSpeed() {
    usi_option.USI_Hash = 1;
    usi_option.playout_limit = 800;
    usi_option.draw_turn = 100;

    ReplayBuffer buffer;
    buffer.max_size = 10000;

    for (int64_t thread_num = 8; thread_num <= 128; thread_num *= 2) {
        buffer.clear();
        auto start = std::chrono::steady_clock::now();
        GameGenerator2 generator(0, thread_num * 2, thread_num, buffer, *nn);
        generator.genGames();
        auto end = std::chrono::steady_clock::now();
        auto ela = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "thread_num = " << std::setw(4) << thread_num << ", elapsed = " << ela.count() << ", speed = "
                  << (usi_option.draw_turn * thread_num * 2 * 1000.0) / ela.count() << " pos / sec" << std::endl;
    }
}