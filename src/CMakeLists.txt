cmake_minimum_required(VERSION 3.10)
project(Miacis)

#共通する設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

list(APPEND CMAKE_PREFIX_PATH ../../libtorch-1.7.0)
find_package(Torch REQUIRED)

#各ゲームで共通する部分のソース
set(SRCS hash_table.cpp
        replay_buffer.cpp
        game_generator.cpp
        neural_network.cpp
        neural_network_modules.cpp
        supervised_learn.cpp
        learn.cpp
        alphazero_learn.cpp
        searcher.cpp
        searcher_for_play.cpp
        searcher_for_mate.cpp
        hyperparameter_loader.cpp
        segment_tree.cpp
        common.cpp
        search_nn/mcts_net/mcts_net.cpp
        search_nn/mcts_net/hash_table_for_mcts_net.cpp
        search_nn/proposed_model_lstm/proposed_model_lstm.cpp
        search_nn/proposed_model_transformer/proposed_model_transformer.cpp
        search_nn/stacked_lstm/stacked_lstm.cpp
        search_nn/state_encoder.cpp
        search_nn/simple_MLP/simple_mlp.cpp
        search_nn/common.cpp
        search_nn/base_model/base_model.cpp)

#将棋用のバイナリをコンパイルするのに必要なソース
set(SHOGI_SRCS shogi/main.cpp
        shogi/test.cpp
        shogi/bitboard.cpp
        shogi/move.cpp
        shogi/piece.cpp
        shogi/position.cpp
        shogi/square.cpp
        shogi/interface.cpp
        shogi/game.cpp
        shogi/book.cpp)

#オセロ用のバイナリをコンパイルするのに必要なソース
set(OTHELLO_SRCS othello/main.cpp
                 othello/position.cpp
                 othello/square.cpp
                 othello/piece.cpp
                 othello/interface.cpp
                 othello/game.cpp)

add_executable(Miacis_shogi_scalar      ${SRCS} ${SHOGI_SRCS})
target_compile_definitions(Miacis_shogi_scalar      PUBLIC SHOGI)

add_executable(Miacis_othello_scalar ${SRCS} ${OTHELLO_SRCS})
target_compile_definitions(Miacis_othello_scalar PUBLIC OTHELLO)

if (MSVC)
    target_link_libraries(Miacis_shogi_scalar        "${TORCH_LIBRARIES}")
    target_link_libraries(Miacis_othello_scalar      "${TORCH_LIBRARIES}")
else()
    target_link_libraries(Miacis_shogi_scalar        pthread stdc++fs "${TORCH_LIBRARIES}")
    target_link_libraries(Miacis_othello_scalar      pthread stdc++fs "${TORCH_LIBRARIES}")
endif()