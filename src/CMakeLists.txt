cmake_minimum_required(VERSION 3.10)
project(Miacis)

#共通する設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

list(APPEND CMAKE_PREFIX_PATH ~/libtorch)
find_package(Torch REQUIRED)

#各ゲームで共通する部分のソース
set(SRCS uct_hash_table.cpp
         replay_buffer.cpp
         game_generator.cpp
         neural_network.cpp
         supervised_learn.cpp
         learn.cpp
         alphazero_learn.cpp
         searcher.cpp
         searcher_for_play.cpp
         searcher_for_generate.cpp
         hyperparameter_manager.cpp
         segment_tree.cpp)

#将棋用のバイナリをコンパイルするのに必要なソース
set(SHOGI_SRCS shogi/main.cpp
               shogi/test.cpp
               shogi/bitboard.cpp
               shogi/move.cpp
               shogi/piece.cpp
               shogi/position.cpp
               shogi/square.cpp
               shogi/usi.cpp
               shogi/load_game.cpp)

#オセロ用のバイナリをコンパイルするのに必要なソース
set(OTHELLO_SRCS othello/main.cpp
                 othello/position.cpp
                 othello/square.cpp
                 othello/piece.cpp
                 othello/game.cpp
        othello/interface.cpp)

add_executable(Miacis_shogi_scalar      ${SRCS} ${SHOGI_SRCS})
add_executable(Miacis_shogi_categorical ${SRCS} ${SHOGI_SRCS})
target_compile_definitions(Miacis_shogi_scalar      PUBLIC SHOGI)
target_compile_definitions(Miacis_shogi_categorical PUBLIC SHOGI)
target_compile_definitions(Miacis_shogi_categorical PUBLIC USE_CATEGORICAL)

add_executable(Miacis_othello_scalar      ${SRCS} ${OTHELLO_SRCS})
add_executable(Miacis_othello_categorical ${SRCS} ${OTHELLO_SRCS})
target_compile_definitions(Miacis_othello_scalar      PUBLIC OTHELLO)
target_compile_definitions(Miacis_othello_categorical PUBLIC OTHELLO)
target_compile_definitions(Miacis_othello_categorical PUBLIC USE_CATEGORICAL)

target_link_libraries(Miacis_shogi_scalar      pthread stdc++fs "${TORCH_LIBRARIES}")
target_link_libraries(Miacis_shogi_categorical pthread stdc++fs "${TORCH_LIBRARIES}")
target_link_libraries(Miacis_othello_scalar      pthread stdc++fs "${TORCH_LIBRARIES}")
target_link_libraries(Miacis_othello_categorical pthread stdc++fs "${TORCH_LIBRARIES}")