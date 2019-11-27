cmake_minimum_required(VERSION 3.10)
project(Miacis)

#共通する設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

list(APPEND CMAKE_PREFIX_PATH ~/libtorch)
find_package(Torch REQUIRED)

#使用するソースも共通なので変数でまとめる
set(SRCS bitboard.cpp
         game.cpp
         main.cpp
         move.cpp
         piece.cpp
         position_shogi.cpp
         square.cpp
         test.cpp
         uct_hash_table.cpp
         usi.cpp
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

add_executable(Miacis_scalar      ${SRCS})
add_executable(Miacis_categorical ${SRCS})
target_compile_definitions(Miacis_categorical PUBLIC USE_CATEGORICAL)

target_link_libraries(Miacis_scalar      pthread stdc++fs "${TORCH_LIBRARIES}")
target_link_libraries(Miacis_categorical pthread stdc++fs "${TORCH_LIBRARIES}")