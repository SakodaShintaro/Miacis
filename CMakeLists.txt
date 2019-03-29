cmake_minimum_required(VERSION 3.10)
project(Miacis)

#共通する設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(Torch REQUIRED)

#使用するソースも共通なので変数でまとめる
set(SRCS bitboard.cpp
         game.cpp
         main.cpp
         move.cpp
         piece.cpp
         position.cpp
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
         hyperparameter_manager.cpp)

add_executable(TorchSa ${SRCS})
add_executable(TorchCa ${SRCS})
target_compile_definitions(TorchCa PUBLIC USE_CATEGORICAL)

target_link_libraries(TorchSa pthread stdc++fs "${TORCH_LIBRARIES}")
target_link_libraries(TorchCa pthread stdc++fs "${TORCH_LIBRARIES}")