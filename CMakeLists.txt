cmake_minimum_required(VERSION 3.10)
project(Miacis)

#共通する設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

list(APPEND CMAKE_PREFIX_PATH ~/pytorch)
find_package(Torch REQUIRED)

#使用するソースも共通なので変数でまとめる
set(SRCS alphazero_trainer.cpp
         base_trainer.cpp
         bitboard.cpp
         supervised_learner.cpp
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
         searcher_for_gen.cpp
         neural_network.cpp
         MCTSearcher.cpp
         parallel_MCTSearcher.cpp)

add_executable(Miacis_sv ${SRCS})
add_executable(Miacis_cv ${SRCS})
add_executable(TorchSa ${SRCS})
add_executable(TorchCa ${SRCS})
target_compile_definitions(Miacis_cv PUBLIC USE_CATEGORICAL)
target_compile_definitions(TorchSa PUBLIC USE_LIBTORCH)
target_compile_definitions(TorchCa PUBLIC USE_LIBTORCH USE_CATEGORICAL)

target_link_libraries(Miacis_sv pthread stdc++fs primitiv)
target_link_libraries(Miacis_cv pthread stdc++fs primitiv)
target_link_libraries(TorchSa pthread stdc++fs "${TORCH_LIBRARIES}")
target_link_libraries(TorchCa pthread stdc++fs "${TORCH_LIBRARIES}")

