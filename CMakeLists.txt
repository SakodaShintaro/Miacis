cmake_minimum_required(VERSION 3.10)
project(Miacis)

list(APPEND CMAKE_PREFIX_PATH /home/sakoda/libtorch)
find_package(Torch REQUIRED)

include_directories(/home/sakoda/libtorch/include)
include_directories(/home/sakoda/libtorch/include/torch/csrc/api/include)

#link_directories(/home/sakoda/libtorch/lib)
#link_directories(${TORCH_LIBRARIES})
#message(STATUS "Pytorch status:")
#message(STATUS "    libraries: ${TORCH_LIBRARIES}")

#共通する設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#使用するソースも共通なので変数でまとめる
set(SRCS alphazero_trainer.cpp
         base_trainer.cpp
         bitboard.cpp
         bonanza_method_trainer.cpp
         game.cpp
         main.cpp
         move.cpp
         piece.cpp
         position.cpp
         square.cpp
         test.cpp
         uct_hash_table.cpp
         usi.cpp
         replay_buffer.cpp)

add_executable(Miacis_sv ${SRCS})
add_executable(Miacis_cv ${SRCS})
target_compile_definitions(Miacis_cv PUBLIC USE_CATEGORICAL)

target_link_libraries(Miacis_sv pthread stdc++fs primitiv ${TORCH_LIBRARIES})
target_link_libraries(Miacis_cv pthread stdc++fs primitiv ${TORCH_LIBRARIES})