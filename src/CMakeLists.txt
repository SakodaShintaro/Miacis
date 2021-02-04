cmake_minimum_required(VERSION 3.10)
project(Miacis)

# 共通する設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# LibTorch
list(APPEND CMAKE_PREFIX_PATH ../../libtorch-1.7.0)
find_package(Torch REQUIRED)

# TRTorch
set(TRTORCH_INCLUDE ../../trtorch/include)
file(GLOB TRTORCH_LIBRARIES ../../trtorch/lib/*.so)

# 各ゲームで共通する部分のソース
set(SRCS hash_table.cpp
         replay_buffer.cpp
         game_generator.cpp
         neural_network.cpp
         learn.cpp
         supervised_learn.cpp
         reinforcement_learn.cpp
         searcher.cpp
         searcher_for_play.cpp
         searcher_for_mate.cpp
         hyperparameter_loader.cpp
         segment_tree.cpp
         common.cpp
         timer.cpp
         learning_model.cpp
         infer_model.cpp
         )

# 将棋用のバイナリをコンパイルするのに必要なソース
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

# オセロ用のバイナリをコンパイルするのに必要なソース
set(OTHELLO_SRCS othello/main.cpp
                 othello/position.cpp
                 othello/square.cpp
                 othello/piece.cpp
                 othello/interface.cpp
                 othello/game.cpp)

# 実行ファイルの定義
add_executable(Miacis_shogi_scalar      ${SRCS} ${SHOGI_SRCS})
add_executable(Miacis_shogi_categorical ${SRCS} ${SHOGI_SRCS})
target_compile_definitions(Miacis_shogi_scalar      PUBLIC SHOGI)
target_compile_definitions(Miacis_shogi_categorical PUBLIC SHOGI USE_CATEGORICAL)

add_executable(Miacis_othello_scalar      ${SRCS} ${OTHELLO_SRCS})
add_executable(Miacis_othello_categorical ${SRCS} ${OTHELLO_SRCS})
target_compile_definitions(Miacis_othello_scalar      PUBLIC OTHELLO)
target_compile_definitions(Miacis_othello_categorical PUBLIC OTHELLO USE_CATEGORICAL)

set(BIN Miacis_shogi_scalar Miacis_shogi_categorical Miacis_othello_scalar Miacis_othello_categorical)

if (MSVC)
    foreach(bin IN ITEMS ${BIN})
        target_link_libraries(${bin} ${TORCH_LIBRARIES} ${TRTORCH_LIBRARIES})
        target_include_directories(${bin} PRIVATE ${TRTORCH_INCLUDE})
    endforeach()
else()
    foreach(bin IN ITEMS ${BIN})
        target_link_libraries(${bin} pthread stdc++fs ${TORCH_LIBRARIES} ${TRTORCH_LIBRARIES})
        target_include_directories(${bin} PRIVATE ${TRTORCH_INCLUDE})
    endforeach()
endif()