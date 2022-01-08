cmake_minimum_required(VERSION 3.10)
project(Miacis)

# 共通する設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# LibTorch
list(APPEND CMAKE_PREFIX_PATH ../../libtorch-1.10.1)
find_package(Torch REQUIRED)

# TensorRT
set(TENSORRT_INCLUDE ../../TensorRT-8.2.2.1/include)
file(GLOB TENSORRT_LIBRARIES ../../TensorRT-8.2.2.1/lib/*.so)

# 各ゲームで共通する部分のソース
file(GLOB COMMON_SRCS ./*.cpp)
file(GLOB MODEL_SRCS ./model/*.cpp)
file(GLOB LEARN_SRCS ./learn/*.cpp)
file(GLOB SEARCH_SRCS ./search/*.cpp)
set(SRCS ${COMMON_SRCS} ${MODEL_SRCS} ${LEARN_SRCS} ${SEARCH_SRCS})

# 将棋用のバイナリをコンパイルするのに必要なソース
file(GLOB SHOGI_SRCS ./shogi/*.cpp)

# オセロ用のバイナリをコンパイルするのに必要なソース
file(GLOB OTHELLO_SRCS ./othello/*.cpp)

# オセロ用のバイナリをコンパイルするのに必要なソース
file(GLOB GO_SRCS ./go/*.cpp)

# 実行ファイルの定義
add_executable(Miacis_shogi_scalar ${SRCS} ${SHOGI_SRCS})
add_executable(Miacis_shogi_categorical ${SRCS} ${SHOGI_SRCS})
# add_executable(Miacis_shogi_dlshogi ${SRCS} ${SHOGI_SRCS})
target_compile_definitions(Miacis_shogi_scalar PUBLIC SHOGI)
target_compile_definitions(Miacis_shogi_categorical PUBLIC SHOGI USE_CATEGORICAL)
# target_compile_definitions(Miacis_shogi_dlshogi PUBLIC SHOGI DLSHOGI)

# add_executable(Miacis_othello_scalar ${SRCS} ${OTHELLO_SRCS})
# add_executable(Miacis_othello_categorical ${SRCS} ${OTHELLO_SRCS})
# target_compile_definitions(Miacis_othello_scalar PUBLIC OTHELLO)
# target_compile_definitions(Miacis_othello_categorical PUBLIC OTHELLO USE_CATEGORICAL)

# add_executable(Miacis_go_categorical ${SRCS} ${GO_SRCS})
# target_compile_definitions(Miacis_go_categorical PUBLIC GO USE_CATEGORICAL)

# foreachで回すためにリスト化
# set(BIN Miacis_shogi_scalar
#         Miacis_shogi_categorical
#         Miacis_shogi_dlshogi
#         Miacis_othello_scalar
#         Miacis_othello_categorical
#         Miacis_go_categorical)
set(BIN Miacis_shogi_scalar
        Miacis_shogi_categorical)

if (MSVC)
    foreach (bin IN ITEMS ${BIN})
        target_link_libraries(${bin} ${TORCH_LIBRARIES})
        target_include_directories(${bin} PRIVATE)
    endforeach ()
else ()
    foreach (bin IN ITEMS ${BIN})
        target_link_libraries(${bin} pthread stdc++fs ${TORCH_LIBRARIES} ${TENSORRT_LIBRARIES})
        target_include_directories(${bin} PRIVATE ${TENSORRT_INCLUDE})
    endforeach ()
endif ()
