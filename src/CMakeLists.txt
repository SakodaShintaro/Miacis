cmake_minimum_required(VERSION 3.10)
project(Miacis)

# 共通する設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# LibTorch
list(APPEND CMAKE_PREFIX_PATH ../../libtorch-1.13.1)
find_package(Torch REQUIRED)

# TensorRT
set(TENSORRT_INCLUDE ../../TensorRT-8.2.2.1/include)
file(GLOB TENSORRT_LIBRARIES ../../TensorRT-8.2.2.1/lib/*.so)

# TorchTensorRT
set(TORCHTRT_INCLUDE ../../torch_tensorrt-1.3.0/include)
file(GLOB TORCHTRT_LIBRARIES ../../torch_tensorrt-1.3.0/lib/*.so)

# 各ゲームで共通する部分のソース
file(GLOB COMMON_SRCS ./*.cpp)
file(GLOB MODEL_SRCS ./model/*.cpp)
file(GLOB LEARN_SRCS ./learn/*.cpp)
file(GLOB SEARCH_SRCS ./search/*.cpp)
set(SRCS ${COMMON_SRCS} ${MODEL_SRCS} ${LEARN_SRCS} ${SEARCH_SRCS})

# 将棋用のバイナリをコンパイルするのに必要なソース
file(GLOB SHOGI_SRCS ./shogi/*.cpp)

# 実行ファイルの定義
add_executable(Miacis_shogi_scalar ${SRCS} ${SHOGI_SRCS})
add_executable(Miacis_shogi_categorical ${SRCS} ${SHOGI_SRCS})
target_compile_definitions(Miacis_shogi_scalar PUBLIC SHOGI)
target_compile_definitions(Miacis_shogi_categorical PUBLIC SHOGI USE_CATEGORICAL)

set(BIN Miacis_shogi_scalar
    Miacis_shogi_categorical)

if(MSVC)
    foreach(bin IN ITEMS ${BIN})
        target_link_libraries(${bin} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARIES} ${TORCHTRT_LIBRARIES})
        target_include_directories(${bin} PRIVATE ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE} ${TORCHTRT_INCLUDE})
    endforeach()
else()
    foreach(bin IN ITEMS ${BIN})
        target_link_libraries(${bin} pthread stdc++fs ${CUDA_LIBRARIES} ${TENSORRT_LIBRARIES} ${TORCH_LIBRARIES} ${TORCHTRT_LIBRARIES})
        target_include_directories(${bin} PRIVATE ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE} ${TORCH_INCLUDE_DIRS} ${TORCHTRT_INCLUDE})
    endforeach()
endif()
