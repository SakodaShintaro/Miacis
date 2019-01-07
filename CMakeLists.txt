cmake_minimum_required(VERSION 3.10)
project(Miacis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native -DUSE_SEARCH_STACK -DUSE_NN")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(Miacis
        alphazero_trainer.cpp
        base_trainer.cpp
        bitboard.cpp
        bonanza_method_trainer.cpp
        game.cpp
        hash_table.cpp
        main.cpp
        MCTSearcher.cpp
        move.cpp
        piece.cpp
        piece_state.cpp
        position.cpp
        square.cpp
        test.cpp
        types.cpp
        uct_hash_table.cpp
        usi.cpp)
target_link_libraries(Miacis pthread stdc++fs primitiv)