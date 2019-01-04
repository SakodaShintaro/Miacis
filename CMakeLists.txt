cmake_minimum_required(VERSION 3.10)
project(Miacis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native -DUSE_SEARCH_STACK")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(Miacis
        alphabeta_searcher.cpp
        alphazero_trainer.cpp
        base_trainer.cpp
        bitboard.cpp
        bonanza_method_trainer.cpp
        calcScore.cpp
        game.cpp
        hash_table.cpp
        main.cpp
        MCTSearcher.cpp
        move.cpp
        move_picker.cpp
        network.cpp
        piece.cpp
        piece_state.cpp
        position.cpp
        rootstrap_trainer.cpp
        square.cpp
        test.cpp
        treestrap_trainer.cpp
        types.cpp
        uct_hash_table.cpp
        usi.cpp)
target_link_libraries(Miacis pthread stdc++fs)