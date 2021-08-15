#ifndef MIACIS_RANDOM_GAME_GENERATOR_HPP
#define MIACIS_RANDOM_GAME_GENERATOR_HPP

#include "game.hpp"
#include "model/infer_model.hpp"
#include "replay_buffer.hpp"
#include "search_options.hpp"
#include "searcher.hpp"
#include "searcher_for_mate.hpp"
#include <atomic>
#include <mutex>
#include <stack>
#include <utility>

//ランダムに自己対局をしてデータを生成するクラス
class RandomGameGenerator {
public:
    RandomGameGenerator(const SearchOptions& search_options, ReplayBuffer& rb)
        : stop_signal(false), search_options_(search_options), replay_buffer_(rb){};

    //生成してリプレイバッファに送り続ける関数
    void start();

    //停止信号。止めたいときは外部からこれをtrueにする
    bool stop_signal;

private:
    //1回行動する関数
    void step();

    //探索に関するオプション
    const SearchOptions& search_options_;

    //データを送るReplayBufferへの参照
    ReplayBuffer& replay_buffer_;

    //管理している対局データ
    Game game_;

    //現在思考している局面
    Position position_;
};

#endif //MIACIS_RANDOM_GAME_GENERATOR_HPP