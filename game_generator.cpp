//
// Created by sakoda on 19/01/30.
//

#include <atomic>
#include <thread>
#include "game_generator.hpp"
#include "usi_options.hpp"

void GameGenerator::genGames() {
    //GPUスレッドを生成
    std::thread gpu_thread(&GameGenerator::gpuFunc, this);

    running_ = true;

    for (int64_t i = 0; i < thread_num_; i++) {
        searchers_.emplace_back(usi_option.USI_Hash, i, *this);
    }

    //生成スレッドを生成
    std::vector<std::thread> threads;
    for (int64_t i = 0; i < thread_num_; i++) {
        threads.emplace_back(&GameGenerator::genSlave, this, i);
    }

    //生成スレッドが終わるのを待つ
    for (auto& th : threads) {
        th.join();
    }

    //GPUスレッドへ停止信号を送る
    running_ = false;

    //GPUスレッドが終わるのを待つ
    gpu_thread.join();
}

void GameGenerator::genSlave(int64_t id) {
    //生成
    while (true) {
        int64_t num = game_num_--;
        std::cout << "num = " << num << std::endl;
        if (num <= 0) {
            break;
        }

        Game game;
        Position pos;

        while (true) {
            //id番目のsearcherを使って探索
            std::cout << id << " : " << pos.turn_number() << std::endl;
            auto result = searchers_[id].think(pos);
            if (result.first == NULL_MOVE) {
                break;
            }
            pos.doMove(result.first);
            game.moves.push_back(result.first);
            game.teachers.push_back(result.second);
        }
    }
}

void GameGenerator::gpuFunc() {
    bool enough_batch_size = true;

    while (running_) {
        lock_expand_.lock();
        if (current_features_.empty()) {
            lock_expand_.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            continue;
        }

        if (!enough_batch_size && current_features_.size() < thread_num_ / 2) {
            lock_expand_.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            enough_batch_size = true;
            continue;
        }

        enough_batch_size = false;

        //現在のキューを保存
        auto eval_features = current_features_;
        auto eval_hash_index_queue = current_hash_index_queue_;
        auto eval_thread_ids = current_thread_ids_;

        //カレントキューを入れ替える
        current_queue_index_ ^= 1;
        current_features_ = features_[current_queue_index_];
        current_features_.clear();
        current_hash_index_queue_ = hash_index_queues_[current_queue_index_];
        current_hash_index_queue_.clear();
        current_thread_ids_ = thread_ids_[current_queue_index_];
        current_thread_ids_.clear();
        lock_expand_.unlock();

        auto result = evaluator_.policyAndValueBatch(eval_features);
        auto policies = result.first;
        auto values = result.second;

        for (int32_t i = 0; i < eval_hash_index_queue.size(); i++) {
            //何番目のスレッドのどの位置に書き込むかを取得
            auto& current_node = searchers_[eval_thread_ids[i]].hash_table_[eval_hash_index_queue[i]];

            //policyを設定
            std::vector<float> legal_moves_policy(static_cast<unsigned long>(current_node.child_num));
            for (int32_t j = 0; j < current_node.child_num; j++) {
                legal_moves_policy[j] = policies[i][current_node.legal_moves[j].toLabel()];
            }
            current_node.nn_rates = softmax(legal_moves_policy);

            //valueを設定
            current_node.value = values[i];
            current_node.evaled = true;
        }
    }
}

void GameGenerator::clearEvalQueue() {
    current_queue_index_ = 0;
    for (int32_t i = 0; i < 2; i++) {
        features_[i].clear();
        hash_index_queues_[i].clear();
    }
    current_features_ = features_[current_queue_index_];
    current_hash_index_queue_ = hash_index_queues_[current_queue_index_];
}