#include "replay_buffer.hpp"
#include "common.hpp"
#include "include_switch.hpp"
#include <iomanip>
#include <random>
#include <thread>

const std::string ReplayBuffer::KIFU_SAVE_DIR = "./learn_kifu/";

std::vector<LearningData> ReplayBuffer::makeBatch(int64_t batch_size) {
    //ロックの確保
    mutex_.lock();

    while (first_wait_ > 0) {
        mutex_.unlock();
        std::cout << "\rwait_remain = " << first_wait_ << "  " << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(10));
        mutex_.lock();
    }

    //現時点のpriorityの和を取得し,そこまでの範囲の一様分布生成器を作る
    float sum = segment_tree_.getSum();
    std::uniform_real_distribution<float> dist(0.0, sum);

    //データを取得
    std::vector<LearningData> data;
    pre_indices_.clear();
    for (int64_t i = 0; i < batch_size; i++) {
        //データの取り出し及びインデックスを保存
        uint64_t index = segment_tree_.getIndex(dist(engine));
        data.push_back(data_[index]);
        pre_indices_.push_back(index);
    }

    //ロックの解放
    mutex_.unlock();
    return data;
}

void ReplayBuffer::push(Game& game) {
    std::cerr << "No Implementation" << std::endl;
    std::exit(1);
}

void ReplayBuffer::update(const std::vector<float>& loss) {
    mutex_.lock();

    if (loss.size() != pre_indices_.size()) {
        std::cout << "loss.size() != pre_indices_.size()" << std::endl;
        std::exit(1);
    }
    for (uint64_t i = 0; i < loss.size(); i++) {
        segment_tree_.update(pre_indices_[i], std::pow(loss[i], alpha_));
    }

    mutex_.unlock();
}

void ReplayBuffer::fillByKifu(const std::string& file_path, float rate_threshold) {
    std::vector<LearningData> data = loadData(file_path, data_augmentation_, rate_threshold);
    std::cout << "data.size() = " << data.size() << ", max_size_ = " << max_size_ << std::endl;
    for (int64_t i = 0; i < std::min(max_size_, (int64_t)data.size()); i++) {
        //このデータを入れる位置を取得
        int64_t change_index = segment_tree_.getIndexToPush();

        //そこのデータを入れ替える
        data_[change_index] = data[i];

        //segment_treeのpriorityを更新
        segment_tree_.update(change_index, 3.0f);

        //データを加えた数をカウント
        //拡張したデータは一つとして数えた方が良いかもしれない？
        total_num_++;
        if (first_wait_ > 0) {
            first_wait_--;
        }
    }
}