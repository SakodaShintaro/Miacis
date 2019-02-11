#ifndef MIACIS_REPLAY_BUFFER_HPP
#define MIACIS_REPLAY_BUFFER_HPP
#include"neural_network.hpp"
#include"game.hpp"
#include<mutex>

class ReplayBuffer{
public:
    //ミニバッチを作って返す関数
    void makeBatch(int32_t batch_size, std::vector<float>& inputs, std::vector<uint32_t>& policy_labels,
                   std::vector<ValueTeacher>& value_teachers);

    //データを入れる関数
    void push(Game& game);

    //checkGenSpeedで使うもの
    void clear();
    int64_t size() { return data_.size(); }

    //最初に待つ量
    uint64_t first_wait;

    //最大サイズ
    uint64_t max_size;

    //TD(λ)のパラメータ
    double lambda;

private:
    //実際のデータ
    std::vector<std::tuple<std::string, uint32_t, ValueTeacher>> data_;

    //排他制御用
    std::mutex mutex_;
};

#endif //MIACIS_REPLAY_BUFFER_HPP