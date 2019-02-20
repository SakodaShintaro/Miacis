#ifndef MIACIS_REPLAY_BUFFER_HPP
#define MIACIS_REPLAY_BUFFER_HPP
#include"neural_network.hpp"
#include"game.hpp"
#include<mutex>

class ReplayBuffer{
public:
    ReplayBuffer(int64_t first_wait, int64_t max_size, float lambda) : first_wait_(first_wait), max_size_(max_size), lambda_(lambda) {}

    //ミニバッチを作って返す関数
    void makeBatch(int64_t batch_size, std::vector<float>& inputs, std::vector<PolicyTeacherType>& policy_teachers,
                   std::vector<ValueTeacherType>& value_teachers);

    //データを入れる関数
    void push(Game& game);

    //checkGenSpeedで使うもの
    void clear();
    int64_t size() { return data_.size(); }

private:
    //実際のデータ
    std::vector<std::tuple<std::string, uint32_t, ValueTeacherType>> data_;

    //最初に待つ量
    int64_t first_wait_;

    //最大サイズ
    int64_t max_size_;

    //TD(λ)のパラメータ
    double lambda_;

    //排他制御用
    std::mutex mutex_;
};

#endif //MIACIS_REPLAY_BUFFER_HPP