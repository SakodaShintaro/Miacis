#ifndef MIACIS_REPLAY_BUFFER_HPP
#define MIACIS_REPLAY_BUFFER_HPP
#include"neural_network.hpp"
#include<mutex>

class ReplayBuffer{
public:
    //ミニバッチを作って返す関数
    std::tuple<std::vector<float>, std::vector<uint32_t>, std::vector<ValueTeacher>> makeBatch(int32_t batch_size);

    //データを入れる関数
    void push(const Position& pos, Move move, float value);
    void push(const Position& pos, TeacherType teacher);
    void push(const std::string& sfen, TeacherType teacher);

    //サイズの設定をする関数
    void setSize(int64_t max_size);

    //中身を確認する関数
    void show();
private:
    //実際のデータ
    std::vector<std::tuple<std::string, uint32_t, ValueTeacher>> data_;

    //最大サイズ
    uint64_t max_size_;

    //排他制御用
    std::mutex mutex_;
};

#endif //MIACIS_REPLAY_BUFFER_HPP