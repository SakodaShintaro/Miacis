#ifndef MIACIS_REPLAY_BUFFER_HPP
#define MIACIS_REPLAY_BUFFER_HPP
#include"neural_network.hpp"
#include"game.hpp"
#include"segment_tree.hpp"
#include<mutex>
#include<queue>
#include<experimental/filesystem>
#ifdef _MSC_VER
#include<direct.h>
#elif __GNUC__
#include<sys/stat.h>
#endif


class ReplayBuffer{
public:
    ReplayBuffer(int64_t first_wait, int64_t max_size, float lambda) : first_wait_(first_wait), max_size_(max_size),
    lambda_(lambda), segment_tree_(max_size), data_(max_size) {
        std::experimental::filesystem::remove_all("./learn_kifu");

        //棋譜を保存するディレクトリの作成
#ifdef _MSC_VER
        _mkdir("./learn_kifu");
#elif __GNUC__
        mkdir("./learn_kifu", ACCESSPERMS);
#endif
    }

    //ミニバッチを作って返す関数
    void makeBatch(int64_t batch_size, std::vector<float>& inputs, std::vector<PolicyTeacherType>& policy_teachers,
                   std::vector<ValueTeacherType>& value_teachers);

    //データを入れる関数
    void push(Game& game);

    //ミニバッチを学習した結果を用いてpriorityを更新する関数
    void update(const std::vector<float>& loss);

    //checkGenSpeedで使うもの
    void clear();
    int64_t size() { return data_.size(); }

private:
    //実際のデータ
    std::vector<std::tuple<std::string, TeacherType>> data_;

    //対応するpriorityを持ったセグメント木
    SegmentTree segment_tree_;

    //最初に待つ量
    int64_t first_wait_;

    //最大サイズ
    int64_t max_size_;

    //TD(λ)のパラメータ
    double lambda_;

    //排他制御用
    std::mutex mutex_;

    //更新に利用するため前回使用したindexらを保存しておく
    std::vector<uint64_t> pre_indices_;
};

#endif //MIACIS_REPLAY_BUFFER_HPP