#ifndef MIACIS_REPLAY_BUFFER_HPP
#define MIACIS_REPLAY_BUFFER_HPP
#include"neural_network.hpp"
#include"game.hpp"
#include"segment_tree.hpp"
#include<mutex>
#include<experimental/filesystem>
#ifdef _MSC_VER
#include<direct.h>
#elif __GNUC__
#include<sys/stat.h>
#endif

class ReplayBuffer{
public:
    ReplayBuffer(int64_t first_wait, int64_t max_size, double lambda, double alpha) : first_wait_(first_wait), max_size_(max_size),
    segment_tree_(max_size), data_(max_size), lambda_(lambda),  alpha_(alpha) {
        //棋譜を保存するディレクトリの削除
        std::experimental::filesystem::remove_all(save_dir);

        //棋譜を保存するディレクトリの作成
#ifdef _MSC_VER
        _mkdir(save_dir.c_str());
#elif __GNUC__
        mkdir(save_dir.c_str(), ACCESSPERMS);
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

    //priorityを累乗するパラメータ
    double alpha_;

    //排他制御用
    std::mutex mutex_;

    //更新に利用するため前回使用したindexらを保存しておく
    std::vector<uint64_t> pre_indices_;

    //学習に用いた棋譜を保存するディレクトリへのパス
    static const std::string save_dir;
};

#endif //MIACIS_REPLAY_BUFFER_HPP