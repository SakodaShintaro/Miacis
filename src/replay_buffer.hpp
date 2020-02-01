#ifndef MIACIS_REPLAY_BUFFER_HPP
#define MIACIS_REPLAY_BUFFER_HPP
#include"neural_network.hpp"
#include"game.hpp"
#include"segment_tree.hpp"
#include"learn.hpp"
#include<mutex>
#include<experimental/filesystem>
#ifdef _MSC_VER
#include<direct.h>
#elif __GNUC__
#include<sys/stat.h>
#endif

class ReplayBuffer{
public:
    ReplayBuffer(int64_t first_wait, int64_t max_size, int64_t output_interval, double lambda, double alpha, bool data_augmentation) :
        data_(max_size), segment_tree_(max_size), first_wait_(first_wait), max_size_(max_size), total_num_(0),
        output_interval_(output_interval), lambda_(lambda),  alpha_(alpha), data_augmentation_(data_augmentation) {
        //棋譜を保存するディレクトリの削除
        std::experimental::filesystem::remove_all(KIFU_SAVE_DIR);

        //棋譜を保存するディレクトリの作成
#ifdef _MSC_VER
        _mkdir(KIFU_SAVE_DIR.c_str());
#elif __GNUC__
        mkdir(KIFU_SAVE_DIR.c_str(), ACCESSPERMS);
#endif
    }

    //ミニバッチを作って返す関数
    std::vector<LearningData> makeBatch(int64_t batch_size);

    //データを入れる関数
    void push(Game& game);

    //ミニバッチを学習した結果を用いてpriorityを更新する関数
    void update(const std::vector<float>& loss);

    //checkGenSpeedで使うもの
    int64_t totalNum() { return total_num_; }

private:
    //実際のデータ
    std::vector<LearningData> data_;

    //対応するpriorityを持ったセグメント木
    SegmentTree segment_tree_;

    //最初に待つ量
    int64_t first_wait_;

    //最大サイズ
    int64_t max_size_;

    //今までに追加されたデータの総数
    int64_t total_num_;

    //棋譜に書き出す間隔
    int64_t output_interval_;

    //TD(λ)のパラメータ
    double lambda_;

    //priorityを累乗するパラメータ
    double alpha_;

    //データ拡張をするかどうか
    bool data_augmentation_;

    //排他制御用
    std::mutex mutex_;

    //更新に利用するため前回使用したindexらを保存しておく
    std::vector<uint64_t> pre_indices_;

    //学習に用いた棋譜を保存するディレクトリへのパス
    static const std::string KIFU_SAVE_DIR;
};

#endif //MIACIS_REPLAY_BUFFER_HPP