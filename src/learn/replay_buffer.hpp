﻿#ifndef MIACIS_REPLAY_BUFFER_HPP
#define MIACIS_REPLAY_BUFFER_HPP
#include "../model/model_common.hpp"
#include "../shogi/game.hpp"
#include "learn.hpp"
#include "segment_tree.hpp"
#include <mutex>
#ifdef _MSC_VER
#include <direct.h>
#elif __GNUC__
#include <sys/stat.h>
#endif

class ReplayBuffer {
public:
    ReplayBuffer(int64_t first_wait, int64_t max_size, int64_t output_interval, float lambda, float alpha, bool data_augmentation)
        : data_(max_size), segment_tree_(max_size), first_wait_(first_wait), max_size_(max_size), total_num_(0),
          output_interval_(output_interval), lambda_(lambda), alpha_(alpha), data_augmentation_(data_augmentation) {
        //棋譜を保存するディレクトリの削除・作成
#ifdef _MSC_VER
        std::filesystem::remove_all(KIFU_SAVE_DIR);
        _mkdir(KIFU_SAVE_DIR.c_str());
#elif __GNUC__
        std::experimental::filesystem::remove_all(KIFU_SAVE_DIR);
        mkdir(KIFU_SAVE_DIR.c_str(), ACCESSPERMS);
#endif
    }

    //ミニバッチを作って返す関数
    std::vector<LearningData> makeBatch(int64_t batch_size);

    //データを入れる関数
    void push(Game& game);

    //既存の棋譜データからリプレイバッファを埋める関数
    void fillByKifu(const std::string& file_path, float rate_threshold);

    //ミニバッチを学習した結果を用いてpriorityを更新する関数
    void update(const std::vector<float>& loss);

    //checkGenSpeedで使うもの
    int64_t totalNum() const { return total_num_; }

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
    float lambda_;

    //priorityを累乗するパラメータ
    float alpha_;

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