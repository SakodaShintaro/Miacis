#ifndef MIACIS_REPLAY_BUFFER_HPP
#define MIACIS_REPLAY_BUFFER_HPP
#include"neural_network.hpp"
#include"game.hpp"
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
    lambda_(lambda), segment_tree_(max_size), priority_time_bonus_(0.0), data_(max_size) {
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
    //1点更新,区間和
    class SegmentTree {
    public:
        explicit SegmentTree(uint64_t n) {
            n_ = 1ull << MSB64(n);
            sum_.resize(2 * n_ - 1);
            min_.resize(2 * n_ - 1);
        }

        void update(uint64_t x, float v) {
            sum_[x + n_ - 1] = v;
            min_[x + n_ - 1] = v;
            for (uint64_t i = (x + n_ - 2) / 2; ; i = (i - 1) / 2) {
                sum_[i] = sum_[2 * i + 1] + sum_[2 * i + 2];
                min_[i] = std::min(min_[2 * i + 1], min_[2 * i + 2]);
                if (i == 0) {
                    break;
                }
            }
        }

        uint64_t getIndex(float value, uint64_t k = 0) {
            if (k >= n_ - 1) {
                //最下段まで来ていたらindexを返す
                return k - (n_ - 1);
            }
            return (value <= sum_[k] ? getIndex(value, 2 * k + 1) : getIndex(value - sum_[k], 2 * k + 2));
        }

        uint64_t getMinIndex(uint64_t k = 0) {
            if (k >= n_ - 1) {
                //最下段まで来ていたらindexを返す
                return k - (n_ - 1);
            }
            return (min_[2 * k + 1] <= min_[2 * k + 2] ? getMinIndex(2 * k + 1) : getMinIndex(2 * k + 2));
        }

        float getSum() {
            return sum_.front();
        }

        float operator[](uint64_t i) {
            return sum_[i + n_ - 1];
        }

        void print() {
            for (uint64_t i = 0; i < n_; i++) {
                std::cout << sum_[i + n_ - 1] << " \n"[i == n_ - 1];
            }
        }

    private:
        //2のべき乗
        uint64_t n_;
        std::vector<float> sum_, min_;
    };
    SegmentTree segment_tree_;

    float priority_time_bonus_;

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