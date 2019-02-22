#ifndef MIACIS_REPLAY_BUFFER_HPP
#define MIACIS_REPLAY_BUFFER_HPP
#include"neural_network.hpp"
#include"game.hpp"
#include<mutex>
#include<queue>

class ReplayBuffer{
public:
    ReplayBuffer(int64_t first_wait, int64_t max_size, float lambda) : first_wait_(first_wait), max_size_(max_size),
    lambda_(lambda), segment_tree_(max_size), priority_time_bonus_(0.0), data_(max_size) {
        for (int64_t i = 0; i < max_size; i++) {
            priority_queue_.push(Element(0.0, i));
        }
    }

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
    std::vector<std::tuple<std::string, TeacherType>> data_;

    //対応するpriorityを持ったセグメント木
    //1点更新,区間和
    class SegmentTree {
    public:
        explicit SegmentTree(uint64_t n) {
            n_ = 1ull << MSB64(n);
            nodes_.resize(2 * n_ - 1);
        }

        void update(uint64_t x, float v) {
            nodes_[x + n_ - 1] = v;
            for (uint64_t i = (x + n_ - 2) / 2; ; i = (i - 1) / 2) {
                nodes_[i] = nodes_[2 * i + 1] + nodes_[2 * i + 2];
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
            return (value <= nodes_[k] ? getIndex(value, 2 * k + 1) : getIndex(value, 2 * k + 2));
        }

        float getSum() {
            return nodes_.front();
        }

    private:
        //2のべき乗
        uint64_t n_;
        std::vector<float> nodes_;
    };
    SegmentTree segment_tree_;

    float priority_time_bonus_;

    struct Element {
        Element(float p, int32_t i) : priority(p), index(i) {}
        float priority;
        int32_t index;
        bool operator<(const Element& rhs) const {
            return priority > rhs.priority;
        }
    };
    std::priority_queue<Element> priority_queue_;

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