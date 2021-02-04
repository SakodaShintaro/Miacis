#ifndef MIACIS_GAME_GENERATOR_HPP
#define MIACIS_GAME_GENERATOR_HPP

#include "game.hpp"
#include "replay_buffer.hpp"
#include "search_options.hpp"
#include "searcher.hpp"
#include "searcher_for_mate.hpp"
#include "infer_model.hpp"
#include <atomic>
#include <mutex>
#include <stack>
#include <utility>

//自己対局をしてデータを生成するクラス
//一つのGPUに対して割り当てられる
class GameGenerator {
public:
    GameGenerator(const SearchOptions& search_options, int64_t worker_num, float Q_dist_lambda, int64_t noise_mode,
                  float noise_epsilon, float noise_alpha, ReplayBuffer& rb, int64_t gpu_id)
        : stop_signal(false), search_options_(search_options), worker_num_(worker_num), Q_dist_lambda_(Q_dist_lambda),
          noise_mode_(noise_mode), noise_epsilon_(noise_epsilon), noise_alpha_(noise_alpha), replay_buffer_(rb),
          neural_network_(), gpu_id_(gpu_id), gpu_queues_(search_options_.thread_num_per_gpu) {
        assert(0 <= noise_mode_ && noise_mode_ < NOISE_MODE_SIZE);
    };

    //生成してリプレイバッファに送り続ける関数
    void genGames();

    //排他制御用のmutex。強化学習時に外部からアクセスできるようpublicに置いている
    std::mutex gpu_mutex;

    //評価パラメータの読み込みが必要かどうかのシグナル
    bool need_load;

    //停止信号。止めたいときは外部からこれをtrueにする
    bool stop_signal;

private:
    enum NoiseMode { DIRICHLET, ONEHOT, NOISE_MODE_SIZE };
    //ディリクレ分布に従ったものを返す関数
    static std::vector<float> dirichletDistribution(uint64_t k, float alpha);

    //onehotベクトルとしてのノイズを返す関数
    static std::vector<float> onehotNoise(uint64_t k);

    //gpu_queue_に溜まっている入力を処理する関数
    void evalWithGPU(int64_t thread_id);

    //各スレッドのエントリーポイント
    void genSlave(int64_t thread_id);

    //探索に関するオプション
    const SearchOptions& search_options_;

    //Workerを並列に走らせる数
    const int64_t worker_num_;

    //探索結果の分布として価値のsoftmax分布を混ぜる割合([0,1])
    //0で探索回数を正規化した分布(AlphaZeroと同じ), 1で完全に価値のsoftmax分布
    const float Q_dist_lambda_;

    //ノイズモード
    const int64_t noise_mode_;

    //ノイズを混合する割合
    const float noise_epsilon_;

    //ディリクレノイズ時のalpha
    const float noise_alpha_;

    //データを送るReplayBufferへの参照
    ReplayBuffer& replay_buffer_;

    //局面評価に用いるネットワーク
    InferModel neural_network_;

    //CUDAがスレッドごとに紐付くのでgpu_id_を明に保持する必要がある
    int64_t gpu_id_;

    //評価要求を受け付けるQueue
    std::vector<GPUQueue> gpu_queues_;
};

//一つのGPUに対して複数生成されるWorker
class GenerateWorker {
public:
    GenerateWorker(const SearchOptions& search_options, GPUQueue& gpu_queue, float Q_dist_lambda, ReplayBuffer& rb);

    //現在のposition_に対して探索する準備を行う関数。ルート局面の展開や数手の詰み探索など
    void prepareForCurrPos();

    //1回選択してGPUに評価要求を溜める関数
    void select();

    //GPUによって計算された結果をbackupする関数
    void backup();

    //現在の局面を規定回数探索し終わった後、結果を取り出す関数
    OneTurnElement resultForCurrPos();

private:
    //探索回数などを見て探索を続けるかどうかを判定する関数
    bool shouldStop();

    //探索に関するオプション
    const SearchOptions& search_options_;

    //評価要求を投げる先
    GPUQueue& gpu_queue_;

    //探索結果の分布として価値のsoftmax分布を混ぜる割合([0,1])
    //0で探索回数を正規化した分布(AlphaZeroと同じ), 1で完全に価値のsoftmax分布
    const float Q_dist_lambda_;

    //データを送るReplayBufferへの参照
    ReplayBuffer& replay_buffer_;

    //管理している対局データ
    Game game_;

    //現在思考している局面
    Position position_;

    //置換表
    HashTable hash_table_;

    //探索クラス
    Searcher searcher_;

    //漸進的に更新されてしまうのでルート局面の生のValue出力を保存しておく
    //ルートノードのValueは更新する意味がないのでそのように変更すれば保存しておく必要もないのだが
    ValueType root_raw_value_;

    //詰み探索エージェント
    SearcherForMate mate_searcher_;
};

#endif //MIACIS_GAME_GENERATOR_HPP