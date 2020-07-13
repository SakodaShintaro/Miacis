#include "mcts_net.hpp"
#include <stack>

MCTSNet::MCTSNet(const SearchOptions& search_options) : search_options_(search_options),
                                                        hash_table_(search_options.USI_Hash * 1024 * 1024 / 10000) {}

Move MCTSNet::think(Position& root, int64_t time_limit) {
    //思考を行う
    //時間制限、あるいはノード数制限に基づいて何回やるかを決める
    //基本的にはノード数制限のみを考えるイメージで最初は
    for (int64_t _ = 0; _ < search_options_.search_limit; _++) {
        //1回探索する

        //到達したノードの履歴
        std::stack<Index> indices;

        //(1)選択
        //特徴量作成
        //Policy Networkに入力し、方策を推論
        //行動をサンプリング

        //(2)評価
        //Embed Networkに入力し、埋め込みを取得
        //置換表に登録

        //(3)バックアップ
        while (!indices.empty()) {
            Index top = indices.top();
            indices.pop();

            //Backup Networkにより更新(差分更新)
        }
    }

    //最終的な行動決定
    //Readout Networkにより最終決定

    //SoftMaxでの最大を行動選択
}