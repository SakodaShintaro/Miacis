#include "proposed_model_transformer.hpp"
#include "../../common.hpp"
#include "../common.hpp"

ProposedModelTransformerImpl::ProposedModelTransformerImpl(const SearchOptions& search_options) : BaseModel(search_options) {
    torch::nn::TransformerOptions options(StateEncoderImpl::HIDDEN_DIM, 4, 3, 3);
    transformer_ = register_module("transformer_", torch::nn::Transformer(options));
    policy_head_ = register_module("policy_head_", torch::nn::Linear(StateEncoderImpl::HIDDEN_DIM, POLICY_DIM));
}

Move ProposedModelTransformerImpl::think(Position& root, int64_t time_limit) {
    //バッチ化している関数と共通化している都合上、盤面をvector化
    std::vector<Position> positions;
    positions.push_back(root);

    //出力方策の系列を取得
    std::vector<torch::Tensor> policy_logits = search(positions);

    //合法手だけマスクをかける
    std::vector<Move> moves = root.generateAllMoves();
    std::vector<float> logits;
    for (const Move& move : moves) {
        logits.push_back(policy_logits.back()[0][0][move.toLabel()].item<float>());
    }

    if (root.turnNumber() <= search_options_.random_turn) {
        //Softmaxの確率に従って選択
        std::vector<float> masked_policy = softmax(logits, 1.0f);
        int32_t move_id = randomChoose(masked_policy);
        return moves[move_id];
    } else {
        //最大のlogitを持つ行動を選択
        int32_t move_id = std::max_element(logits.begin(), logits.end()) - logits.begin();
        return moves[move_id];
    }
}

std::vector<torch::Tensor> ProposedModelTransformerImpl::search(std::vector<Position>& positions) {
    //探索をして出力方策の系列を得る
    std::vector<torch::Tensor> policy_logits;

    //バッチサイズを取得しておく
    const int64_t batch_size = positions.size();

    //何回探索したら戻すか
    constexpr int64_t RESET_NUM = 3;

    std::vector<torch::Tensor> history;

    //ルート局面の特徴量
    torch::Tensor root_x = embed(positions);

    for (int64_t m = 0; m <= search_options_.search_limit; m++) {
        //盤面を戻す
        if (m != 0 && m % RESET_NUM == 0) {
            for (int64_t i = 0; i < batch_size; i++) {
                for (int64_t _ = 0; _ < RESET_NUM; _++) {
                    positions[i].undo();
                }
            }
        }

        //現局面の特徴を抽出
        torch::Tensor x = embed(positions);

        //ここまでの探索から最終行動決定
        policy_logits.push_back(inferPolicy(root_x, history));

        if (m == search_options_.search_limit) {
            break;
        }

        //探索行動を決定
        torch::Tensor sim_policy_logit = sim_policy_head_->forward(x);

        //行動をサンプリングして盤面を動かす
        for (int64_t i = 0; i < batch_size; i++) {
            std::vector<Move> moves = positions[i].generateAllMoves();
            std::vector<float> logits(moves.size());
            for (uint64_t j = 0; j < moves.size(); j++) {
                logits[j] = sim_policy_logit[0][i][moves[j].toLabel()].item<float>();
            }
            std::vector<float> softmaxed = softmax(logits, 1.0f);
            int64_t move_index = randomChoose(softmaxed);
            positions[i].doMove(moves[move_index]);
        }

        //positional encodingを取得
        torch::Tensor pe = positionalEncoding(m);

        //現在の特徴量を追加
        history.push_back(x + pe);
    }

    for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t _ = 0; _ < search_options_.search_limit % RESET_NUM; _++) {
            positions[i].undo();
        }
    }

    return policy_logits;
}

torch::Tensor ProposedModelTransformerImpl::inferPolicy(const torch::Tensor& x, const std::vector<torch::Tensor>& history) {
    //xをキーとして推論
    torch::Tensor src =
        (history.empty() ? torch::zeros({ 1, x.size(1), StateEncoderImpl::HIDDEN_DIM }) : torch::cat(history, 0)).to(device_);
    torch::Tensor y = transformer_->forward(src, x);
    y = policy_head_->forward(y);
    return y;
}

torch::Tensor ProposedModelTransformerImpl::positionalEncoding(int64_t pos) const {
    //参考1) https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    //参考2) https://qiita.com/omiita/items/07e69aef6c156d23c538
    //shape (StateEncoderImpl::HIDDEN_DIM)のものを返す
    torch::Tensor pe = torch::zeros({ StateEncoderImpl::HIDDEN_DIM });
    for (int64_t i = 0; i < StateEncoderImpl::HIDDEN_DIM; i++) {
        float exponent = static_cast<float>(i / 2 * 2) / StateEncoderImpl::HIDDEN_DIM;
        float div = std::pow(10000, exponent);
        pe[i] = (i % 2 == 0 ? std::sin(pos / div) : std::cos(pos / div));
    }
    return pe.to(device_);
}