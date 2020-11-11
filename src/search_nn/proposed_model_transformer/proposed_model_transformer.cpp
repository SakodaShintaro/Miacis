#include "proposed_model_transformer.hpp"
#include "../../common.hpp"
#include "../common.hpp"

constexpr int64_t COMPRESSED_DIM = 512;
constexpr int64_t MAX_PE_LENGTH = 50;

ProposedModelTransformerImpl::ProposedModelTransformerImpl(const SearchOptions& search_options) : BaseModel(search_options) {
    first_compressor_ = register_module("first_compressor_", torch::nn::Linear(StateEncoderImpl::HIDDEN_DIM, COMPRESSED_DIM));
    torch::nn::TransformerOptions options(COMPRESSED_DIM, 4, 3, 3);
    transformer_ = register_module("transformer_", torch::nn::Transformer(options));
    policy_head_ = register_module("policy_head_", torch::nn::Linear(COMPRESSED_DIM, POLICY_DIM));
    value_head_ = register_module("value_head_", torch::nn::Linear(COMPRESSED_DIM, 1));
    positional_encoding_ = register_parameter(
        "positional_encoding_", torch::randn({ MAX_PE_LENGTH, COMPRESSED_DIM }, torch::TensorOptions().requires_grad(true)));
}

std::vector<std::tuple<torch::Tensor, torch::Tensor>> ProposedModelTransformerImpl::search(std::vector<Position>& positions) {
    //探索をして出力方策の系列を得る
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> policy_and_value;

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
        policy_and_value.push_back(infer(first_compressor_->forward(root_x), history));

        if (m == search_options_.search_limit) {
            break;
        }

        //探索行動を決定
        torch::Tensor sim_policy_logit = base_policy_head_->forward(x);

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
        history.push_back(first_compressor_->forward(x) + pe);
    }

    for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t _ = 0; _ < search_options_.search_limit % RESET_NUM; _++) {
            positions[i].undo();
        }
    }

    return policy_and_value;
}

std::tuple<torch::Tensor, torch::Tensor> ProposedModelTransformerImpl::infer(const torch::Tensor& x,
                                                                             const std::vector<torch::Tensor>& history) {
    //xをキーとして推論
    torch::Tensor src = (history.empty() ? torch::zeros({ 1, x.size(1), COMPRESSED_DIM }) : torch::cat(history, 0)).to(device_);
    torch::Tensor y = transformer_->forward(src, x);

    torch::Tensor policy_logit = policy_head_->forward(y);
    torch::Tensor value = torch::tanh(value_head_->forward(y));
    return std::make_tuple(policy_logit, value);
}

torch::Tensor ProposedModelTransformerImpl::positionalEncoding(int64_t pos) const {
    return positional_encoding_[pos];
    //参考1) https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    //参考2) https://qiita.com/omiita/items/07e69aef6c156d23c538
    //shape (COMPRESSED_DIM)のものを返す
    torch::Tensor pe = torch::zeros({ COMPRESSED_DIM });
    for (int64_t i = 0; i < COMPRESSED_DIM; i++) {
        float exponent = static_cast<float>(i / 2 * 2) / COMPRESSED_DIM;
        float div = std::pow(10000, exponent);
        pe[i] = (i % 2 == 0 ? std::sin(pos / div) : std::cos(pos / div));
    }
    return pe.to(device_);
}