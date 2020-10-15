#include "transformer_model.hpp"
#include "../../common.hpp"
#include "../common.hpp"

//ネットワークの設定
static constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * StateEncoderImpl::LAST_CHANNEL_NUM;
static constexpr int32_t HIDDEN_SIZE = 512;
static constexpr int32_t NUM_LAYERS = 1;

const std::string TransformerModelImpl::MODEL_PREFIX = "transformer_model";
const std::string TransformerModelImpl::DEFAULT_MODEL_NAME = TransformerModelImpl::MODEL_PREFIX + ".model";

TransformerModelImpl::TransformerModelImpl(const SearchOptions& search_options)
    : search_options_(std::move(search_options)), device_(torch::kCUDA), fp16_(false), freeze_encoder_(true) {
    encoder_ = register_module("encoder_", StateEncoder());

    torch::nn::TransformerOptions options(HIDDEN_DIM, 4);
    transformer_ = register_module("transformer_", torch::nn::Transformer(options));
}

Move TransformerModelImpl::think(Position& root, int64_t time_limit) {
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

std::vector<torch::Tensor> TransformerModelImpl::search(std::vector<Position>& positions) {
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
        torch::Tensor sim_policy_logit = inferPolicy(x, history);

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

        //現在の特徴量を追加
        history.push_back(x);
    }

    for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t _ = 0; _ < search_options_.search_limit % RESET_NUM; _++) {
            positions[i].undo();
        }
    }

    return policy_logits;
}

void TransformerModelImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}

torch::Tensor TransformerModelImpl::embed(const std::vector<float>& inputs) {
    torch::Tensor x = encoder_->embed(inputs, device_, fp16_, freeze_encoder_);
    x = x.view({ 1, -1, HIDDEN_DIM });
    return x;
}

torch::Tensor TransformerModelImpl::embed(const std::vector<Position>& positions) {
    std::vector<float> features;
    for (const auto& position : positions) {
        std::vector<float> f = position.makeFeature();
        features.insert(features.end(), f.begin(), f.end());
    }
    return embed(features);
}

torch::Tensor TransformerModelImpl::inferPolicy(const torch::Tensor& x, const std::vector<torch::Tensor>& history) {
    //xをキーとして推論
    torch::Tensor memory = (history.empty() ? torch::zeros({ 1, 1, HIDDEN_DIM }) : torch::cat(history, 0)).to(device_);
    torch::Tensor policy = transformer_->forward(x, memory);
    return policy;
}