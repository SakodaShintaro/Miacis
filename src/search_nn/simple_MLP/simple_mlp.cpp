#include "simple_mlp.hpp"
#include "../../common.hpp"
#include "../common.hpp"

SimpleMLPImpl::SimpleMLPImpl(const SearchOptions& search_options) : BaseModel(search_options) {
    search_options_.search_limit = 0;
}

torch::Tensor SimpleMLPImpl::inferPolicy(const Position& pos) {
    std::vector<float> inputs = pos.makeFeature();
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    return forward(x);
}

torch::Tensor SimpleMLPImpl::forward(const torch::Tensor& x) {
    torch::Tensor y = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    y = encoder_->forward(y);
    y = y.view({ -1, StateEncoderImpl::HIDDEN_DIM });
    return base_policy_head_->forward(y);
}

void SimpleMLPImpl::save() {
    torch::save(encoder_, "encoder.model");
    torch::save(base_policy_head_, "policy_head.model");
}

std::vector<torch::Tensor> SimpleMLPImpl::search(std::vector<Position>& positions) {
    if (search_options_.search_limit != 0) {
        std::cerr << "SimpleMLPではsearch_options.search_limitは0でなければならない" << std::endl;
        std::exit(1);
    }
    torch::Tensor x = embed(positions);
    torch::Tensor y = base_policy_head_->forward(x);
    return { y };
}