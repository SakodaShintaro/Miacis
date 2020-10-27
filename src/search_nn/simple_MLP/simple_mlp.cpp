#include "simple_mlp.hpp"
#include "../../common.hpp"
#include "../common.hpp"

SimpleMLPImpl::SimpleMLPImpl(const SearchOptions& search_options) : BaseModel(search_options) {
    search_options_.search_limit = 0;
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