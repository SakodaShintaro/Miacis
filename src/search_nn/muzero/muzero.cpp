#include "muzero.hpp"
#include "../../common.hpp"

//ネットワークの設定
static constexpr int64_t HIDDEN_DIM = BOARD_WIDTH * BOARD_WIDTH * StateEncoderImpl::LAST_CHANNEL_NUM;
static constexpr int64_t ACTION_FEATURE_CHANNEL_NUM = 10;

const std::string MuZeroImpl::MODEL_PREFIX = "muzero";
const std::string MuZeroImpl::DEFAULT_MODEL_NAME = MuZeroImpl::MODEL_PREFIX + ".model";

MuZeroImpl::MuZeroImpl(SearchOptions search_options)
    : search_options_(std::move(search_options)), device_(torch::kCUDA), fp16_(false) {
    encoder_ = register_module("encoder_", StateEncoder());
    policy_ = register_module("policy_", torch::nn::Linear(HIDDEN_DIM, POLICY_DIM));
    value_ = register_module("value_", torch::nn::Linear(HIDDEN_DIM, 1));
    env_model_ = register_module("env_model_", StateEncoder(StateEncoderImpl::LAST_CHANNEL_NUM + ACTION_FEATURE_CHANNEL_NUM));
}

Move MuZeroImpl::think(Position& root, int64_t time_limit) {
    std::cout << "まだ未実装" << std::endl;
    std::exit(1);

    //この局面を推論して探索せずにそのまま出力

    //ニューラルネットワークによる推論
    torch::Tensor policy = inferPolicy(root);

    //合法手だけマスクをかける
    std::vector<Move> moves = root.generateAllMoves();
    std::vector<float> logits;
    for (const Move& move : moves) {
        logits.push_back(policy[0][move.toLabel()].item<float>());
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

std::vector<torch::Tensor> MuZeroImpl::loss(const std::vector<LearningData>& data) {
    //現状バッチサイズは1のみに対応
    assert(data.size() == 1);

    std::cout << "まだ未実装" << std::endl;
    std::exit(1);

    //局面を構築して推論
    Position root;
    root.fromStr(data.front().position_str);
    torch::Tensor policy_logit = inferPolicy(root);

    //policyの教師信号
    std::vector<float> policy_teachers(POLICY_DIM, 0.0);
    for (const std::pair<int32_t, float>& e : data.front().policy) {
        policy_teachers[e.first] = e.second;
    }

    torch::Tensor policy_teacher = torch::tensor(policy_teachers).to(device_).view({ -1, POLICY_DIM });

    //損失を計算
    std::vector<torch::Tensor> loss(1);
    torch::Tensor log_softmax = torch::log_softmax(policy_logit, 1);
    torch::Tensor clipped = torch::clamp_min(log_softmax, -20);
    loss[0] = (-policy_teacher * clipped).sum().view({ 1 });

    return loss;
}

torch::Tensor MuZeroImpl::encodeAction(Move move) {
#ifdef SHOGI
    static const ArrayMap<int32_t, PieceNum> PieceToNum({
        { BLACK_PAWN, 0 },
        { BLACK_LANCE, 1 },
        { BLACK_KNIGHT, 2 },
        { BLACK_SILVER, 3 },
        { BLACK_GOLD, 4 },
        { BLACK_BISHOP, 5 },
        { BLACK_ROOK, 6 },
        { BLACK_KING, 7 },
        { BLACK_PAWN_PROMOTE, 8 },
        { BLACK_LANCE_PROMOTE, 9 },
        { BLACK_KNIGHT_PROMOTE, 10 },
        { BLACK_SILVER_PROMOTE, 11 },
        { BLACK_BISHOP_PROMOTE, 12 },
        { BLACK_ROOK_PROMOTE, 13 },
    });

    //各moveにつき9×9×MOVE_FEATURE_CHANNEL_NUMの特徴マップを得る
    std::vector<float> move_features(9 * 9 * ACTION_FEATURE_CHANNEL_NUM, 0.0);

    //この行動の手番
    Color color = pieceToColor(move.subject());

    //1ch:toの位置に1を立てる
    Square to = (color == BLACK ? move.to() : InvSquare[move.to()]);
    move_features[SquareToNum[to]] = 1;

    //2ch:fromの位置に1を立てる.持ち駒から打つ手ならなし
    //3ch:持ち駒から打つ手なら全て1
    if (move.isDrop()) {
        for (Square sq : SquareList) {
            move_features[2 * SQUARE_NUM + SquareToNum[sq]] = 1;
        }
    } else {
        Square from = (color == BLACK ? move.from() : InvSquare[move.from()]);
        move_features[SQUARE_NUM + SquareToNum[from]] = 1;
    }

    //4ch:成りなら全て1
    if (move.isPromote()) {
        for (Square sq : SquareList) {
            move_features[3 * SQUARE_NUM + SquareToNum[sq]] = 1;
        }
    }

    //5ch以降:駒の種類に合わせたところだけ全て1
    for (Square sq : SquareList) {
        Piece p = (color == BLACK ? move.subject() : oppositeColor(move.subject()));
        move_features[(4 + PieceToNum[p]) * SQUARE_NUM + SquareToNum[sq]] = 1;
    }

    torch::Tensor move_features_tensor =
        (fp16_ ? torch::tensor(move_features).to(device_, torch::kHalf) : torch::tensor(move_features).to(device_));
    move_features_tensor = move_features_tensor.view({ -1, ACTION_FEATURE_CHANNEL_NUM, 9, 9 });
    return move_features_tensor;
#elif OTHELLO
    return torch::zeros({ 1 });
#endif
}

torch::Tensor MuZeroImpl::embed(const std::vector<float>& inputs) {
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    x = encoder_->forward(x);
    return x;
}

torch::Tensor MuZeroImpl::inferPolicy(const Position& pos) {
    std::vector<FloatType> inputs = pos.makeFeature();
    torch::Tensor x = (fp16_ ? torch::tensor(inputs).to(device_, torch::kHalf) : torch::tensor(inputs).to(device_));
    x = x.view({ -1, INPUT_CHANNEL_NUM, BOARD_WIDTH, BOARD_WIDTH });
    x = encoder_->forward(x);
    x = x.view({ -1, HIDDEN_DIM });
    return policy_->forward(x);
}

torch::Tensor MuZeroImpl::predictTransition(const torch::Tensor& state_representations,
                                            const torch::Tensor& move_representations) {
    torch::Tensor x = torch::cat({ state_representations, move_representations }, 1);
    x = env_model_->forward(x);
#ifdef SHOGI
    //手番が変わることを考慮して180度ひっくり返す
    x = torch::rot90(x, 2, { 2, 3 });
#elif OTHELLO
#endif

    return x;
}

void MuZeroImpl::setGPU(int16_t gpu_id, bool fp16) {
    device_ = (torch::cuda::is_available() ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));
    fp16_ = fp16;
    (fp16_ ? to(device_, torch::kHalf) : to(device_, torch::kFloat));
}