#include "random_game_generator.hpp"
#include <thread>
#include <trtorch/trtorch.h>

void RandomGameGenerator::start() {
    //停止信号が来るまで生成し続けるループ
    while (!stop_signal) {
        step();
    }
}

void RandomGameGenerator::step() {
    //ランダムに行動を選択して次の局面へ遷移
    static std::mt19937_64 engine(std::random_device{}());
    std::vector<Move> moves = position_.generateAllMoves();
    std::uniform_int_distribution<int64_t> dist(0, moves.size() - 1);
    int64_t index = dist(engine);

    //教師データを作成
    OneTurnElement element;

    //valueのセット
    element.score = (MIN_SCORE + MAX_SCORE) / 2;

    //policyのセット
    //行動を選択
    element.policy_teacher.push_back({ moves[index].toLabel(), 1.0f });
    element.move = moves[index];

    //NNの出力を適当にセット
    element.nn_output_policy.resize(POLICY_DIM, 1.0);
    element.nn_output_value = ValueType{};

    //動かす
    position_.doMove(element.move);
    game_.elements.push_back(element);

    float score{};
    if (position_.isFinish(score, false) || position_.turnNumber() > search_options_.draw_turn) {
        //決着したので最終結果を設定
        game_.result = (position_.color() == BLACK ? score : MAX_SCORE + MIN_SCORE - score);

        //データを送る
        replay_buffer_.push(game_);

        //次の対局へ向かう
        position_.init();
        game_.elements.clear();
    }
}