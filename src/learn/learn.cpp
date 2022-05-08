#include "learn.hpp"
#include "../shogi/game.hpp"
#include "../shogi/position.hpp"
#include <iomanip>
#include <random>
#include <sstream>

std::array<float, LOSS_TYPE_NUM> validation(InferModel& model, const std::vector<LearningData>& valid_data, uint64_t batch_size) {
    std::array<float, LOSS_TYPE_NUM> losses{};
    for (uint64_t index = 0; index < valid_data.size();) {
        std::vector<LearningData> curr_data;
        while (index < valid_data.size() && curr_data.size() < batch_size) {
            curr_data.push_back(valid_data[index++]);
        }

        std::array<std::vector<float>, LOSS_TYPE_NUM> loss = model.validLoss(curr_data);
        for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
            losses[i] += std::accumulate(loss[i].begin(), loss[i].end(), 0.0f);
        }
    }

    //データサイズで割って平均
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        losses[i] /= valid_data.size();
    }

    return losses;
}

std::array<float, LOSS_TYPE_NUM> validationWithSave(InferModel& model, const std::vector<LearningData>& valid_data,
                                                    uint64_t batch_size) {
    std::ofstream ofs("valid_loss.tsv");
    std::array<float, LOSS_TYPE_NUM> losses{};
    Position pos;
    for (uint64_t index = 0; index < valid_data.size();) {
        std::vector<LearningData> curr_data;
        while (index < valid_data.size() && curr_data.size() < batch_size) {
            curr_data.push_back(valid_data[index++]);
        }

        std::array<std::vector<float>, LOSS_TYPE_NUM> loss = model.validLoss(curr_data);

        //各データについて処理
        for (uint64_t i = 0; i < curr_data.size(); i++) {
            const LearningData& datum = curr_data[i];
            pos.fromStr(datum.position_str);
            const auto [label, prob] = datum.policy[0];
            Move move = pos.labelToMove(label);
            ofs << datum.position_str << "\t" << move.toPrettyStr() << "\t" << prob << "\t" << datum.value << "\t" << loss[0][i]
                << "\t" << loss[1][i] << std::endl;
        }

        for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
            losses[i] += std::accumulate(loss[i].begin(), loss[i].end(), 0.0f);
        }
    }

    //データサイズで割って平均
    for (int64_t i = 0; i < LOSS_TYPE_NUM; i++) {
        losses[i] /= valid_data.size();
    }

    return losses;
}

std::vector<LearningData> deleteDuplicate(std::vector<LearningData>& data_buffer) {
    std::sort(data_buffer.begin(), data_buffer.end(),
              [](LearningData& lhs, LearningData& rhs) { return lhs.position_str < rhs.position_str; });

    std::vector<LearningData> remain;
    remain.reserve(data_buffer.size());
    //統計情報
    int64_t redundant_num = 0;
    float value_sum = 0;
    std::map<int64_t, float> policy_map;

    for (uint64_t i = 0; i < data_buffer.size(); i++) {
        const LearningData& curr = data_buffer[i];
        redundant_num++;
#ifdef USE_CATEGORICAL
        value_sum += (MIN_SCORE + (curr.value + 0.5) * VALUE_WIDTH);
#else
        value_sum += curr.value;
#endif
        for (auto [label, prob] : curr.policy) {
            policy_map[label] += prob;
        }

        if (i == data_buffer.size() - 1 || curr.position_str != data_buffer[i + 1].position_str) {
            LearningData datum;
            datum.position_str = curr.position_str;
            float v = value_sum / redundant_num;
#ifdef USE_CATEGORICAL
            datum.value = valueToIndex(v);
#else
            datum.value = v;
#endif
            for (auto [label, prob_sum] : policy_map) {
                datum.policy.push_back({ label, prob_sum / redundant_num });
            }

            //            if (redundant_num > 100) {
            //                std::cout << "redundant_num = " << redundant_num << std::endl;
            //                Position pos;
            //                pos.fromStr(datum.position_str);
            //                pos.print();
            //                std::cout << "v = " << v << std::endl;
            //                for (auto [label, prob] : datum.policy) {
            //                    std::cout << label << " " << prob << std::endl;
            //                }
            //            }

            remain.push_back(datum);

            redundant_num = 0;
            value_sum = 0;
            policy_map.clear();
        }
    }

    return remain;
}

std::vector<LearningData> loadData(const std::string& file_path, bool data_augmentation, float rate_threshold) {
    //棋譜を読み込めるだけ読み込む
    std::vector<Game> games = loadGames(file_path, rate_threshold);

    //データを局面単位にバラす
    std::vector<LearningData> data_buffer;
    for (const Game& game : games) {
        Position pos;
        for (const OneTurnElement& e : game.elements) {
            const Move& move = e.move;
            uint32_t label = move.toLabel();
            std::string position_str = pos.toStr();
            for (int64_t i = 0; i < (data_augmentation ? Position::DATA_AUGMENTATION_PATTERN_NUM : 1); i++) {
                LearningData datum{};
                datum.policy.push_back({ Move::augmentLabel(label, i), 1.0 });
#ifdef USE_CATEGORICAL
                datum.value = valueToIndex((pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result));
#else
                datum.value = (float)(pos.color() == BLACK ? game.result : MAX_SCORE + MIN_SCORE - game.result);
#endif
                datum.position_str = Position::augmentStr(position_str, i);
                data_buffer.push_back(datum);
            }
            pos.doMove(move);
        }
    }

    //重複を削除して返す
    return deleteDuplicate(data_buffer);
}

// make move
Move make_move_label(const uint16_t move16, const Color color) {
    // xxxxxxxx x1111111  移動先
    // xx111111 1xxxxxxx  移動元。駒打ちの際には、PieceType + SquareNum - 1
    // x1xxxxxx xxxxxxxx  1 なら成り
    uint16_t to_sq = move16 & 0b1111111;
    uint16_t from_sq = (move16 >> 7) & 0b1111111;

    Square to = SquareList[to_sq];

    if (from_sq < SQUARE_NUM) {
        Square from = SquareList[from_sq];
        bool promote = (move16 & 0b100000000000000) > 0;
        return Move(to, from, false, promote);
    } else {
        // 持ち駒の場合
        const int hand_piece = from_sq - (uint16_t)SQUARE_NUM;
        Piece p = coloredPiece(color, DLShogiPieceKindList[hand_piece]);
        return dropMove(to, p);
    }
}

// make result
inline float make_result(const uint8_t result, const Color color) {
    const GameResult gameResult = (GameResult)(result & 0x3);
    if (gameResult == Draw) return (MAX_SCORE + MIN_SCORE) / 2;

    if ((color == BLACK && gameResult == BlackWin) || (color == WHITE && gameResult == WhiteWin)) {
        return MAX_SCORE;
    } else {
        return MIN_SCORE;
    }
}

std::vector<LearningData> __hcpe_decode_with_value(const size_t len, char* ndhcpe, bool data_augmentation) {
    HuffmanCodedPosAndEval* hcpe = reinterpret_cast<HuffmanCodedPosAndEval*>(ndhcpe);

    std::vector<LearningData> data_buffer;

    Position pos;
    for (size_t i = 0; i < len; i++, hcpe++) {
        pos.fromHCP(hcpe->hcp);
        std::string position_str = pos.toStr();

        Move move = make_move_label(hcpe->bestMove16, pos.color());
        move = pos.transformValidMove(move);
        uint32_t label = move.toLabel();

        float score = 1.0f / (1.0f + expf(-(float)hcpe->eval * 0.0013226f)) * (MAX_SCORE - MIN_SCORE) + MIN_SCORE;
        float result = make_result(hcpe->gameResult, pos.color());
        float target_value = result;

        for (int64_t i = 0; i < (data_augmentation ? Position::DATA_AUGMENTATION_PATTERN_NUM : 1); i++) {
            LearningData datum{};
            datum.policy.push_back({ Move::augmentLabel(label, i), 1.0 });
#ifdef USE_CATEGORICAL
            datum.value = valueToIndex(target_value);
#else
            datum.value = target_value;
#endif
            datum.position_str = Position::augmentStr(position_str, i);
            data_buffer.push_back(datum);
        }
    }

    return deleteDuplicate(data_buffer);
}

std::vector<LearningData> loadHCPE(const std::string& file_path, bool data_augmentation) {
    std::ifstream hcpe_file(file_path, std::ios::binary);
    hcpe_file.seekg(0, std::ios_base::end);
    const size_t file_size = hcpe_file.tellg();
    hcpe_file.seekg(0, std::ios_base::beg);
    std::unique_ptr<char[]> blob(new char[file_size]);
    hcpe_file.read(blob.get(), file_size);
    const size_t len = file_size / sizeof(HuffmanCodedPosAndEval);
    return __hcpe_decode_with_value(len, blob.get(), data_augmentation);
}
