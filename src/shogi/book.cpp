#include <climits>
#include "book.hpp"

void YaneBook::open(const std::string& file_name) {
    std::ifstream ifs(file_name);
    std::string input;
    std::string curr_sfen;
    while (std::getline(ifs, input)) {
        if (input[0] == '#') {
            continue;
        }
        if (input.substr(0, 4) == "sfen") {
            curr_sfen = input.substr(5);
            curr_sfen.pop_back(); //改行文字をなくす
            curr_sfen = curr_sfen.substr(0, curr_sfen.rfind(' ')); //最後の手数部分を消去する
        } else {
            int64_t start = 0;
            std::vector<std::string> vec;
            for (int64_t i = 0; i <= input.size(); i++) {
                if (i == input.size() || input[i] == ' ') {
                    vec.push_back(input.substr(start, i - start));
                    start = i;
                    continue;
                }
            }

            YaneBookEntry entry;
            entry.move = stringToMove(vec[0]);
            entry.counter_move = stringToMove(vec[1]);
            entry.score = std::stoll(vec[2]);
            entry.depth = std::stoll(vec[3]);
            entry.selected_num = std::stoll(vec[4]);
            book_[curr_sfen].push_back(entry);
        }
    }
}

Move YaneBook::pickOne(const Position& pos, float temperature) {
    std::string pos_str = pos.toStr();
    pos_str = pos_str.substr(0, pos_str.rfind(' '));

    int64_t best_score = LLONG_MIN;
    for (const YaneBookEntry& entry : book_[pos_str]) {
        best_score = std::max(best_score, entry.score);
    }

    std::vector<Move> best_score_moves;
    for (const YaneBookEntry& entry : book_[pos_str]) {
        best_score = std::max(best_score, entry.score);
        if (entry.score == best_score) {
            best_score_moves.push_back(entry.move);
        }
    }

    std::mt19937_64 engine(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist(0, best_score_moves.size() - 1);
    return best_score_moves[dist(engine)];
}

bool YaneBook::hasEntry(const Position& pos) {
    std::string pos_str = pos.toStr();
    pos_str = pos_str.substr(0, pos_str.rfind(' '));
    return !book_[pos_str].empty();
}