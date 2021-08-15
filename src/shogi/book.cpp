#include "book.hpp"
#include "../search/searcher_for_play.hpp"
#include <climits>

namespace Shogi {

void Book::open(const std::string& file_name) {
    std::ifstream ifs(file_name);
    std::string input;
    std::string curr_sfen;
    while (std::getline(ifs, input)) {
        if (input[0] == '#') {
            continue;
        }
        if (input.substr(0, 4) == "sfen") {
            curr_sfen = removeTurnNumber(input.substr(5));
        } else {
            //指し手
            //move policy value select_num
            int64_t start = 0;
            std::vector<std::string> vec;
            for (uint64_t i = 0; i <= input.size(); i++) {
                if (i == input.size() || input[i] == ' ') {
                    vec.push_back(input.substr(start, i - start));
                    start = i;
                    continue;
                }
            }

            book_[curr_sfen].moves.push_back(stringToMove(vec[0]));
            book_[curr_sfen].policies.push_back(std::stof(vec[1]));
            book_[curr_sfen].values.push_back(std::stof(vec[2]));
            book_[curr_sfen].select_num.push_back(std::stoll(vec[3]));
        }
    }
}

void Book::write(const std::string& file_name) {
    std::ofstream ofs(file_name);
    ofs << std::fixed;
    for (const auto& p : book_) {
        ofs << "sfen " << p.first << " 0" << std::endl;
        for (uint64_t i = 0; i < p.second.moves.size(); i++) {
            ofs << p.second.moves[i] << " " << p.second.policies[i] << " " << p.second.values[i] << " " << p.second.select_num[i]
                << std::endl;
        }
    }
}

bool Book::hasEntry(const Position& pos) {
    const std::string pos_str = removeTurnNumber(pos.toStr());
    return book_.count(pos_str) > 0;
}

BookEntry& Book::getEntry(const Position& pos) {
    const std::string pos_str = removeTurnNumber(pos.toStr());
    return book_[pos_str];
}

Move Book::pickOne(const Position& pos, float temperature) {
    //温度は0にならないようにする
    temperature = std::max(temperature, 0.00001f);

    //適切に取ってきて価値のソフトマックスをかけた値を考える
    const BookEntry& entry = book_[removeTurnNumber(pos.toStr())];
    std::vector<float> softmaxed = softmax(entry.values, temperature);

    return entry.moves[randomChoose(softmaxed)];
}

std::string Book::removeTurnNumber(const std::string& sfen) {
    return sfen.substr(0, sfen.rfind(' ')); //最後の手数部分を消去する
}

} // namespace Shogi