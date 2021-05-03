#include "book.hpp"
#include "../searcher_for_play.hpp"
#include <climits>

namespace Shogi {

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
            curr_sfen.pop_back();                                  //改行文字をなくす
            curr_sfen = curr_sfen.substr(0, curr_sfen.rfind(' ')); //最後の手数部分を消去する
        } else {
            int64_t start = 0;
            std::vector<std::string> vec;
            for (uint64_t i = 0; i <= input.size(); i++) {
                if (i == input.size() || input[i] == ' ') {
                    vec.push_back(input.substr(start, i - start));
                    start = i;
                    continue;
                }
            }

            YaneBookEntry entry{};
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

    std::uniform_int_distribution<int64_t> dist(0, best_score_moves.size() - 1);
    return best_score_moves[dist(engine)];
}

bool YaneBook::hasEntry(const Position& pos) {
    std::string pos_str = pos.toStr();
    pos_str = pos_str.substr(0, pos_str.rfind(' '));
    return !book_[pos_str].empty();
}

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

void Book::updateOne(int64_t think_sec) {
    SearchOptions search_options;
    search_options.print_interval = think_sec * 2000;
    search_options.USI_Hash = 8096;
    search_options.use_book = false;

    Position pos;
    std::vector<Move> selected_moves;
    while (true) {
        std::string key_sfen = removeTurnNumber(pos.toStr());
        if (book_.count(key_sfen) == 0) {
            break;
        }

        const BookEntry& book_entry = book_[key_sfen];

        //選択回数の合計
        int64_t sum = std::accumulate(book_entry.select_num.begin(), book_entry.select_num.end(), (int64_t)0);

        //UCBを計算し、一番高い行動を選択
        int64_t max_index = -1;
        float max_value = -1;
        for (uint64_t i = 0; i < book_entry.moves.size(); i++) {
            float U = std::sqrt(sum + 1) / (book_entry.select_num[i] + 1);
            float ucb = search_options.Q_coeff_x1000 / 1000.0 * book_entry.values[i] +
                        search_options.C_PUCT_x1000 / 1000.0 * book_entry.policies[i] * U;
            if (ucb > max_value) {
                max_index = i;
                max_value = ucb;
            }
        }

        Move best_move = pos.transformValidMove(book_entry.moves[max_index]);
        selected_moves.push_back(best_move);
        pos.doMove(best_move);
    }

    //-------------
    //    展開部
    //-------------
    //この局面を探索する
    pos.print();
    SearcherForPlay searcher(search_options);
    searcher.think(pos, think_sec * 1000);

    //結果を取得
    const HashTable& searched = searcher.hashTable();
    const HashEntry& root_node = searched[searched.root_index];

    //展開
    BookEntry& book_entry = book_[removeTurnNumber(pos.toStr())];
    book_entry.moves = root_node.moves;
    book_entry.policies = root_node.nn_policy;
    book_entry.values.resize(book_entry.moves.size());
    for (uint64_t i = 0; i < book_entry.moves.size(); i++) {
        book_entry.values[i] = searched.expQfromNext(root_node, i);
    }
    book_entry.select_num.assign(book_entry.moves.size(), 1);
    float value = *max_element(book_entry.values.begin(), book_entry.values.end());

    //この局面を登録
    //backupする
    while (!selected_moves.empty()) {
        //局面を戻し、そこに相当するエントリを取得
        pos.undo();
        std::string sfen = removeTurnNumber(pos.toStr());
        BookEntry& curr_entry = book_[sfen];

        //価値を反転
        value = -value;

        //最終手を取得
        Move last_move = selected_moves.back();
        selected_moves.pop_back();

        //更新
        for (uint64_t i = 0; i < curr_entry.moves.size(); i++) {
            if (pos.transformValidMove(curr_entry.moves[i]) != last_move) {
                continue;
            }

            //この手の価値を更新
            float alpha = 1.0f / (++curr_entry.select_num[i]);
            curr_entry.values[i] += alpha * (value - curr_entry.values[i]);
            break;
        }
    }
}

bool Book::hasEntry(const Position& pos) {
    std::string pos_str = removeTurnNumber(pos.toStr());
    return book_.count(pos_str) > 0;
}

Move Book::pickOne(const Position& pos, float temperature) {
    //温度は0にならないようにする
    temperature = std::max(temperature, 0.00001f);

    //適切に取ってきて価値のソフトマックスをかけた値を考える
    const BookEntry& entry = book_[removeTurnNumber(pos.toStr())];
    std::vector<float> softmaxed = softmax(entry.values, temperature);

    std::cout << std::fixed;
    for (uint64_t i = 0; i < entry.moves.size(); i++) {
        std::cout << entry.moves[i] << " " << softmaxed[i] << std::endl;
    }
    return entry.moves[randomChoose(softmaxed)];
}

std::string Book::removeTurnNumber(const std::string& sfen) {
    return sfen.substr(0, sfen.rfind(' ')); //最後の手数部分を消去する
}

} // namespace Shogi