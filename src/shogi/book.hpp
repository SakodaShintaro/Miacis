#ifndef MIACIS_BOOK_HPP
#define MIACIS_BOOK_HPP

#include "position.hpp"

struct YaneBookEntry {
    Move move;
    Move counter_move;
    int64_t score;
    int64_t depth;
    int64_t selected_num;
};

class YaneBook {
public:
    void open(const std::string& file_name);
    bool hasEntry(const Position& pos);
    Move pickOne(const Position& pos, float temperature);
private:
    std::unordered_map<std::string, std::vector<YaneBookEntry>> book_;
};

#endif //MIACIS_BOOK_HPP