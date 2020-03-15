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

struct BookEntry {
    std::vector<Move> moves;
    std::vector<float> policies;
    std::vector<float> values;
    std::vector<int64_t> select_num;
};

class Book {
public:
    void open(const std::string& file_name);
    void write(const std::string& file_name);
    void updateOne(int64_t think_sec);
    bool hasEntry(const Position& pos);
    Move pickOne(const Position& pos, float temperature);
private:
    static std::string removeTurnNumber(const std::string& sfen);
    std::unordered_map<std::string, BookEntry> book_;
};

#endif //MIACIS_BOOK_HPP