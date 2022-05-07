#ifndef MIACIS_SHOGI_BOOK_HPP
#define MIACIS_SHOGI_BOOK_HPP

#include "position.hpp"

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
    bool hasEntry(const Position& pos);
    BookEntry& getEntry(const Position& pos);
    Move pickOne(const Position& pos, float temperature);

private:
    static std::string removeTurnNumber(const std::string& sfen);
    std::unordered_map<std::string, BookEntry> book_;
};

#endif //MIACIS_SHOGI_BOOK_HPP