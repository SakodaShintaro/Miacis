#ifndef MIACIS_SEARCHER_FOR_MATE_HPP
#define MIACIS_SEARCHER_FOR_MATE_HPP

#include "hash_table.hpp"
#include "search_options.hpp"

class SearcherForMate {
public:
    SearcherForMate(HashTable& hash_table, const SearchOptions& search_options);
    bool stop_signal;
    void mateSearch(Position pos, int32_t depth_limit);
private:
    bool search(Position& pos, int32_t depth);

    HashTable& hash_table_;
    const SearchOptions& search_options_;
};

#endif //MIACIS_SEARCHER_FOR_MATE_HPP