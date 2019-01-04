#pragma once
#ifndef PV_TABLE_HPP
#define PV_TABLE_HPP

#include"types.hpp"
#include"move.hpp"
#include<climits>

class PVTable {
private:
    Move pv_[DEPTH_MAX / PLY + 1][DEPTH_MAX / PLY + 1];
    int pv_length_[DEPTH_MAX / PLY + 1];

public:
    PVTable() {
        pv_length_[0] = 0;
    }

    Move operator[](int ply) const {
        return pv_[0][ply];
    }

    size_t size() const {
        return (size_t)pv_length_[0];
    }

    const Move* begin() const {
        return &pv_[0][0];
    }

    const Move* end() const {
        return begin() + size();
    }

    void closePV(int ply) {
        pv_length_[ply] = ply;
    }
    
    void clear() {
        for (int i = 0; i <= DEPTH_MAX; ++i) {
            pv_length_[i] = INT_MAX;
            for (int j = 0; j <= DEPTH_MAX; ++j) {
                pv_[i][j] = NULL_MOVE;
            }
        }
    }

    void reset() {
        //論理的なクリア
        pv_length_[0] = 0;
    }

    void update(const Move& move, int distance_from_root) {
        assert(0 <= distance_from_root && distance_from_root <= DEPTH_MAX);

        int length = pv_length_[distance_from_root + 1];

        assert(0 <= length && length <= DEPTH_MAX);

        pv_[distance_from_root][distance_from_root] = move;
        pv_length_[distance_from_root] = length;

        for (int i = distance_from_root + 1; i < length; ++i) {
            pv_[distance_from_root][i] = pv_[distance_from_root + 1][i];
        }
    }
};

#endif // !PV_TABLE_HPP