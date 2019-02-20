#include "hyperparameter_manager.hpp"
#include <fstream>
#include <iostream>
#include <cassert>

const std::string BAD_INIT = "bad_init";

void HyperparameterManager::add(const std::string& name, int64_t lower_limit, int64_t upper_limit) {
    //最初は条件を満たさないように初期化
    int_map[name][VALUE] = lower_limit - 1;
    int_map[name][UPPER_LIMIT] = upper_limit;
    int_map[name][LOWER_LIMIT] = lower_limit;
}

void HyperparameterManager::add(const std::string& name, float lower_limit, float upper_limit) {
    float_map[name][VALUE] = lower_limit - 1.0f;
    float_map[name][UPPER_LIMIT] = upper_limit;
    float_map[name][LOWER_LIMIT] = lower_limit;
}

void HyperparameterManager::add(const std::string& name) {
    string_map[name] = BAD_INIT;
}

void HyperparameterManager::load(const std::string& file_path) {
    //オプションをファイルから読み込む
    std::ifstream ifs(file_path);
    if (!ifs) {
        std::cerr << "fail to open alphazero_settings.txt" << std::endl;
        assert(false);
    }
    std::string name;
    while (ifs >> name) {
        if (int_map.count(name)) {
            ifs >> int_map[name][VALUE];
        } else if (float_map.count(name)) {
            ifs >> float_map[name][VALUE];
        } else if (string_map.count(name)) {
            ifs >> string_map[name];
        } else {
            std::cerr << "There is not such parameter : " << name << std::endl;
            assert(false);
        }
    }
}

bool HyperparameterManager::check() {
    for (const auto& p : int_map) {
        if (!(p.second[LOWER_LIMIT] <= p.second[VALUE] && p.second[VALUE] <= p.second[UPPER_LIMIT])) {
            std::cerr << "Invalid value : " << p.first << ", " << p.second[VALUE] << std::endl;
            return false;
        }
    }
    for (const auto& p : float_map) {
        if (!(p.second[LOWER_LIMIT] <= p.second[VALUE] && p.second[VALUE] <= p.second[UPPER_LIMIT])) {
            std::cerr << "Invalid value : " << p.first << ", " << p.second[VALUE] << std::endl;
            return false;
        }
    }
    for (const auto& p : string_map) {
        if (p.second == BAD_INIT) {
            std::cerr << "Invalid value : " << p.first << ", " << p.second << std::endl;
            return false;
        }
    }
    return true;
}

template <>
int64_t HyperparameterManager::get(const std::string& name){
    if (int_map.count(name)) {
        return int_map[name][VALUE];
    } else {
        std::cerr << "There is not such key : " << name << std::endl;
        return -1;
    }
}

template <>
float HyperparameterManager::get(const std::string& name){
    if (float_map.count(name)) {
        return float_map[name][VALUE];
    } else {
        std::cerr << "There is not such key : " << name << std::endl;
        return -1;
    }
}

template <>
std::string HyperparameterManager::get(const std::string& name){
    if (string_map.count(name)) {
        return string_map[name];
    } else {
        std::cerr << "There is not such key : " << name << std::endl;
        return "";
    }
}