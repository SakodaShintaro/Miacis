#ifndef MIACIS_LEARN_HPP
#define MIACIS_LEARN_HPP

#include"neural_network.hpp"
#include<string>
#include<chrono>

//経過時間を取得する関数
std::string elapsedTime(const std::chrono::steady_clock::time_point& start);
double elapsedHours(const std::chrono::steady_clock::time_point& start);

//data_bufからミニバッチを取得する関数
std::tuple<std::vector<float>, std::vector<PolicyTeacherType>, std::vector<ValueTeacherType>>
getBatch(const std::vector<std::pair<std::string, TeacherType>>& data_buf, int64_t index, int64_t batch_size);

//validationを行う関数
std::array<float, 2> validation(const std::vector<std::pair<string, TeacherType>>& validation_data);

//棋譜からの教師あり学習
void supervisedLearn();

//AlphaZero式の強化学習
void alphaZero();

#endif //MIACIS_LEARN_HPP