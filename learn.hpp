#ifndef MIACIS_LEARN_HPP
#define MIACIS_LEARN_HPP

#include"neural_network.hpp"
#include<string>
#include<chrono>

//経過時間を取得する関数
std::string elapsedTime(const std::chrono::steady_clock::time_point& start);
double elapsedHours(const std::chrono::steady_clock::time_point& start);

//標準出力とファイルストリームに同時に出力するためのクラス
//参考)https://aki-yam.hatenablog.com/entry/20080630/1214801872
class dout {
private:
    std::ostream &os1, &os2;
public:
    explicit dout (std::ostream &_os1, std::ostream &_os2) : os1(_os1), os2(_os2) {};
    template <typename T>
    dout& operator<< (const T &rhs)  { os1 << rhs;  os2 << rhs; return *this; };
    dout& operator<< (std::ostream& (*__pf)(std::ostream&))  { __pf(os1); __pf(os2); return *this; };
    /*!<  Interface for manipulators, such as \c std::endl and \c std::setw
      For more information, see ostream header */
};

//教師データを読み込む関数
std::vector<std::pair<std::string, TeacherType>> loadData(const std::string& file_path);

//validationを行う関数
#ifdef USE_CATEGORICAL
std::array<float, 3> validation(const std::vector<std::pair<std::string, TeacherType>>& validation_data);
#else
std::array<float, 2> validation(const std::vector<std::pair<std::string, TeacherType>>& validation_data);
#endif

//棋譜からの教師あり学習
void supervisedLearn();

//AlphaZero式の強化学習
void alphaZero();

#endif //MIACIS_LEARN_HPP