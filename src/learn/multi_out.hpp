#ifndef MULTI_OUT_HPP
#define MULTI_OUT_HPP

#include <fstream>

//標準出力とファイルストリームに同時に出力するためのクラス
//参考)https://aki-yam.hatenablog.com/entry/20080630/1214801872
class dout {
private:
    std::ostream &os1, &os2;

public:
    explicit dout(std::ostream& _os1, std::ostream& _os2) : os1(_os1), os2(_os2){};
    template<typename T> dout& operator<<(const T& rhs) {
        os1 << rhs;
        os2 << rhs;
        return *this;
    };
    dout& operator<<(std::ostream& (*__pf)(std::ostream&)) {
        __pf(os1);
        __pf(os2);
        return *this;
    };
    /*!<  Interface for manipulators, such as \c std::endl and \c std::setw
      For more information, see ostream header */
};

class tout {
private:
    std::ostream &os1, &os2, &os3;

public:
    explicit tout(std::ostream& _os1, std::ostream& _os2, std::ostream& _os3) : os1(_os1), os2(_os2), os3(_os3){};
    template<typename T> tout& operator<<(const T& rhs) {
        os1 << rhs;
        os2 << rhs;
        os3 << rhs;
        return *this;
    };
    tout& operator<<(std::ostream& (*__pf)(std::ostream&)) {
        __pf(os1);
        __pf(os2);
        __pf(os3);
        return *this;
    };
    /*!<  Interface for manipulators, such as \c std::endl and \c std::setw
      For more information, see ostream header */
};

#endif