#ifndef MIACIS_POSITION_HPP
#define MIACIS_POSITION_HPP

#include <vector>

//将棋やオセロなどへ拡張していく基底となるクラス
//環境(Environment)という名前にした方が良い可能性も？
class BasePosition {
public:
    //doMoveは行動を得て動かす関数であり、気持ちとしてはここで宣言したいが
    //行動が各ゲームごとに異なるのでこれは各子クラスで宣言・定義する
    //virtual void doMove(const Move& move);

    //一手戻す関数は純粋仮想関数として宣言
    virtual void undo() = 0;

    //現局面の特徴量を作る関数
    virtual std::vector<float> makeFeature() = 0;

private:
};

#endif //MIACIS_POSITION_HPP