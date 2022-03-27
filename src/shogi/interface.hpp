#ifndef MIACIS_SHOGI_INTERFACE_HPP
#define MIACIS_SHOGI_INTERFACE_HPP

#include "../search/search_options.hpp"
#include "../search/searcher_for_play.hpp"
#include "position.hpp"
#include <functional>
#include <thread>

namespace Shogi {

//USIプロトコルに従うインターフェース
class Interface {
public:
    Interface();
    void loop();
    void usi();
    void isready();
    void setoption();
    void usinewgame();
    void position();
    void go();
    void stop();
    void quit();
    void gameover();

    //setoptionで設定した後テストする関数
    void testSelfPlay(int64_t game_num);

    //onnxをTensorRTエンジンへ変換する
    void convertOnnxToEngine();

private:
    std::unordered_map<std::string, std::function<void()>> command_;
    Position root_;
    std::unique_ptr<SearcherForPlay> searcher_;
    std::thread thread_;
    SearchOptions search_options_;
};

} // namespace Shogi

#endif //MIACIS_SHOGI_INTERFACE_HPP