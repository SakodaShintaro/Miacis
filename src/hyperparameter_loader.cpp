#include "hyperparameter_loader.hpp"
#include <iostream>
#include <sstream>

HyperparameterLoader::HyperparameterLoader(const std::string& file_name) {
    //オプションをファイルから読み込む
    ifs_.open(file_name);
    if (!ifs_) {
        std::cerr << "fail to open " << file_name << std::endl;
        exit(1);
    }
}

template<class T> T HyperparameterLoader::get(const std::string& name) {
    //最初は条件を満たさないように初期化
    std::string line;
    while (getline(ifs_, line)) {
        if (line.front() == '#') {
            //コメントは読み飛ばす
            continue;
        }

        //空白で区切りたいのでstringstreamを使う
        std::stringstream ss;
        ss << line;

        std::string option_name;
        ss >> option_name;
        if (option_name != name) {
            continue;
        }

        //このオプションが見つかった
        //ファイルポインタを先頭に戻す
        ifs_.seekg(0, std::ios_base::beg);

        T value;
        ss >> value;
        return value;
    }

    std::cerr << "There is not such option: " << name << std::endl;
    std::exit(1);
}

template int64_t HyperparameterLoader::get<int64_t>(const std::string& name);
template float HyperparameterLoader::get<float>(const std::string& name);
template bool HyperparameterLoader::get<bool>(const std::string& name);
template std::string HyperparameterLoader::get<std::string>(const std::string& name);