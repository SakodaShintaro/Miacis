#ifndef MIACIS_HYPERPARAMETER_LOADER_HPP
#define MIACIS_HYPERPARAMETER_LOADER_HPP

#include <array>
#include <fstream>
#include <string>

class HyperparameterLoader {
public:
    explicit HyperparameterLoader(const std::string& file_name);
    template<class T> T get(const std::string& name);

private:
    std::ifstream ifs_;
};

#endif //MIACIS_HYPERPARAMETER_LOADER_HPP