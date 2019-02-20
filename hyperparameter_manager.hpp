#ifndef MIACIS_HYPERPARAMETER_MANAGER_HPP
#define MIACIS_HYPERPARAMETER_MANAGER_HPP

#include<string>
#include<unordered_map>

class HyperparameterManager {
public:
    void add(const std::string& name, int64_t lower_limit, int64_t upper_limit);
    void add(const std::string& name, float   lower_limit,   float upper_limit);
    void add(const std::string& name);
    void load(const std::string& file_path);
    bool check();
    template <class T> T get(const std::string& name);
private:
    enum {
        VALUE, UPPER_LIMIT, LOWER_LIMIT,
    };

    static const std::string BAT_INIT;
    std::unordered_map<std::string, std::array<int64_t, 3>> int_map;
    std::unordered_map<std::string, std::array<float, 3>>   float_map;
    std::unordered_map<std::string, std::string>            string_map;
};

#endif //MIACIS_HYPERPARAMETER_MANAGER_HPP