#pragma once
#include <string>
#include <unordered_map>
#include <variant>

namespace mmviz::utils {

// Simple TOML-like config loader for runtime parameters
// Avoids magic numbers scattered across the codebase
class Config {
public:
    using Value = std::variant<int, float, bool, std::string>;

    static Config& instance();
    bool load(const std::string& toml_path);

    int         getInt   (const std::string& key, int         def = 0)      const;
    float       getFloat (const std::string& key, float       def = 0.0f)   const;
    bool        getBool  (const std::string& key, bool        def = false)   const;
    std::string getString(const std::string& key, std::string def = "")      const;

private:
    Config() = default;
    std::unordered_map<std::string, Value> m_data;
};

} // namespace mmviz::utils
