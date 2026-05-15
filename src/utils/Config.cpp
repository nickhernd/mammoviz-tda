#include "utils/Config.h"
#include "utils/Logger.h"
#include <fstream>
#include <sstream>
#include <regex>

namespace mmviz::utils {

Config& Config::instance() {
    static Config inst;
    return inst;
}

// Minimal TOML parser: handles key = value lines and [section] headers
// Ignores tables/arrays; sufficient for flat config files
bool Config::load(const std::string& toml_path) {
    std::ifstream file(toml_path);
    if (!file) {
        LOG_WARN("Config file not found: {} — using defaults", toml_path);
        return false;
    }

    std::string section;
    std::string line;

    while (std::getline(file, line)) {
        // Strip comments and whitespace
        auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);

        // Trim
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t\r\n"));
            s.erase(s.find_last_not_of(" \t\r\n") + 1);
        };
        trim(line);
        if (line.empty()) continue;

        // Section header [name]
        if (line.front() == '[') {
            section = line.substr(1, line.find(']') - 1);
            continue;
        }

        // key = value
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        trim(key);
        trim(val);

        std::string full_key = section.empty() ? key : section + "." + key;

        // Type inference
        if (val == "true")  { m_data[full_key] = true;  continue; }
        if (val == "false") { m_data[full_key] = false; continue; }

        // Strip quotes for strings
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
            m_data[full_key] = val.substr(1, val.size() - 2);
            continue;
        }

        // Try int, then float
        try {
            size_t pos;
            int iv = std::stoi(val, &pos);
            if (pos == val.size()) { m_data[full_key] = iv; continue; }
        } catch (...) {}

        try {
            size_t pos;
            float fv = std::stof(val, &pos);
            if (pos == val.size()) { m_data[full_key] = fv; continue; }
        } catch (...) {}

        m_data[full_key] = val;
    }

    LOG_INFO("Config loaded from: {} ({} keys)", toml_path, m_data.size());
    return true;
}

int Config::getInt(const std::string& key, int def) const {
    auto it = m_data.find(key);
    if (it == m_data.end()) return def;
    if (auto* v = std::get_if<int>(&it->second)) return *v;
    return def;
}

float Config::getFloat(const std::string& key, float def) const {
    auto it = m_data.find(key);
    if (it == m_data.end()) return def;
    if (auto* v = std::get_if<float>(&it->second)) return *v;
    if (auto* v = std::get_if<int>(&it->second))   return (float)*v;
    return def;
}

bool Config::getBool(const std::string& key, bool def) const {
    auto it = m_data.find(key);
    if (it == m_data.end()) return def;
    if (auto* v = std::get_if<bool>(&it->second)) return *v;
    return def;
}

std::string Config::getString(const std::string& key, std::string def) const {
    auto it = m_data.find(key);
    if (it == m_data.end()) return def;
    if (auto* v = std::get_if<std::string>(&it->second)) return *v;
    return def;
}

} // namespace mmviz::utils
