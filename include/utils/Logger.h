#pragma once
#include <string>
#include <format>

namespace mmviz::utils {

enum class LogLevel { DEBUG, INFO, WARN, ERROR };

class Logger {
public:
    static Logger& instance();

    void setLevel(LogLevel level) { m_level = level; }

    template<typename... Args>
    void debug(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::DEBUG, std::format(fmt, std::forward<Args>(args)...));
    }
    template<typename... Args>
    void info(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::INFO, std::format(fmt, std::forward<Args>(args)...));
    }
    template<typename... Args>
    void warn(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::WARN, std::format(fmt, std::forward<Args>(args)...));
    }
    template<typename... Args>
    void error(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::ERROR, std::format(fmt, std::forward<Args>(args)...));
    }

private:
    Logger() = default;
    void log(LogLevel level, const std::string& msg);
    LogLevel m_level = LogLevel::INFO;
};

#define LOG_DEBUG(...) ::mmviz::utils::Logger::instance().debug(__VA_ARGS__)
#define LOG_INFO(...)  ::mmviz::utils::Logger::instance().info(__VA_ARGS__)
#define LOG_WARN(...)  ::mmviz::utils::Logger::instance().warn(__VA_ARGS__)
#define LOG_ERROR(...) ::mmviz::utils::Logger::instance().error(__VA_ARGS__)

} // namespace mmviz::utils
