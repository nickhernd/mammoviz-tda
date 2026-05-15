#include "utils/Logger.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>

namespace mmviz::utils {

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

static const char* levelStr(LogLevel l) {
    switch (l) {
        case LogLevel::DEBUG: return "\033[36mDEBUG\033[0m";
        case LogLevel::INFO:  return "\033[32m INFO\033[0m";
        case LogLevel::WARN:  return "\033[33m WARN\033[0m";
        case LogLevel::ERROR: return "\033[31mERROR\033[0m";
    }
    return "?";
}

void Logger::log(LogLevel level, const std::string& msg) {
    if (level < m_level) return;

    auto now  = std::chrono::system_clock::now();
    auto t    = std::chrono::system_clock::to_time_t(now);
    auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) % 1000;

    std::ostream& out = (level >= LogLevel::WARN) ? std::cerr : std::cout;
    out << std::put_time(std::localtime(&t), "%H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count()
        << " [" << levelStr(level) << "] "
        << msg << "\n";
}

} // namespace mmviz::utils
