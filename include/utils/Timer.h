#pragma once
#include <chrono>

namespace mmviz::utils {

class Timer {
public:
    Timer() : m_start(std::chrono::steady_clock::now()) {}

    void reset() { m_start = std::chrono::steady_clock::now(); }

    double elapsedSeconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - m_start).count();
    }

    double elapsedMs() const { return elapsedSeconds() * 1000.0; }

private:
    std::chrono::steady_clock::time_point m_start;
};

} // namespace mmviz::utils
