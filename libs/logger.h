#pragma once

#include <iostream>
#include <ostream>

namespace Logger {

enum class Level {
    INFO,
    WARN,
    FATAL
};

class Log {
    Level level_;
public:
    Log(Level l = Level::INFO) { level_ = l; }

    template <typename T>
    std::ostream& operator<<(T&& object)
    {
        std::string level;

        switch (level_) {
        case Level::INFO:
            level = "[Info]: ";
            break;
        case Level::WARN:
            level = "[WARN]: ";
            break;
        case Level::FATAL:
            level = "[*FATAL*]: ";
            break;
        default:
            break;
        }

        std::cout << level << object;
        return std::cout;
    }
};

} // Logger namespace