// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tnn/optimizer/graph_matcher/logger.h"

#ifdef _WIN32
#include "windows.h"
#include <ctime>
#else
#include <unistd.h>
#endif

#include <chrono>
#include <stdexcept>
#include <vector>
#include <string>
#include <mutex>
#include <sstream>
#include <string.h>

#include "tnn/core/macro.h"

namespace TNN_NS {

using namespace std;

static const int LEVEL_D = 0;
static const int LEVEL_I = 1;
static const int LEVEL_W = 2;
static const int LEVEL_E = 3;
static const int LEVEL_F = 4;

inline int GetLogLevelFromString(const std::string& str_level) {
    if (str_level == Logger::kLogLevelDebug()) {
        return LEVEL_D;
    } else if (str_level == Logger::kLogLevelInfo()) {
        return LEVEL_I;
    } else if (str_level == Logger::kLogLevelWarning()) {
        return LEVEL_W;
    } else if (str_level == Logger::kLogLevelError()) {
        return LEVEL_E;
    } else if (str_level == Logger::kLogLevelFatal()) {
        return LEVEL_F;
    }
    return 5;
}

inline std::string GetLogLevelFromInt(int level) {
    if (level >=0 && level <=4) {
        const char * p = "DIWEF";
        return std::string(1, p[level]);
    }
    return "F";
}

Logger& Logger::instance()
{
    static Logger _instance;

    return _instance;
}

Logger::~Logger() { }

Logger::Logger()
{
#ifdef _WIN32
    pid_ = GetCurrentProcessId();
#else
    pid_ = getpid();
#endif
    level_ = LEVEL_I;
    verbose_level_ = LEVEL_E;
}

void Logger::log(const string& inMessage, const string& inLogLevel)
{
    lock_guard<mutex> guard(mutex_);
    logHelper(inMessage, inLogLevel);
}

void Logger::log(const vector<string>& inMessages, const string& inLogLevel)
{
    lock_guard<mutex> guard(mutex_);
    for (size_t i = 0; i < inMessages.size(); i++) {
        logHelper(inMessages[i], inLogLevel);
    }
}

void Logger::set_level(const string& level) {
    const int lv = GetLogLevelFromString(level);
    level_= lv;
} 

void Logger::set_verbose_level(const string& level) {
    const int lv = GetLogLevelFromString(level);
    verbose_level_ = lv;
} 

std::string Logger::get_level() {
    auto ret = GetLogLevelFromInt(level_);
    return ret;
}

std::string Logger::get_verbose_level() {
    auto ret = GetLogLevelFromInt(verbose_level_);
    return ret;
}

const std::string Logger::get_time_str() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_t = std::chrono::system_clock::to_time_t(now);
    std::string s(200, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now_t));

    auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    auto fraction = now - seconds;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(fraction);

    char buffer[200];
    sprintf(buffer, " %03lld", milliseconds.count());
    return s.substr(0, strlen(s.c_str()))+std::string(buffer);
}

void Logger::logHelper(const std::string& inMessage, const std::string& inLogLevel)
{
    const int level = GetLogLevelFromString(inLogLevel);
    if (level >= level_) {
        output_stream_ <<  Logger::get_time_str() << ": " << inLogLevel << " " << inMessage << endl;
    }

    std::ostream * std_stream = &std::cout;
    if (inLogLevel == kLogLevelError() || inLogLevel == kLogLevelFatal()) {
        std_stream = &std::cerr;
    } 
    if (level >= verbose_level_ ) {
        *std_stream <<  Logger::get_time_str() << ": " << inLogLevel << " " << inMessage << endl;
    }
}	

std::string Logger::str()
{
    return output_stream_.str();
}

}
