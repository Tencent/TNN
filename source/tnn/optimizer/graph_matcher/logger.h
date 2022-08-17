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


#ifndef TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_LOGGER_H_
#define TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_LOGGER_H_


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <mutex>
#include <stack>
#include <sstream>

#include "tnn/core/macro.h"

#if __cplusplus < 201103L
#error This code requires C++11
#endif

// this logger implementation referenced https://cplusplus.com/forum/unices/132540/ and updated for c++11

namespace TNN_NS {

class Logger {
public:
    static std::string kLogLevelDebug() {
      static const std::string _kLogLevelDebug = "D";
      return _kLogLevelDebug;
    }
    static std::string kLogLevelInfo() {
      static const std::string _kLogLevelInfo = "I";
      return _kLogLevelInfo;
    }
    static std::string kLogLevelWarning() {
      static const std::string _kLogLevelWarning = "W";
      return _kLogLevelWarning;
    }
    static std::string kLogLevelError() {
      static const std::string _kLogLevelError = "E";
      return _kLogLevelError;
    }
    static std::string kLogLevelFatal() {
      static const std::string _kLogLevelFatal = "F";
      return _kLogLevelFatal;
    }

    // Returns a reference to the singleton Logger object
    PUBLIC static Logger& instance();

    // Logs a single message at the given log level
    void log(const std::string& inMessage, const std::string& inLogLevel);

    PUBLIC void set_level(const std::string&);
    PUBLIC void set_verbose_level(const std::string&);

    PUBLIC std::string get_level();
    PUBLIC std::string get_verbose_level();

    // Logs a vector of messages at the given log level
    void log(const std::vector<std::string>& inMessages, const std::string& inLogLevel);

    PUBLIC std::string str();

    PUBLIC void renew_session() {
        std::lock_guard<std::mutex> guard(mutex_);
        history_.push_back(this->str());
        output_stream_.str("");
        output_stream_.clear();
        while(history_.size() > 100) {
            history_.erase(history_.begin());
        }
    }

    PUBLIC std::string last(size_t index) {
        if (index < history_.size()) {
            return history_[index];
        } 
        return "";
    }

    PUBLIC size_t last_size() {
        return history_.size();
    };

    PUBLIC void clear() {
        history_.clear();
    }

protected:

    static const std::string get_time_str();

    // Data member for the output stream
    std::stringstream output_stream_;

    int pid_;
    int level_;
    int verbose_level_;

    std::mutex mutex_;

    std::vector<std::string> history_;

    // Logs message. The thread should own a lock on sMutex
    // before calling this function.
    void logHelper(const std::string& inMessage, const std::string& inLogLevel);

    Logger();

public:
    virtual ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};

#ifdef _SOURCE_DIR_LEN
#define __LOG_FORMAT(str_name, max_len, format, ...)                                  \
  char str_name[max_len];                                                     \
  snprintf(str_name, max_len, "%s:%03d " format, &__FILE__[_SOURCE_DIR_LEN+1], __LINE__, ##__VA_ARGS__)
#else
#define __LOG_FORMAT(str_name, max_len, format, ...)                                  \
  char str_name[max_len];                                                     \
  snprintf(str_name, max_len, "%s:%03d " format, __FILE__, __LINE__, ##__VA_ARGS__)
#endif


#define DEBUG(f_, ...) \
  do { \
    __LOG_FORMAT(__ss, 2000, f_, ##__VA_ARGS__); \
    ::TNN_NS::Logger::instance().log(std::string(__ss), ::TNN_NS::Logger::kLogLevelDebug());\
  } while(0)

#define INFO(f_, ...) \
  do { \
    __LOG_FORMAT(__ss, 2000, f_, ##__VA_ARGS__); \
    ::TNN_NS::Logger::instance().log(std::string(__ss), ::TNN_NS::Logger::kLogLevelInfo());\
  } while(0)

#define WARN(f_, ...) \
  do { \
    __LOG_FORMAT(__ss, 2000, f_, ##__VA_ARGS__); \
    ::TNN_NS::Logger::instance().log(std::string(__ss), ::TNN_NS::Logger::kLogLevelWarning());\
  } while(0)

#define ERROR(f_, ...) \
  do { \
    __LOG_FORMAT(__ss, 2000, f_, ##__VA_ARGS__); \
    ::TNN_NS::Logger::instance().log(std::string(__ss), ::TNN_NS::Logger::kLogLevelError());\
  } while(0)

#define ERRORV(f_, vname_, ...) \
  __LOG_FORMAT(vname_, 2000, f_, ##__VA_ARGS__); \
  do { \
    ::TNN_NS::Logger::instance().log(std::string(vname_), ::TNN_NS::Logger::kLogLevelError());\
  } while(0)

#define FATAL(f_, ...) \
  do { \
    __LOG_FORMAT(__ss, 2000, f_, ##__VA_ARGS__); \
    ::TNN_NS::Logger::instance().log(std::string(__ss), ::TNN_NS::Logger::kLogLevelFatal());\
    throw std::runtime_error(std::string(__ss));\
  } while(0)

#define PY_FRAMEINFO(_fname, _lineno, f, lino) \
  const char * _fname = f.c_str();\
  int _lineno = lino;\
  if (f.length() == 0 || lino == -1 ) {\
    pybind11::module inspect_module = pybind11::module::import("inspect");\
    pybind11::object curframe = inspect_module.attr("currentframe")();\
    pybind11::object frameinfo = inspect_module.attr("getframeinfo")(curframe);\
    pybind11::str filename = frameinfo.attr("filename");\
    pybind11::int_ lineno = frameinfo.attr("lineno");\
    _fname = std::string(filename).c_str();\
    _lineno = int(lineno);\
  }

#define PYDEBUG(msg, f, lino) \
  do { \
    char __ss[2000];\
    PY_FRAMEINFO(_fname, _lineno, f, lino);\
    snprintf(__ss, 2000, "%s:%d %s",  _fname, _lineno, msg);\
    ::TNN_NS::Logger::instance().log(std::string(__ss), ::TNN_NS::Logger::kLogLevelDebug());\
  } while(0)

#define PYINFO(msg, f, lino) \
  do { \
    char __ss[2000];\
    PY_FRAMEINFO(_fname, _lineno, f, lino);\
    snprintf(__ss, 2000, "%s:%d %s",  _fname, _lineno, msg);\
    ::TNN_NS::Logger::instance().log(std::string(__ss), ::TNN_NS::Logger::kLogLevelInfo());\
  } while(0)

#define PYWARN(msg, f, lino) \
  do { \
    char __ss[2000];\
    PY_FRAMEINFO(_fname, _lineno, f, lino);\
    snprintf(__ss, 2000, "%s:%d %s",  _fname, _lineno, msg);\
    ::TNN_NS::Logger::instance().log(std::string(__ss), ::TNN_NS::Logger::kLogLevelWarning());\
  } while(0)

#define PYERROR(msg, f, lino) \
  do { \
    char __ss[2000];\
    PY_FRAMEINFO(_fname, _lineno, f, lino);\
    snprintf(__ss, 2000, "%s:%d %s",  _fname, _lineno, msg);\
    ::TNN_NS::Logger::instance().log(std::string(__ss), ::TNN_NS::Logger::kLogLevelError());\
  } while(0)

#define PYFATAL(msg, f, lino) \
  do { \
    char __ss[2000];\
    PY_FRAMEINFO(_fname, _lineno, f, lino);\
    snprintf(__ss, 2000, "%s:%d %s",  _fname, _lineno, msg);\
    ::TNN_NS::Logger::instance().log(std::string(__ss), ::TNN_NS::Logger::kLogLevelFatal());\
    throw std::runtime_error(std::string(__ss));\
  } while(0)

}

#endif // TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_LOGGER_H_

