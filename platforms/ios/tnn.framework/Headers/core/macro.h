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

#ifndef TNN_INCLUDE_TNN_CORE_MACRO_H_
#define TNN_INCLUDE_TNN_CORE_MACRO_H_
#include <stdio.h>
#include <stdlib.h>

// disable Warning 4003 in MSVC: warning C4003: param not enough to call “TNN_NS_”
#pragma warning(disable:4003)
// TNN namespcae
#define TNN_NS__(x) tnn##x
#define TNN_NS_(x) TNN_NS__(x)
#define TNN_NS TNN_NS_()

// TNN profile
#ifndef TNN_PROFILE
#define TNN_PROFILE 0
#endif

// Interface visibility
#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_DLL
#ifdef __GNUC__
#define PUBLIC __attribute__((dllexport))
#else  // __GNUC__
#define PUBLIC __declspec(dllexport)
#endif // __GNUC__
#else // BUILDING_DLL
#ifdef __GNUC__
#define PUBLIC __attribute__((dllimport))
#else
#define PUBLIC __declspec(dllimport)
#endif // __GNUC__
#endif // BUILDING_DLL
#define LOCAL
#else // _WIN32 || __CYGWIN__
#if __GNUC__ >= 4
#define PUBLIC __attribute__((visibility("default")))
#define LOCAL __attribute__((visibility("hidden")))
#else
#define PUBLIC
#define LOCAL
#endif
#endif

// DATAPRECISION
// float IEEE 754
#ifndef FLT_MIN
#define FLT_MIN 1.175494351e-38F
#define FLT_MAX 3.402823466e+38F
#define FLT_EPSILON 1.192092896e-07F
#endif
// int8
#ifndef INT8_MIN
#define INT8_MIN ((int8_t)-128)
#endif
#ifndef INT8_MAX
#define INT8_MAX ((int8_t)127)
#endif

#define DEFAULT_TAG "tnn"

#ifdef _WIN32
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

// Log
#ifdef __ANDROID__
#include <android/log.h>
#define LOGDT(fmt, tag, ...)                                                                                           \
    __android_log_print(ANDROID_LOG_DEBUG, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,         \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LOGIT(fmt, tag, ...)                                                                                           \
    __android_log_print(ANDROID_LOG_INFO, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,          \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LOGET(fmt, tag, ...)                                                                                           \
    __android_log_print(ANDROID_LOG_ERROR, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,         \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define LOGDT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LOGIT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LOGET(fmt, tag, ...)                                                                                           \
    fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#endif  //__ANDROID__

#define LOGD(fmt, ...) LOGDT(fmt, DEFAULT_TAG, ##__VA_ARGS__)
#define LOGI(fmt, ...) LOGIT(fmt, DEFAULT_TAG, ##__VA_ARGS__)
#define LOGE(fmt, ...) LOGET(fmt, DEFAULT_TAG, ##__VA_ARGS__)
#define LOGE_IF(cond, fmt, ...) if(cond) { LOGET(fmt, DEFAULT_TAG, ##__VA_ARGS__); }

// Assert
#include <cassert>
#define ASSERT(x)                                                                                                      \
    {                                                                                                                  \
        int res = (x);                                                                                                 \
        if (!res) {                                                                                                    \
            LOGE("Error: assert failed\n");                                                                              \
            assert(res);                                                                                               \
        }                                                                                                              \
    }

#ifndef DEBUG
#undef LOGDT
#undef LOGD
#define LOGDT(fmt, tag, ...)
#define LOGD(fmt, ...)
#undef ASSERT
#define ASSERT(x)
#endif  // DEBUG

// BREAK_IF
#define BREAK_IF(cond)                                                                                                 \
    if (cond)                                                                                                          \
    break
#ifdef __OPTIMIZE__
#define BREAK_IF_MSG(cond, msg)                                                                                        \
    if (cond)                                                                                                          \
    break
#else
#define BREAK_IF_MSG(cond, msg)                                                                                        \
    if (cond)                                                                                                          \
        LOGD(msg);                                                                                                     \
    if (cond)                                                                                                          \
    break
#endif

// Math
#ifndef UP_DIV
#define UP_DIV(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y))
#endif
#ifndef ROUND_UP
#define ROUND_UP(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y) * (int)(y))
#endif
#ifndef ALIGN_UP4
#define ALIGN_UP4(x) ROUND_UP((x), 4)
#endif
#ifndef ALIGN_UP8
#define ALIGN_UP8(x) ROUND_UP((x), 8)
#endif
#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif
#ifndef ABS
#define ABS(x) ((x) > (0) ? (x) : (-(x)))
#endif

#if (__arm__ || __aarch64__) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#define TNN_USE_NEON
#endif

#if TNN_ARM82

#ifndef TNN_ARM82_SIMU

#if defined(__aarch64__) && defined(TNN_USE_NEON)
#define TNN_ARM82_A64
#elif defined(__arm__)  && defined(TNN_USE_NEON)
#define TNN_ARM82_A32
#else
#define TNN_ARM82_SIMU
#endif

#endif  // TNN_ARM82_SIMU

#else

#ifdef TNN_ARM82_SIMU
#undef TNN_ARM82_SIMU
#endif

#endif  // TNN_ARM82

#if defined(TNN_ARM82_A64) || defined(TNN_ARM82_A32)
#define TNN_ARM82_USE_NEON
#endif

#define RETURN_VALUE_ON_NEQ(status, expected, value)                  \
    do {                                                                                                         \
        auto _status = (status);                                                                         \
        if (_status != (expected)) {                                                                     \
            return (value);                                                                                 \
        }                                                                                                          \
    } while (0)

#define RETURN_ON_NEQ(status, expected)                                         \
    do {                                                                                                        \
        auto _status = (status);                                                                        \
        if (_status != (expected)) {                                                                    \
            return _status;                                                                               \
        }                                                                                                         \
    } while (0)

#define CHECK_PARAM_NULL(param)                                                   \
    do {                                                                                                         \
        if (!param) {                                                                                        \
            return Status(TNNERR_PARAM_ERR, "Error: param is nil");                                                    \
        }                                                                                                          \
    } while (0)


#if defined(__GNUC__) || defined(__clang__)
#define DEPRECATED(msg) __attribute__((deprecated (msg)))
#elif defined(_MSC_VER)
#define DEPRECATED(msg) __declspec(deprecated (msg))
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED
#endif

#endif  // TNN_INCLUDE_TNN_CORE_MACRO_H_
