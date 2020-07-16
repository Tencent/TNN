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

#ifndef TNN_INCLUDE_TNN_API_MACRO_H_
#define TNN_INCLUDE_TNN_API_MACRO_H_

// Interface visibility
#define PUBLIC __attribute__((visibility("default")))

// Log
#define DEFAULT_TAG "converter"

#ifdef __ANDROID__
#include <android/log.h>
#define LOGDT(fmt, tag, ...)                                                                                           \
    __android_log_print(ANDROID_LOG_DEBUG, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,         \
                        __LINE__, ##__VA_ARGS__)
#define LOGIT(fmt, tag, ...)                                                                                           \
    __android_log_print(ANDROID_LOG_INFO, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,          \
                        __LINE__, ##__VA_ARGS__)
#define LOGET(fmt, tag, ...)                                                                                           \
    __android_log_print(ANDROID_LOG_ERROR, tag, ("%s [File %s][Line %d] " fmt), __PRETTY_FUNCTION__, __FILE__,         \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define LOGDT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LOGIT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LOGET(fmt, tag, ...)                                                                                           \
    fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#endif  //__ANDROID__

#define LOGD(fmt, ...) LOGDT(fmt, DEFAULT_TAG, ##__VA_ARGS__)
#define LOGI(fmt, ...) LOGIT(fmt, DEFAULT_TAG, ##__VA_ARGS__)
#define LOGE(fmt, ...) LOGET(fmt, DEFAULT_TAG, ##__VA_ARGS__)

#ifndef DEBUG
#undef LOGD
#define LOGD(fmt, ...)
#endif  // DEBUG

// Assert
#include "assert.h"
#define ASSERT(x)                                                              \
    {                                                                          \
        int res = (x);                                                         \
        if (!res) {                                                            \
            LOGE("");                                                          \
            assert(res);                                                       \
        }                                                                      \
    }

// Math
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)
#define ALIGN_UP8(x) ROUND_UP((x), 8)

#if (__arm__ || __aarch64__) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#define TNN_USE_NEON
#endif

#endif  // TNN_INCLUDE_TNN_API_MACRO_H_
