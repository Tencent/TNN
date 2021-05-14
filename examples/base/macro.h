
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

#ifndef TNN_EXAMPLES_X86_UTILS_MACRO_H_
#define TNN_EXAMPLES_X86_UTILS_MACRO_H_

#include <stdio.h>
#include <string>

#include "tnn/core/status.h"

#define LOGERROR(status)                                \
    fprintf(stderr, "%s", status.description().c_str());

#define BREAK_ON_NEQ(status, expected)                  \
    {                                                   \
        auto _status = (status);                        \
        if (_status != (expected)) {                    \
            LOGERROR(_status);                          \
            break;                                      \
        }                                               \
    } 

#define CHECK_API(status)                                                      \
    do {                                                                       \
        if (status != 0) {                                                     \
            fprintf(stderr, "API ERROR:%d\n", int(status));                    \
            return -1;                                                         \
        }                                                                      \
    } while (0)

#define CHECK_TNN_STATUS(status)                                               \
    do {                                                                       \
        if (status != TNN_NS::TNN_OK) {                                        \
            fprintf(stderr, "TNN API ERROR:0x%x", int(status));                \
            return status;                                                     \
        }                                                                      \
    } while (0)


#endif // TNN_EXAMPLES_X86_UTILS_MACRO_H_
