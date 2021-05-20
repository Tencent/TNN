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

#ifndef TNN_TEST_UNIT_TEST_LAYER_TEST_LAYER_TEST_MACRO_H_
#define TNN_TEST_UNIT_TEST_LAYER_TEST_LAYER_TEST_MACRO_H_

#include <gtest/gtest.h>
#include "tnn/core/macro.h"

namespace TNN_NS {

#ifdef TNN_UNIT_TEST_BENCHMARK

#define BASIC_BATCH_CHANNEL_SIZE testing::Values(1, 16), testing::Values(1, 3, 64, 128), testing::Values(256)

#define UNARY_BATCH_CHANNEL_SIZE testing::Values(1, 16), testing::Values(1, 3, 64, 128), testing::Values(256)

#else

#define BASIC_BATCH_CHANNEL_SIZE                                                                                       \
    testing::Values(1, 2), testing::Values(1, 2, 3, 4, 10, 32), testing::Values(9, 10, 16, 19)

#define UNARY_BATCH_CHANNEL_SIZE                                                                                       \
    testing::Values(1, 2), testing::Values(1, 3, 4), testing::Values(3)

#endif

}  // namespace TNN_NS

#endif  // TNN_TEST_UNIT_TEST_LAYER_TEST_LAYER_TEST_MACRO_H_
