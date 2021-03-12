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

#ifndef TNN_INCLUDE_TNN_UTILS_HALF_UTILS_H_
#define TNN_INCLUDE_TNN_UTILS_HALF_UTILS_H_

#include "tnn/core/macro.h"

namespace TNN_NS {

// @brief convert float to half
extern PUBLIC int ConvertFromFloatToHalf(float *fp32, void *fp16, int count);
// @brief convert half to float
extern PUBLIC int ConvertFromHalfToFloat(void *fp16, float *fp32, int count);

}  // namespace TNN_NS

#endif  // TNN_INCLUDE_TNN_UTILS_HALF_UTILS_H_
