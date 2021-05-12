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

#ifndef TNN_INCLUDE_TNN_UTILS_DIMS_VECTOR_UTILS_H_
#define TNN_INCLUDE_TNN_UTILS_DIMS_VECTOR_UTILS_H_

#include <algorithm>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"

namespace TNN_NS {

class PUBLIC DimsVectorUtils {
public:
    // @brief all dims product, [start_index, end_index)
    // @param dims
    static int Count(const DimsVector &dims, int start_index = 0, int end_index = -1);

    // @brief max of dims0 and dims1, [start_index, end_index)
    static DimsVector Max(const DimsVector &dims0, const DimsVector &dims1, int start_index = 0, int end_index = -1);

    // @brief min of dims0 and dims1, [start_index, end_index)
    static DimsVector Min(const DimsVector &dims0, const DimsVector &dims1, int start_index = 0, int end_index = -1);

    // @brief equal of dims0 and dims1, [start_index, end_index)
    static bool Equal(const DimsVector &dims0, const DimsVector &dims1, int start_index = 0, int end_index = -1);
    
    // @brief NCHW dims vector to NHWC dims vector
    static DimsVector NCHW2NHWC(const DimsVector &dims);

    // @brief NHWC dims vector to NCHW
    static DimsVector NHWC2NCHW(const DimsVector &dims);
};

}  // namespace TNN_NS

#endif  // TNN_INCLUDE_TNN_UTILS_DIMS_VECTOR_UTILS_H_
