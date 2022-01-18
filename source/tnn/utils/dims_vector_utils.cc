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

#include "tnn/utils/dims_utils.h"

#include <cmath>
#include <climits>

namespace TNN_NS {

int DimsVectorUtils::Count(const DimsVector &dims, int start_index, int end_index) {
    if (-1 == end_index || end_index > dims.size()) {
        end_index = static_cast<int>(dims.size());
    }

    int result = 1;
    for (int index = start_index; index < end_index; ++index) {
        result *= dims[index];
    }
    return result;
}

DimsVector DimsVectorUtils::Max(const DimsVector &dims0, const DimsVector &dims1, int start_index, int end_index) {
    DimsVector max_dims;
    DimsVector small_dims;
    if (dims0.size() >= dims1.size()) {
        max_dims   = dims0;
        small_dims = dims1;
    } else {
        max_dims   = dims1;
        small_dims = dims0;
    }

    if (small_dims.size() <= start_index) {
        return max_dims;
    }

    if (-1 == end_index || end_index > small_dims.size()) {
        end_index = static_cast<int>(small_dims.size());
    }

    for (int i = start_index; i < end_index; i++) {
        max_dims[i] = std::max(max_dims[i], small_dims[i]);
    }

    return max_dims;
}

DimsVector DimsVectorUtils::Min(const DimsVector &dims0, const DimsVector &dims1, int start_index, int end_index) {
    DimsVector min_dims;
    DimsVector small_dims;
    if (dims0.size() >= dims1.size()) {
        min_dims   = dims0;
        small_dims = dims1;
    } else {
        min_dims   = dims1;
        small_dims = dims0;
    }

    if (small_dims.size() <= start_index) {
        return small_dims;
    }

    if (-1 == end_index || end_index > small_dims.size()) {
        end_index = static_cast<int>(small_dims.size());
    }

    for (int i = start_index; i < end_index; i++) {
        min_dims[i] = std::min(min_dims[i], small_dims[i]);
    }

    return min_dims;
}

bool DimsVectorUtils::Equal(const DimsVector &dims0, const DimsVector &dims1, int start_index, int end_index) {
    if (dims0.size() == 0 && dims1.size() == 0) {
        return true;
    }
    
    if (dims0.size() <= start_index) {
        return false;
    }

    if (-1 == end_index || end_index > dims0.size()) {
        end_index = static_cast<int>(dims0.size());
    }

    if (dims0.size() != dims1.size()) {
        return false;
    }

    for (int i = start_index; i < end_index; i++) {
        if (dims0[i] != dims1[i]) {
            return false;
        }
    }
    return true;
}

DimsVector DimsVectorUtils::NCHW2NHWC(const DimsVector &dims) {
    ASSERT(dims.size() == 4);
    const int n           = dims[0];
    const int c           = dims[1];
    const int h           = dims[2];
    const int w           = dims[3];
    std::vector<int> nhwc = {n, h, w, c};
    return nhwc;
}

DimsVector DimsVectorUtils::NHWC2NCHW(const DimsVector &dims) {
    ASSERT(dims.size() == 4);
    const int n           = dims[0];
    const int h           = dims[1];
    const int w           = dims[2];
    const int c           = dims[3];
    std::vector<int> nhwc = {n, c, h, w};
    return nhwc;
}

}  // namespace TNN_NS
