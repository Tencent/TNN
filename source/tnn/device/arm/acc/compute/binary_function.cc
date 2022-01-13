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

#include "tnn/device/arm/acc/compute/binary_function.h"

namespace TNN_NS {

void PadShape(const int pad_size, const int dim_size, DimsVector &pad_shape, DimsVector in_shape) {
    int j = 0;
    for (; j < pad_size; j++) {
        pad_shape[j] = 1;
    }
    for (; j < dim_size; j++) {
        pad_shape[j] = in_shape[j - pad_size];
    }
}

void BroadCastTypeFilter(const DimsVector &dims_output, const DimsVector &dims_input, BroadcastType &type) {
    if (DimsVectorUtils::Equal(dims_output, dims_input)) {
        type = BroadcastTypeNormal;
        return;
    }
    if (DimsVectorUtils::Equal(dims_output, dims_input, 1) &&
        DimsVectorUtils::Count(dims_input, 0, 1) == 1) {
        type = BroadcastTypeElement;
        return;
    }
    if (DimsVectorUtils::Equal(dims_output, dims_input, 2) &&
        DimsVectorUtils::Count(dims_input, 0, 2) == 1) {
        type = BroadcastTypeHeightWidth;
        return;
    }
    if (DimsVectorUtils::Equal(dims_output, dims_input, 3) &&
        DimsVectorUtils::Count(dims_input, 0, 3) == 1) {
        type = BroadcastTypeWidth;
        return;
    }
    int broadcast_count = DimsVectorUtils::Count(dims_input);
    if (broadcast_count == 1) {
        type = BroadcastTypeSingle;
    } else if (broadcast_count == dims_output[1]) {
        // broadcast dim = [1, channel, 1...]
        if (dims_input[1] == dims_output[1]) {
            type = BroadcastTypeChannel;
        } else {
            type = BroadcastTypeGeneral;
        }
    } else {
        type = BroadcastTypeGeneral;
    }
    return;
}

void BroadCastInit(const DimsVector &dims, const DimsVector &dims0, const DimsVector &dims1,
                   BroadcastType &type, DimsVector &dims_broadcast, bool &swap_flag) {
    if (DimsVectorUtils::Equal(dims0, dims1)) {
        type = BroadcastTypeNormal;
        dims_broadcast.clear();
    } else if (DimsVectorUtils::Equal(dims0, dims1, 1) &&
              (DimsVectorUtils::Count(dims0, 0, 1) == 1 ||
               DimsVectorUtils::Count(dims1, 0, 1) == 1)) {
        type = BroadcastTypeElement;
        dims_broadcast.clear();
        if (dims0[0] < dims1[0])
            swap_flag = true;
    } else if (DimsVectorUtils::Equal(dims0, dims1, 2) &&
              (DimsVectorUtils::Count(dims0, 0, 2) == 1 ||
               DimsVectorUtils::Count(dims1, 0, 2) == 1)) {
        type = BroadcastTypeHeightWidth;
        dims_broadcast.clear();
        if (dims0[1] < dims1[1])
            swap_flag = true;
    } else if (DimsVectorUtils::Equal(dims0, dims1, 3) &&
              (DimsVectorUtils::Count(dims0, 0, 3) == 1 ||
               DimsVectorUtils::Count(dims1, 0, 3) == 1)) {
        type = BroadcastTypeWidth;
        dims_broadcast.clear();
        if (dims0[1] < dims1[1])
            swap_flag = true;
    } else if (DimsVectorUtils::Equal(dims0, dims)) {
        dims_broadcast = dims1;
    } else {
        dims_broadcast = dims0;
        swap_flag      = true;
    }
}

void BinaryComputeOffset(DimsVector &offset, const DimsVector dims_in, const DimsVector dims_out) {
    DimsVector dims_pad_in;
    dims_pad_in.resize(dims_out.size());
    int pad_size = dims_out.size() - dims_in.size();
    PadShape(pad_size, dims_out.size(), dims_pad_in, dims_in);

    offset.resize(dims_out.size());
    int s = 1;
    for (int i = dims_out.size() - 1; i >= 0; i--) {
        offset[i] = (dims_pad_in[i] == dims_out[i]) ? s : 0;
        s *= dims_pad_in[i];
    }
}

}  // namespace TNN_NS
