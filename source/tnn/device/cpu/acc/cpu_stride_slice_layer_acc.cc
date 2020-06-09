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

#include <algorithm>
#include <cmath>
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(StrideSlice, LAYER_STRIDED_SLICE);

Status CpuStrideSliceLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuStrideSliceLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<StrideSliceLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: StrideSliceLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    int input_channel = input_blob->GetBlobDesc().dims[1];
    int input_height  = input_blob->GetBlobDesc().dims[2];
    int input_width   = input_blob->GetBlobDesc().dims[3];

    auto begins = layer_param->begins;
    std::reverse(begins.begin(), begins.end());
    auto ends = layer_param->ends;
    std::reverse(ends.begin(), ends.end());
    auto strides = layer_param->strides;
    std::reverse(strides.begin(), strides.end());

    for (int i = 0; i < ends.size(); ++i) {
        if (ends[i] == 0) {
            ends[i] = input_blob->GetBlobDesc().dims[i];
        }
    }
    if (output_blob->GetBlobDesc().data_type != DATA_TYPE_INT8) {
        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);
        int out_offset     = 0;
        for (int n = begins[0]; n < ends[0]; n += strides[0]) {
            for (int c = begins[1]; c < ends[1]; c += strides[1]) {
                for (int h = begins[2]; h < ends[2]; h += strides[2]) {
                    for (int w = begins[3]; w < ends[3]; w += strides[3]) {
                        const int in_offset = n * input_channel * input_height * input_width +
                                              c * input_height * input_width + h * input_width + w;
                        output_data[out_offset] = input_data[in_offset];
                        ++out_offset;
                    }
                }
            }
        }
    } else {
        ASSERT(0);
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(StrideSlice, LAYER_STRIDED_SLICE);

}  // namespace TNN_NS
