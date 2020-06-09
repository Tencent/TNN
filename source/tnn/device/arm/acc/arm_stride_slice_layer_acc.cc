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
#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(StrideSlice, LAYER_STRIDED_SLICE);

Status ArmStrideSliceLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<StrideSliceLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: StrideSliceLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto dims_input   = input_blob->GetBlobDesc().dims;
    auto dims_output  = output_blob->GetBlobDesc().dims;
    int input_channel = dims_input[1];
    int input_height  = dims_input[2];
    int input_width   = dims_input[3];
    int output_height = dims_output[2];
    int output_width  = dims_output[3];

    int input_slice  = UP_DIV(dims_input[1], 4);
    int output_slice = UP_DIV(dims_output[1], 4);

    auto begins  = layer_param->begins;
    auto ends    = layer_param->ends;
    auto strides = layer_param->strides;
    std::reverse(begins.begin(), begins.end());
    std::reverse(ends.begin(), ends.end());
    std::reverse(strides.begin(), strides.end());

    for (int i = 0; i < ends.size(); ++i) {
        if (ends[i] == 0) {
            ends[i] = input_blob->GetBlobDesc().dims[i];
        }
    }
    int nn = 0, nc = 0, nh = 0, nw = 0;

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));
        float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));

        for (int n = begins[0]; n < ends[0]; n += strides[0], nn++) {
            auto input_ptr  = input_data + n * input_slice * 4 * input_width * input_height;
            auto output_ptr = output_data + nn * output_slice * 4 * output_width * output_height;
            nc              = 0;
            for (int c = begins[1]; c < ends[1]; c += strides[1], nc++) {
                auto zi = c / 4, ri = c % 4;
                auto zo = nc / 4, ro = nc % 4;
                nh = 0;
                for (int h = begins[2]; h < ends[2]; h += strides[2], nh++) {
                    nw = 0;
                    for (int w = begins[3]; w < ends[3]; w += strides[3], nw++) {
                        const int i_offset = zi * input_width * input_height * 4 + h * input_width * 4 + w * 4 + ri;
                        const int o_offset =
                            zo * output_width * output_height * 4 + nh * output_width * 4 + nw * 4 + ro;
                        output_ptr[o_offset] = input_ptr[i_offset];
                    }
                }
            }
        }
    } else {
        return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR int8/bfp16 StrideSlice, in todo list");
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(StrideSlice, LAYER_STRIDED_SLICE)

}  // namespace TNN_NS
