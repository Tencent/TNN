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
#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_X86_ACC(StrideSlice, LAYER_STRIDED_SLICE);

Status X86StrideSliceLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<StrideSliceLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: StrideSliceLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParam is nil");
    }

    Blob *input_blob   = inputs[0];
    Blob *output_blob  = outputs[0];
    auto dims_input    = input_blob->GetBlobDesc().dims;
    auto dims_output   = output_blob->GetBlobDesc().dims;

    if (dims_output.size() > 4 || dims_input.size() > 4) {
        LOGE("Error: x86 stride slice only support dimension <= 4\n");
        return Status(TNNERR_MODEL_ERR, "Error: x86 stride slice only support dimension <= 4");
    }

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
    
    DimsVector input_strides;
    DimsVector output_strides;
    input_strides.reserve(dims_output.size());
    output_strides.reserve(dims_output.size());

    for (int i = 0; i < dims_output.size() - 1; i++) {
        input_strides.push_back(DimsVectorUtils::Count(dims_input, i + 1));
        output_strides.push_back(DimsVectorUtils::Count(dims_output, i + 1));
    }

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = handle_ptr<float *>(input_blob->GetHandle());
        float *output_data = handle_ptr<float *>(output_blob->GetHandle());

        X86StrideSliceImpl(begins, strides, dims_output, input_strides, output_strides, input_data, output_data);
    } else {
        return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR int8/bfp16 StrideSlice");
    }
    return TNN_OK;
}

REGISTER_X86_ACC(StrideSlice, LAYER_STRIDED_SLICE)

}  // namespace TNN_NS
