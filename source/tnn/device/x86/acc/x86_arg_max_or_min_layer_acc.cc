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
#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"
#include <iostream>

namespace TNN_NS {
DECLARE_X86_ACC(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

Status X86ArgMaxOrMinLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param       = dynamic_cast<ArgMaxOrMinLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];
    auto input_dims  = input_blob->GetBlobDesc().dims;
    int axis         = param->axis;
    int num          = DimsVectorUtils::Count(input_dims, 0, axis);
    int channels     = input_dims[axis];
    int stride       = DimsVectorUtils::Count(input_dims, axis + 1);
    if (stride == 0) {
        stride = 1;
    }
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT32) {
        auto input_ptr  = handle_ptr<float *>(input_blob->GetHandle());
        auto output_ptr = handle_ptr<int *>(output_blob->GetHandle());
        for (int n = 0; n < num; ++n) {
            for (int s = 0; s < stride; ++s) {
                int guard_index = 0;
                for (int c = 1; c < channels; ++c) {
                    float guard_value = input_ptr[n * stride * channels + guard_index * stride + s];
                    float cur_value   = input_ptr[n * stride * channels + c * stride + s];
                    if (param->mode == 0) {
                        // ArgMin
                        guard_index = cur_value < guard_value ? c : guard_index;
                    } else {
                        // ArgMax
                        guard_index = cur_value > guard_value ? c : guard_index;
                    }
                };  // end for loop
                output_ptr[n * stride + s] = guard_index;
                // std::cout << output_ptr[n * stride + s] << " ";
            }
        }  // end for
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }  // end if
    return TNN_OK;
}

REGISTER_X86_ACC(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

}  // namespace TNN_NS
