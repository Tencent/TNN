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

#include "tnn/device/x86/acc/x86_arg_max_or_min_layer_acc.h"

namespace TNN_NS {

Status X86ArgMaxOrMinLayerAcc::Init(Context *context, LayerParam *param, LayerResource* resource, const std::vector<Blob*> &inputs,
                                    const std::vector<Blob *> &outputs) {
    auto param_list = dynamic_cast<ArgMaxOrMinLayerParam *>(param);
    auto input_dims = inputs[0]->GetBlobDesc().dims;

    num_        = DimsVectorUtils::Count(input_dims, 0, param_list->axis);
    channels_   = input_dims[param_list->axis];
    stride_     = DimsVectorUtils::Count(input_dims, param_list->axis + 1);
    stride_     = stride_ == 0 ? 1 : stride_;

    if (param_list->mode == 0) op_ = std::make_shared<X86_ARG_MIN_OP>();
    else                       op_ = std::make_shared<X86_ARG_MAX_OP>();

    return TNN_OK;
}

Status X86ArgMaxOrMinLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status X86ArgMaxOrMinLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    auto input_blob     = inputs[0];
    auto output_blob    = outputs[0];

    // dimension：num,axis,stride -> num, 1, stride
    // SIMD: 1 stride ->  8-16 stride
    // maintain " tmp, val, index "
    // val = max(val, cur), mask = dif(tmp, val), index = select_mask(mask, cur_index, index)
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_ptr = static_cast<float *>(input_blob->GetHandle().base);
        float *output_ptr = static_cast<float *>(output_blob->GetHandle().base);
        for (int n = 0; n < num_; n++) {
            for (int s = 0; s < stride_; s++) {
                int guard_index = 0;
                int ptr_base = n * stride_ * channels_ + s;
                float gurad_value = 0.0;
                op_->Init();
                for (int c = 0; c < channels_; c++) {
                    (*op_)(c, input_ptr[ptr_base]);
                    ptr_base += stride_;
                }
                output_ptr[n * stride_ + s] = (float)op_->get_idx();
            }
        }
    } else {
        LOGE("Error : layer acc dont support datatype : %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_X86_ACC(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

} // namespace TNN_NS