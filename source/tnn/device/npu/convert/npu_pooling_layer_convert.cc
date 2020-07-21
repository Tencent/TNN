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

#include "graph/attr_value.h"
#include "graph/op/nn_defs.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER(Pool, LAYER_POOLING)

Status NpuPoolLayer::Convert() {
    // parameter and weight of the pooling layer
    auto param = dynamic_cast<PoolingLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    // max pooling type 0; other avg
    int pool_mode = param->pool_type;
    if (pool_mode != 0) {
        pool_mode = 1;
    }
    int pad_mode = 0;
    Status ret   = NpuUtils::GetPadMode(pad_mode, param->pad_type);
    if (ret != TNN_OK)
        return ret;

    int stride_w    = param->strides[0];
    int stride_h    = param->strides[1];
    int pad_w_begin = param->pads[0];
    int pad_w_end   = param->pads[1];
    int pad_h_begin = param->pads[2];
    int pad_h_end   = param->pads[3];
    int kernel_w    = param->kernels[0];
    int kernel_h    = param->kernels[1];

    auto output = std::make_shared<ge::op::Pooling>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_attr_mode(pool_mode);
    if (kernel_h == 0 || kernel_w == 0) {
        output->set_attr_global_pooling(ge::AttrValue::BOOL{true});
    } else {
        output->set_attr_window(ge::AttrValue::LIST_INT({kernel_h, kernel_w}));
    }
    output->set_attr_pad_mode(pad_mode);
    output->set_attr_pad(ge::AttrValue::LIST_INT({pad_h_begin, pad_h_end, pad_w_begin, pad_w_end}));
    output->set_attr_stride(ge::AttrValue::LIST_INT({stride_h, stride_w}));
    output->set_attr_ceil_mode(0);
    output->set_attr_data_mode(1);
    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(Pool, LAYER_POOLING)

}  // namespace TNN_NS
