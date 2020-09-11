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
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER(PriorBox, LAYER_PRIOR_BOX)

Status NpuPriorBoxLayer::Convert() {
    auto param = dynamic_cast<PriorBoxLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    int img_height = param->img_h;
    int img_width  = param->img_w;
    if (img_height == 0 || img_width == 0) {
        img_height = input_ops_[1]->GetShape()[2];
        img_width  = input_ops_[1]->GetShape()[3];
    }
    auto output = std::make_shared<hiai::op::PriorBox>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_attr_min_size(param->min_sizes);
    output->set_attr_max_size(param->max_sizes);
    output->set_attr_aspect_ratio(param->aspect_ratios);

    output->set_attr_flip(param->flip);
    output->set_attr_clip(param->clip);
    output->set_attr_variance(param->variances);
    output->set_attr_step_h(param->step_h);
    output->set_attr_step_w(param->step_w);
    output->set_attr_offset(param->offset);
    output->set_attr_img_h(img_height);
    output->set_attr_img_w(img_width);

    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(PriorBox, LAYER_PRIOR_BOX)

}  // namespace TNN_NS