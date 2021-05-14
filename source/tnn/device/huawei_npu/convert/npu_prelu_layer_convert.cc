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

DECLARE_NPU_LAYER_WEIGHT(Prelu, LAYER_PRELU)

Status NpuPreluLayer::Convert() {
    auto param = dynamic_cast<PReluLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    auto resource = dynamic_cast<PReluLayerResource *>(resource_);
    CHECK_PARAM_NULL(resource);

    // check slope
    bool channel_shared = param->channel_shared;
    if (!channel_shared) {
        const float *slope_data = resource->slope_handle.force_to<float *>();
        float temp_data         = slope_data[0];
        bool val_is_same        = true;
        for (int i = 1; i < resource->slope_handle.GetDataCount(); ++i) {
            if (temp_data != slope_data[i]) {
                val_is_same = false;
                break;
            }
        }
        if (val_is_same) {
            channel_shared = true;
        }
    }

    if ((!channel_shared) && NpuUtils::VersionCompare(npu_version_, "100.320.010.023", VCT_BIGEQUAL)) {
        // use hiai::op::PRelu
        auto output = std::make_shared<hiai::op::PRelu>(outputs_name_[0]);

        std::shared_ptr<ge::op::Const> slope_const = std::make_shared<ge::op::Const>(layer_name_ + "_slope");
        ge::Shape data_shape({1, (int64_t)resource->slope_handle.GetDataCount(), 1, 1});
        NpuUtils::CreateAttrValue(slope_const, data_shape, resource->slope_handle);
        weight_ops_.push_back(slope_const);

        output->set_input_weight(*slope_const);
        output->set_input_x(*input_ops_[0]->GetOperator());

        ADD_OUTPUT_OP(output)
    } else {
        // use hiai::op::Activation
        auto output = std::make_shared<hiai::op::Activation>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        const float *slope_data = resource->slope_handle.force_to<float *>();
        if (channel_shared) {
            // if channel shared
            output->set_attr_negative_slope(slope_data[0]);
        } else {
            LOGE("Error: huawei_npu currently only supports shared-channel prelu in this rom version (%s)\n",
                 npu_version_.c_str());
            return Status(TNNERR_LAYER_ERR,
                          "Error: huawei_npu currently only supports shared-channel prelu in this rom version");
        }
        output->set_attr_mode(5);
        ADD_OUTPUT_OP(output)
    }
}

REGISTER_NPU_LAYER(Prelu, LAYER_PRELU)

}  // namespace TNN_NS
