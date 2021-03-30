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

DECLARE_NPU_LAYER_WEIGHT(Gather, LAYER_GATHER)

Status NpuGatherLayer::Convert() {
    auto param = dynamic_cast<GatherLayerParam*>(param_);
    CHECK_PARAM_NULL(param);
    auto resource = dynamic_cast<GatherLayerResource*>(resource_);
    if ((param->data_in_resource || param->indices_in_resource) && !resource) {
        return Status(TNNERR_MODEL_ERR, "Gather resource is invalid");
    }

    auto output = std::make_shared<hiai::op::GatherV2D>(outputs_name_[0]);

    // set data
    if (param->data_in_resource) {
        DimsVector data_dims                      = resource->data.GetBufferDims();
        std::shared_ptr<ge::op::Const> data_const = std::make_shared<ge::op::Const>(layer_name_ + "_data");
        ge::Shape data_shape(NpuUtils::Int32VecToTVec<int64_t>(data_dims));
        NpuUtils::CreateAttrValue(data_const, data_shape, resource->data);
        weight_ops_.push_back(data_const);
        output->set_input_x(*data_const);
    } else {
        output->set_input_x(*input_ops_[0]->GetOperator());
    }

    // set indices
    if (param->indices_in_resource) {
        DimsVector indices_dims = resource->indices.GetBufferDims();
        int length              = resource->indices.GetBytesSize();

        std::shared_ptr<ge::op::Const> indices_const = std::make_shared<ge::op::Const>(layer_name_ + "_indices");
        if (indices_dims.size() == 0 && length == 4) {
            std::vector<int> vec = {resource->indices.force_to<int*>()[0]};
            ge::TensorDesc const_desc(ge::Shape(), ge::FORMAT_NCHW, ge::DT_INT32);
            NpuUtils::CreateAttrArray(indices_const, vec, const_desc, 1);
        } else {
            ge::Shape indices_shape(NpuUtils::Int32VecToTVec<int64_t>(indices_dims));
            NpuUtils::CreateAttrValue(indices_const, indices_shape, resource->indices);
        }
        weight_ops_.push_back(indices_const);
        output->set_input_indices(*indices_const);
    } else {
        if (input_ops_.size() < 2) {
            LOGE("gather layer don't have indics resource\n");
            return Status(TNNERR_MODEL_ERR, "gather layer don't have indics resource\n");
        }
        output->set_input_indices(*input_ops_[1]->GetOperator());
    }

    // set axis
    output->set_attr_axis(param->axis);

    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(Gather, LAYER_GATHER)

}  // namespace TNN_NS
