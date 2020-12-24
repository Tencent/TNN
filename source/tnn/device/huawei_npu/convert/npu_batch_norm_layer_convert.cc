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

#include <tnn/utils/data_type_utils.h>
#include "graph/attr_value.h"
#include "graph/op/nn_defs.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(BatchNorm, LAYER_BATCH_NORM)

Status InitBnVectorData(std::vector<float> &mean_data, std::vector<float> &variance_data,
                        std::vector<float> &scale_data, std::vector<float> &bias_data, float *scale_data_fp32,
                        float *bias_data_fp32, int channel, bool share_channel) {
    if (nullptr == scale_data_fp32) {
        return Status(TNNERR_NULL_PARAM, "scale data ptr is null");
    }
    for (int i = 0; i < channel; i++) {
        mean_data.push_back(0.0f);
        variance_data.push_back(1.0f);
        int index = i;
        if (share_channel) {
            index = 0;
        }
        scale_data.push_back(scale_data_fp32[index]);
        if (nullptr == bias_data_fp32) {
            bias_data.push_back(0.0f);
        } else {
            bias_data.push_back(bias_data_fp32[index]);
        }
    }
    return TNN_OK;
}

Status NpuBatchNormLayer::Convert() {
    auto resource = dynamic_cast<BatchNormLayerResource *>(resource_);
    if (!resource) {
        return Status(TNNERR_MODEL_ERR, "Error: BatchNorm layer resource is nil");
    }

    Status ret = TNN_OK;

    // channel is the second element of NCHW
    int channel = input_ops_[0]->GetShape()[1];
    bool share_channel =
        resource->scale_handle.GetBytesSize() == DataTypeUtils::GetBytesSize(resource->scale_handle.GetDataType());
    // fixed - set to be 0 and 1
    std::vector<float> mean_data;
    std::vector<float> variance_data;
    // here needs to consider float 16
    std::vector<float> scale_data;
    std::vector<float> bias_data;

    if (resource->scale_handle.GetDataType() != DATA_TYPE_FLOAT) {
        // if filter handle is half,it needs to be converted to float first.
        auto scale_data_fp32 = GetFloatFromRawBuffer(resource->scale_handle);
        auto bias_data_fp32  = GetFloatFromRawBuffer(resource->bias_handle);

        if (scale_data_fp32 == nullptr) {
            return Status(TNNERR_NPU_LOAD_ERROR, "In NPU, when convert to 16, pointer is null");
        }
        ret = InitBnVectorData(mean_data, variance_data, scale_data, bias_data, scale_data_fp32.get(),
                               bias_data_fp32.get(), channel, share_channel);
    } else {
        ret = InitBnVectorData(mean_data, variance_data, scale_data, bias_data,
                               resource->scale_handle.force_to<float *>(), resource->bias_handle.force_to<float *>(),
                               channel, share_channel);
    }
    RETURN_ON_NEQ(ret, TNN_OK);

    ge::Shape shape({channel});
    ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);

    auto mean_const = std::make_shared<ge::op::Const>(layer_name_ + "_mean");
    RETURN_ON_NEQ(NpuUtils::CreateAttrArray(mean_const, mean_data, desc, channel), TNN_OK);

    auto variance_const = std::make_shared<ge::op::Const>(layer_name_ + "_variance");
    RETURN_ON_NEQ(NpuUtils::CreateAttrArray(variance_const, variance_data, desc, channel), TNN_OK);

    auto scale_const = std::make_shared<ge::op::Const>(layer_name_ + "_scale");
    auto bias_const  = std::make_shared<ge::op::Const>(layer_name_ + "_bias");
    RETURN_ON_NEQ(NpuUtils::CreateAttrArray(scale_const, scale_data, desc, channel), TNN_OK);
    RETURN_ON_NEQ(NpuUtils::CreateAttrArray(bias_const, bias_data, desc, channel), TNN_OK);

    weight_ops_.push_back(mean_const);
    weight_ops_.push_back(variance_const);
    weight_ops_.push_back(scale_const);
    weight_ops_.push_back(bias_const);

    auto output = std::make_shared<ge::op::BatchNormExt2>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_input_variance(*variance_const);
    output->set_input_mean(*mean_const);
    output->set_input_scale(*scale_const);
    output->set_input_offset(*bias_const);
    output->set_attr_mode(1);
    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(BatchNorm, LAYER_BATCH_NORM)
REGISTER_NPU_LAYER(BatchNorm, LAYER_SCALE)

}  // namespace TNN_NS
