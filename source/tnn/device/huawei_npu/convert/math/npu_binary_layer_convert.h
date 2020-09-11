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

#ifndef TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_MATH_NPU_BINARY_LAYER_CONVERT_H_
#define TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_MATH_NPU_BINARY_LAYER_CONVERT_H_
#include <tnn/device/huawei_npu/convert/npu_base_layer_convert.h>
#include <tnn/device/huawei_npu/convert/npu_utils.h>
#include <tnn/layer/base_layer.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "graph/attr_value.h"
#include "graph/op/nn_defs.h"
#include "tnn/core/abstract_device.h"
#include "tnn/core/blob.h"
#include "tnn/core/context.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"

namespace TNN_NS {

class NpuBinaryLayer : public NpuBaseLayer {
public:
    NpuBinaryLayer(LayerType layer_type) : NpuBaseLayer(layer_type){};
    virtual ~NpuBinaryLayer() {}

protected:
    template <class T>
    Status BinaryConvert() {
        auto param    = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
        auto resource = dynamic_cast<EltwiseLayerResource *>(resource_);
        CHECK_PARAM_NULL(param);

        int input_size = input_ops_.size();
        if (!((input_size == 1 && resource) || input_size == 2)) {
            return Status(TNNERR_LAYER_ERR, "Error: the Binary layer input number is not correct");
        }

        auto output = std::make_shared<T>(outputs_name_[0]);
        if (input_size == 2) {
            output->set_input_x1(*input_ops_[0]->GetOperator());
            output->set_input_x2(*input_ops_[1]->GetOperator());
        } else {
            auto weight_const             = std::make_shared<ge::op::Const>(layer_name_ + "_weight");
            std::vector<int> weight_shape = resource->element_shape;
            std::vector<int> input_shape  = input_ops_[0]->GetShape();
            Status calculate_ret = NpuUtils::CalculateBroadcastSize(weight_shape, resource, input_shape);
            if (calculate_ret != TNN_OK) {
                return calculate_ret;
            }
            ge::Shape weight_shape_op({weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]});
            NpuUtils::CreateAttrValue(weight_const, weight_shape_op, resource->element_handle);
            weight_ops_.push_back(weight_const);

            if (param->weight_input_index == 0) {
                // weight const
                output->set_input_x1(*weight_const);
                output->set_input_x2(*input_ops_[0]->GetOperator());
            } else {
                // make input as the sub
                output->set_input_x1(*input_ops_[0]->GetOperator());
                output->set_input_x2(*weight_const);
            }
        }
        ADD_OUTPUT_OP(output)
    }

private:
    std::vector<shared_ptr<ge::Operator>> weight_ops_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_MATH_NPU_BINARY_LAYER_CONVERT_H_
