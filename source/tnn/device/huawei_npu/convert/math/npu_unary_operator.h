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

#ifndef TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_MATH_NPU_UNARY_OPERATOR_H_
#define TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_MATH_NPU_UNARY_OPERATOR_H_

#include <tnn/core/layer_type.h>
#include <tnn/device/huawei_npu/convert/npu_base_layer_convert.h>

namespace TNN_NS {

class NpuUnaryLayer : public NpuBaseLayer {
public:
    NpuUnaryLayer(LayerType layer_type) : NpuBaseLayer(layer_type){};
    virtual ~NpuUnaryLayer() {}

protected:
    template <class T>
    Status UnaryConvert() {
        int input_size = input_ops_.size();
        if (input_size >= 2) {
            printf("the Unary input size is not correct\n");
            return Status(TNNERR_PARAM_ERR, "Error: the Unary layer count is not correct");
        }
        auto output = std::make_shared<T>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        ADD_OUTPUT_OP(output)
    }
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_MATH_NPU_UNARY_OPERATOR_H_
