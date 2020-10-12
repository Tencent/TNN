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

#include <graph/op/all_ops.h>
#include <tnn/core/status.h>
#include "npu_binary_layer_convert.h"
#include "tnn/device/huawei_npu/convert/npu_base_layer_convert.h"
#include "tnn/device/huawei_npu/convert/npu_utils.h"
#include "tnn/utils/npu_common_utils.h"

namespace TNN_NS {

class NpuMulLayer : public NpuBinaryLayer {
public:
    NpuMulLayer(LayerType ignore) : NpuBinaryLayer(LAYER_MUL) {}
    ~NpuMulLayer() {}

protected:
    Status Convert() {
        auto param    = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
        auto resource = dynamic_cast<EltwiseLayerResource *>(resource_);
        CHECK_PARAM_NULL(param);

        int input_size = input_ops_.size();
        if (!((input_size == 1 && resource) || input_size == 2)) {
            return Status(TNNERR_MODEL_ERR, "Error: the Multiply layer input number is not correct");
        }

        auto output = std::make_shared<ge::op::Mul>(outputs_name_[0]);

        if (input_size == 2) {
            output->set_input_x(*input_ops_[0]->GetOperator());
            output->set_input_y(*input_ops_[1]->GetOperator());
        } else {
            std::shared_ptr<ge::op::Const> weight_const = nullptr;
            RETURN_ON_NEQ(GetBinaryWeight(weight_const), TNN_OK);

            output->set_input_x(*weight_const);
            output->set_input_y(*input_ops_[0]->GetOperator());
        }

        ADD_OUTPUT_OP(output);
    }
};

REGISTER_NPU_LAYER(Mul, LAYER_MUL)

}  // namespace TNN_NS
