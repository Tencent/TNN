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
//#include <graph/op/all_ops.h>
//#include "npu_binary_layer_convert.h"
//#include "tnn/device/npu/convert/npu_base_layer_convert.h"
//#include "tnn/device/npu/convert/npu_utils.h"
//
// namespace tnn {
// class NpuMulLayer : public NpuBinaryLayer {
// public:
//    NpuMulLayer(LayerType ignore) : NpuBinaryLayer(LAYER_MUL) {}
//    ~NpuMulLayer() {}
//
// protected:
//    Status Convert() {
//        return NpuBinaryLayer::BinaryConvert<ge::op::Mul>();
//    }
//};
// REGISTER_NPU_LAYER(Mul, LAYER_MUL);
//}  // namespace tnn

#include <graph/op/all_ops.h>
#include <tnn/core/status.h>
#include "tnn/device/npu/convert/npu_base_layer_convert.h"
#include "tnn/device/npu/convert/npu_utils.h"

namespace tnn {
DECLARE_NPU_LAYER_WEIGHT(Mul, LAYER_MUL);
Status NpuMulLayer::Convert() {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error:layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: the layer param is nil");
    }
    int input_size = input_ops_.size();
    if (input_size != 2) {
        printf("the layer nae is %s \n", layer_name_.c_str());
        printf("the mul input size is not correct\n");
        return Status(TNNERR_PARAM_ERR, "Error: the layer count is not correct");
    }
    auto output = std::make_shared<ge::op::Mul>(outputs_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_input_y(*input_ops_[1]->GetOperator());
    std::shared_ptr<OperatorInfo> output_op = std::make_shared<OperatorInfo>(output);
    output_ops_.push_back(output_op);
    return SetOutputOps();
}  // namespace tnn
REGISTER_NPU_LAYER(Mul, LAYER_MUL);
}  // namespace tnn