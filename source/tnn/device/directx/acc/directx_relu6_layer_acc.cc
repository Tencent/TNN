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

#include "tnn/device/directx/acc/directx_unary_layer_acc.h"

namespace TNN_NS {

namespace directx {

DECLARE_DIRECTX_UNARY_ACC(Relu6);

Status DirectXRelu6LayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Relu Acc\n");
    Status ret = DirectXUnaryLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    kernel_name_ = "unary_op_relu6";

    return TNN_OK;
}

DirectXRelu6LayerAcc::~DirectXRelu6LayerAcc() {}

REGISTER_DIRECTX_ACC(Relu6, LAYER_RELU6)
REGISTER_DIRECTX_LAYOUT(LAYER_RELU6, DATA_FORMAT_NHC4W4);

} // namespace directx

}  // namespace TNN_NS
