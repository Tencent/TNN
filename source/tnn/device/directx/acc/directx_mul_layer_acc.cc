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

#include "tnn/device/directx/acc/directx_binary_layer_acc.h"

namespace TNN_NS {

namespace directx {

DECLARE_DIRECTX_BINARY_ACC(Mul);

Status DirectXMulLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Mul Acc\n");
    Status ret = DirectXBinaryLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    if (use_buffer_) {
        kernel_name_ = "binary_op_mul";
    } else {
        kernel_name_ = "binary_op_texture_mul";
    }

    return TNN_OK;
}

DirectXMulLayerAcc::~DirectXMulLayerAcc() {}

REGISTER_DIRECTX_ACC(Mul, LAYER_MUL)
REGISTER_DIRECTX_LAYOUT(LAYER_MUL, DATA_FORMAT_NHC4W4);

} // namespace directx

}  // namespace TNN_NS
