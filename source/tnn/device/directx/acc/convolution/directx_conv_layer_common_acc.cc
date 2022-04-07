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

#include "tnn/device/directx/acc/convolution/directx_conv_layer_common_acc.h"
#include "tnn/utils/string_utils_inner.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
namespace directx {

bool DirectXConvLayerCommonAcc::IsPrefered(const ConvLayerParam *param, const std::vector<Blob *> &,
                                          const std::vector<Blob *> &) {
    if (!param) {
        return false;
    }

    return true;
}

Status DirectXConvLayerCommonAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                      const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Conv Common Acc\n");

    Status ret = DirectXConvLayerAccImpl::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    conv_type_ = CT_CONV_COMMON;

    ret = AllocateWeightsBias(resource);
    RETURN_ON_NEQ(ret, TNN_OK);

    return TNN_OK;
}

DirectXConvLayerCommonAcc::~DirectXConvLayerCommonAcc() {}

Status DirectXConvLayerCommonAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("DirecctX Conv Common Acc Reshape\n");
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int output_height = DimsFunctionUtils::GetDim(output_dims, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims, 3);

    const int input_height   = DimsFunctionUtils::GetDim(input_dims, 2);
    const int input_width    = DimsFunctionUtils::GetDim(input_dims, 3);

    int input_imageshape[2]  = {input_width, input_height};
    int output_imageshape[2] = {output_width, output_height};
    int kernel_shape[2]      = {conv_params_.kernel_w, conv_params_.kernel_h};
    int stride_shape[2]      = {conv_params_.stride_w, conv_params_.stride_h};
    int padding_shape[2]     = {conv_params_.pad_w, conv_params_.pad_h};
    int dilation_shape[2]    = {conv_params_.dilation_w, conv_params_.dilation_h};

    return TNN_OK;
}

Status DirectXConvLayerCommonAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

}  // namespace directx
}  // namespace TNN_NS
