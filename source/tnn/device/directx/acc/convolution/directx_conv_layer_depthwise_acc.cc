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

#include "tnn/device/directx/acc/convolution/directx_conv_layer_depthwise_acc.h"

namespace TNN_NS {
namespace directx {

bool DirectXConvLayerDepthwiseAcc::IsPrefered(const ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                             const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    return param->group == DimsFunctionUtils::GetDim(inputs[0]->GetBlobDesc().dims, 1) &&
           param->group == DimsFunctionUtils::GetDim(outputs[0]->GetBlobDesc().dims, 1);
}

Status DirectXConvLayerDepthwiseAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                         const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Conv Depthwise Acc\n");

    conv_type_ = CT_CONV_DEPTHWISE;

    Status ret = DirectXConvLayerAccImpl::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    if (1 == conv_params_.stride_w && 1 == conv_params_.stride_h) {
        stride_is_1_ = true;
    }

    ret = AllocateWeightsBias(resource);
    RETURN_ON_NEQ(ret, TNN_OK);

    return CreateCB(inputs, outputs);
}

DirectXConvLayerDepthwiseAcc::~DirectXConvLayerDepthwiseAcc() {}

Status DirectXConvLayerDepthwiseAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Conv Depthwise Acc Reshape\n");
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int output_height = DimsFunctionUtils::GetDim(output_dims, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims, 3);

    const int input_height   = DimsFunctionUtils::GetDim(input_dims, 2);
    const int input_width    = DimsFunctionUtils::GetDim(input_dims, 3);
    const int input_channels = DimsFunctionUtils::GetDim(input_dims, 1);

    int input_imageshape[2]  = {input_width, input_height};
    int output_imageshape[2] = {output_width, output_height};
    int kernel_shape[2]      = {conv_params_.kernel_w, conv_params_.kernel_h};
    int stride_shape[2]      = {conv_params_.stride_w, conv_params_.stride_h};
    int padding_shape[2]     = {conv_params_.pad_w, conv_params_.pad_h};
    int dilation_shape[2]    = {conv_params_.dilation_w, conv_params_.dilation_h};

    return CreateCB(inputs, outputs);
}

Status DirectXConvLayerDepthwiseAcc::CreateCB(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto & in_dims = inputs[0]->GetBlobDesc().dims;
    auto & out_dims = outputs[0]->GetBlobDesc().dims;

    if (in_dims.size() < 4 || out_dims.size() < 4) {
        LOGE("Expect shape lenghts > 4 for input and output.\n");
        return Status(TNNERR_DX_LAYER_ERR, "Expect shape lenghts > 4 for input and output.");
    }
    typedef struct launch_param {
        DirectX::XMUINT4 in_shape;
        DirectX::XMUINT4 out_shape;
        DirectX::XMUINT4 kernel_wh;
        DirectX::XMUINT4 stride_wh;
        DirectX::XMUINT4 padding_wh;
        DirectX::XMUINT4 dilation_wh;
        DirectX::XMUINT4 activation_type;
    } launch_param_t;

    launch_param_t args;
    args.in_shape  = DirectX::XMUINT4(DimsFunctionUtils::GetDim(in_dims, 0), DimsFunctionUtils::GetDim(in_dims, 1),
                                      DimsFunctionUtils::GetDim(in_dims, 2), DimsFunctionUtils::GetDim(in_dims, 3));
    args.out_shape = DirectX::XMUINT4(DimsFunctionUtils::GetDim(out_dims, 0), DimsFunctionUtils::GetDim(out_dims, 1),
                                      DimsFunctionUtils::GetDim(out_dims, 2), DimsFunctionUtils::GetDim(out_dims, 3));
    args.kernel_wh = DirectX::XMUINT4(conv_params_.kernel_w, conv_params_.kernel_h, 0, 0);
    args.stride_wh = DirectX::XMUINT4(conv_params_.stride_w, conv_params_.stride_h, 0, 0);
    args.padding_wh = DirectX::XMUINT4(conv_params_.pad_w, conv_params_.pad_h, 0, 0);
    args.dilation_wh = DirectX::XMUINT4(conv_params_.dilation_w, conv_params_.dilation_h, 0, 0);
    args.activation_type = DirectX::XMUINT4(conv_params_.activation_type, 0,0,0);

    return CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer_);
}


Status DirectXConvLayerDepthwiseAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    std::shared_ptr<DirectXMemory> in_memory, out_memory;
    RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(inputs[0], in_memory), TNN_OK);
    RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(outputs[0], out_memory), TNN_OK);

    auto in_srv = in_memory->GetSRV();
    auto weight_srv = weights_->GetSRV();
    auto bias_srv = bias_->GetSRV();
    auto out_uav = out_memory->GetUAV();

    std::string kernel_name;
    auto &out_dims = outputs[0]->GetBlobDesc().dims;
    Status ret;

    int image_width;
    int image_height;

    if (stride_is_1_) {
        kernel_name = "conv_depthwise_s1_texture";
    } else {
        kernel_name = "conv_depthwise_texture";
    }

    image_width = UP_DIV(DimsFunctionUtils::GetDim(out_dims, 1), 4) * UP_DIV(DimsFunctionUtils::GetDim(out_dims, 3), 4);
    image_height = DimsFunctionUtils::GetDim(out_dims, 0) * DimsFunctionUtils::GetDim(out_dims, 2);

//    LOGD("kernel name: %s\n",kernel_name.c_str());
    std::shared_ptr<ID3D11ComputeShader> cs;
    ret = GetShaderByName(kernel_name, cs);
    RETURN_ON_NEQ(ret, TNN_OK);

    ret = DispatchShader(cs, {in_srv, weight_srv, bias_srv}, {out_uav}, {const_buffer_.get()}, {image_width, image_height, 1});

    return ret;
}

}  // namespace directx
}  // namespace TNN_NS
