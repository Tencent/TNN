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

#include "tnn/device/directx/acc/convolution/directx_conv_layer_1x1_acc.h"

#include <directxpackedvector.h>

#include "tnn/core/macro.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
namespace directx {

bool DirectXConvLayer1x1Acc::IsPrefered(const ConvLayerParam *param, const std::vector<Blob *> &,
                                       const std::vector<Blob *> &) {
    if (!param) {
        return false;
    }
    return param->group == 1 && param->kernels[0] == 1 && param->kernels[1] == 1 && param->dialations[0] == 1 && 
           param->dialations[1] == 1 && param->pads[0] == 0 && param->pads[1] == 0;
}

Status DirectXConvLayer1x1Acc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                   const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Conv 1x1 Acc\n");

    conv_type_ = CT_CONV_1x1;

    // AccImpl init first
    Status ret = DirectXConvLayerAccImpl::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    if (1 == conv_params_.stride_w && 1 == conv_params_.stride_h) {
        stride_is_1_ = true;
    }

    if (!stride_is_1_) {
        LOGE("dx conv1x1 not supports stride other than 1.");
        return Status(TNNERR_DX_ACC_INIT_ERR, "dx conv1x1 not supports stride other than 1.");
    }

    ret = AllocateWeightsBias(resource);
    RETURN_ON_NEQ(ret, TNN_OK);

    return CreateCB(inputs, outputs);
}

DirectXConvLayer1x1Acc::~DirectXConvLayer1x1Acc() {}

Status DirectXConvLayer1x1Acc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Conv 1x1 Acc Reshape\n");

    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int input_channel_blocks = UP_DIV(DimsFunctionUtils::GetDim(input_dims, 1), 4);

    const int output_channels = DimsFunctionUtils::GetDim(output_dims, 1);
    const int output_channel_blocks = UP_DIV(output_channels, 4);

    return CreateCB(inputs, outputs);
}

Status DirectXConvLayer1x1Acc::CreateCB(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto & in_dims = inputs[0]->GetBlobDesc().dims;
    auto & out_dims = outputs[0]->GetBlobDesc().dims;

    if (in_dims.size() < 4 || out_dims.size() < 4) {
        LOGE("Expect shape lenghts > 4 for input and output.\n");
        return Status(TNNERR_DX_LAYER_ERR, "Expect shape lenghts > 4 for input and output.");
    }
    typedef struct launch_param {
        DirectX::XMUINT4 in_shape;
        DirectX::XMUINT4 out_shape;
        DirectX::XMUINT4 stride_wh;
        DirectX::XMUINT4 fused_relu;
    } launch_param_t;

    launch_param_t args;
    args.in_shape  = DirectX::XMUINT4(DimsFunctionUtils::GetDim(in_dims, 0), DimsFunctionUtils::GetDim(in_dims, 1),
                                      DimsFunctionUtils::GetDim(in_dims, 2), DimsFunctionUtils::GetDim(in_dims, 3));
    args.out_shape = DirectX::XMUINT4(DimsFunctionUtils::GetDim(out_dims, 0), DimsFunctionUtils::GetDim(out_dims, 1),
                                      DimsFunctionUtils::GetDim(out_dims, 2), DimsFunctionUtils::GetDim(out_dims, 3));
    args.stride_wh = DirectX::XMUINT4(conv_params_.stride_w, conv_params_.stride_h, 0, 0);
    args.fused_relu= DirectX::XMUINT4(conv_params_.activation_type, 0, 0 ,0);

    return CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer_);
}

Status DirectXConvLayer1x1Acc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {


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

    if (use_buffer_) {
        int BLOCK_A;

        auto out_c = DimsFunctionUtils::GetDim(outputs[0]->GetBlobDesc().dims, 1);
        if (out_c <= 16) {
            BLOCK_A = 128;
            kernel_name = "conv1x1_128x16";
        } else if (out_c <= 64) {
            BLOCK_A = 64;
            kernel_name = "conv1x1_64x32";
        } else {
            BLOCK_A = 32;
            kernel_name = "conv1x1_32x64";
        }

        std::shared_ptr<ID3D11ComputeShader> cs;
        ret = GetShaderByName(kernel_name, cs);
        RETURN_ON_NEQ(ret, TNN_OK);

        auto &in_dims = inputs[0]->GetBlobDesc().dims;
        const int NHW = DimsVectorUtils::Count(in_dims) / DimsFunctionUtils::GetDim(in_dims, 1);

        ret = DispatchShader(cs, {in_srv, weight_srv, bias_srv}, {out_uav}, {const_buffer_.get()}, {UP_DIV(NHW, BLOCK_A)});
    } else {
        int image_width;
        int image_height;

        if (stride_is_1_) {
            kernel_name = "conv1x1_s1_texture";
        } else {
            kernel_name = "conv1x1_texture";
        }

        image_width = UP_DIV(DimsFunctionUtils::GetDim(out_dims, 1), 4) * UP_DIV(DimsFunctionUtils::GetDim(out_dims, 3), 4);
        image_height = DimsFunctionUtils::GetDim(out_dims, 0) * DimsFunctionUtils::GetDim(out_dims, 2);

//        LOGD("kernel name: %s\n",kernel_name.c_str());
        std::shared_ptr<ID3D11ComputeShader> cs;
        ret = GetShaderByName(kernel_name, cs);
        RETURN_ON_NEQ(ret, TNN_OK);

        ret = DispatchShader(cs, {in_srv, weight_srv, bias_srv}, {out_uav}, {const_buffer_.get()}, {image_width, image_height, 1});

    }

    return ret;
}

}  // namespace directx
}  // namespace TNN_NS
