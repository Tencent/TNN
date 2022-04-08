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

#include <directxpackedvector.h>

#include "tnn/utils/string_utils_inner.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/device/directx/directx_memory.h"

namespace TNN_NS {
namespace directx {

bool DirectXConvLayerCommonAcc::IsPrefered(const ConvLayerParam *param, const std::vector<Blob *> &,
                                          const std::vector<Blob *> &) {
    if (!param) {
        return false;
    }

    return param->pads[0] == param->pads[2] && param->strides[0] == param->strides[1] && 
            param->dialations[0] == 1 && param->dialations[1] == 1;
}

Status DirectXConvLayerCommonAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                      const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Conv Common Acc\n");

    Status ret = DirectXConvLayerAccImpl::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    conv_type_ = CT_CONV_COMMON;

    ret = AllocateWeightsBias(resource);
    RETURN_ON_NEQ(ret, TNN_OK);

    return PreCompute(inputs, outputs);
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

    return PreCompute(inputs, outputs);
}


void precompute_warp_offsets(int *warp_offsets, int loop_k, ConvParam param, DimsVector input_dims) {

    auto &GetDim = DimsFunctionUtils::GetDim;

    int cstride = GetDim(input_dims, 2) * GetDim(input_dims, 3);
    for( int c = 0, i = 0 ; c < param.input_channel ; ++c ) {
        for( int r = 0 ; r < param.kernel_h; ++r ) {
            for( int s = 0 ; s < param.kernel_w ; ++s, ++i ) {
                if( i >= loop_k ) 
                    return;
                warp_offsets[i] = c*cstride + r* GetDim(input_dims, 3) + s;
            }
        }
    }
}

void precompute_filter_offsets(int * filter_offsets,const int loop_k, ConvParam param, DimsVector input_dims) {
    auto &GetDim = DimsFunctionUtils::GetDim;

    int rs = param.kernel_h * param.kernel_w;
    int cstride = GetDim(input_dims, 2) * GetDim(input_dims, 3);

    int num_offsets = std::max(loop_k, rs);

    for( int i = 0 ; i < num_offsets; ++i ) {
        int c, t, r, s;
        c = i / rs;
        t = i % rs;

        r = t / param.kernel_w;
        s = t % param.kernel_w;

        int new_i = i + loop_k;
        int new_c, new_t, new_r, new_s;

        new_c = new_i / rs;
        new_t = new_i % rs;

        new_r = new_t / param.kernel_w;
        new_s = new_t % param.kernel_w;

        int c_diff = new_c - c;
        int r_diff = new_r - r;
        int s_diff = new_s - s;

        int offset = c_diff*cstride + r_diff*GetDim(input_dims, 3) + s_diff;
        filter_offsets[i] = offset;
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
void precompute_ptr_offsets(
    int * edge_masks,
    int * ptr_offsets,  ConvParam param, DimsVector input_dims, DimsVector output_dims)
{
    auto &GetDim = DimsFunctionUtils::GetDim;

    const int R = param.kernel_h;
    const int S = param.kernel_w;
    const int IH = GetDim(input_dims, 2);
    const int IW = GetDim(input_dims, 3);
    const int OH = GetDim(output_dims, 2);
    const int OW = GetDim(output_dims, 3);

    const int pad = param.pad_h;
    const int stride = param.stride_h;
    const int h_stride = GetDim(input_dims, 3);

    for(int h=0;h<OH;h++){
        for(int w=0;w<OW;w++){

            int offset = (h*stride - pad) * h_stride + w * stride - pad;
            ptr_offsets[h*OW+ w] = offset;

            const int rStart =  - pad;
            const int sStart =  - pad; 

            int m = 0;

            for( int r = 0 ; r < R; ++r ) {
                for( int s = 0 ; s < S; ++s ) {
                    int ih = h*stride + rStart + r;
                    int iw = w*stride + sStart + s;
                    
                    int in_image = ih >= 0 && ih < IH && w >= 0 && w < IW;
                    int flag = in_image ? 1u : 0u;
                    m = m | (flag << (r*R + s));
                }
            }

            edge_masks[h * OW + w] = m;

        }
    }
}

Status DirectXConvLayerCommonAcc::PreCompute(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto & in_dims = inputs[0]->GetBlobDesc().dims;
    auto & out_dims = outputs[0]->GetBlobDesc().dims;

    if (in_dims.size() < 4 || out_dims.size() < 4) {
        LOGE("Expect shape lenghts > 4 for input and output.\n");
        return Status(TNNERR_DX_LAYER_ERR, "Expect shape lenghts > 4 for input and output.");
    }

    int oh = DimsFunctionUtils::GetDim(out_dims, 2);
    int ow = DimsFunctionUtils::GetDim(out_dims, 3);
    int THREADS_PER_BLOCK = 64;
    int LOOP_K = 4;
    int elements = ROUND_UP(oh * ow, 64);

    std::shared_ptr<int> ptr_offset = std::shared_ptr<int>(new int[elements], [](void *p){delete[] p;});
    std::shared_ptr<int> mask = std::shared_ptr<int>(new int[elements], [](void *p){delete[] p;});
    std::shared_ptr<int> filter_offset = std::shared_ptr<int>(new int[9], [](void *p){delete[] p;});
    std::shared_ptr<int> warp_offset = std::shared_ptr<int>(new int[4], [](void *p){delete[] p;});

    precompute_filter_offsets(filter_offset.get(), LOOP_K, conv_params_, in_dims);
    precompute_warp_offsets(warp_offset.get(), LOOP_K, conv_params_, in_dims);
    precompute_ptr_offsets(mask.get(), ptr_offset.get(), conv_params_, in_dims, out_dims);

    ptr_offset_ = DirectXMemory::CreateBufferMemoryFromHost(ptr_offset.get(), {elements}, DATA_TYPE_INT32, DATA_FORMAT_NCHW);
    mask_ = DirectXMemory::CreateBufferMemoryFromHost(mask.get(), {elements}, DATA_TYPE_INT32, DATA_FORMAT_NCHW);
    filter_offset_ = DirectXMemory::CreateBufferMemoryFromHost(filter_offset.get(), {9}, DATA_TYPE_INT32, DATA_FORMAT_NCHW);
    warp_offset_ = DirectXMemory::CreateBufferMemoryFromHost(warp_offset.get(), {4}, DATA_TYPE_INT32, DATA_FORMAT_NCHW);

    if (!ptr_offset_ || !mask_ || !filter_offset_ || !warp_offset_) {
        LOGE("Create dx buffer from host failed.\n");
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "Create dx buffer from host failed.");
    }

    typedef struct launch_param {
        DirectX::XMUINT4 in_shape;
        DirectX::XMUINT4 out_shape;
        DirectX::XMUINT4 filter_kcrs;
        DirectX::XMUINT4 fused_relu;
    } launch_param_t;

    launch_param_t args;
    args.in_shape  = DirectX::XMUINT4(in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
    args.out_shape = DirectX::XMUINT4(out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
    args.filter_kcrs = DirectX::XMUINT4(out_dims[1], in_dims[1], conv_params_.kernel_h, conv_params_.kernel_w);
    args.fused_relu = DirectX::XMUINT4(0,0,0,0);

    return CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer_);
}
Status DirectXConvLayerCommonAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    auto in_memory = DirectXMemory::CreateRefMemoryFromBlob(inputs[0]); 
    auto out_memory = DirectXMemory::CreateRefMemoryFromBlob(outputs[0]); 

    auto in_srv = in_memory->GetSRV();
    auto weight_srv = weights_->GetSRV();
    auto bias_srv = bias_->GetSRV();
    auto ptr_of_srv = ptr_offset_->GetSRV();
    auto mask_srv = mask_->GetSRV();
    auto filter_of_srv = filter_offset_->GetSRV();
    auto warp_of_srv = warp_offset_->GetSRV();

    auto out_uav = out_memory->GetUAV();

    int BLOCK_A;
    int BLOCK_B;
    std::string kernel_name;

    BLOCK_A = 64;
    BLOCK_B = 64;
    kernel_name = "conv";


    std::shared_ptr<ID3D11ComputeShader> cs;
    Status ret = GetShaderByName(kernel_name, cs);
    RETURN_ON_NEQ(ret, TNN_OK);

    auto &out_dims = outputs[0]->GetBlobDesc().dims;
    const int OHW = DimsFunctionUtils::GetDim(out_dims, 2) * DimsFunctionUtils::GetDim(out_dims, 3); 
    const int OC = DimsFunctionUtils::GetDim(out_dims, 1);

    ret = DispatchShader(cs, {in_srv, weight_srv, bias_srv, ptr_of_srv, mask_srv, filter_of_srv, warp_of_srv}, 
                             {out_uav}, {const_buffer_.get()}, {UP_DIV(OHW, BLOCK_A), UP_DIV(OC, BLOCK_B)});

    return ret;
}

}  // namespace directx
}  // namespace TNN_NS
