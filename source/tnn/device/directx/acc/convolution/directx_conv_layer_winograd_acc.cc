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

#include "tnn/device/directx/acc/convolution/directx_conv_layer_winograd_acc.h"

#include "tnn/utils/winograd_generator.h"

namespace TNN_NS {
namespace directx {

#define UNIT 2

    bool DirectXConvLayerWinogradAcc::IsPrefered(const ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
        if (!param) {
            return false;
        }

        if (param->group != 1) {
            return false;
        }

        DirectXRuntime *directx_runtime = DirectXRuntime::GetInstance();
        auto texture_2d_max_size        = directx_runtime->GetTexture2DMaxSize();
        int texture_2d_max_height       = texture_2d_max_size[1];

        auto input_dims = inputs[0]->GetBlobDesc().dims;

        if (UP_DIV(param->output_channel, 4) * 16 > texture_2d_max_height ||
            DimsFunctionUtils::GetDim(input_dims, 0) * UP_DIV(DimsFunctionUtils::GetDim(input_dims, 2), 2) * 16 >
                texture_2d_max_height) {
            return false;
        }

        return param->kernels[0] == 3 && param->kernels[1] == 3 && param->dialations[0] == 1 &&
               param->dialations[1] == 1 && param->strides[0] == 1 && param->strides[1] == 1 &&
               param->output_channel >= 32 && param->input_channel >= 32 &&
               DimsFunctionUtils::GetDim(input_dims, 3) * 1.0f / param->output_channel <= 4;
    }

    Status DirectXConvLayerWinogradAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                             const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
        LOGD("Init Conv Winograd Acc \n");

        conv_type_ = CT_CONV_WINOGRAD;

        Status ret = DirectXConvLayerAccImpl::Init(context, param, resource, inputs, outputs);
        RETURN_ON_NEQ(ret, TNN_OK);

        ConvLayerResource *conv_resource = dynamic_cast<ConvLayerResource *>(resource_);

        auto input_dims          = inputs[0]->GetBlobDesc().dims;
        auto output_dims         = outputs[0]->GetBlobDesc().dims;
        const int input_channel  = DimsFunctionUtils::GetDim(input_dims, 1);
        const int output_channel = DimsFunctionUtils::GetDim(output_dims, 1);

        // convert filter
        ret = ConvertWinogradTransformWeigths(conv_resource->filter_handle, weights_, input_channel, output_channel);
        RETURN_ON_NEQ(ret, TNN_OK);

        // convert bias
        ret = ConvertChannelWeights(conv_resource->bias_handle, bias_, conv_params_.output_channel,
                                    conv_params_.has_bias, false);
        RETURN_ON_NEQ(ret, TNN_OK);

        ret = AllocateWinogradMatrixVAndM(input_dims, output_dims);
        RETURN_ON_NEQ(ret, TNN_OK);

        return TNN_OK;
    }

    DirectXConvLayerWinogradAcc::~DirectXConvLayerWinogradAcc() {}

    Status DirectXConvLayerWinogradAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
        return CreateCB(inputs, outputs);
    }

    Status DirectXConvLayerWinogradAcc::ConvertWinogradTransformWeigths(RawBuffer &raw_handle,
                                                                        shared_ptr<DirectXMemory> &dx_handle,
                                                                        int input_channel, int output_channel) {
        const int kernel_size = conv_params_.kernel_w;
        int unit_output       = UNIT;
        int unit_input        = UNIT + kernel_size - 1;
        WinogradGenerator generator(unit_output, kernel_size, 1.0f);
        auto transform_weight =
            generator.allocTransformWeight(output_channel, input_channel, kernel_size, kernel_size, 4, 4);
        // if filter handle is half, need convert to float first.
        auto filter_data = GetFloatFromRawBuffer(raw_handle);
        if (filter_data == nullptr) {
            return Status(TNNERR_DX_ACC_INIT_ERR, "pointer is null");
        }
        generator.transformWeight(transform_weight, filter_data.get(), output_channel, input_channel, kernel_size,
                                  kernel_size);

        auto dims = std::get<1>(transform_weight);

        int image_height = DimsFunctionUtils::GetDim(dims, 0) * DimsFunctionUtils::GetDim(dims, 1);
        int image_width  = DimsFunctionUtils::GetDim(dims, 2) * DimsFunctionUtils::GetDim(dims, 3);

        auto dx_mem = DirectXMemory::CreateTextureMemoryFromHost(nullptr, dims, image_width, image_height,
                                                                 DATA_TYPE_FLOAT, DATA_FORMAT_NHC4W4);
        if (!dx_mem) {
            LOGE("CreateTextureMemoryFromHost failed\n");
            return Status(TNNERR_DX_TEXTURE_ALOCATE_ERR, "create directx texture memory failed.");
        }

        auto transform_weight_ptr = std::get<0>(transform_weight).get();

        Status ret = UpdateConvWGFilterTexture2D(transform_weight_ptr, dims, image_width, image_height, dx_mem);
        RETURN_ON_NEQ(ret, TNN_OK);

        weights_ = std::move(dx_mem);

        return TNN_OK;
    }

    Status DirectXConvLayerWinogradAcc::AllocateWinogradMatrixVAndM(DimsVector input_dims, DimsVector output_dims) {
        const int batch          = DimsFunctionUtils::GetDim(output_dims, 0);
        const int output_channel = DimsFunctionUtils::GetDim(output_dims, 1);
        const int output_height  = DimsFunctionUtils::GetDim(output_dims, 2);
        const int output_width   = DimsFunctionUtils::GetDim(output_dims, 3);

        const int input_channel         = DimsFunctionUtils::GetDim(input_dims, 1);
        const int output_channel_blocks = UP_DIV(output_channel, 4);
        const int input_channel_blocks  = UP_DIV(input_channel, 4);

        const int round_up_ouptut_width  = UP_DIV(output_width, 2);
        const int round_up_output_height = UP_DIV(output_height, 2);

        int v_width       = input_channel_blocks * round_up_ouptut_width;
        int v_height      = 16 * batch * round_up_output_height;
        DimsVector v_dims = {16 * batch, input_channel_blocks * 4, round_up_output_height, round_up_ouptut_width};
        auto dx_v = DirectXMemory::CreateTextureMemoryFromHost(nullptr, v_dims, v_width, v_height, DATA_TYPE_FLOAT,
                                                               DATA_FORMAT_NHC4W4);
        if (!dx_v) {
            LOGE("CreateTextureMemoryFromHost failed\n");
            return Status(TNNERR_DX_TEXTURE_ALOCATE_ERR, "create directx texture memory failed.");
        }
        dx_v_ = std::move(dx_v);

        int m_width       = output_channel_blocks * round_up_ouptut_width;
        int m_height      = v_height;
        DimsVector m_dims = {16 * batch, output_channel_blocks * 4, round_up_output_height, round_up_ouptut_width};
        auto dx_m = DirectXMemory::CreateTextureMemoryFromHost(nullptr, m_dims, m_width, m_height, DATA_TYPE_FLOAT,
                                                               DATA_FORMAT_NHC4W4);
        if (!dx_m) {
            LOGE("CreateTextureMemoryFromHost failed\n");
            return Status(TNNERR_DX_TEXTURE_ALOCATE_ERR, "create directx texture memory failed.");
        }
        dx_m_ = std::move(dx_m);

        return TNN_OK;
    }

    Status DirectXConvLayerWinogradAcc::CreateCB(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
        auto &in_dims  = inputs[0]->GetBlobDesc().dims;
        auto &out_dims = outputs[0]->GetBlobDesc().dims;

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
        args.in_shape = DirectX::XMUINT4(DimsFunctionUtils::GetDim(in_dims, 0), DimsFunctionUtils::GetDim(in_dims, 1),
                                         DimsFunctionUtils::GetDim(in_dims, 2), DimsFunctionUtils::GetDim(in_dims, 3));
        args.out_shape =
            DirectX::XMUINT4(DimsFunctionUtils::GetDim(out_dims, 0), DimsFunctionUtils::GetDim(out_dims, 1),
                             DimsFunctionUtils::GetDim(out_dims, 2), DimsFunctionUtils::GetDim(out_dims, 3));
        args.kernel_wh       = DirectX::XMUINT4(conv_params_.kernel_w, conv_params_.kernel_h, 0, 0);
        args.stride_wh       = DirectX::XMUINT4(conv_params_.stride_w, conv_params_.stride_h, 0, 0);
        args.padding_wh      = DirectX::XMUINT4(conv_params_.pad_w, conv_params_.pad_h, 0, 0);
        args.dilation_wh     = DirectX::XMUINT4(conv_params_.dilation_w, conv_params_.dilation_h, 0, 0);
        args.activation_type = DirectX::XMUINT4(conv_params_.activation_type, 0, 0, 0);

        return CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer_);
    }

    Status DirectXConvLayerWinogradAcc::DoForward(const std::vector<Blob *> &inputs,
                                                  const std::vector<Blob *> &outputs) {
        std::shared_ptr<DirectXMemory> in_memory, out_memory;
        RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(inputs[0], in_memory), TNN_OK);
        RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(outputs[0], out_memory), TNN_OK);

        auto input_dims  = inputs[0]->GetBlobDesc().dims;
        auto output_dims = outputs[0]->GetBlobDesc().dims;

        const int batch          = DimsFunctionUtils::GetDim(output_dims, 0);
        const int output_channel = DimsFunctionUtils::GetDim(output_dims, 1);
        const int output_height  = DimsFunctionUtils::GetDim(output_dims, 2);
        const int output_width   = DimsFunctionUtils::GetDim(output_dims, 3);

        const int input_channel = DimsFunctionUtils::GetDim(input_dims, 1);
        const int input_height  = DimsFunctionUtils::GetDim(input_dims, 2);
        const int input_width   = DimsFunctionUtils::GetDim(input_dims, 3);

        const int round_up_ouptut_width  = UP_DIV(output_width, 2);
        const int round_up_output_height = UP_DIV(output_height, 2);
        const int batch_round_h          = batch * round_up_output_height;
        const int output_channel_blocks  = UP_DIV(output_channel, 4);
        const int input_channel_blocks   = UP_DIV(input_channel, 4);
        const int round_up_64_output_hw  = UP_DIV(round_up_output_height * round_up_ouptut_width, 64);

        int v_width  = input_channel_blocks * round_up_ouptut_width;
        int v_height = 16 * batch * round_up_output_height;
        int m_width  = output_channel_blocks * round_up_ouptut_width;
        int m_height = v_height;

        // kernel WinogradTransformSource
        auto in_srv = in_memory->GetSRV();
        auto v_uav  = dx_v_->GetUAV();

        std::string kernel_name;
        Status ret;

        kernel_name = "WinogradTransformToMatrixV";
        LOGD("kernel name: %s\n", kernel_name.c_str());
        std::shared_ptr<ID3D11ComputeShader> cs;
        ret = GetShaderByName(kernel_name, cs);
        RETURN_ON_NEQ(ret, TNN_OK);

        int THREADS_BLOCK_A = 8;
        int THREADS_BLOCK_B = 8;

        ret = DispatchShader(cs, {in_srv}, {v_uav}, {const_buffer_.get()},
                             {UP_DIV(v_width, THREADS_BLOCK_A), UP_DIV(v_height / 16, THREADS_BLOCK_B)});
        // printf("input\n");
        // in_memory->Dump();
        // printf("matrix_v\n");
        // dx_v_->Dump();

        // kernel MatrixInnerProduct
        auto v_srv       = dx_v_->GetSRV();
        auto weights_srv = weights_->GetSRV();
        auto m_uav       = dx_m_->GetUAV();

        kernel_name = "WinogradMatrixInnerProduct";
        LOGD("kernel name: %s\n", kernel_name.c_str());
        ret = GetShaderByName(kernel_name, cs);
        RETURN_ON_NEQ(ret, TNN_OK);

        ret = DispatchShader(cs, {v_srv, weights_srv}, {m_uav}, {const_buffer_.get()},
                             {UP_DIV(output_channel_blocks, 16), 16 * batch * round_up_64_output_hw});
        // printf("weight\n");
        // weights_->Dump();
        // printf("matrix_m\n");
        // dx_m_->Dump();

        // kernel TransformFromMatrixM
        auto m_srv    = dx_m_->GetSRV();
        auto bias_srv = bias_->GetSRV();
        auto out_uav  = out_memory->GetUAV();

        kernel_name = "WinogradTransformFromMatrixM";
        LOGD("kernel name: %s\n", kernel_name.c_str());
        ret = GetShaderByName(kernel_name, cs);
        RETURN_ON_NEQ(ret, TNN_OK);

        ret = DispatchShader(cs, {m_srv, bias_srv}, {out_uav}, {const_buffer_.get()},
                             {UP_DIV(m_width, THREADS_BLOCK_A), UP_DIV(m_height / 16, THREADS_BLOCK_B)});

        return ret;
    }

}  // namespace directx
}  // namespace TNN_NS
