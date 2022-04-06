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

#include "tnn/device/directx/acc/directx_pooling_layer_acc.h"
#include "tnn/utils/string_utils_inner.h"
#include "tnn/utils/dims_function_utils.h"
#include <directxmath.h>


namespace TNN_NS {

namespace directx {

#define LowOpParallelismThre 256
#define HighOpIntensityThre 128

Status DirectXPoolingLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Pooling Acc\n");
    Status ret = DirectXLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret,TNN_OK);

    run_3d_ndrange_ = true;
    kernel_name_    = "pooling";

    PoolingLayerParam *pooling_param = dynamic_cast<PoolingLayerParam *>(param);
    if (!pooling_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    const int batch         = DimsFunctionUtils::GetDim(output_dims_, 0);
    const int output_height = DimsFunctionUtils::GetDim(output_dims_, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims_, 3);
    const int channels      = DimsFunctionUtils::GetDim(output_dims_, 1);

    const int kernel_height = pooling_param->kernels[1];
    const int kernel_width  = pooling_param->kernels[0];

    const int channel_blocks = UP_DIV(channels, 4);

    bool run_local_work = batch * output_height * output_width * channel_blocks < LowOpParallelismThre &&
        kernel_width * kernel_height >= HighOpIntensityThre;
    if (run_local_work) {
        kernel_name_ += "Local";
    }

    if (pooling_param->pool_type != 0) {  // 0:max_pooling  other:average pooling
        kernel_name_+= "_avg";
    } else {
        kernel_name_+= "_max";
    }

    return TNN_OK;
}

Status DirectXPoolingLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Pooling Acc Reshape\n");
    Status ret = DirectXLayerAcc::Reshape(inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    return CalcParam(inputs, outputs);
}

Status DirectXPoolingLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    Status status = DirectXLayerAcc::Forward(inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);

    auto d3d_context = GetID3DContext();

    auto in_memory = DirectXMemory::CreateRefMemoryFromBlob(inputs[0]);
    auto out_memory = DirectXMemory::CreateRefMemoryFromBlob(outputs[0]);

    auto in_srv = in_memory->GetSRV();
    auto out_uav = out_memory->GetUAV();

    std::shared_ptr<ID3D11ComputeShader> cs;
    Status ret = GetShaderByName(kernel_name_, cs);
    RETURN_ON_NEQ(ret, TNN_OK);

    const int batch         = DimsFunctionUtils::GetDim(output_dims_, 0);
    const int output_height = DimsFunctionUtils::GetDim(output_dims_, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims_, 3);
    const int channels      = DimsFunctionUtils::GetDim(output_dims_, 1);

    const int channel_blocks    = UP_DIV(channels, 4);

    ret = DispatchShader(cs, {in_srv}, {out_uav}, {const_buffer_.get()}, {batch * output_height,output_width,channel_blocks});

    return ret;
}

DirectXPoolingLayerAcc::~DirectXPoolingLayerAcc() {}

Status DirectXPoolingLayerAcc::CalcParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    PoolingLayerParam *pooling_param = dynamic_cast<PoolingLayerParam *>(param_);
    if (!pooling_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    if (pooling_param->pad_type == 1) {  // VALID Type
        pooling_param->pads[0] = 0;
        pooling_param->pads[2] = 0;
    }
    const int pad_height = pooling_param->pads[2];
    const int pad_width  = pooling_param->pads[0];

    const int stride_height = pooling_param->strides[1];
    const int stride_width  = pooling_param->strides[0];

    const int kernel_height = pooling_param->kernels[1];
    const int kernel_width  = pooling_param->kernels[0];

    typedef struct launch_param {
        DirectX::XMINT4 id;     //input_dim
        DirectX::XMINT4 od;     //output_dim
        DirectX::XMINT2 pad;    //pad_wh
        DirectX::XMINT2 stride; //stride_wh
        DirectX::XMINT2 kernel; //kernel_wh
    } launch_param_t;

    launch_param_t args;
    args.id = DirectX::XMINT4(DimsFunctionUtils::GetDim(input_dims_, 0), DimsFunctionUtils::GetDim(input_dims_, 1),
                               DimsFunctionUtils::GetDim(input_dims_, 2), DimsFunctionUtils::GetDim(input_dims_, 3));
    args.od = DirectX::XMINT4(DimsFunctionUtils::GetDim(output_dims_, 0), DimsFunctionUtils::GetDim(output_dims_, 1),
                               DimsFunctionUtils::GetDim(output_dims_, 2), DimsFunctionUtils::GetDim(output_dims_, 3));
    args.pad = DirectX::XMINT2(pad_width,pad_height);
    args.stride = DirectX::XMINT2(stride_width,stride_height);
    args.kernel = DirectX::XMINT2(kernel_width,kernel_height);

    Status ret = CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer_);
    RETURN_ON_NEQ(ret, TNN_OK);

    return TNN_OK;
}

REGISTER_DIRECTX_ACC(Pooling, LAYER_POOLING)
REGISTER_DIRECTX_LAYOUT(LAYER_POOLING, DATA_FORMAT_NHC4W4);

} // namespace directx

}  // namespace TNN_NS
