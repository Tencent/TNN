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

#include "tnn/device/directx/acc/directx_layer_acc.h"

namespace TNN_NS {

namespace directx {

class DirectXBatchNormLayerAcc : public DirectXLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~DirectXBatchNormLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    Status ConvertWeights(float *weights_data_ptr, int output_channel);

    Status CreateCB(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) ;

    bool share_channel_ = false;
    std::shared_ptr<DirectXMemory> bn_scale_ = nullptr;
    std::shared_ptr<DirectXMemory> bn_bias_ = nullptr;
    std::shared_ptr<ID3D11Buffer> const_buffer_;

};

Status DirectXBatchNormLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init BatchNorm Acc\n");
    Status ret = DirectXLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    auto input_dims = inputs[0]->GetBlobDesc().dims;
    int channels    = DimsFunctionUtils::GetDim(input_dims, 1);

    BatchNormLayerResource *batchnorm_resource = dynamic_cast<BatchNormLayerResource *>(resource);
    if (batchnorm_resource == nullptr) {
        LOGE("BatchNormLayerResource is null!\n");
        return Status(TNNERR_MODEL_ERR, "BatchNormLayerResource is null");
    }

    RawBuffer &scale_handle = batchnorm_resource->scale_handle;
    RawBuffer &bias_handle  = batchnorm_resource->bias_handle;
    DataType data_type      = scale_handle.GetDataType();

    share_channel_ = scale_handle.GetBytesSize() == DataTypeUtils::GetBytesSize(data_type);
    bool has_bias  = bias_handle.GetBytesSize() != 0;

    ret = ConvertChannelWeights(scale_handle, bn_scale_, channels, true, share_channel_);
    RETURN_ON_NEQ(ret, TNN_OK);

    // get bias
    ret = ConvertChannelWeights(bias_handle, bn_bias_, channels, has_bias, share_channel_);
    RETURN_ON_NEQ(ret, TNN_OK);

    return CreateCB(inputs, outputs);
}

DirectXBatchNormLayerAcc::~DirectXBatchNormLayerAcc() {}

Status DirectXBatchNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("BatchNorm Layer Reshape\n");
    ASSERT(inputs.size() == 1);
    Status ret = DirectXLayerAcc::Reshape(inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    return CreateCB(inputs, outputs);
}


Status DirectXBatchNormLayerAcc::CreateCB(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto & in0_dims = inputs[0]->GetBlobDesc().dims;
    auto & out_dims = outputs[0]->GetBlobDesc().dims;

//    if (in_dims.size() < 4 || out_dims.size() < 4) {
//        LOGE("Expect shape lenghts > 4 for input and output.\n");
//        return Status(TNNERR_DX_LAYER_ERR, "Expect shape lenghts > 4 for input and output.");
//    }
    typedef struct launch_param {
        DirectX::XMUINT4 out_shape;
    } launch_param_t;

    launch_param_t args;
    args.out_shape = DirectX::XMUINT4(DimsFunctionUtils::GetDim(out_dims, 0), DimsFunctionUtils::GetDim(out_dims, 1),
                                      DimsFunctionUtils::GetDim(out_dims, 2), DimsFunctionUtils::GetDim(out_dims, 3));

    return CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer_);
}

Status DirectXBatchNormLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    std::shared_ptr<DirectXMemory> in_memory, out_memory;
    RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(inputs[0], in_memory), TNN_OK);
    RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(outputs[0], out_memory), TNN_OK);

    auto in_srv = in_memory->GetSRV();
    auto scale_srv = bn_scale_->GetSRV();
    auto bias_srv = bn_bias_->GetSRV();
    auto out_uav = out_memory->GetUAV();

    std::string kernel_name;
    auto &out_dims = outputs[0]->GetBlobDesc().dims;
    Status ret;

//    kernel_name = "batchnorm_texture";
    kernel_name = "batchnorm_2d_texture";

    LOGD("kernel name: %s\n",kernel_name.c_str());
    std::shared_ptr<ID3D11ComputeShader> cs;
    ret = GetShaderByName(kernel_name, cs);
    RETURN_ON_NEQ(ret, TNN_OK);

    const int batch         = DimsFunctionUtils::GetDim(output_dims_, 0);
    const int output_height = DimsFunctionUtils::GetDim(output_dims_, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims_, 3);
    const int channels      = DimsFunctionUtils::GetDim(output_dims_, 1);

    const int channel_blocks    = UP_DIV(channels, 4);
    int image_width = UP_DIV(DimsFunctionUtils::GetDim(out_dims, 1), 4) * DimsFunctionUtils::GetDim(out_dims, 3);
    int image_height = DimsFunctionUtils::GetDim(out_dims, 0) * DimsFunctionUtils::GetDim(out_dims, 2);

    ret = DispatchShader(cs, {in_srv, scale_srv, bias_srv}, {out_uav}, {const_buffer_.get()}, {UP_DIV(image_width, 4),UP_DIV(image_height, 4),1});

//    ret = DispatchShader(cs, {in_srv, scale_srv, bias_srv}, {out_uav}, {const_buffer_.get()}, {batch * output_height,output_width,channel_blocks});

    return ret;
}

REGISTER_DIRECTX_ACC(BatchNorm, LAYER_BATCH_NORM)
REGISTER_DIRECTX_LAYOUT(LAYER_BATCH_NORM, DATA_FORMAT_NHC4W4);

} // namespace directx

}  // namespace TNN_NS
