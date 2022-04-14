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

class DirectXPReluLayerAcc : public DirectXLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~DirectXPReluLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    Status ConvertWeights(float *weights_data_ptr, int output_channel);

    Status CreateCB(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) ;

    bool share_channel_ = false;
    shared_ptr<DirectXMemory> prelu_scope_ = nullptr;
    std::shared_ptr<ID3D11Buffer> const_buffer_;

};

Status DirectXPReluLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init PRelu Acc\n");
    Status ret = DirectXLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    auto input_dims = inputs[0]->GetBlobDesc().dims;
    int channels    = DimsFunctionUtils::GetDim(input_dims, 1);

    auto layer_param = dynamic_cast<PReluLayerParam *>(param);
    if (layer_param == nullptr) {
        LOGE("PReluLayerParam is null!\n");
        return Status(TNNERR_MODEL_ERR, "PReluLayerParam is null");
    }
    share_channel_ = layer_param->channel_shared;

    auto layer_res = dynamic_cast<PReluLayerResource *>(resource);
    if (layer_res == nullptr) {
        LOGE("PReluLayerResource is null!\n");
        return Status(TNNERR_MODEL_ERR, "PReluLayerResource is null");
    }
    RawBuffer &scope_handle = layer_res->slope_handle;

    ConvertChannelWeights(scope_handle, prelu_scope_, channels, true, share_channel_);

    return CreateCB(inputs, outputs);
}

DirectXPReluLayerAcc::~DirectXPReluLayerAcc() {}

Status DirectXPReluLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("PRelu Acc Reshape\n");
    ASSERT(inputs.size() == 1);
    Status ret = DirectXLayerAcc::Reshape(inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    return CreateCB(inputs, outputs);
}

Status DirectXPReluLayerAcc::CreateCB(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto & in_dims = inputs[0]->GetBlobDesc().dims;
    auto & out_dims = outputs[0]->GetBlobDesc().dims;

    if (in_dims.size() < 4 || out_dims.size() < 4) {
        LOGE("Expect shape lenghts > 4 for input and output.\n");
        return Status(TNNERR_DX_LAYER_ERR, "Expect shape lenghts > 4 for input and output.");
    }
    typedef struct launch_param {
        DirectX::XMUINT4 in_shape;
        DirectX::XMUINT4 out_shape;
    } launch_param_t;

    launch_param_t args;
    args.in_shape  = DirectX::XMUINT4(in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
    args.out_shape = DirectX::XMUINT4(out_dims[0], out_dims[1], out_dims[2], out_dims[3]);

    return CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer_);
}

Status DirectXPReluLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    auto in_memory = DirectXMemory::CreateRefMemoryFromBlob(inputs[0]);
    auto out_memory = DirectXMemory::CreateRefMemoryFromBlob(outputs[0]);

    auto in_srv = in_memory->GetSRV();
    auto scope_srv = prelu_scope_->GetSRV();
    auto out_uav = out_memory->GetUAV();

    std::string kernel_name;
    auto &out_dims = outputs[0]->GetBlobDesc().dims;
    Status ret;

    kernel_name = "prelu_texture";

    int image_width = UP_DIV(DimsFunctionUtils::GetDim(out_dims, 1), 4) * DimsFunctionUtils::GetDim(out_dims, 3);
    int image_height = DimsFunctionUtils::GetDim(out_dims, 0) * DimsFunctionUtils::GetDim(out_dims, 2);

    LOGD("kernel name: %s\n",kernel_name.c_str());
    std::shared_ptr<ID3D11ComputeShader> cs;
    ret = GetShaderByName(kernel_name, cs);
    RETURN_ON_NEQ(ret, TNN_OK);

    ret = DispatchShader(cs, {in_srv, scope_srv}, {out_uav}, {const_buffer_.get()}, {image_width, image_height, 1});

    return ret;
}


REGISTER_DIRECTX_ACC(PRelu, LAYER_PRELU)
REGISTER_DIRECTX_LAYOUT(LAYER_PRELU, DATA_FORMAT_NHC4W4);

} // namespace directx

}  // namespace TNN_NS