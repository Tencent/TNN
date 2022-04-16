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

class DirectXInnerProductLayerAcc : public DirectXLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~DirectXInnerProductLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    Status ConvertWeights(float *weights_data_ptr, int weight_w, int weight_h);

    Status CreateCB(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    int num_output_ = 0;
    int transpose_ = 0;
    int axis_ = 0;
    shared_ptr<DirectXMemory> innerproduct_weights_ = nullptr;
    shared_ptr<DirectXMemory> innerproduct_bias_ = nullptr;
    std::shared_ptr<ID3D11Buffer> const_buffer_;

};

Status DirectXInnerProductLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init InnerProduct Acc\n");
    Status ret = DirectXLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    use_buffer_ = false;

    InnerProductLayerParam *innerproduct_param = dynamic_cast<InnerProductLayerParam *>(param);
    CHECK_PARAM_NULL(innerproduct_param);

    num_output_  = innerproduct_param->num_output;
    int has_bias = innerproduct_param->has_bias;
    transpose_   = innerproduct_param->transpose;
    axis_        = innerproduct_param->axis;

    InnerProductLayerResource *innerproduct_resource = dynamic_cast<InnerProductLayerResource *>(resource);
    CHECK_PARAM_NULL(innerproduct_resource);
    RawBuffer &weight_handle = innerproduct_resource->weight_handle;
    RawBuffer &bias_handle   = innerproduct_resource->bias_handle;
    DataType data_type       = weight_handle.GetDataType();

    // get weights
    int weights_height = weight_handle.GetBytesSize() / DataTypeUtils::GetBytesSize(data_type) / num_output_;
    int weights_width  = num_output_;
    if (weight_handle.GetDataType() == DATA_TYPE_FLOAT) {
        // get float pointer from raw buffer.
        float *weights_data_ptr = weight_handle.force_to<float *>();
        if (weights_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(weights_data_ptr, weights_width, weights_height);
        RETURN_ON_NEQ(ret, TNN_OK);
    } else {
        LOGE("directx weight only support float data type now!\n");
    }

    // get bias
    ret = ConvertChannelWeights(innerproduct_resource->bias_handle, innerproduct_bias_, num_output_, has_bias);
    RETURN_ON_NEQ(ret, TNN_OK);

    return CreateCB(inputs, outputs);
}

DirectXInnerProductLayerAcc::~DirectXInnerProductLayerAcc() {}

Status DirectXInnerProductLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("InnerProduct Layer Reshape\n");
    ASSERT(inputs.size() == 1);
    Status ret = DirectXLayerAcc::Reshape(inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    auto input_dims     = inputs[0]->GetBlobDesc().dims;
    auto output_dims    = outputs[0]->GetBlobDesc().dims;
    auto output_height  = DimsFunctionUtils::GetDim(output_dims, 2);
    auto output_width   = DimsFunctionUtils::GetDim(output_dims, 3);
    auto input_height   = DimsFunctionUtils::GetDim(input_dims, 2);
    auto input_width    = DimsFunctionUtils::GetDim(input_dims, 3);
    // now only support axis is channel, output width and output height is 1.
    if (axis_ != 1 || output_height != 1 || output_width != 1) {
        LOGE("Invalid InnerParameter param or input/output size!\n");
        return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "Invalid InnerParameter param or input/output size!");
    }

    return CreateCB(inputs, outputs);
}

Status DirectXInnerProductLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    auto in_memory = DirectXMemory::CreateRefMemoryFromBlob(inputs[0]);
    auto out_memory = DirectXMemory::CreateRefMemoryFromBlob(outputs[0]);

    Status ret;
    auto in_srv = in_memory->GetSRV();
    auto weight_srv = innerproduct_weights_->GetSRV();
    auto bias_srv = innerproduct_bias_->GetSRV();
    auto out_uav = out_memory->GetUAV();
    std::string kernel_name;
    auto &out_dims = outputs[0]->GetBlobDesc().dims;

    if (use_buffer_) {
        LOGE("directx innerproduct only support using texture2d now!\n");
    } else {

        int N = num_output_;
        int M = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims, 0, axis_);
        int K = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims, axis_);

        int image_width = UP_DIV(N, 4);
        int image_height = M;

        kernel_name = "innerproduct_texture";

        LOGD("kernel name: %s\n",kernel_name.c_str());
        std::shared_ptr<ID3D11ComputeShader> cs;
        ret = GetShaderByName(kernel_name, cs);
        RETURN_ON_NEQ(ret, TNN_OK);

        ret = DispatchShader(cs, {in_srv, weight_srv, bias_srv}, {out_uav}, {const_buffer_.get()}, {image_width, image_height, 1});
    }

    return ret;
}

Status DirectXInnerProductLayerAcc::ConvertWeights(float *weights_data_ptr, int weight_w, int weight_h) {

    // traspose
    shared_ptr<float> weights_data_ptr_trans(new float[weight_w * weight_h]);
    for (size_t i = 0; i < weight_h; i++) {
        for (size_t j = 0; j < weight_w; j++) {
            weights_data_ptr_trans.get()[j + i * weight_w] = weights_data_ptr[i + j * weight_h];
        }
    }

    DimsVector weight_shape{weight_h, weight_w, 1, 1};

    if (use_buffer_) {
        LOGE("directx innerproduct only support using texture2d now!\n");
    } else {

        DimsVector weight_imageshape{(int)(UP_DIV(weight_w, 4)), weight_h};

        auto dx_mem = DirectXMemory::CreateTextureMemoryFromHost(
            nullptr, weight_shape, weight_imageshape[0], weight_imageshape[1], DATA_TYPE_FLOAT, DATA_FORMAT_NHC4W4);
        if (!dx_mem) {
            LOGE("CreateTextureMemoryFromHost failed\n");
            return Status(TNNERR_DX_TEXTURE_ALOCATE_ERR, "create directx texture memory failed.");
        }

        Status ret = UpdateTexture2D(weights_data_ptr_trans.get(), weight_shape, dx_mem);

        RETURN_ON_NEQ(ret, TNN_OK);

        innerproduct_weights_ = std::move(dx_mem);
    }

    return TNN_OK;
}

Status DirectXInnerProductLayerAcc::CreateCB(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto & in_dims = inputs[0]->GetBlobDesc().dims;
    auto & out_dims = outputs[0]->GetBlobDesc().dims;

//    if (in_dims.size() < 4 || out_dims.size() < 4) {
//        LOGE("Expect shape lenghts > 4 for input and output.\n");
//        return Status(TNNERR_DX_LAYER_ERR, "Expect shape lenghts > 4 for input and output.");
//    }

    int N = num_output_;
    int M = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims, 0, axis_);
    int K = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims, axis_);

    typedef struct launch_param {
        DirectX::XMUINT4 innerproduct_shape;
    } launch_param_t;

    launch_param_t args;
    args.innerproduct_shape = DirectX::XMUINT4(N, M, K, 0);

    return CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer_);
}

REGISTER_DIRECTX_ACC(InnerProduct, LAYER_INNER_PRODUCT)
REGISTER_DIRECTX_LAYOUT(LAYER_INNER_PRODUCT, DATA_FORMAT_NHC4W4);

} // namespace directx
}  // namespace TNN_NS
