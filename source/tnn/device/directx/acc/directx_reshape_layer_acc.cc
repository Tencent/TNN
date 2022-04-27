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

class DirectXReshapeLayerAcc : public DirectXLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~DirectXReshapeLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:

    Status CreateCB(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

//    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) override;
//    virtual std::vector<DataType> SupportDataType(int dims_size, BlobType blob_type) override;

    std::shared_ptr<ID3D11Buffer> const_buffer_ = nullptr;
//    std::shared_ptr<ID3D11Buffer> inter_buffer_ = nullptr;
    int input_dims_size_ = 0;
    int output_dims_size_ = 0;

};

Status DirectXReshapeLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                   const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Reshape Acc\n");
    Status ret = DirectXLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    int reshape_type = -1;
    ReshapeLayerParam *reshape_param = dynamic_cast<ReshapeLayerParam *>(param_);
    if (!reshape_param) {
        FlattenLayerParam *flatten_param = dynamic_cast<FlattenLayerParam *>(param_);
        if(!flatten_param) {
            LOGE("Error: layer param is null\n");
            return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
        } else {
            reshape_type = 0;
        }
    } else {
        reshape_type = reshape_param->reshape_type;
    }

    auto input = inputs[0];
    auto output = outputs[0];

    auto input_dims = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;
    input_dims_size_ = input_dims.size();
    output_dims_size_ = output_dims.size();

    if (reshape_type != 0)
    {
        LOGE("directx only support reshape type = 0 now. Unsupport reshape type(%d)\n", reshape_type);
        return Status(TNNERR_MODEL_ERR, "directx only support reshape type = 0 now.\n");
    }

    return CreateCB(inputs, outputs);
}

DirectXReshapeLayerAcc::~DirectXReshapeLayerAcc() {}

Status DirectXReshapeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Reshape Acc Reshape\n");
    Status ret = DirectXLayerAcc::Reshape(inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    return CreateCB(inputs, outputs);
}

Status DirectXReshapeLayerAcc::CreateCB(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto & in0_dims = inputs[0]->GetBlobDesc().dims;
    auto & out_dims = outputs[0]->GetBlobDesc().dims;

    //    if (in_dims.size() < 4 || out_dims.size() < 4) {
    //        LOGE("Expect shape lenghts > 4 for input and output.\n");
    //        return Status(TNNERR_DX_LAYER_ERR, "Expect shape lenghts > 4 for input and output.");
    //    }
    typedef struct launch_param {
        DirectX::XMUINT4 in_shape;
        DirectX::XMUINT4 out_shape;
    } launch_param_t;

    launch_param_t args;
    args.in_shape  = DirectX::XMUINT4(DimsFunctionUtils::GetDim(input_dims_, 0), DimsFunctionUtils::GetDim(input_dims_, 1),
                                      DimsFunctionUtils::GetDim(input_dims_, 2), DimsFunctionUtils::GetDim(input_dims_, 3));
    args.out_shape = DirectX::XMUINT4(DimsFunctionUtils::GetDim(output_dims_, 0), DimsFunctionUtils::GetDim(output_dims_, 1),
                                      DimsFunctionUtils::GetDim(output_dims_, 2), DimsFunctionUtils::GetDim(output_dims_, 3));

    return CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer_);
}

Status DirectXReshapeLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    std::shared_ptr<DirectXMemory> in_memory, out_memory;
    RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(inputs[0], in_memory), TNN_OK);
    RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(outputs[0], out_memory), TNN_OK);

    shared_ptr<DirectXMemory> inter_buffer = DirectXMemory::CreateBufferMemoryFromHost(
        nullptr, input_dims_, DATA_TYPE_FLOAT, DATA_FORMAT_NCHW);
    if (!inter_buffer) {
        LOGE("param transfer to GPU failed.");
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "param transfer to GPU failed.");
    }

    // reshape image to buffer
    auto in_srv = in_memory->GetSRV();
    auto inter_uav = inter_buffer->GetUAV();

    std::string kernel_name;
    Status ret;

    kernel_name = "reshape_image2buffer";

//    LOGD("kernel name: %s\n",kernel_name.c_str());
    std::shared_ptr<ID3D11ComputeShader> cs;
    ret = GetShaderByName(kernel_name, cs);
    RETURN_ON_NEQ(ret, TNN_OK);

    int batch_in, channel_in, height_in, width_in;
    batch_in            = DimsFunctionUtils::GetDim(input_dims_, 0);
    channel_in          = DimsFunctionUtils::GetDim(input_dims_, 1);
    height_in           = DimsFunctionUtils::GetDim(input_dims_, 2);
    width_in            = DimsFunctionUtils::GetDim(input_dims_, 3);
    int image_width_in  = UP_DIV(channel_in, 4) * width_in;
    int image_height_in = batch_in * height_in;

    ret = DispatchShader(cs, {in_srv}, {inter_uav}, {const_buffer_.get()},  {image_width_in, image_height_in, 1});

    // reshape buffer to image
    auto inter_srv = inter_buffer->GetSRV();
    auto out_uav = out_memory->GetUAV();

    kernel_name = "reshape_buffer2image";

//    LOGD("kernel name: %s\n",kernel_name.c_str());
    ret = GetShaderByName(kernel_name, cs);
    RETURN_ON_NEQ(ret, TNN_OK);

    int batch_out, channel_out, height_out, width_out;
    batch_out            = DimsFunctionUtils::GetDim(output_dims_, 0);
    channel_out          = DimsFunctionUtils::GetDim(output_dims_, 1);
    height_out           = DimsFunctionUtils::GetDim(output_dims_, 2);
    width_out            = DimsFunctionUtils::GetDim(output_dims_, 3);
    int image_width_out  = UP_DIV(channel_out, 4) * width_out;
    int image_height_out = batch_out * height_out;

    ret = DispatchShader(cs, {inter_srv}, {out_uav}, {const_buffer_.get()},  {UP_DIV(image_width_out, 4),UP_DIV(image_height_out, 4), 1});

    return ret;
}

//std::vector<DataFormat> DirectXReshapeLayerAcc::SupportDataFormat(DataType data_type,
//                                                                 int dims_size,
//                                                                 BlobType blob_type) {
//    std::vector<DataFormat> support_list;
//    if (data_type == DATA_TYPE_INT32) {
//        // reshape layer blob may contain shape info
//        support_list.push_back(DATA_FORMAT_NHC4W4);
//    } else if (dims_size >= 2 && dims_size <= 6) { // only support up to 6 dims
//        support_list.push_back(DATA_FORMAT_NHC4W4);
//        // output blob support nchw
//        if (blob_type == BLOB_OUTPUT) {
//            support_list.push_back(DATA_FORMAT_NCHW);
//        }
//    }
//    return support_list;
//}
//
//std::vector<DataType> DirectXReshapeLayerAcc::SupportDataType(int dims_size, BlobType blob_type) {
//    if (blob_type == BLOB_INPUT) {
//        // reshape layer blob may contain shape info
//        return {DATA_TYPE_FLOAT, DATA_TYPE_HALF, DATA_TYPE_INT32};
//    } else {
//        return {DATA_TYPE_FLOAT, DATA_TYPE_HALF};
//    }
//}

REGISTER_DIRECTX_ACC(Reshape, LAYER_RESHAPE)
REGISTER_DIRECTX_LAYOUT(LAYER_RESHAPE, DATA_FORMAT_NHC4W4);

} // namespace directx

}  // namespace TNN_NS
