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

typedef enum { BUFFER_COPY = 0, IMAGE_COPY, TWO_INPUTS_CHANNEL_4X, TWO_INPUTS_CHANNEL_MOD_123 } ConcatKernelType;

class DirectXConcatLayerAcc : public DirectXLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~DirectXConcatLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;


private:

    Status CreateCB(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    int CalculateAxis(int axis, int dims_size);

    int axis_                                               = 1;
    bool do_image_concat_                                   = true;
    ConcatKernelType concat_type_                           = TWO_INPUTS_CHANNEL_MOD_123;
    std::shared_ptr<ID3D11Buffer> const_buffer_;

};

bool CheckIsTwoInputs(const size_t input_size, const int axis) {
    return input_size == 2 && axis == 1;
}

Status DirectXConcatLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Concat Acc\n");
    Status ret = DirectXLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    ConcatLayerParam *concat_param = dynamic_cast<ConcatLayerParam *>(param);
    CHECK_PARAM_NULL(concat_param);

    axis_            = CalculateAxis(concat_param->axis, outputs[0]->GetBlobDesc().dims.size());
    do_image_concat_ = true;

    if (axis_ == 1) {
        for (size_t i = 0; i < inputs.size() - 1; ++i) {
            int channel = DimsFunctionUtils::GetDim(inputs[i]->GetBlobDesc().dims, 1);
            if (channel % 4 != 0) {
                do_image_concat_ = false;
                break;
            }
        }
    }

    LOGD("do_image_concat: %s\n", do_image_concat_ ? "true" : "false");

    // choose kernel type
    if (CheckIsTwoInputs(inputs.size(), axis_)) {
        if (do_image_concat_) {
            concat_type_ = TWO_INPUTS_CHANNEL_4X;
        } else {
            concat_type_ = TWO_INPUTS_CHANNEL_MOD_123;
        }
    } else {
        LOGE("directx concat only support 2 inputs & axis=1 now.\n");
        return Status(TNNERR_MODEL_ERR, "directx concat only support 2 inputs & axis=1 now.\n");
    }

    return CreateCB(inputs, outputs);
}

DirectXConcatLayerAcc::~DirectXConcatLayerAcc() {}

Status DirectXConcatLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Concat Acc Reshape\n");
    Status ret = DirectXLayerAcc::Reshape(inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    return CreateCB(inputs, outputs);
}

Status DirectXConcatLayerAcc::CreateCB(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto & in0_dims = inputs[0]->GetBlobDesc().dims;
    auto & out_dims = outputs[0]->GetBlobDesc().dims;

    //    if (in_dims.size() < 4 || out_dims.size() < 4) {
    //        LOGE("Expect shape lenghts > 4 for input and output.\n");
    //        return Status(TNNERR_DX_LAYER_ERR, "Expect shape lenghts > 4 for input and output.");
    //    }
    typedef struct launch_param {
        DirectX::XMUINT4 in0_shape;
        DirectX::XMUINT4 out_shape;
        DirectX::XMUINT4 param;
    } launch_param_t;

    launch_param_t args;
    args.in0_shape  = DirectX::XMUINT4(DimsFunctionUtils::GetDim(in0_dims, 0), DimsFunctionUtils::GetDim(in0_dims, 1),
                                       DimsFunctionUtils::GetDim(in0_dims, 2), DimsFunctionUtils::GetDim(in0_dims, 3));
    args.out_shape = DirectX::XMUINT4(DimsFunctionUtils::GetDim(out_dims, 0), DimsFunctionUtils::GetDim(out_dims, 1),
                                      DimsFunctionUtils::GetDim(out_dims, 2), DimsFunctionUtils::GetDim(out_dims, 3));
    args.param = DirectX::XMUINT4(DimsFunctionUtils::GetDim(in0_dims, 1) % 4, 0, 0, 0);

    return CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer_);
}

Status DirectXConcatLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    std::shared_ptr<DirectXMemory> in0_memory, in1_memory, out_memory;
    RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(inputs[0], in0_memory), TNN_OK);
    RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(inputs[1], in1_memory), TNN_OK);
    RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(outputs[0], out_memory), TNN_OK);

    auto in0_srv = in0_memory->GetSRV();
    auto in1_srv = in1_memory->GetSRV();
    auto out_uav = out_memory->GetUAV();

    std::string kernel_name;
    auto &out_dims = outputs[0]->GetBlobDesc().dims;
    Status ret;

    if (do_image_concat_) {
        kernel_name = "concat_channel4x_texture";
    } else {
        kernel_name = "concat_channel_texture";
    }

    LOGD("kernel name: %s\n",kernel_name.c_str());
    std::shared_ptr<ID3D11ComputeShader> cs;
    ret = GetShaderByName(kernel_name, cs);
    RETURN_ON_NEQ(ret, TNN_OK);

    const int batch         = DimsFunctionUtils::GetDim(output_dims_, 0);
    const int output_height = DimsFunctionUtils::GetDim(output_dims_, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims_, 3);
    const int channels      = DimsFunctionUtils::GetDim(output_dims_, 1);

    const int channel_blocks    = UP_DIV(channels, 4);

    ret = DispatchShader(cs, {in0_srv, in1_srv}, {out_uav}, {const_buffer_.get()}, {UP_DIV(batch * output_height, 4), UP_DIV(output_width, 4), channel_blocks});

    return ret;
}

int DirectXConcatLayerAcc::CalculateAxis(int axis, int dims_size) {
    if (dims_size <= 4 || axis == 0) {
        return axis;
    }

    return 2;
}

REGISTER_DIRECTX_ACC(Concat, LAYER_CONCAT)
REGISTER_DIRECTX_LAYOUT(LAYER_CONCAT, DATA_FORMAT_NHC4W4);

} // namespace directx

}  // namespace TNN_NS
