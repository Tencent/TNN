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

#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_1x1_fuse_add.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
/*
ArmConvInt8Layer1x1FuseAdd used for 1x1 conv fused with add
*/
bool ArmConvInt8Layer1x1FuseAdd::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                     const std::vector<Blob *> &outputs) {
    if (param->group != 1 || param->kernels[0] != 1 || param->kernels[1] != 1 || param->strides[0] != 1 ||
        param->strides[1] != 1 || param->pads[0] != 0 || param->pads[1] != 0 || param->pads[2] != 0 ||
        param->pads[3] != 0) {
        return false;
    } else if (param->fusion_type == FusionType_None) {
        return false;
    } else {
        return true;
    }
}

ArmConvInt8Layer1x1FuseAdd::~ArmConvInt8Layer1x1FuseAdd() {}

Status ArmConvInt8Layer1x1FuseAdd::allocateBufferAddScale(const std::vector<Blob *> &inputs,
                                                          const std::vector<Blob *> &outputs) {
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    // alloc add scale buffer
    if (!buffer_add_scale_.GetBytesSize()) {
        auto dims_output = outputs[0]->GetBlobDesc().dims;
        int total_byte_size =
            ROUND_UP(dims_output[1], 4) * DataTypeUtils::GetBytesSize(conv_res->scale_handle.GetDataType());

        const float *i_scale =
            reinterpret_cast<BlobInt8 *>(inputs[1])->GetIntResource()->scale_handle.force_to<float *>();

        const float *o_scale =
            reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.force_to<float *>();

        int scale_len_i = reinterpret_cast<BlobInt8 *>(inputs[1])->GetIntResource()->scale_handle.GetDataCount();
        int scale_len_o = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.GetDataCount();
        RawBuffer temp_buffer(total_byte_size);
        float *temp_ptr = temp_buffer.force_to<float *>();
        for (int i = 0; i < dims_output[1]; i++) {
            int scale_idx_i = scale_len_i == 1 ? 0 : i;
            int scale_idx_o = scale_len_o == 1 ? 0 : i;

            if (o_scale[scale_idx_o] >= FLT_MIN)
                temp_ptr[i] = i_scale[scale_idx_i] / o_scale[scale_idx_o];
            else
                temp_ptr[i] = 0.0;
        }
        buffer_add_scale_ = temp_buffer;
    }

    return TNN_OK;
}

Status ArmConvInt8Layer1x1FuseAdd::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmConvInt8Layer1x1::Init(context, param, resource, inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferAddScale(inputs, outputs), TNN_OK);

    return TNN_OK;
}

Status ArmConvInt8Layer1x1FuseAdd::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    auto input  = inputs[0];
    auto output = outputs[0];

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

#ifdef __aarch64__
    const int mr        = 8;
#else
    const int mr        = 4;
#endif
    const int nr        = 8;
    const int kr        = 1;
    auto dims_input     = input->GetBlobDesc().dims;
    auto dims_output    = output->GetBlobDesc().dims;
    const int batch     = dims_output[0];
    int ic              = dims_input[1];
    int oc              = dims_output[1];
    int8_t *input_data  = reinterpret_cast<int8_t *>(GetBlobHandlePtr(input->GetHandle()));
    int8_t *add_input   = reinterpret_cast<int8_t *>(GetBlobHandlePtr(inputs[1]->GetHandle()));
    int8_t *output_data = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));

    struct Q8GemmAddContext context = {.k         = ic,
                                       .k_stride  = ic,
                                       .n         = oc,
                                       .n_stride  = ROUND_UP(oc, 8),
                                       .a         = input_data,
                                       .a_stride  = ROUND_UP(ic, 4),  // input_pixel_stride
                                       .packed_w  = reinterpret_cast<int8_t *>(k_param_->fil_ptr),
                                       .c         = output_data,
                                       .c_stride  = ROUND_UP(oc, 4),
                                       .scales    = reinterpret_cast<float *>(k_param_->scale),
                                       .relu      = conv_param->activation_type == ActivationType_ReLU,
                                       .add_input = add_input,
                                       .add_scale = buffer_add_scale_.force_to<float *>()};
    ComputeQ8GemmAdd(&context, dims_output[2] * dims_output[3], oc, mr, nr);

    return TNN_OK;
}

}  // namespace TNN_NS
