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

#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_1x1.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
/*
ArmConvInt8Layer1x1 used for 1x1 conv with small c and big h*w
*/
bool ArmConvInt8Layer1x1::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                     const std::vector<Blob *> &outputs) {
    if (param->group != 1 || param->kernels[0] != 1 || param->kernels[1] != 1 || param->strides[0] != 1 ||
        param->strides[1] != 1 || param->pads[0] != 0 || param->pads[1] != 0 || param->pads[2] != 0 ||
        param->pads[3] != 0) {
        return false;
    }
    auto dims_input         = inputs[0]->GetBlobDesc().dims;
    const int input_channel = dims_input[1];
    const int h             = dims_input[2];
    const int w             = dims_input[3];
    if (input_channel <= 32 && h * w > param->output_channel) {
        return true;
    }

    return false;
}

ArmConvInt8Layer1x1::~ArmConvInt8Layer1x1() {}

static inline void packWeightBias(const size_t nc, const size_t kc, const uint32_t nr, const uint32_t np,
                                  const uint32_t kr, const int8_t *const k, const int32_t *const b,
                                  void *const packed_w) {
    union {
        void *const as_void_ptr;
        int8_t *as_int8_ptr;
        int32_t *as_int32_ptr;
    } packed = {packed_w};

    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
        const size_t nr_block_size = nc - nr_block_start < nr ? (nc - nr_block_start) : nr;
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            *(packed.as_int32_ptr++) = b ? b[nr_block_start + nr_block_offset] : 0.0f;
        }
        packed.as_int32_ptr += (nr - nr_block_size);
        for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
            const size_t kr_block_size = kc - kr_block_start < kr ? (kc - kr_block_start) : kr;
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
                    const int8_t kv = k[(nr_block_start + nr_block_offset) * kc + (kr_block_start + kr_block_offset)];
                    *(packed.as_int8_ptr++) = kv;
                }
                packed.as_int8_ptr += (kr - kr_block_size);
            }
            packed.as_int8_ptr += ((nr - nr_block_size) & (np - 1)) * kr;
        }
    }
}

Status ArmConvInt8Layer1x1::allocateBufferWeightBias(const std::vector<Blob *> &inputs,
                                                     const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto dims_input           = inputs[0]->GetBlobDesc().dims;
    auto dims_output          = outputs[0]->GetBlobDesc().dims;
    const int input_channels  = dims_input[1];
    const int output_channels = dims_output[1];

    int nr                           = 8;
    int kr                           = 1;
    const int n_stride               = ROUND_UP(output_channels, nr);
    const int k_stride               = input_channels;
    const size_t packed_weights_size = (sizeof(uint8_t) * k_stride + sizeof(int32_t)) * n_stride;
    const int8_t *k                  = conv_res->filter_handle.force_to<int8_t *>();
    const int32_t *b                 = conv_res->bias_handle.force_to<int32_t *>();
    RawBuffer temp_buffer(packed_weights_size);
    int8_t *packed_w = temp_buffer.force_to<int8_t *>();
    buffer_weight_   = temp_buffer;

    packWeightBias(output_channels, input_channels, nr, nr, kr, k, b, (void *)packed_w);
    return TNN_OK;
}

Status ArmConvInt8Layer1x1::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferWeightBias(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferScale(inputs, outputs), TNN_OK);

    // init base k_param_
    k_param_->scale   = buffer_scale_.force_to<float *>();
    k_param_->fil_ptr = buffer_weight_.force_to<void *>();

    return TNN_OK;
}

Status ArmConvInt8Layer1x1::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
    int8_t *output_data = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));

    struct Q8GemmContext context = {.k        = ic,
                                    .k_stride = ic,
                                    .n        = oc,
                                    .n_stride = ROUND_UP(oc, 8),
                                    .a        = input_data,
                                    .a_stride = ROUND_UP(ic, 4),  // input_pixel_stride
                                    .packed_w = reinterpret_cast<int8_t *>(k_param_->fil_ptr),
                                    .c        = output_data,
                                    .c_stride = ROUND_UP(oc, 4),
                                    .scales   = reinterpret_cast<float *>(k_param_->scale),
                                    .relu     = conv_param->activation_type == ActivationType_ReLU};
    size_t output_size           = k_param_->ow * k_param_->oh * k_param_->oc_r4;
    ComputeQ8Gemm(&context, dims_output[2] * dims_output[3], oc, mr, nr);

    return TNN_OK;
}

}  // namespace TNN_NS
