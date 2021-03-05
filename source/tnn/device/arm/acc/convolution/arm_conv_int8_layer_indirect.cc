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

#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_indirect.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
bool ArmConvInt8LayerIndirect::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                          const std::vector<Blob *> &outputs) {
    if (param->group != 1 || param->kernels[0] == 1 || param->kernels[1] == 1) {
        return false;
    }
    return true;
}

ArmConvInt8LayerIndirect::~ArmConvInt8LayerIndirect() {}

static inline void packWeightBias(const size_t output_channel, const size_t kernel_size, const size_t input_channel,
                                  const int32_t nr, const int32_t kr, const int8_t *const weight_addr,
                                  const int32_t *const bias_addr, void *const packed_w) {
    union {
        void *const as_void_ptr;
        int8_t *as_int8_ptr;
        int32_t *as_int32_ptr;
    } packed = {packed_w};

    for (size_t nr_block_start = 0; nr_block_start < output_channel; nr_block_start += nr) {
        const size_t nr_block_size = (output_channel - nr_block_start < nr) ? (output_channel - nr_block_start) : nr;
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            *(packed.as_int32_ptr++) = bias_addr ? bias_addr[nr_block_start + nr_block_offset] : 0.0f;
        }
        packed.as_int32_ptr += (nr - nr_block_size);
        for (size_t ki = 0; ki < kernel_size; ki++) {
            for (size_t kr_block_start = 0; kr_block_start < input_channel; kr_block_start += kr) {
                const size_t kr_block_size =
                    (input_channel - kr_block_start < kr) ? (input_channel - kr_block_start) : kr;
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                    for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
                        const int8_t kv = weight_addr[((nr_block_start + nr_block_offset) * input_channel + kr_block_start + kr_block_offset) * kernel_size + ki];
                        *(packed.as_int8_ptr++) = kv;
                    }
                    packed.as_int8_ptr += (kr - kr_block_size);
                }
                packed.as_int8_ptr += (nr - nr_block_size) * kr;
            }
        }
    }
}

Status ArmConvInt8LayerIndirect::allocateBufferWeightBias(const std::vector<Blob *> &inputs,
                                                          const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto dims_input           = inputs[0]->GetBlobDesc().dims;
    auto dims_output          = outputs[0]->GetBlobDesc().dims;
    const int input_channels  = dims_input[1];
    const int output_channels = dims_output[1];

    int kw                           = conv_param->kernels[0];
    int kh                           = conv_param->kernels[1];
    const int kernel_size            = kh * kw;
    int nr                           = 8;
    int kr                           = 1;
    const int n_stride               = ROUND_UP(output_channels, nr);
    const int k_stride               = input_channels * kh * kw;
    const size_t packed_weights_size = (sizeof(int8_t) * k_stride + sizeof(int32_t)) * n_stride;
    const int8_t *k                  = conv_res->filter_handle.force_to<int8_t *>();
    const int32_t *b                 = conv_res->bias_handle.force_to<int32_t *>();
    RawBuffer temp_buffer(packed_weights_size);
    int8_t *packed_w = temp_buffer.force_to<int8_t *>();
    buffer_weight_   = temp_buffer;
    memset(packed_w, 0, packed_weights_size);
    packWeightBias(output_channels, kernel_size, input_channels, nr, kr, k, b, (void *)packed_w);
    return TNN_OK;
}

Status ArmConvInt8LayerIndirect::initIndirectionBuffer(const std::vector<Blob *> &inputs,
                                                       const std::vector<Blob *> &outputs, size_t output_tile_size,
                                                       size_t tiled_output_size) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    auto dims_input         = inputs[0]->GetBlobDesc().dims;
    auto dims_output        = outputs[0]->GetBlobDesc().dims;
    //const int8_t *input     = reinterpret_cast<int8_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    const size_t ic         = dims_input[1];
    const size_t ic_r4      = ROUND_UP(ic, 4);
    const size_t ih         = dims_input[2];
    const size_t iw         = dims_input[3];
    const size_t oc         = dims_output[1];
    const size_t oh         = dims_output[2];
    const size_t ow         = dims_output[3];
    const size_t kh         = conv_param->kernels[1];
    const size_t kw         = conv_param->kernels[0];
    const size_t stride_h   = conv_param->strides[1];
    const size_t stride_w   = conv_param->strides[0];
    const size_t dilation_h = conv_param->dialations[1];
    const size_t dilation_w = conv_param->dialations[0];
    const size_t pad_h      = conv_param->pads[1];
    const size_t pad_w      = conv_param->pads[0];

    const size_t output_size    = oh * ow;
    const size_t kernel_size    = kh * kw;
    auto *indirection_buffer = indirection_buffer_.force_to<int32_t *>();
    //int8_t *zero                = zero_buffer_.force_to<int8_t *>() + zero_buffer_offset_;
    for (size_t output_tile_start = 0; output_tile_start < tiled_output_size; output_tile_start += output_tile_size) {
        for (size_t output_tile_offset = 0; output_tile_offset < output_tile_size; output_tile_offset++) {
            const size_t tiled_output_index = output_tile_start + output_tile_offset;
            const size_t output_index = (tiled_output_index < output_size - 1) ? tiled_output_index : (output_size - 1);
            const size_t output_y     = output_index / ow;
            const size_t output_x     = output_index % ow;
            for (size_t kernel_y = 0; kernel_y < kh; kernel_y++) {
                const size_t input_y = output_y * stride_h + kernel_y * dilation_h - pad_h;
                if (input_y < ih) {
                    for (size_t kernel_x = 0; kernel_x < kw; kernel_x++) {
                        const size_t input_x = output_x * stride_w + kernel_x * dilation_w - pad_w;
                        const size_t index = output_tile_start * kernel_size + (kernel_y * kw + kernel_x) * output_tile_size + output_tile_offset;
                        if (input_x < iw) {
                            indirection_buffer[index] = (input_y * iw + input_x) * ic_r4;
                        } else {
                            indirection_buffer[index] = -1;
                        }
                    }
                } else {
                    for (size_t kernel_x = 0; kernel_x < kw; kernel_x++) {
                        const size_t index = output_tile_start * kernel_size + (kernel_y * kw + kernel_x) * output_tile_size + output_tile_offset;
                        indirection_buffer[index] = -1;
                    }
                }
            }
        }
    }

    return TNN_OK;
}

Status ArmConvInt8LayerIndirect::allocateBufferInput(const std::vector<Blob *> &inputs,
                                                     const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
#ifdef __aarch64__
    const int mr = 8;
#else
    const int mr = 4;
#endif
    const size_t kh         = conv_param->kernels[1];
    const size_t kw         = conv_param->kernels[0];
    auto dims_input         = inputs[0]->GetBlobDesc().dims;
    auto dims_output        = outputs[0]->GetBlobDesc().dims;
    const size_t ic         = dims_input[1];
    const size_t ic_r4      = ROUND_UP(ic, 4);
    const size_t oh         = dims_output[2];
    const size_t ow         = dims_output[3];
    const size_t ih         = dims_input[2];
    const size_t iw         = dims_input[3];
    const size_t oc         = dims_output[1];
    const size_t stride_h   = conv_param->strides[1];
    const size_t stride_w   = conv_param->strides[0];
    const size_t dilation_h = conv_param->dialations[1];
    const size_t dilation_w = conv_param->dialations[0];
    const size_t pad_h      = conv_param->pads[1];
    const size_t pad_w      = conv_param->pads[0];

    const size_t kernel_size             = kh * kw;
    const size_t output_size             = oh * ow;
    const size_t tiled_output_size       = ROUND_UP(output_size, mr);
    const size_t output_tile_size        = mr;
    const size_t indirection_buffer_size = sizeof(int32_t) * tiled_output_size * kernel_size;
    {
        RawBuffer temp_buffer(indirection_buffer_size);
        indirection_buffer_ = temp_buffer;
    }
    {
        const size_t zero_buffer_size = sizeof(int8_t) * ic_r4 + 8;
        RawBuffer temp_buffer(zero_buffer_size);
        zero_buffer_        = temp_buffer;
        zero_buffer_offset_ = 8;
    }
    RETURN_ON_NEQ(initIndirectionBuffer(inputs, outputs, mr, tiled_output_size), TNN_OK);
    blob_allocated_ = true;
    return TNN_OK;
}

Status ArmConvInt8LayerIndirect::Init(Context *context, LayerParam *param, LayerResource *resource,
                                      const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferWeightBias(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferScale(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferInput(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(setFusionParam(inputs, outputs), TNN_OK);

    // init base k_param_
    k_param_->scale   = buffer_scale_.force_to<float *>();
    k_param_->fil_ptr = buffer_weight_.force_to<void *>();
    blob_allocated_   = false;

    return TNN_OK;
}

Status ArmConvInt8LayerIndirect::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    auto input     = inputs[0];
    auto output    = outputs[0];
    auto add_input = (conv_param->fusion_type == FusionType_None) ? nullptr : inputs[1];

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

#ifdef __aarch64__
    const int mr = 8;
#else
    const int mr = 4;
#endif
    const int nr           = 8;
    const int kr           = 1;
    auto dims_input        = input->GetBlobDesc().dims;
    auto dims_output       = output->GetBlobDesc().dims;
    const int batch_size   = dims_output[0];
    int oh                 = dims_output[2];
    int ow                 = dims_output[3];
    int ic                 = dims_input[1];
    int ic_r4              = ROUND_UP(ic, 4);
    int oc                 = dims_output[1];
    int oc_r4              = ROUND_UP(oc, 4);
    int m_stride           = ROUND_UP(oh * ow, mr);
    int n_stride           = ROUND_UP(oc, nr);
    int kh                 = conv_param->kernels[1];
    int kw                 = conv_param->kernels[0];
    int8_t *input_data     = reinterpret_cast<int8_t *>(GetBlobHandlePtr(input->GetHandle()));
    int8_t *output_data    = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));
    int8_t *add_input_data = add_input ? reinterpret_cast<int8_t *>(GetBlobHandlePtr(add_input->GetHandle())) : nullptr;

    struct Q8ConvContext context = {.ks         = kh * kw,
                                    .kc         = ic,
                                    .kc_stride  = ic * kh * kw,
                                    .m          = oh * ow,
                                    .m_stride   = m_stride,
                                    .n          = oc,
                                    .n_stride   = n_stride,
                                    .indirect_a = indirection_buffer_.force_to<const int32_t *>(),
                                    .packed_w   = reinterpret_cast<int8_t *>(k_param_->fil_ptr),
                                    .c          = output_data,
                                    .c_stride   = oc_r4,
                                    .scales     = reinterpret_cast<float *>(k_param_->scale),
                                    .relu       = relu_,
                                    .add_input  = add_input_data,
                                    .add_scale  = buffer_add_scale_.force_to<float *>(),
                                    .zero       = zero_buffer_.force_to<int8_t *>() + zero_buffer_offset_,
                                    .real_input = reinterpret_cast<int8_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()))};
    ComputeQ8Conv(&context, oh * ow, oc, mr, nr);
    return TNN_OK;
}

}  // namespace TNN_NS
