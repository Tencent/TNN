//  Copyright © 2019 tencent. All rights reserved.

#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_winograd.h"
#include "tnn/device/arm/acc/compute/winograd_function_int8.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

bool ArmConvInt8LayerWinograd::isPrefered(ConvLayerParam *param,
                                          const std::vector<Blob *> &inputs,
                                          const std::vector<Blob *> &outputs) {
	return false;
#if !defined(TNN_USE_NEON) || !defined(__aarch64__)
    return false;
#else
    if (inputs[0]->GetBlobDesc().data_type != DATA_TYPE_INT8) {
        return false;
    }
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    auto hw_ok       = dims_output[2] >= 2 && dims_output[3] >= 2;

    //auto channel_ok = dims_output[1] % 8 == 0 && dims_input[1] % 8 == 0;
    auto channel_ok = (dims_input[1] % 16 == 0) || (dims_input[1] % 16 > 7);
    auto param_ok = param->strides[0] == 1 && param->strides[1] == 1 &&
                    param->dialations[1] == 1 && param->group == 1 &&
                    param->kernels[0] == 3 && param->kernels[1] == 3 &&
                    param->pads[0] == param->pads[2];
    bool res = hw_ok && channel_ok && param_ok;
    //if (res) {
    //    printf ("use winograd: %s, (%d, %d, %d, %d)\n", param->name.c_str(), dims_output[0], dims_output[1], dims_output[2], dims_output[3]);
    //}
    return res;
#endif
}

Status ArmConvInt8LayerWinograd::Init(Context *context, LayerParam *param,
                                      LayerResource *resource,
                                      const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(setFusionParam(inputs, outputs), TNN_OK);

    return this->Reshape(inputs, outputs);
}

ArmConvInt8LayerWinograd::~ArmConvInt8LayerWinograd() {}

Status ArmConvInt8LayerWinograd::allocateBufferWeight(
    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
#if !defined(TNN_USE_NEON) || !defined(__aarch64__)
    return TNNERR_LAYER_ERR;
#else
    if (!buffer_weight_.GetBytesSize()) {
        ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
        ConvLayerResource *conv_res =
            dynamic_cast<ConvLayerResource *>(resource_);
        auto filter = conv_res->filter_handle.force_to<int8_t *>();

        auto ic     = inputs[0]->GetBlobDesc().dims[1];
        auto oc     = outputs[0]->GetBlobDesc().dims[1];
        auto ic_r16 = ROUND_UP(ic, 16);
        auto oc_r4  = ROUND_UP(oc, 4);
        const int buffer_size =
            ic_r16 * oc_r4 * 16 * sizeof(int8_t) + NEON_KERNEL_EXTRA_LOAD;
            //ic_r16 * oc * 16 * sizeof(int8_t) + NEON_KERNEL_EXTRA_LOAD;

        RawBuffer temp_buffer(buffer_size);
        memset(temp_buffer.force_to<void *>(), 0, buffer_size);
        buffer_weight_ = temp_buffer;
        /// aloha temp hack here, using int8 weight convert, wait for convert in
        /// quantization
        weight_convert(filter, buffer_weight_.force_to<int8_t *>(), ic, oc);
    }

    return TNN_OK;
#endif
}

Status ArmConvInt8LayerWinograd::allocateBufferBias(
    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);

    if (!buffer_bias_.GetBytesSize()) {
        auto dims_output = outputs[0]->GetBlobDesc().dims;
        int total_byte_size =
            ROUND_UP(dims_output[1], 4) *
            DataTypeUtils::GetBytesSize(conv_res->bias_handle.GetDataType());

        const int bias_handle_size = conv_res->bias_handle.GetBytesSize();
        const int32_t *bias_handle_data =
            conv_res->bias_handle.force_to<int32_t *>();

        RawBuffer temp_buffer(total_byte_size);
        memset(temp_buffer.force_to<void*>(), 0, total_byte_size);
        for (int i = 0; i < bias_handle_size / sizeof(int32_t); i++) {
            temp_buffer.force_to<int32_t *>()[i] =
                conv_res->bias_handle.force_to<int32_t *>()[i] * 4;
        }

        buffer_bias_ = temp_buffer;
    }

    return TNN_OK;
}

Status ArmConvInt8LayerWinograd::allocateBufferScale(
    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);

    if (!buffer_scale_.GetBytesSize()) {
        auto dims_output = outputs[0]->GetBlobDesc().dims;
        int total_byte_size =
            ROUND_UP(dims_output[1], 4) *
            DataTypeUtils::GetBytesSize(conv_res->scale_handle.GetDataType());

        const int scale_handle_size = conv_res->scale_handle.GetBytesSize();
        const float *w_scale = conv_res->scale_handle.force_to<float *>();

        const float *o_scale = reinterpret_cast<BlobInt8 *>(outputs[0])
                                   ->GetIntResource()
                                   ->scale_handle.force_to<float *>();

        int scale_len_w = conv_res->scale_handle.GetDataCount();
        int scale_len_o = reinterpret_cast<BlobInt8 *>(outputs[0])
                              ->GetIntResource()
                              ->scale_handle.GetDataCount();
        RawBuffer temp_buffer(total_byte_size);
        memset(temp_buffer.force_to<void*>(), 0, total_byte_size);
        float *temp_ptr = temp_buffer.force_to<float *>();
        for (int i = 0; i < dims_output[1]; i++) {
            int scale_idx_w = scale_len_w == 1 ? 0 : i;
            int scale_idx_o = scale_len_o == 1 ? 0 : i;

            if (o_scale[scale_idx_o] >= FLT_MIN)
                // div 4 due to winograd
                temp_ptr[i] = w_scale[scale_idx_w] / 4.0 / o_scale[scale_idx_o];
            else
                temp_ptr[i] = 0.0;
        }
        buffer_scale_ = temp_buffer;
    }

    return TNN_OK;
}

Status ArmConvInt8LayerWinograd::allocateBufferWinoBuf(
    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto ic                 = inputs[0]->GetBlobDesc().dims[1];
    auto output_dims        = outputs[0]->GetBlobDesc().dims;
    auto oc                 = output_dims[1];
    auto oh                 = output_dims[2];
    auto ow                 = output_dims[3];
    auto ic_r16             = ROUND_UP(ic, 16);
    auto oc_r4              = ROUND_UP(oc, 4);
    const int hw_tile_size_ = 4 * 4;
    const int hw_tile_cnt_  = 2 * 2;
    // buffer for padding temp
    int max_num_threads = OMP_CORES_;
    if (!buffer_srcpad_.GetBytesSize()) {
        const int buffer_size =
            hw_tile_size_ * ic * 4 * sizeof(int8_t) * max_num_threads;

        RawBuffer temp_buffer(buffer_size);
        memset(temp_buffer.force_to<void *>(), 0, buffer_size);
        buffer_srcpad_ = temp_buffer;
    }

    // buffer for src winograd convert
    if (!buffer_srcwino_.GetBytesSize()) {
        const int buffer_size = hw_tile_cnt_ * hw_tile_size_ * ic_r16 *
                                    sizeof(int8_t) * max_num_threads +
                                NEON_KERNEL_EXTRA_LOAD;

        RawBuffer temp_buffer(buffer_size);
        memset(temp_buffer.force_to<void *>(), 0, buffer_size);
        buffer_srcwino_ = temp_buffer;
    }

    // buffer for dst winograd convert
    if (!buffer_dstwino_.GetBytesSize()) {
        const int buffer_size = hw_tile_cnt_ * hw_tile_size_ * 4 *
                                sizeof(int32_t) * max_num_threads;

        RawBuffer temp_buffer(buffer_size);
        memset(temp_buffer.force_to<void *>(), 0, buffer_size);
        buffer_dstwino_ = temp_buffer;
    }
    return TNN_OK;
}

Status ArmConvInt8LayerWinograd::Reshape(const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {
    allocateBufferWeight(inputs, outputs);
    allocateBufferBias(inputs, outputs);
    allocateBufferScale(inputs, outputs);
    allocateBufferWinoBuf(inputs, outputs);
    return TNN_OK;
}

Status ArmConvInt8LayerWinograd::DoForward(const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {
#if !defined(TNN_USE_NEON) || !defined(__aarch64__)
    return TNNERR_LAYER_ERR;
#else
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    auto dims_input            = inputs[0]->GetBlobDesc().dims;
    auto dims_output           = outputs[0]->GetBlobDesc().dims;
    auto add_input = (conv_param->fusion_type == FusionType_None) ? nullptr : inputs[1];
    auto oc_r4                 = ROUND_UP(dims_output[1], 4);
    auto ic_r4                 = ROUND_UP(dims_input[1], 4);
    auto *src    = reinterpret_cast<int8_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    auto *dst    = reinterpret_cast<int8_t *>(GetBlobHandlePtr(outputs[0]->GetHandle());
    int8_t* add_input_data = add_input ? reinterpret_cast<int8_t *>(GetBlobHandlePtr(add_input->GetHandle())) : nullptr;
    auto *bias   = buffer_bias_.force_to<int32_t *>();
    auto *scale  = buffer_scale_.force_to<float *>();
    auto *weight = buffer_weight_.force_to<int8_t *>();
    auto *src_pad_buf  = buffer_srcpad_.force_to<int8_t *>();
    auto *src_wino_buf = buffer_srcwino_.force_to<int8_t *>();
    auto *dst_wino_buf = buffer_dstwino_.force_to<int32_t *>();
    for (int n = 0; n < dims_output[0]; ++n) {
        auto src_n = src + n * dims_input[3] * dims_input[2] * ic_r4;
        auto dst_n = dst + n * dims_output[3] * dims_output[2] * oc_r4;
        int8_t* add_input_n = nullptr;
        if (add_input) {
            auto dims_add_input        = inputs[1]->GetBlobDesc().dims;
            add_input_n = add_input_data + n * dims_add_input[3] * dims_add_input[2] * oc_r4;
        }
        kernel4x4(ic_r4, dims_input[2], dims_input[3], oc_r4,
                  dims_output[2], dims_output[3], src_n, weight, dst_n, scale,
                  bias, conv_param->pads[0], src_pad_buf, src_wino_buf,
                  dst_wino_buf, relu_, add_input_n, buffer_add_scale_.force_to<float *>());
        //if (conv_param->activation_type == ActivationType_ReLU) {
        //    ReluInt8(dst_n, dst_n,
        //                oc_r4 * dims_output[2] * dims_output[3]);
        //}
    }
    return TNN_OK;
#endif
}
}  // namespace TNN_NS
