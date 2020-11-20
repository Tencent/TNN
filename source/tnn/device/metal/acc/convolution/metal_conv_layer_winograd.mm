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

#include "tnn/device/metal/acc/convolution/metal_conv_layer_winograd.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/half_utils.h"
#include "tnn/utils/winograd_generator.h"

namespace TNN_NS {
bool MetalConvLayerWinograd::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs) {
    //        return false;
    if (!param) {
        return false;
    }

    if (param->group != 1) {
        return false;
    }

    if (param->group != 1 || param->kernels[0] != 3 || param->kernels[1] != 3 || param->dialations[0] != 1 ||
        param->dialations[1] != 1 || param->strides[0] != 1 || param->strides[1] != 1) {
        return false;
    }

    auto iw = inputs[0]->GetBlobDesc().dims[3];
    auto ih = inputs[0]->GetBlobDesc().dims[2];
    auto ic = ROUND_UP(inputs[0]->GetBlobDesc().dims[1], 4);
    auto oc = ROUND_UP(outputs[0]->GetBlobDesc().dims[1], 4);
    return ic * oc * ih / iw >= 2048;
}

MetalConvLayerWinograd::~MetalConvLayerWinograd() {}

Status MetalConvLayerWinograd::AllocateBufferWeight(const std::vector<Blob *> &inputs,
                                                    const std::vector<Blob *> &outputs) {
    id<MTLDevice> device        = [TNNMetalDeviceImpl sharedDevice];
    ConvLayerParam *conv_param  = dynamic_cast<ConvLayerParam *>(param_);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);

    auto dims_input          = inputs[0]->GetBlobDesc().dims;
    auto dims_output         = outputs[0]->GetBlobDesc().dims;
    const int input_channel  = dims_input[1];
    const int output_channel = dims_output[1];

    Status status = TNN_OK;
    if (!buffer_weight_) {
        const int dst_unit = 2;

        const int kw = conv_param->kernels[0];
        const int kh = conv_param->kernels[1];

        const int weight_bytes_count = conv_res->filter_handle.GetBytesSize();
        const float *weight_fp32     = conv_res->filter_handle.force_to<float *>();
        const uint16_t *weight_fp16  = conv_res->filter_handle.force_to<uint16_t *>();
        const DataType data_type     = conv_res->filter_handle.GetDataType();

        if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
            LOGE("Error: DataType %d not support\n", data_type);
            return Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        }

        //转float
        if (data_type == DATA_TYPE_HALF) {
            weight_fp32 = new float[weight_bytes_count / 2];
            ConvertFromHalfToFloat((void *)weight_fp16, (float *)weight_fp32, weight_bytes_count / 2);
        }

        //预处理
        WinogradGenerator generator(dst_unit, kh, 1.0f);
        auto pack_weight_fp32 = generator.allocTransformWeight(output_channel, input_channel, kh, kw, 4, 4);
        generator.transformWeight(pack_weight_fp32, weight_fp32, output_channel, input_channel, kh, kw);

        auto pack_weight_fp32_data = get<0>(pack_weight_fp32).get();
        auto pack_weight_fp32_dims = get<1>(pack_weight_fp32);
        int pack_weight_count      = DimsVectorUtils::Count(pack_weight_fp32_dims);

#if TNN_METAL_FULL_PRECISION
        {
            buffer_weight_ = [device newBufferWithBytes:pack_weight_fp32_data
                                                 length:pack_weight_count * sizeof(float)
                                                options:MTLResourceCPUCacheModeWriteCombined];
        }
#else
        {
            auto pack_weight_fp16_data = new uint16_t[pack_weight_count];
            ConvertFromFloatToHalf((float *)pack_weight_fp32_data, (void *)pack_weight_fp16_data, pack_weight_count);
            buffer_weight_ = [device newBufferWithBytes:pack_weight_fp16_data
                                                 length:pack_weight_count * sizeof(uint16_t)
                                                options:MTLResourceCPUCacheModeWriteCombined];
            delete[] pack_weight_fp16_data;
        }

#endif
        if (data_type == DATA_TYPE_HALF) {
            delete[] weight_fp32;
        }
    }
    return status;
}

Status MetalConvLayerWinograd::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    id<MTLDevice> device       = [TNNMetalDeviceImpl sharedDevice];
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);

    const int src_unit = 2 + conv_param->kernels[1] - 1;
    const int dst_nit  = 2;

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    auto ow = dims_output[3];
    auto oh = dims_output[2];
    auto uw = UP_DIV(ow, dst_nit);
    auto uh = UP_DIV(oh, dst_nit);
    auto us = UP_DIV(uw * uh, 4);
    auto iz = UP_DIV(dims_input[1], 4);
    auto oz = UP_DIV(dims_output[1], 4);

    // buffer_param_
    {
        MetalWinogradParams metal_params;
        metal_params.activation = conv_param->activation_type;
        metal_params.has_bias   = conv_param->bias;

        SetDefaultMetalParams(metal_params, dims_input, dims_output);

        metal_params.pad_x       = conv_param->pads[0];
        metal_params.pad_y       = conv_param->pads[2];
        metal_params.unit_width  = uw;
        metal_params.unit_height = uh;
        metal_params.unit        = dst_nit;

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalWinogradParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    // buffer_shape_
    {
        MetalMatMul4x4Params metal_params;
        metal_params.output_width  = us;
        metal_params.output_height = oz;
        metal_params.multi_length  = iz;
        metal_params.group         = src_unit * src_unit;

        buffer_shape_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalMatMul4x4Params)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    // save threads size
    {
        input_transform_threads_.width   = uw;
        input_transform_threads_.height  = uh;
        input_transform_threads_.depth   = iz;
        matmul_threads_.width            = us;
        matmul_threads_.height           = oz;
        matmul_threads_.depth            = src_unit * src_unit;
        output_transform_threads_.width  = uw;
        output_transform_threads_.height = uh;
        output_transform_threads_.depth  = oz;
    }

    {
        int data_byte_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);

        // accquire space
        int is = src_unit * src_unit * us * iz * 16 * data_byte_size;
        int os = src_unit * src_unit * us * oz * 16 * data_byte_size;
#if TNN_METAL_DEBUG
        buffer_temp_input_  = [device newBufferWithLength:is options:MTLResourceCPUCacheModeDefaultCache];
        buffer_temp_output_ = [device newBufferWithLength:os options:MTLResourceCPUCacheModeDefaultCache];
#else
        buffer_temp_input_  = [device newBufferWithLength:is options:MTLResourceStorageModePrivate];
        buffer_temp_output_ = [device newBufferWithLength:os options:MTLResourceStorageModePrivate];
#endif
    }

    return TNN_OK;
}

Status MetalConvLayerWinograd::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_output = output->GetBlobDesc().dims;

    auto context_impl = context_->getMetalContextImpl();
    auto encoder      = [context_impl encoder];
    encoder.label = GetKernelLabel();

    Status status = TNN_OK;
    MetalBandwidth bandwidth;

    do {
        { // transform
            status = [context_impl load:@"winograd_transform_source2_3_1" encoder:encoder bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);

            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                        offset:(NSUInteger)input->GetHandle().bytes_offset
                       atIndex:0];
            [encoder setBuffer:buffer_temp_input_ offset:0 atIndex:1];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
            status = [context_impl dispatchEncoder:encoder threads:input_transform_threads_ bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
        }
        { // gemm
            status = [context_impl load:@"matmul4x4" encoder:encoder bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);

            [encoder setBuffer:buffer_temp_input_ offset:0 atIndex:0];
            [encoder setBuffer:buffer_temp_output_ offset:0 atIndex:1];
            [encoder setBuffer:buffer_weight_ offset:0 atIndex:2];
            [encoder setBuffer:buffer_shape_ offset:0 atIndex:3];
            status = [context_impl dispatchEncoder:encoder threads:matmul_threads_ bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
        }
        { // transform
            status = [context_impl load:@"winograd_transform_dest2_3_1" encoder:encoder bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);

            [encoder setBuffer:buffer_temp_output_ offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                        offset:(NSUInteger)output->GetHandle().bytes_offset
                       atIndex:1];
            [encoder setBuffer:buffer_bias_ offset:0 atIndex:2];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:3];
            status = [context_impl dispatchEncoder:encoder threads:output_transform_threads_ bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
        }
    } while (0);

    [encoder endEncoding];
    [context_impl commit];
    TNN_PRINT_ENCODER(context_, encoder, this);
    return status;
}

} // namespace TNN_NS
