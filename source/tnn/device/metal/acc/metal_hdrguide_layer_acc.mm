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

#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/acc/metal_layer_acc.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils.h"

namespace TNN_NS {

class MetalHDRGuideLayerAcc : public MetalLayerAcc {
public:
    virtual ~MetalHDRGuideLayerAcc(){};

    Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    id<MTLBuffer> buffer_ccm_weight_ = nil;
    id<MTLBuffer> buffer_ccm_bias_   = nil;
    id<MTLBuffer> buffer_shifts_     = nil;
    id<MTLBuffer> buffer_slopes_     = nil;
    id<MTLBuffer> buffer_projection_ = nil;

    Status AllocateBufferCCMWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    Status AllocateBufferCCMBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    Status AllocateBufferSlopes(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    Status AllocateBufferShifts(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    Status AllocateBufferProjection(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
};

Status MetalHDRGuideLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalHDRGuideLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                  const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    auto layer_res = dynamic_cast<HdrGuideLayerResource *>(resource_);
    if (!layer_res) {
        return Status(TNNERR_MODEL_ERR, "Error: HDRGuideLayerResource is nil");
    }

    Status status = TNN_OK;
    // buffer_param_
    {
        auto metal_params = GetDefaultMetalParams(dims_input, dims_output);
        buffer_param_     = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(metal_params)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    status = AllocateBufferCCMWeight(inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }

    status = AllocateBufferCCMBias(inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }

    status = AllocateBufferSlopes(inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }

    status = AllocateBufferShifts(inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }

    status = AllocateBufferProjection(inputs, outputs);
    return status;
}

Status MetalHDRGuideLayerAcc::AllocateBufferCCMWeight(const std::vector<Blob *> &inputs,
                                                      const std::vector<Blob *> &outputs) {
    auto layer_res = dynamic_cast<HdrGuideLayerResource *>(resource_);
    if (!layer_res) {
        return Status(TNNERR_MODEL_ERR, "Error: HDRGuideLayerResource is nil");
    }

    Status status = TNN_OK;
    if (!buffer_ccm_weight_) {
        int kw = 1;
        int kh = 1;

        const int input_channel  = 3;
        const int output_channel = 3;

        const int group = 1;

        buffer_ccm_weight_ = AllocatePackedGOIHW16MetalBufferFormRawBuffer(
            layer_res->ccm_weight_handle, {output_channel, input_channel, kh, kw}, group, status);
    }
    return status;
}

Status MetalHDRGuideLayerAcc::AllocateBufferCCMBias(const std::vector<Blob *> &inputs,
                                                    const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto layer_res       = dynamic_cast<HdrGuideLayerResource *>(resource_);
    if (!layer_res) {
        return Status(TNNERR_MODEL_ERR, "Error: HDRGuideLayerResource is nil");
    }

    if (!buffer_ccm_bias_) {
        const float *ccm_bias_data = layer_res->ccm_bias_handle.force_to<float *>();
        const DataType data_type   = layer_res->ccm_bias_handle.GetDataType();

        if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
            LOGE("Error: DataType %d not support\n", data_type);
            return Status(TNNERR_MODEL_ERR, "k_handle DataType is not supported");
        }

#if TNN_METAL_FULL_PRECISION
        if (data_type == DATA_TYPE_FLOAT) {
            float *ccm_bias_data_fp32 = (float *)ccm_bias_data;

            //补齐
            float data_fill_4[4] = {
                ccm_bias_data_fp32[0],
                ccm_bias_data_fp32[1],
                ccm_bias_data_fp32[2],
                0,
            };

            buffer_ccm_bias_ = [device newBufferWithBytes:(const void *)data_fill_4
                                                   length:4 * sizeof(float)
                                                  options:MTLResourceCPUCacheModeWriteCombined];
        } else if (data_type == DATA_TYPE_HALF) {
            uint16_t *ccm_bias_data_fp16 = (uint16_t *)ccm_bias_data;

            //补齐
            uint16_t data_fill_4[4] = {
                ccm_bias_data_fp16[0],
                ccm_bias_data_fp16[1],
                ccm_bias_data_fp16[2],
                0,
            };

            // convert to float
            float *data_fp32_data = new float[4];
            ConvertFromHalfToFloat((void *)data_fill_4, data_fp32_data, 4);

            buffer_ccm_bias_ = [device newBufferWithBytes:(const void *)data_fp32_data
                                                   length:4 * sizeof(float)
                                                  options:MTLResourceCPUCacheModeWriteCombined];
            delete[] data_fp32_data;
        }
#else
        if (data_type == DATA_TYPE_FLOAT) {
            //补齐
            float data_fill_4[4] = {
                ccm_bias_data[0],
                ccm_bias_data[1],
                ccm_bias_data[2],
                0.0f,
            };

            // convert to half
            uint16_t *data_fp16_data = new uint16_t[4];
            ConvertFromFloatToHalf((float *)data_fill_4, (void *)data_fp16_data, 4);

            buffer_ccm_bias_ = [device newBufferWithBytes:(const void *)data_fp16_data
                                                   length:4 * sizeof(uint16_t)
                                                  options:MTLResourceCPUCacheModeWriteCombined];
            delete[] data_fp16_data;
        } else if (data_type == DATA_TYPE_HALF) {
            uint16_t *ccm_bias_data_fp16 = (uint16_t *)ccm_bias_data;

            //补齐
            uint16_t data_fill_4[4] = {
                ccm_bias_data_fp16[0],
                ccm_bias_data_fp16[1],
                ccm_bias_data_fp16[2],
                0,
            };

            buffer_ccm_bias_ = [device newBufferWithBytes:(const void *)data_fill_4
                                                   length:4 * sizeof(uint16_t)
                                                  options:MTLResourceCPUCacheModeWriteCombined];
        }
#endif
    }
    return TNN_OK;
}

Status MetalHDRGuideLayerAcc::AllocateBufferSlopes(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto layer_res       = dynamic_cast<HdrGuideLayerResource *>(resource_);
    if (!layer_res) {
        return Status(TNNERR_MODEL_ERR, "Error: HDRGuideLayerResource is nil");
    }
    if (!buffer_slopes_) {
        const float *slope_data  = layer_res->slopes_handle.force_to<float *>();
        const DataType data_type = layer_res->ccm_bias_handle.GetDataType();

        if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
            LOGE("Error: DataType %d not support\n", data_type);
            return Status(TNNERR_MODEL_ERR, "k_handle DataType is not supported");
        }

#if TNN_METAL_FULL_PRECISION
        if (data_type == DATA_TYPE_FLOAT) {
            float *slope_data_fp32 = (float *)slope_data;

            //补齐
            float data_fill_4[16] = {
                slope_data_fp32[0], slope_data_fp32[4], slope_data_fp32[8],  0,
                slope_data_fp32[1], slope_data_fp32[5], slope_data_fp32[9],  0,
                slope_data_fp32[2], slope_data_fp32[6], slope_data_fp32[10], 0,
                slope_data_fp32[3], slope_data_fp32[7], slope_data_fp32[11], 0,
            };

            buffer_slopes_ = [device newBufferWithBytes:(const void *)data_fill_4
                                                 length:16 * sizeof(float)
                                                options:MTLResourceCPUCacheModeWriteCombined];
        } else if (data_type == DATA_TYPE_HALF) {
            uint16_t *slope_data_fp16 = (uint16_t *)slope_data;

            //补齐
            uint16_t data_fill_4[16] = {
                slope_data_fp16[0], slope_data_fp16[4], slope_data_fp16[8],  0,
                slope_data_fp16[1], slope_data_fp16[5], slope_data_fp16[9],  0,
                slope_data_fp16[2], slope_data_fp16[6], slope_data_fp16[10], 0,
                slope_data_fp16[3], slope_data_fp16[7], slope_data_fp16[11], 0,
            };

            // convert to float
            float *data_fp32_data = new float[16];
            ConvertFromHalfToFloat((void *)data_fill_4, data_fp32_data, 16);

            buffer_slopes_ = [device newBufferWithBytes:(const void *)data_fp32_data
                                                 length:16 * sizeof(float)
                                                options:MTLResourceCPUCacheModeWriteCombined];
            delete[] data_fp32_data;
        }
#else
        if (data_type == DATA_TYPE_FLOAT) {
            //补齐
            float data_fill_4[16] = {
                slope_data[0], slope_data[4], slope_data[8],  0.0f, slope_data[1], slope_data[5], slope_data[9],  0.0f,
                slope_data[2], slope_data[6], slope_data[10], 0.0f, slope_data[3], slope_data[7], slope_data[11], 0.0f,
            };

            // convert to half
            uint16_t *data_fp16_data = new uint16_t[16];
            ConvertFromFloatToHalf((float *)data_fill_4, (void *)data_fp16_data, 16);

            buffer_slopes_ = [device newBufferWithBytes:(const void *)data_fp16_data
                                                 length:16 * sizeof(uint16_t)
                                                options:MTLResourceCPUCacheModeWriteCombined];
            delete[] data_fp16_data;
        } else if (data_type == DATA_TYPE_HALF) {
            uint16_t *slope_data_fp16 = (uint16_t *)slope_data;

            //补齐
            uint16_t data_fill_4[16] = {
                slope_data_fp16[0], slope_data_fp16[4], slope_data_fp16[8],  0,
                slope_data_fp16[1], slope_data_fp16[5], slope_data_fp16[9],  0,
                slope_data_fp16[2], slope_data_fp16[6], slope_data_fp16[10], 0,
                slope_data_fp16[3], slope_data_fp16[7], slope_data_fp16[11], 0,
            };

            buffer_slopes_ = [device newBufferWithBytes:(const void *)data_fill_4
                                                 length:16 * sizeof(uint16_t)
                                                options:MTLResourceCPUCacheModeWriteCombined];
        }
#endif
    }
    return TNN_OK;
}

Status MetalHDRGuideLayerAcc::AllocateBufferShifts(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto layer_res       = dynamic_cast<HdrGuideLayerResource *>(resource_);
    if (!layer_res) {
        return Status(TNNERR_MODEL_ERR, "Error: HDRGuideLayerResource is nil");
    }

    if (!buffer_shifts_) {
        const float *shifts_data = layer_res->shifts_handle.force_to<float *>();
        const DataType data_type = layer_res->shifts_handle.GetDataType();

        if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
            LOGE("Error: DataType %d not support\n", data_type);
            return Status(TNNERR_MODEL_ERR, "k_handle DataType is not supported");
        }

#if TNN_METAL_FULL_PRECISION
        if (data_type == DATA_TYPE_FLOAT) {
            float *shifts_data_fp32 = (float *)shifts_data;

            //补齐
            float data_fill_4[16] = {
                shifts_data_fp32[0], shifts_data_fp32[4], shifts_data_fp32[8],  0,
                shifts_data_fp32[1], shifts_data_fp32[5], shifts_data_fp32[9],  0,
                shifts_data_fp32[2], shifts_data_fp32[6], shifts_data_fp32[10], 0,
                shifts_data_fp32[3], shifts_data_fp32[7], shifts_data_fp32[11], 0,
            };

            buffer_shifts_ = [device newBufferWithBytes:(const void *)data_fill_4
                                                 length:16 * sizeof(float)
                                                options:MTLResourceCPUCacheModeWriteCombined];

        } else if (data_type == DATA_TYPE_HALF) {
            uint16_t *shifts_data_fp16 = (uint16_t *)shifts_data;

            //补齐
            uint16_t data_fill_4[16] = {
                shifts_data_fp16[0], shifts_data_fp16[4], shifts_data_fp16[8],  0,
                shifts_data_fp16[1], shifts_data_fp16[5], shifts_data_fp16[9],  0,
                shifts_data_fp16[2], shifts_data_fp16[6], shifts_data_fp16[10], 0,
                shifts_data_fp16[3], shifts_data_fp16[7], shifts_data_fp16[11], 0,
            };

            // convert to float
            float *data_fp32_data = new float[16];
            ConvertFromHalfToFloat((void *)data_fill_4, data_fp32_data, 16);

            buffer_shifts_ = [device newBufferWithBytes:(const void *)data_fp32_data
                                                 length:16 * sizeof(float)
                                                options:MTLResourceCPUCacheModeWriteCombined];
            delete[] data_fp32_data;
        }
#else
        if (data_type == DATA_TYPE_FLOAT) {
            //补齐
            float data_fill_4[16] = {
                shifts_data[0], shifts_data[4], shifts_data[8],  0.0f,           shifts_data[1],  shifts_data[5],
                shifts_data[9], 0.0f,           shifts_data[2],  shifts_data[6], shifts_data[10], 0.0f,
                shifts_data[3], shifts_data[7], shifts_data[11], 0.0f,
            };

            // convert to half
            uint16_t *data_fp16_data = new uint16_t[16];
            ConvertFromFloatToHalf((float *)data_fill_4, (void *)data_fp16_data, 16);

            buffer_shifts_ = [device newBufferWithBytes:(const void *)data_fp16_data
                                                 length:16 * sizeof(uint16_t)
                                                options:MTLResourceCPUCacheModeWriteCombined];
            delete[] data_fp16_data;
        } else if (data_type == DATA_TYPE_HALF) {
            uint16_t *shifts_data_fp16 = (uint16_t *)shifts_data;

            //补齐
            uint16_t data_fill_4[16] = {
                shifts_data_fp16[0], shifts_data_fp16[4], shifts_data_fp16[8],  0,
                shifts_data_fp16[1], shifts_data_fp16[5], shifts_data_fp16[9],  0,
                shifts_data_fp16[2], shifts_data_fp16[6], shifts_data_fp16[10], 0,
                shifts_data_fp16[3], shifts_data_fp16[7], shifts_data_fp16[11], 0,
            };

            buffer_shifts_ = [device newBufferWithBytes:(const void *)data_fill_4
                                                 length:16 * sizeof(uint16_t)
                                                options:MTLResourceCPUCacheModeWriteCombined];
        }
#endif
    }

    return TNN_OK;
}

Status MetalHDRGuideLayerAcc::AllocateBufferProjection(const std::vector<Blob *> &inputs,
                                                       const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto layer_res       = dynamic_cast<HdrGuideLayerResource *>(resource_);
    if (!layer_res) {
        return Status(TNNERR_MODEL_ERR, "Error: HDRGuideLayerResource is nil");
    }

    if (!buffer_projection_) {
        const float *projection_weight_data = layer_res->projection_weight_handle.force_to<float *>();
        const float *projection_bias_data   = layer_res->projection_bias_handle.force_to<float *>();
        const DataType data_type            = layer_res->projection_weight_handle.GetDataType();

        if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
            LOGE("Error: DataType %d not support\n", data_type);
            return Status(TNNERR_MODEL_ERR, "b_handle DataType is not supported");
        }

#if TNN_METAL_FULL_PRECISION
        if (data_type == DATA_TYPE_FLOAT) {
            float *projection_weight_data_fp32 = (float *)projection_weight_data;
            float *projection_bias_data_fp32   = (float *)projection_bias_data;
            //补齐
            float projection_data[4] = {
                projection_weight_data_fp32[0],
                projection_weight_data_fp32[1],
                projection_weight_data_fp32[2],
                projection_bias_data_fp32[0],
            };

            buffer_projection_ = [device newBufferWithBytes:(const void *)projection_data
                                                     length:4 * sizeof(float)
                                                    options:MTLResourceCPUCacheModeWriteCombined];
        } else if (data_type == DATA_TYPE_HALF) {
            uint16_t *projection_weight_data_fp16 = (uint16_t *)projection_weight_data;
            uint16_t *projection_bias_data_fp16   = (uint16_t *)projection_bias_data;
            //补齐
            uint16_t projection_data[4] = {
                projection_weight_data_fp16[0],
                projection_weight_data_fp16[1],
                projection_weight_data_fp16[2],
                projection_bias_data_fp16[0],
            };

            // convert to float
            float *data_fp32_data = new float[4];
            ConvertFromHalfToFloat((void *)projection_data, data_fp32_data, 4);

            buffer_projection_ = [device newBufferWithBytes:(const void *)data_fp32_data
                                                     length:4 * sizeof(float)
                                                    options:MTLResourceCPUCacheModeWriteCombined];
            delete[] data_fp32_data;
        }
#else
        if (data_type == DATA_TYPE_FLOAT) {
            //补齐
            float projection_data[4] = {
                projection_weight_data[0],
                projection_weight_data[1],
                projection_weight_data[2],
                projection_bias_data[0],
            };

            // convert to half
            uint16_t *data_fp16_data = new uint16_t[4];
            ConvertFromFloatToHalf((float *)projection_data, (void *)data_fp16_data, 4);

            buffer_projection_ = [device newBufferWithBytes:(const void *)data_fp16_data
                                                     length:4 * sizeof(uint16_t)
                                                    options:MTLResourceCPUCacheModeWriteCombined];
            delete[] data_fp16_data;
        } else if (data_type == DATA_TYPE_HALF) {
            uint16_t *projection_weight_data_fp16 = (uint16_t *)projection_weight_data;
            uint16_t *projection_bias_data_fp16   = (uint16_t *)projection_bias_data;
            //补齐
            uint16_t projection_data[4] = {
                projection_weight_data_fp16[0],
                projection_weight_data_fp16[1],
                projection_weight_data_fp16[2],
                projection_bias_data_fp16[0],
            };

            buffer_projection_ = [device newBufferWithBytes:(const void *)projection_data
                                                     length:4 * sizeof(uint16_t)
                                                    options:MTLResourceCPUCacheModeWriteCombined];
        }
#endif
    }

    return TNN_OK;
}

Status MetalHDRGuideLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto context_impl = context_->getMetalContextImpl();
    if (!context_impl) {
        LOGE("context_impl is nil\n");
        return Status(TNNERR_CONTEXT_ERR, "MetalHDRGuideLayerAcc context_impl is nil");
    }

    auto encoder = [context_impl encoder];
    if (!encoder) {
        LOGE("encoder is nil\n");
        return Status(TNNERR_CONTEXT_ERR, "MetalHDRGuideLayerAcc encoder is nil");
    }

    encoder.label = GetKernelLabel();

    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_output  = output->GetBlobDesc().dims;
    auto output_width = dims_output[3], output_height = dims_output[2],
         output_slice = UP_DIV(dims_output[1], 4) * dims_output[0];

    Status status = TNN_OK;
    MetalBandwidth bandwidth;

    do {
        status = [context_impl load:@"hdr_guide" encoder:encoder bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);

        MTLSize threads = {(NSUInteger)output_height * output_width, (NSUInteger)output_slice, 1};

        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                    offset:(NSUInteger)input->GetHandle().bytes_offset
                   atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                    offset:(NSUInteger)output->GetHandle().bytes_offset
                   atIndex:1];
        [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
        [encoder setBuffer:buffer_ccm_weight_ offset:0 atIndex:3];
        [encoder setBuffer:buffer_ccm_bias_ offset:0 atIndex:4];
        [encoder setBuffer:buffer_slopes_ offset:0 atIndex:5];
        [encoder setBuffer:buffer_shifts_ offset:0 atIndex:6];
        [encoder setBuffer:buffer_projection_ offset:0 atIndex:7];

        status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
    } while (0);

    [encoder endEncoding];
    [context_impl commit];
    TNN_PRINT_ENCODER(context_, encoder, this);
    return status;
}

REGISTER_METAL_ACC(HDRGuide, LAYER_HDRGUIDE);

} // namespace TNN_NS
