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
#include "tnn/device/metal/acc/metal_gather_layer_acc.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/half_utils.h"

namespace TNN_NS {

Status MetalGatherLayerAcc::UpdateBlobDataType(const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs) {
    int blob_idx = 0;
    auto layer_param = dynamic_cast<GatherLayerParam *>(param_);

    if (!layer_param->data_in_resource) {
#if TNN_METAL_FULL_PRECISION
        inputs[blob_idx++]->GetBlobDesc().data_type  = DATA_TYPE_FLOAT;
#else
        inputs[blob_idx++]->GetBlobDesc().data_type  = DATA_TYPE_HALF;
#endif
    }

    if (!layer_param->indices_in_resource) {
        inputs[blob_idx++]->GetBlobDesc().data_type = DATA_TYPE_INT32;
    }

#if TNN_METAL_FULL_PRECISION
    outputs[0]->GetBlobDesc().data_type  = DATA_TYPE_FLOAT;
#else
    outputs[0]->GetBlobDesc().data_type  = DATA_TYPE_HALF;
#endif

    return TNN_OK;
}

Status MetalGatherLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

std::vector<DataFormat> MetalGatherLayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    return {DATA_FORMAT_NCHW};
}

Status MetalGatherLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto layer_param     = dynamic_cast<GatherLayerParam *>(param_);
    auto layer_resource  = dynamic_cast<GatherLayerResource *>(resource_);
    if ((layer_param->data_in_resource || layer_param->indices_in_resource) && !layer_resource) {
        return Status(TNNERR_MODEL_ERR, "Gather resource is invalid");
    }
    const auto axis = layer_param->axis;

    DimsVector input_data_dims;
    void *input_data = nullptr;
    if (layer_param->data_in_resource) {
        input_data_dims = layer_resource->data.GetBufferDims();
        input_data = layer_resource->data.force_to<void *>();
    } else {
        input_data_dims = (*(inputs.begin()))->GetBlobDesc().dims;
    }

    DimsVector indices_dims;
    void *indices_data = nullptr;
    if (layer_param->indices_in_resource) {
        indices_dims = layer_resource->indices.GetBufferDims();
        indices_data = layer_resource->indices.force_to<void *>();
    } else {
        indices_dims = (*(inputs.rbegin()))->GetBlobDesc().dims;
    }

    auto dims_output = outputs[0]->GetBlobDesc().dims;

    //input_data_dims[1]   = UP_DIV(input_data_dims[1], 4);
    const int inner_size = DimsFunctionUtils::GetDimProduct(input_data_dims, axis+1);
    const int outer_size = DimsFunctionUtils::GetDimProduct(input_data_dims, 0, axis);
    int input_axis_size  = DimsFunctionUtils::GetDim(input_data_dims, axis);
    int output_axis_size = DimsVectorUtils::Count(indices_dims);
    if (DimsVectorUtils::Count(indices_dims) == 1 && dims_output.size() < input_data_dims.size()) {
        dims_output.insert(dims_output.begin()+axis, 1);
        output_axis_size = DimsFunctionUtils::GetDim(dims_output, axis);
    }
    // buffer_param_
    {
        MetalGatherParams metal_params;
        metal_params.inner_size = inner_size;
        metal_params.outer_size = outer_size;
        metal_params.input_axis_size  = input_axis_size;
        metal_params.output_axis_size = output_axis_size;
        //metal_params.input_slice  = input_data_dims[1];
        //metal_params.output_slice = UP_DIV(dims_output[1], 4);

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalGatherParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    // buffer data
    if (layer_param->data_in_resource && buffer_data_ == nil) {
        const auto data_count = DimsVectorUtils::Count(input_data_dims);
        const auto data_type  = layer_resource->data.GetDataType();
        auto data_type_size   = 0;
        std::shared_ptr<void> data_cast_type = nullptr;
#if TNN_METAL_FULL_PRECISION
        if (data_type == DATA_TYPE_HALF) {
            data_cast_type.reset(new float[data_count], [](float *p){delete[] p;});
            if (ConvertFromHalfToFloat(input_data, (float *)data_cast_type.get(), data_count) != 0) {
                LOGE("Error: DataType %d not support\n", data_type);
                return Status(TNNERR_MODEL_ERR, "Convert Data in LayerRerouece from half to float failed!");
            }
            input_data = data_cast_type.get();
        }
        data_type_size = sizeof(float);
#else
        if (data_type == DATA_TYPE_FLOAT) {
            data_cast_type.reset(new uint16_t[data_count], [](uint16_t *p){delete[] p;});
            if (ConvertFromFloatToHalf((float *)input_data, data_cast_type.get(), data_count) != 0) {
                LOGE("Error: DataType %d not support\n", data_type);
                return Status(TNNERR_MODEL_ERR, "Convert Data in LayerRerouece from float to half failed!");
            }
            input_data = data_cast_type.get();
        }
        data_type_size = sizeof(uint16_t);
#endif
        buffer_data_ = [device newBufferWithBytes:(const void *)input_data
                                           length:data_type_size*data_count
                                          options:MTLResourceCPUCacheModeWriteCombined];
    }
    // buffer indices
    if (layer_param->indices_in_resource && buffer_indices_ == nil) {
        buffer_indices_ = [device newBufferWithBytes:(const void *)indices_data
                                              length:sizeof(int)*output_axis_size
                                             options:MTLResourceCPUCacheModeWriteCombined];
    }
    // save threads shape
    threads_shape_ = MTLSizeMake(inner_size, output_axis_size, outer_size);
    return TNN_OK;
}

std::string MetalGatherLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<GatherLayerParam *>(param_);
    /*
    if (layer_param->axis == 1)
        return "gather_axis_1";
    return "gather_common";
    */
    return "gather_common_nchw";
}

Status MetalGatherLayerAcc::SetKernelEncoderParam(id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    int blob_idx = 0;
    auto layer_param     = dynamic_cast<GatherLayerParam *>(param_);
    if (layer_param->data_in_resource) {
        [encoder setBuffer:buffer_data_
                    offset:(NSUInteger)0
                   atIndex:0];
    } else {
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputs[blob_idx]->GetHandle().base
                    offset:(NSUInteger)inputs[blob_idx]->GetHandle().bytes_offset
                   atIndex:0];
        blob_idx += 1;
    }

    if (layer_param->indices_in_resource) {
        [encoder setBuffer:buffer_indices_
                    offset:(NSUInteger)0
                   atIndex:1];
    } else {
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputs[blob_idx]->GetHandle().base
                    offset:(NSUInteger)inputs[blob_idx]->GetHandle().bytes_offset
                   atIndex:1];
        blob_idx += 1;
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)outputs[0]->GetHandle().base
                        offset:(NSUInteger)outputs[0]->GetHandle().bytes_offset
                       atIndex:2];
    [encoder setBuffer:buffer_param_ offset:0 atIndex:3];

    return TNN_OK;
}

Status MetalGatherLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    size = threads_shape_;
    return TNN_OK;
}

Status MetalGatherLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(Gather, LAYER_GATHER);
REGISTER_METAL_LAYOUT(LAYER_GATHER, DATA_FORMAT_NCHW);

} // namespace TNN_NS
