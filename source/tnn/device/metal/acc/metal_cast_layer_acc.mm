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
#include "tnn/device/metal/acc/metal_cast_layer_acc.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
const static auto isFloat = [](DataType data_type) {
        return data_type == DATA_TYPE_HALF || data_type == DATA_TYPE_FLOAT;
};

Status MetalCastLayerAcc::UpdateBlobDataType(const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<CastLayerParam *>(param_);
    DataType to_datatype = static_cast<DataType>(layer_param->to);
#if TNN_METAL_FULL_PRECISION
    outputs[0]->GetBlobDesc().data_type = isFloat(to_datatype)? DATA_TYPE_FLOAT : to_datatype;
#else
    outputs[0]->GetBlobDesc().data_type = isFloat(to_datatype)? DATA_TYPE_HALF  : to_datatype;
#endif
    return TNN_OK;
}

Status MetalCastLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalCastLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto layer_param     = dynamic_cast<CastLayerParam *>(param_);
    const auto dims_input = inputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalCastParams metal_params;
        metal_params.input_size  = DimsFunctionUtils::GetDimProduct(dims_input, 2);
        metal_params.input_slice = UP_DIV(dims_input[1], 4);
        metal_params.batch       = dims_input[0];

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalCastParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

std::string MetalCastLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<CastLayerParam *>(param_);
    const auto input_data_type  = inputs[0]->GetBlobDesc().data_type;
    const auto output_data_type = outputs[0]->GetBlobDesc().data_type;

    if (input_data_type == output_data_type) {
        const auto size_bytes = DataTypeUtils::GetBytesSize(input_data_type);
        if (size_bytes == 2)
            return "cast_same_bytes2";
        else if(size_bytes == 4)
            return "cast_same_bytes4";
    } else if (isFloat(input_data_type) && output_data_type == DATA_TYPE_INT32) {
        return "cast_ftype_to_int32";
    } else if (input_data_type == DATA_TYPE_INT32 && isFloat(output_data_type)) {
        return "cast_int32_to_ftype";
    } else if (input_data_type == DATA_TYPE_INT32 && output_data_type == DATA_TYPE_UINT32) {
        return "cast_int32_to_uint32";
    } else if (input_data_type == DATA_TYPE_UINT32 && output_data_type == DATA_TYPE_INT32) {
        return "cast_uint32_to_int32";
    }

    LOGE("unsupport data type to cast\n");
    return "";
}

Status MetalCastLayerAcc::SetKernelEncoderParam(id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

Status MetalCastLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    const auto dims_input = inputs[0]->GetBlobDesc().dims;
    size = MTLSizeMake(DimsFunctionUtils::GetDimProduct(dims_input, 2),
                        UP_DIV(dims_input[1], 4),
                        dims_input[0]);
    return TNN_OK;
}

Status MetalCastLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto context_impl = context_->getMetalContextImpl();
    auto encoder = [context_impl encoder];
    
    MTLSize threads;
    auto status = ComputeThreadSize(inputs, outputs, threads);
    RETURN_ON_NEQ(status, TNN_OK);
    
    do {
        auto kernel_name = KernelName(inputs, outputs);
        
        MetalBandwidth bandwidth;
        status = [context_impl load:[NSString stringWithUTF8String:kernel_name.c_str()]
                            encoder:encoder
                          bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
        
        status = SetKernelEncoderParam(encoder, inputs, outputs);
        BREAK_IF(status != TNN_OK);
        status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
    } while (0);

    [encoder endEncoding];
    
    if (status == TNN_OK) {
        [context_impl commit];
        TNN_PRINT_ENCODER(context_, encoder, this);
    }
    return status;
}

REGISTER_METAL_ACC(Cast, LAYER_CAST);
REGISTER_METAL_LAYOUT(LAYER_CAST, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
