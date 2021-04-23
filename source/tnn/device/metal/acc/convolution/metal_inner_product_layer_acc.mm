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

#include "tnn/device/metal/acc/convolution/metal_inner_product_layer_acc.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils_inner.h"

namespace TNN_NS {

Status MetalInnerProductLayerAcc::Init(Context *context, LayerParam *param,
                               LayerResource *resource,
                               const std::vector<Blob *> &inputs,
                               const std::vector<Blob *> &outputs) {
    Status status = isSupported(param, resource, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    return MetalLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status MetalInnerProductLayerAcc::isSupported(LayerParam *param, LayerResource *resource,
                                      const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    
    auto layer_param = dynamic_cast<InnerProductLayerParam *>(param);
    if (!layer_param || layer_param->axis != 1) {
        LOGE("MetalInnerProductLayerAcc do not support axis!=1 \n");
        return Status(TNNERR_LAYER_ERR, "MetalInnerProductLayerAcc do not support axis!=1");
    }
    
    auto layer_res = dynamic_cast<InnerProductLayerResource *>(resource);
    if (!layer_res) {
        LOGE("InnerProductLayerResource is invalid \n");
        return Status(TNNERR_LAYER_ERR, "InnerProductLayerResource is invalid");
    }
    return TNN_OK;
}

MetalInnerProductLayerAcc::~MetalInnerProductLayerAcc() {}

Status
MetalInnerProductLayerAcc::AllocateBufferWeight(const std::vector<Blob *> &inputs,
                                             const std::vector<Blob *> &outputs) {
    Status status = isSupported(param_, resource_, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    
    auto layer_res = dynamic_cast<InnerProductLayerResource *>(resource_);
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output  = outputs[0]->GetBlobDesc().dims;
    const int input_channel = dims_input[1];
    const int output_channel = dims_output[1];
    
    const int kh = DimsFunctionUtils::GetDim(dims_input, 2);
    const int kw = DimsFunctionUtils::GetDim(dims_input, 3);
    
    if (!buffer_weight_) {
        buffer_weight_ = AllocatePackedGOIHW16MetalBufferFormRawBuffer(layer_res->weight_handle,
                                                                {output_channel, input_channel, kh, kw},
                                                                1, status);
    }
    return status;
}

Status
MetalInnerProductLayerAcc::AllocateBufferBias(const std::vector<Blob *> &inputs,
                                           const std::vector<Blob *> &outputs) {
    Status status = isSupported(param_, resource_, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    
    auto layer_param  = dynamic_cast<InnerProductLayerParam *>(param_);
    auto layer_res = dynamic_cast<InnerProductLayerResource *>(resource_);

    // buffer_bias_
    if (!buffer_bias_) {
        if (layer_param->has_bias) {
            auto dims_output = outputs[0]->GetBlobDesc().dims;
            buffer_bias_ = AllocateMetalBufferFormRawBuffer1D(layer_res->bias_handle,
                                                              dims_output[1], status);
        } else {
            //防止bind时候为空
            buffer_bias_ = buffer_weight_;
        }
    }
    return status;
}

Status
MetalInnerProductLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                       const std::vector<Blob *> &outputs) {
    Status status = isSupported(param_, resource_, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    
    id<MTLDevice> device       = [TNNMetalDeviceImpl sharedDevice];
    auto param = dynamic_cast<InnerProductLayerParam *>(param_);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalInnerProductParams metal_params;
        metal_params.has_bias   = param->has_bias;

        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        buffer_param_ =
            [device newBufferWithBytes:(const void *)(&metal_params)
                                length:sizeof(MetalInnerProductParams)
                               options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalInnerProductLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                  const std::vector<Blob *> &outputs) {
    Status status = isSupported(param_, resource_, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    
    return MetalConvLayerCommon::Reshape(inputs, outputs);
}

std::string MetalInnerProductLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "inner_product";
}

Status MetalInnerProductLayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
    [encoder setBuffer:buffer_weight_
                offset:0
               atIndex:3];
    [encoder setBuffer:buffer_bias_
                offset:0
               atIndex:4];
    return TNN_OK;
}

Status MetalInnerProductLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    return MetalLayerAcc::ComputeThreadSize(inputs, outputs, size);
}

Status MetalInnerProductLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                  const std::vector<Blob *> &outputs) {
    Status status = isSupported(param_, resource_, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(InnerProduct, LAYER_INNER_PRODUCT);
REGISTER_METAL_LAYOUT(LAYER_INNER_PRODUCT, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
