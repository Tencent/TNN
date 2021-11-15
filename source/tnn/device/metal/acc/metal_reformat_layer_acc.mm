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

#include "tnn/device/metal/acc/metal_reformat_layer_acc.h"

#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/half_utils_inner.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

Status MetalReformatLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(MetalLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);

    auto reformat_param = dynamic_cast<ReformatLayerParam *>(param);
    CHECK_PARAM_NULL(reformat_param);

    if (reformat_param->src_format == DATA_FORMAT_NC4HW4 && reformat_param->dst_format == DATA_FORMAT_NCHW) {
        // nc4hw4 to nchw
        ;
    } else if (reformat_param->src_format == DATA_FORMAT_NCHW && reformat_param->dst_format == DATA_FORMAT_NC4HW4) {
        // nchw to nc4hw4
        ;
    } else {
        LOGE("MetalReformatLayerAcc::Init Error: src_fmt: %d, dst_fmt: %d, src_type: %d, dst_type: %d\n",
             reformat_param->src_format, reformat_param->dst_format, reformat_param->src_type,
             reformat_param->dst_type);
        return Status(TNNERR_MODEL_ERR, "MetalReformatLayerAcc::Init unsupport reformat type");
    }
    return AllocateBufferParam(inputs, outputs);
}

Status MetalReformatLayerAcc::UpdateBlobDataType(const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs) {
    // TODO: check how to set datatype for reformat
    outputs[0]->GetBlobDesc().data_type = inputs[0]->GetBlobDesc().data_type;
    return TNN_OK;
}

std::vector<DataFormat> MetalReformatLayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    std::vector<DataFormat> support_list{DATA_FORMAT_NC4HW4, DATA_FORMAT_NCHW};
    return support_list;
}

Status MetalReformatLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                  const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto dims     = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalImageConverterParams metal_params;
        metal_params.width   = DimsFunctionUtils::GetDimProduct(dims, 3);
        metal_params.height  = DimsFunctionUtils::GetDim(dims, 2);
        metal_params.size    = metal_params.height * metal_params.width;
        metal_params.channel = dims[1];
        metal_params.slice   = UP_DIV(metal_params.channel, 4);
        metal_params.batch   = dims[0];
        buffer_param_        = [device newBufferWithBytes:(const void *)(&metal_params)
                                                length:sizeof(metal_params)
                                               options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}
    
Status MetalReformatLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                                const std::vector<Blob *> &outputs,
                                                MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = MTLSizeMake(DimsFunctionUtils::GetDimProduct(dims_output, 2), UP_DIV(dims_output[1], 4), dims_output[0]);
    return TNN_OK;
}
    
std::string MetalReformatLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto reformat_param = dynamic_cast<ReformatLayerParam *>(param_);
    const auto data_type = inputs[0]->GetBlobDesc().data_type;
    if (reformat_param->src_format == DATA_FORMAT_NCHW) {
        return data_type == DATA_TYPE_INT32? "nchw_buffer_nc4hw4_buffer_int32" : "nchw_buffer_nc4hw4_buffer";
    }
    return data_type == DATA_TYPE_INT32? "nc4hw4_buffer_nchw_buffer_int32" : "nc4hw4_buffer_nchw_buffer";
}
    
Status MetalReformatLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(Reformat, LAYER_REFORMAT)
REGISTER_METAL_LAYOUT(LAYER_REFORMAT, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS

