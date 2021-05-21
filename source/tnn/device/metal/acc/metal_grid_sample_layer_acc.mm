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

#include "tnn/device/metal/acc/metal_layer_acc.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

class MetalGridSampleLayerAcc : public MetalLayerAcc {
public:
    virtual ~MetalGridSampleLayerAcc(){};
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual std::string KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status ComputeThreadSize(const std::vector<Blob *> &inputs,
                             const std::vector<Blob *> &outputs,
                             MTLSize &size);
    virtual Status SetKernelEncoderParam(id<MTLComputeCommandEncoder> encoder,
                                 const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs);
};

Status MetalGridSampleLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalGridSampleLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                           const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto *layer_param = dynamic_cast<GridSampleLayerParam *>(param_);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);

        buffer_param_ =
            [device newBufferWithBytes:(const void *)(&metal_params)
                                length:sizeof(MetalParams)
                               options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalGridSampleLayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    auto input_blob  = inputs[0];
    auto grid_blob   = inputs[1];
    auto output_blob = outputs[0];

    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input_blob->GetHandle().base
                offset:(NSUInteger)input_blob->GetHandle().bytes_offset
               atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)grid_blob->GetHandle().base
                offset:(NSUInteger)grid_blob->GetHandle().bytes_offset
               atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output_blob->GetHandle().base
                offset:(NSUInteger)output_blob->GetHandle().bytes_offset
               atIndex:2];
    [encoder setBuffer:buffer_param_ offset:0 atIndex:3];

    return TNN_OK;
}

Status MetalGridSampleLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, false);
    return TNN_OK;
}

std::string MetalGridSampleLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "grid_sample";
}

Status MetalGridSampleLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    auto param      = dynamic_cast<GridSampleLayerParam *>(param_);
    auto input_blob = inputs[0];
    auto grid_blob  = inputs[1];

    const auto& input_dims = input_blob->GetBlobDesc().dims;
    const auto& grid_dims  = grid_blob->GetBlobDesc().dims;
    if (!(input_dims.size() == 4 &&  grid_dims.size() == 4 && param->mode == 2 && param->pad_type == 0 && param->align_corners == 0)) {
        LOGE("Error: Metal GridSample layer acc doesn't support GridSample input dim size:(%lu), grid dim size:(%lu), or param:(%d, %d, %d)\n",
            input_dims.size(), grid_dims.size(), param->mode, param->pad_type, param->align_corners);
        return Status(TNNERR_MODEL_ERR, "Error: Metal GridSample layer acc doesn't support.\n");
    }
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(GridSample, LAYER_GRIDSAMPLE);
REGISTER_METAL_LAYOUT(LAYER_GRIDSAMPLE, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
